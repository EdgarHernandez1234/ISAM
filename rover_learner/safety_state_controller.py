#!/usr/bin/env python3
"""
safety_state_controller.py

A modular, importable safety/task state classifier for live rover demos.

It consumes:
- Roboflow inference results (detections with class, confidence, and bounding box)
- LiDAR min_distance_m (optional)
- Frame geometry (w, h) to compute ROIs

It outputs a debounced (hysteresis) state:
  Approach | Bypass | Harvest | Low_Hazard | High_Hazard

Design principles:
- Only High_Hazard is "critical" (hard stop).
- Low_Hazard is a soft bypass / no-scoop zone (system stays live).
- Object hazard => hard_stop (High_Hazard)
- Human hazard + Bypass => soft_bypass (Low_Hazard / no-scoop)

Works with rover_learner's roboflow_provider.RFInference, but is defensive and will
operate with any object exposing a `.detections` iterable where each detection has:
  - .cls (str)
  - .conf (float)
  - .box with x1,y1,x2,y2 floats

Author: ALAM integration helper
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---- Optional import of shared box / inference types ----
try:
    from roboflow_provider import BoxXYXY, RFInference, RFDetection, default_scoop_roi as _default_scoop_roi  # type: ignore
except Exception:  # pragma: no cover
    @dataclass(frozen=True)
    class BoxXYXY:
        x1: float
        y1: float
        x2: float
        y2: float

        def area(self) -> float:
            return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

        def intersects(self, other: "BoxXYXY") -> bool:
            return not (self.x2 <= other.x1 or self.x1 >= other.x2 or self.y2 <= other.y1 or self.y1 >= other.y2)

        def iou(self, other: "BoxXYXY") -> float:
            ix1 = max(self.x1, other.x1)
            iy1 = max(self.y1, other.y1)
            ix2 = min(self.x2, other.x2)
            iy2 = min(self.y2, other.y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                return 0.0
            union = self.area() + other.area() - inter
            return float(inter / max(union, 1e-9))

    # Duck-typed placeholders
    class RFDetection:  # pragma: no cover
        cls: str
        conf: float
        box: BoxXYXY

    class RFInference:  # pragma: no cover
        detections: Iterable[RFDetection]

    def _default_scoop_roi(frame_w: int, frame_h: int) -> BoxXYXY:  # pragma: no cover
        return BoxXYXY(0.35 * frame_w, 0.55 * frame_h, 0.65 * frame_w, 0.95 * frame_h)


class SafetyState(str, Enum):
    APPROACH = "Approach"
    BYPASS = "Bypass"
    HARVEST = "Harvest"
    LOW_HAZARD = "Low_Hazard"
    HIGH_HAZARD = "High_Hazard"


class HazardLevel(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    HIGH = "HIGH"


@dataclass(frozen=True)
class ROIConfig:
    """
    ROI definitions in relative image coordinates [0..1].

    You can tune these based on camera mount and the actual "scoop workspace".
    """
    # Forward path corridor (drive / approach)
    path_x1: float = 0.25
    path_y1: float = 0.30
    path_x2: float = 0.75
    path_y2: float = 0.88

    # Scoop workspace is imported from roboflow_provider.default_scoop_roi by default.
    # You can override with these if desired (set use_default_scoop_roi=False).
    scoop_x1: float = 0.35
    scoop_y1: float = 0.55
    scoop_x2: float = 0.65
    scoop_y2: float = 0.95
    use_default_scoop_roi: bool = True

    def path_roi(self, w: int, h: int) -> BoxXYXY:
        return BoxXYXY(self.path_x1 * w, self.path_y1 * h, self.path_x2 * w, self.path_y2 * h)

    def scoop_roi(self, w: int, h: int) -> BoxXYXY:
        if self.use_default_scoop_roi:
            return _default_scoop_roi(w, h)
        return BoxXYXY(self.scoop_x1 * w, self.scoop_y1 * h, self.scoop_x2 * w, self.scoop_y2 * h)


@dataclass(frozen=True)
class Thresholds:
    # Confidence minimums per class label
    conf_object_hazard: float = 0.45
    conf_human_hazard: float = 0.40
    conf_bypass_zone: float = 0.40
    conf_harvestable: float = 0.45

    # IOU threshold for "in ROI" membership
    min_iou_roi: float = 0.05

    # Distance gates (meters)
    stop_dist_object_m: float = 1.20
    harvest_dist_min_m: float = 0.35
    harvest_dist_max_m: float = 0.80
    approach_dist_min_m: float = 0.80
    approach_dist_max_m: float = 3.00

    # Optional: hard stop on any obstacle distance alone (disabled by default)
    # If set (e.g., 0.6), then min_distance_m <= this triggers High_Hazard even without a detection.
    stop_on_any_dist_m: Optional[float] = None


@dataclass(frozen=True)
class ClassLabels:
    bypass_zone: str = "Bypass"
    harvestable: str = "harvestable"
    human_hazard: str = "human hazard"
    object_hazard: str = "object hazard"


@dataclass(frozen=True)
class Debounce:
    enter_frames: int = 2
    exit_frames: int = 3


@dataclass(frozen=True)
class StateConfig:
    roi: ROIConfig = ROIConfig()
    thr: Thresholds = Thresholds()
    labels: ClassLabels = ClassLabels()
    debounce: Debounce = Debounce()


@dataclass
class StateOutput:
    state: SafetyState
    hazard_level: HazardLevel
    task_intent: SafetyState  # APPROACH/HARVEST/BYPASS (even when hazards override)
    hard_stop: bool
    soft_bypass: bool
    reasons: Tuple[str, ...] = ()
    # Optional: useful for overlays/logging
    debug: Dict[str, Any] = field(default_factory=dict)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _detections(inf: Any) -> List[Any]:
    dets = getattr(inf, "detections", None)
    if dets is None:
        return []
    try:
        return list(dets)
    except Exception:
        return []


def _det_cls(d: Any) -> str:
    return str(getattr(d, "cls", ""))


def _det_conf(d: Any) -> float:
    try:
        return float(getattr(d, "conf", 0.0))
    except Exception:
        return 0.0


def _det_box(d: Any) -> Optional[BoxXYXY]:
    b = getattr(d, "box", None)
    if b is None:
        return None
    # Ensure attributes exist
    for k in ("x1", "y1", "x2", "y2"):
        if not hasattr(b, k):
            return None
    try:
        return BoxXYXY(float(b.x1), float(b.y1), float(b.x2), float(b.y2))
    except Exception:
        return None


def _in_roi(d: Any, roi: BoxXYXY, *, min_iou: float) -> bool:
    box = _det_box(d)
    if box is None:
        return False
    if not box.intersects(roi):
        return False
    return box.iou(roi) >= min_iou


def _max_conf_by_class(dets: Iterable[Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for d in dets:
        c = _det_cls(d)
        conf = _det_conf(d)
        prev = out.get(c, 0.0)
        if conf > prev:
            out[c] = conf
    return out


def _any_in_roi(
    dets: Iterable[Any],
    cls_name: str,
    roi: BoxXYXY,
    *,
    conf_min: float,
    min_iou: float,
) -> bool:
    for d in dets:
        if _det_cls(d) != cls_name:
            continue
        if _det_conf(d) < conf_min:
            continue
        if _in_roi(d, roi, min_iou=min_iou):
            return True
    return False


class _DebouncedFlag:
    """
    Debounces a raw boolean with independent enter/exit frame counts.
    """
    def __init__(self, *, enter_frames: int, exit_frames: int) -> None:
        self.enter_frames = max(1, int(enter_frames))
        self.exit_frames = max(1, int(exit_frames))
        self.active: bool = False
        self._enter = 0
        self._exit = 0

    def update(self, raw_true: bool) -> bool:
        if raw_true:
            self._enter += 1
            self._exit = 0
            if not self.active and self._enter >= self.enter_frames:
                self.active = True
        else:
            self._exit += 1
            self._enter = 0
            if self.active and self._exit >= self.exit_frames:
                self.active = False
        return self.active

    def reset(self) -> None:
        self.active = False
        self._enter = 0
        self._exit = 0


class SafetyStateController:
    """
    Stateful classifier. Call `update()` once per frame.
    """
    def __init__(self, config: Optional[StateConfig] = None) -> None:
        self.cfg = config or StateConfig()

        df = self.cfg.debounce
        self._high = _DebouncedFlag(enter_frames=df.enter_frames, exit_frames=df.exit_frames)
        self._low = _DebouncedFlag(enter_frames=df.enter_frames, exit_frames=df.exit_frames)
        self._harvest = _DebouncedFlag(enter_frames=df.enter_frames, exit_frames=df.exit_frames)
        self._approach = _DebouncedFlag(enter_frames=df.enter_frames, exit_frames=df.exit_frames)

        self._last: Optional[StateOutput] = None

    def reset(self) -> None:
        self._high.reset()
        self._low.reset()
        self._harvest.reset()
        self._approach.reset()
        self._last = None

    def update(
        self,
        *,
        inf: Any,
        frame_w: int,
        frame_h: int,
        min_distance_m: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> StateOutput:
        """
        Compute and debounce state for the current frame.

        Parameters
        ----------
        inf : RFInference-like
            Must expose `.detections` iterable.
        frame_w, frame_h : int
            Frame dimensions.
        min_distance_m : float | None
            LiDAR forward-sector minimum distance.
        timestamp : float | None
            Optional; stored into debug only.

        Returns
        -------
        StateOutput
        """
        cfg = self.cfg
        dets = _detections(inf)
        max_conf = _max_conf_by_class(dets)

        path_roi = cfg.roi.path_roi(frame_w, frame_h)
        scoop_roi = cfg.roi.scoop_roi(frame_w, frame_h)

        lbl = cfg.labels
        thr = cfg.thr

        # ROI membership flags
        obj_in_path = _any_in_roi(dets, lbl.object_hazard, path_roi, conf_min=thr.conf_object_hazard, min_iou=thr.min_iou_roi)
        obj_in_scoop = _any_in_roi(dets, lbl.object_hazard, scoop_roi, conf_min=thr.conf_object_hazard, min_iou=thr.min_iou_roi)

        human_in_path = _any_in_roi(dets, lbl.human_hazard, path_roi, conf_min=thr.conf_human_hazard, min_iou=thr.min_iou_roi)
        human_in_scoop = _any_in_roi(dets, lbl.human_hazard, scoop_roi, conf_min=thr.conf_human_hazard, min_iou=thr.min_iou_roi)

        bypass_in_path = _any_in_roi(dets, lbl.bypass_zone, path_roi, conf_min=thr.conf_bypass_zone, min_iou=thr.min_iou_roi)
        bypass_in_scoop = _any_in_roi(dets, lbl.bypass_zone, scoop_roi, conf_min=thr.conf_bypass_zone, min_iou=thr.min_iou_roi)

        harvest_in_path = _any_in_roi(dets, lbl.harvestable, path_roi, conf_min=thr.conf_harvestable, min_iou=thr.min_iou_roi)
        harvest_in_scoop = _any_in_roi(dets, lbl.harvestable, scoop_roi, conf_min=thr.conf_harvestable, min_iou=thr.min_iou_roi)

        d = _safe_float(min_distance_m)

        # ---- RAW conditions (no debounce) ----
        # High hazard: object hazard is hard stop
        raw_high = bool(obj_in_path or obj_in_scoop)

        # Optional: hard stop on distance alone (if enabled)
        if thr.stop_on_any_dist_m is not None and d is not None and d <= thr.stop_on_any_dist_m:
            raw_high = True

        # Low hazard: soft bypass / no-scoop zones (human hazard + bypass zone)
        raw_low = bool(human_in_path or human_in_scoop or bypass_in_path or bypass_in_scoop)

        # Harvest: only if no hazards and in range
        in_harvest_range = (d is not None) and (thr.harvest_dist_min_m <= d <= thr.harvest_dist_max_m)
        raw_harvest = bool(harvest_in_scoop and in_harvest_range and (not raw_high) and (not raw_low))

        # Approach: harvestable exists but not in harvest state; within approach distance window (if LiDAR present)
        in_approach_range = (d is None) or (thr.approach_dist_min_m <= d <= thr.approach_dist_max_m)
        raw_approach = bool((harvest_in_path or max_conf.get(lbl.harvestable, 0.0) >= thr.conf_harvestable) and in_approach_range and (not raw_high) and (not raw_low) and (not raw_harvest))

        # ---- Debounced flags ----
        high_active = self._high.update(raw_high)
        # If high is active, low is irrelevant (but still update so it doesn't get stuck)
        low_active = self._low.update(raw_low and (not high_active))
        harvest_active = self._harvest.update(raw_harvest and (not high_active) and (not low_active))
        approach_active = self._approach.update(raw_approach and (not high_active) and (not low_active) and (not harvest_active))

        # ---- Task intent + final state ----
        # Intent ignores hazards; hazards override for state.
        if harvest_active:
            intent = SafetyState.HARVEST
        elif approach_active:
            intent = SafetyState.APPROACH
        else:
            intent = SafetyState.BYPASS

        if high_active:
            state = SafetyState.HIGH_HAZARD
            hazard_level = HazardLevel.HIGH
        elif low_active:
            state = SafetyState.LOW_HAZARD
            hazard_level = HazardLevel.LOW
        else:
            state = intent
            hazard_level = HazardLevel.NONE

        hard_stop = bool(high_active)
        soft_bypass = bool(low_active)

        # Reasons (human-readable, stable)
        reasons: List[str] = []
        if obj_in_path:
            reasons.append("OBJ_HAZARD_IN_PATH_ROI")
        if obj_in_scoop:
            reasons.append("OBJ_HAZARD_IN_SCOOP_ROI")
        if thr.stop_on_any_dist_m is not None and d is not None and d <= thr.stop_on_any_dist_m:
            reasons.append(f"LIDAR_STOP_ANY(d={d:.2f}<= {thr.stop_on_any_dist_m:.2f})")

        if human_in_path:
            reasons.append("HUMAN_HAZARD_IN_PATH_ROI")
        if human_in_scoop:
            reasons.append("HUMAN_HAZARD_IN_SCOOP_ROI")
        if bypass_in_path:
            reasons.append("BYPASS_ZONE_IN_PATH_ROI")
        if bypass_in_scoop:
            reasons.append("BYPASS_ZONE_IN_SCOOP_ROI")

        if harvest_in_scoop and in_harvest_range and not (hard_stop or soft_bypass):
            reasons.append("HARVESTABLE_IN_SCOOP_AND_IN_RANGE")
        elif harvest_in_path and not (hard_stop or soft_bypass):
            reasons.append("HARVESTABLE_IN_PATH")

        debug: Dict[str, Any] = {
            "ts": timestamp,
            "min_distance_m": d,
            "max_conf": dict(max_conf),
            "path_roi": path_roi,
            "scoop_roi": scoop_roi,
            "raw": {
                "high": raw_high,
                "low": raw_low,
                "harvest": raw_harvest,
                "approach": raw_approach,
            },
            "active": {
                "high": high_active,
                "low": low_active,
                "harvest": harvest_active,
                "approach": approach_active,
            },
            "roi_hits": {
                "object_hazard": {"path": obj_in_path, "scoop": obj_in_scoop},
                "human_hazard": {"path": human_in_path, "scoop": human_in_scoop},
                "bypass_zone": {"path": bypass_in_path, "scoop": bypass_in_scoop},
                "harvestable": {"path": harvest_in_path, "scoop": harvest_in_scoop},
            },
        }

        out = StateOutput(
            state=state,
            hazard_level=hazard_level,
            task_intent=intent,
            hard_stop=hard_stop,
            soft_bypass=soft_bypass,
            reasons=tuple(reasons),
            debug=debug,
        )
        self._last = out
        return out
