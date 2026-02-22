#!/usr/bin/env python3
"""
roboflow_provider.py — Roboflow Hosted API provider (cloud inference)

Provider module in the same spirit as camera_provider / lidar_provider.

Requirements:
  pip install inference-sdk
  export ROBOFLOW_API_KEY="..."

Model id format:
  "<project>/<version>"   e.g., "object-detection-in-sand/3"

Notes:
- Hosted inference adds latency. This provider uses min_interval_s + last-good cache.
- Fail-soft by default: returns last-good result on transient errors if available.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration  # type: ignore
except Exception:  # pragma: no cover
    InferenceHTTPClient = None  # type: ignore
    InferenceConfiguration = None  # type: ignore


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


@dataclass(frozen=True)
class RFDetection:
    cls: str
    conf: float
    box: BoxXYXY
    cls_id: Optional[int] = None


@dataclass(frozen=True)
class RFInference:
    ts: float
    ok: bool
    detections: Tuple[RFDetection, ...]
    raw: Optional[Dict[str, Any]] = None
    error: str = ""


def default_scoop_roi(frame_w: int, frame_h: int) -> BoxXYXY:
    # Lower-middle ROI; tune once mount is fixed
    return BoxXYXY(0.35 * frame_w, 0.55 * frame_h, 0.65 * frame_w, 0.95 * frame_h)


def _parse_predictions(raw: Dict[str, Any]) -> Tuple[RFDetection, ...]:
    preds = raw.get("predictions", [])
    if not isinstance(preds, list):
        return tuple()

    out: List[RFDetection] = []
    for p in preds:
        if not isinstance(p, dict):
            continue

        cls = str(p.get("class", "unknown"))
        try:
            conf = float(p.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        try:
            cx = float(p["x"])
            cy = float(p["y"])
            ww = float(p["width"])
            hh = float(p["height"])
        except Exception:
            continue

        box = BoxXYXY(cx - ww / 2.0, cy - hh / 2.0, cx + ww / 2.0, cy + hh / 2.0)

        cls_id = p.get("class_id", None)
        try:
            cls_id_i = int(cls_id) if cls_id is not None else None
        except Exception:
            cls_id_i = None

        out.append(RFDetection(cls=cls, conf=conf, box=box, cls_id=cls_id_i))

    return tuple(out)


class RoboflowProvider:
    def __init__(
        self,
        *,
        model_id: str,
        api_url: str = "https://detect.roboflow.com",
        api_key: Optional[str] = None,
        api_key_env: str = "ROBOFLOW_API_KEY",
        min_interval_s: float = 0.50,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        use_api_v0: bool = True,
        use_last_good_on_error: bool = True,
    ):
        self.model_id = str(model_id)
        self.api_url = str(api_url).rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.min_interval_s = float(min_interval_s)

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_api_v0 = bool(use_api_v0)
        self.use_last_good_on_error = bool(use_last_good_on_error)

        self._client = None
        self._last_call_ts = 0.0
        self._last_good: Optional[RFInference] = None

        if InferenceHTTPClient is not None and self.api_key:
            self._client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)

            # Defensive: inference-sdk versions differ
            try:
                if self.use_api_v0 and hasattr(self._client, "select_api_v0"):
                    self._client.select_api_v0()
            except Exception:
                pass

            # Optional thresholds if supported by SDK version
            try:
                if InferenceConfiguration is not None and (confidence_threshold is not None or iou_threshold is not None):
                    cfg = InferenceConfiguration(confidence_threshold=confidence_threshold, iou_threshold=iou_threshold)
                    if hasattr(self._client, "configure"):
                        self._client.configure(cfg)
            except Exception:
                pass

    def available(self) -> bool:
        return self._client is not None

    def infer(self, image: Any) -> RFInference:
        now = time.time()

        if (now - self._last_call_ts) < self.min_interval_s and self._last_good is not None:
            return self._last_good
        self._last_call_ts = now

        if self._client is None:
            return RFInference(
                ts=now,
                ok=False,
                detections=(),
                raw=None,
                error="Roboflow client unavailable (install inference-sdk and set ROBOFLOW_API_KEY)",
            )

        try:
            raw = self._client.infer(image, model_id=self.model_id)
            if not isinstance(raw, dict):
                return RFInference(ts=now, ok=False, detections=(), raw=None, error="Unexpected inference output type")
            dets = _parse_predictions(raw)
            inf = RFInference(ts=now, ok=True, detections=dets, raw=raw, error="")
            self._last_good = inf
            return inf
        except Exception as e:
            if self.use_last_good_on_error and self._last_good is not None:
                return self._last_good
            return RFInference(ts=now, ok=False, detections=(), raw=None, error=str(e))

    # ---- helper rules ----

    def any_class_present(self, inf: RFInference, classes: Sequence[str], *, conf_min: float = 0.25) -> bool:
        s = set(classes)
        return any((d.cls in s and d.conf >= conf_min) for d in inf.detections)

    def hazards_in_roi(
        self,
        inf: RFInference,
        *,
        roi: BoxXYXY,
        hazard_classes: Sequence[str],
        conf_min: float = 0.25,
        min_iou: float = 0.05,
    ) -> Tuple[RFDetection, ...]:
        hc = set(hazard_classes)
        out: List[RFDetection] = []
        for d in inf.detections:
            if d.cls not in hc or d.conf < conf_min:
                continue
            if d.box.intersects(roi) and d.box.iou(roi) >= min_iou:
                out.append(d)
        return tuple(out)

    def large_rocks_in_roi(
        self,
        inf: RFInference,
        *,
        roi: BoxXYXY,
        rock_class: str = "rock",
        frame_w: int = 640,
        frame_h: int = 360,
        conf_min: float = 0.25,
        area_frac_thresh: float = 0.015,
    ) -> Tuple[RFDetection, ...]:
        frame_area = float(max(frame_w, 1) * max(frame_h, 1))
        out: List[RFDetection] = []
        for d in inf.detections:
            if d.cls != rock_class or d.conf < conf_min:
                continue
            if not d.box.intersects(roi):
                continue
            if (d.box.area() / frame_area) >= area_frac_thresh:
                out.append(d)
        return tuple(out)

    def close(self) -> None:
        pass
