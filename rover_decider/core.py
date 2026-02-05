"""rover_decider.core

Pure-Python decision helpers for the rover demo.

Key outputs you want to log per decision frame:
- min_distance_m (LiDAR)
- YOLO pred_conf
- YOLO looks_dirty (0/1) OR class id string
- action (SCOOP/BYPASS)
- safety state (NORMAL/SAFE_HOLD/STOP)

Feature extraction (pure Python):
- x1 = clamp(min_distance_m, 0, 5)
- x2 = pred_conf
- x3 = looks_dirty
- x4 = lidar_valid (0/1) [optional]

All functions here are stdlib-only to keep unit tests lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple


# -----------------------------
# Defaults / knobs
# -----------------------------

MAX_LIDAR_FEATURE_M = 5.0

# Safety thresholds based on *too close* distances.
# (Far-away distances are handled by "max_scoop_dist" in the action policy.)
DEFAULT_SAFE_HOLD_DIST_M = 0.25
DEFAULT_STOP_DIST_M = 0.10

# Confidence threshold for deciding (policy-level; not a "safety" threshold)
DEFAULT_CONF_THRESH = 0.60


# -----------------------------
# Time helper
# -----------------------------

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# -----------------------------
# Numeric helpers
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def lidar_is_valid(min_distance_m: Optional[float]) -> bool:
    """Heuristic validity check for a LiDAR min-distance scalar."""
    if min_distance_m is None:
        return False
    try:
        v = float(min_distance_m)
    except (TypeError, ValueError):
        return False

    # Filter common invalid/sentinel values.
    if v <= 0.0:
        return False
    if v > 1000.0:
        return False
    return True


def compute_safety_state(
    min_distance_m: Optional[float],
    safe_hold_dist_m: float = DEFAULT_SAFE_HOLD_DIST_M,
    stop_dist_m: float = DEFAULT_STOP_DIST_M,
) -> str:
    """Distance-based safety state.

    - STOP: dangerously close
    - SAFE_HOLD: invalid LiDAR or close
    - NORMAL: otherwise
    """
    if not lidar_is_valid(min_distance_m):
        return "SAFE_HOLD"

    d = float(min_distance_m)
    if d <= float(stop_dist_m):
        return "STOP"
    if d <= float(safe_hold_dist_m):
        return "SAFE_HOLD"
    return "NORMAL"


# -----------------------------
# YOLO helper
# -----------------------------

def looks_dirty_from_class(pred_class: str) -> int:
    """Map a YOLO top-1 class string to looks_dirty 0/1.

    This keeps your current policy consistent:
      dirty/trash/plastic => looks_dirty=1
    """
    cls = (pred_class or "").lower()
    if any(w in cls for w in ("dirty", "dirt", "trash", "plastic", "contamin")):
        return 1
    return 0


# -----------------------------
# Feature extraction
# -----------------------------

def extract_features(
    min_distance_m: Optional[float],
    pred_conf: float,
    looks_dirty: int,
    include_lidar_valid: bool = True,
) -> Dict[str, float]:
    """Compute x1..x4 features in pure Python."""
    lv = 1 if lidar_is_valid(min_distance_m) else 0

    # If invalid, treat distance as "max" so the model sees "far/unknown".
    dist_for_x1 = float(min_distance_m) if lv else MAX_LIDAR_FEATURE_M

    x1 = clamp(dist_for_x1, 0.0, MAX_LIDAR_FEATURE_M)
    x2 = clamp(float(pred_conf), 0.0, 1.0)
    x3 = 1.0 if int(looks_dirty) else 0.0

    feats = {"x1": x1, "x2": x2, "x3": x3}
    if include_lidar_valid:
        feats["x4"] = float(lv)
    return feats


# -----------------------------
# Action selection
# -----------------------------

def choose_action(
    looks_dirty: int,
    pred_conf: float,
    min_distance_m: Optional[float],
    conf_thresh: float = DEFAULT_CONF_THRESH,
    max_scoop_dist: float = 2.5,
    safe_hold_dist_m: float = DEFAULT_SAFE_HOLD_DIST_M,
    stop_dist_m: float = DEFAULT_STOP_DIST_M,
) -> Tuple[str, str, str]:
    """Return (action, reason, safety_state).

    Policy:
    1) Safety state computed from LiDAR distance (invalid/too close)
       - if safety_state != NORMAL => BYPASS
    2) If looks_dirty==1 => BYPASS
    3) If pred_conf < conf_thresh => BYPASS
    4) If distance > max_scoop_dist => BYPASS
    5) Else => SCOOP
    """
    safety_state = compute_safety_state(
        min_distance_m,
        safe_hold_dist_m=safe_hold_dist_m,
        stop_dist_m=stop_dist_m,
    )

    if safety_state != "NORMAL":
        return "BYPASS", f"safety state {safety_state}", safety_state

    if int(looks_dirty) == 1:
        return "BYPASS", "looks dirty/contaminated", safety_state

    if float(pred_conf) < float(conf_thresh):
        return "BYPASS", f"low confidence ({pred_conf:.2f} < {conf_thresh})", safety_state

    if not lidar_is_valid(min_distance_m):
        # Defensive: should not happen if safety_state NORMAL, but keep consistent.
        return "BYPASS", "invalid LiDAR", "SAFE_HOLD"

    d = float(min_distance_m)
    if d > float(max_scoop_dist):
        return "BYPASS", f"too far away ({d:.2f} m > {max_scoop_dist} m)", safety_state

    return "SCOOP", "clean and confident within distance", safety_state


# -----------------------------
# Convenience struct
# -----------------------------

@dataclass(frozen=True)
class DecisionResult:
    pred_class: str
    pred_conf: float
    looks_dirty: int
    min_distance_m: Optional[float]
    action: str
    reason: str
    safety_state: str

    @staticmethod
    def build(
        pred_class: str,
        pred_conf: float,
        min_distance_m: Optional[float],
        conf_thresh: float = DEFAULT_CONF_THRESH,
        max_scoop_dist: float = 2.5,
        safe_hold_dist_m: float = DEFAULT_SAFE_HOLD_DIST_M,
        stop_dist_m: float = DEFAULT_STOP_DIST_M,
    ) -> "DecisionResult":
        looks_dirty = looks_dirty_from_class(pred_class)
        action, reason, safety_state = choose_action(
            looks_dirty=looks_dirty,
            pred_conf=pred_conf,
            min_distance_m=min_distance_m,
            conf_thresh=conf_thresh,
            max_scoop_dist=max_scoop_dist,
            safe_hold_dist_m=safe_hold_dist_m,
            stop_dist_m=stop_dist_m,
        )
        return DecisionResult(
            pred_class=str(pred_class),
            pred_conf=float(pred_conf),
            looks_dirty=int(looks_dirty),
            min_distance_m=min_distance_m if min_distance_m is None else float(min_distance_m),
            action=action,
            reason=reason,
            safety_state=safety_state,
        )
