"""
rover_learner.navigation.types

Shared datatypes for navigation + mobility.

Keep these small, dependency-free, and stable: higher layers (behaviors, planners,
controllers) should depend on these types, not on each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple


class NavMode(str, Enum):
    """High-level navigation mode selected by a mission controller."""
    SEARCH_ROUTE = "SEARCH_ROUTE"
    GO_HOME = "GO_HOME"
    GO_LASER = "GO_LASER"
    DOCK_LASER = "DOCK_LASER"
    IDLE = "IDLE"


class NavStatus(str, Enum):
    """Lifecycle state of a nav behavior/controller."""
    RUNNING = "RUNNING"
    ARRIVED = "ARRIVED"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Pose2D:
    """Planar pose in meters + radians."""
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0
    # Quality is a lightweight confidence heuristic (0..1). You can interpret it however you want.
    quality: float = 1.0


@dataclass(frozen=True)
class Twist2D:
    """Differential-drive-friendly velocity command."""
    v_mps: float = 0.0      # linear velocity (m/s)
    w_rps: float = 0.0      # angular velocity (rad/s)


@dataclass(frozen=True)
class Waypoint:
    """2D waypoint target."""
    x_m: float
    y_m: float
    # Optional metadata (e.g., label, station id, priority).
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NavObservation:
    """
    Minimal observation bundle used by planners and behaviors.

    Keep this small and easy to mock in tests.
    """
    timestamp_s: float
    pose: Pose2D
    min_distance_m: Optional[float] = None   # LiDAR forward-sector minimum
    # Optional vision hooks (e.g., fiducial bearing, or a safety classifier result)
    vision: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class NavProposal:
    """
    Proposed motion output from navigation.

    IMPORTANT: This is only a proposal — safety layers may clamp/override.
    """
    twist: Twist2D
    status: NavStatus = NavStatus.RUNNING
    done: bool = False
    reasons: Tuple[str, ...] = ()
    debug: Dict[str, Any] = field(default_factory=dict)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_angle_rad(a: float) -> float:
    """
    Wrap angle to [-pi, +pi].
    """
    import math
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return a


def pose_distance_m(a: Pose2D, b: Pose2D) -> float:
    import math
    return float(math.hypot(b.x_m - a.x_m, b.y_m - a.y_m))
