"""
rover_learner.navigation.waypoint_follower

A small waypoint follower for differential-drive robots.

- Pure math (unit test friendly).
- Outputs a Twist2D command that drives toward a waypoint.
- Does not do obstacle avoidance (that's a separate modifier).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .config import WaypointFollowerConfig
from .types import Pose2D, Twist2D, Waypoint, clamp, wrap_angle_rad


@dataclass(frozen=True)
class WaypointFollowOutput:
    twist: Twist2D
    arrived: bool
    reasons: Tuple[str, ...]
    debug: Dict[str, float]


def compute_twist_to_waypoint(
    *,
    pose: Pose2D,
    waypoint: Waypoint,
    cfg: WaypointFollowerConfig,
) -> WaypointFollowOutput:
    import math

    dx = float(waypoint.x_m) - float(pose.x_m)
    dy = float(waypoint.y_m) - float(pose.y_m)
    dist = float(math.hypot(dx, dy))

    if dist <= cfg.waypoint_radius_m:
        return WaypointFollowOutput(
            twist=Twist2D(0.0, 0.0),
            arrived=True,
            reasons=("WAYPOINT_ARRIVED",),
            debug={"dist": dist, "heading_err": 0.0},
        )

    goal_heading = float(math.atan2(dy, dx))
    heading_err = wrap_angle_rad(goal_heading - float(pose.yaw_rad))

    # Angular velocity
    w = clamp(cfg.heading_kp * heading_err, -cfg.max_w_rps, cfg.max_w_rps)

    # Speed schedule: slow down near target
    # scale = min(1, dist/slow_down_radius)
    scale = clamp(dist / max(1e-6, cfg.slow_down_radius_m), 0.0, 1.0)
    v = clamp(cfg.cruise_v_mps * scale, 0.0, cfg.max_v_mps)
    if v > 0.0:
        v = max(cfg.min_v_mps, v)

    reasons = []

    # If we're facing away, pivot instead of driving forward
    if abs(heading_err) >= cfg.heading_err_stop_rad:
        v = 0.0
        reasons.append("PIVOT_TO_HEADING")

    return WaypointFollowOutput(
        twist=Twist2D(float(v), float(w)),
        arrived=False,
        reasons=tuple(reasons),
        debug={"dist": dist, "heading_err": heading_err, "goal_heading": goal_heading, "scale": scale},
    )
