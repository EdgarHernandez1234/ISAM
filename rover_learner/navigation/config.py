"""
rover_learner.navigation.config

All navigation tunables live here. Keep it pure-data so you can:
- override via CLI/JSON/YAML in higher layers
- unit test controllers with deterministic parameters
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DifferentialDriveParams:
    wheel_base_m: float = 0.40          # distance between wheels (m)
    wheel_radius_m: float = 0.06        # wheel radius (m)
    ticks_per_rev: int = 2048           # encoder ticks per wheel revolution


@dataclass(frozen=True)
class PoseTrackerConfig:
    drive: DifferentialDriveParams = DifferentialDriveParams()
    max_dt_s: float = 0.25              # reject encoder updates larger than this (stale)
    use_imu_yaw: bool = False           # if True, fuse IMU yaw into heading
    imu_yaw_alpha: float = 0.15         # complementary filter blend (0=ignore IMU, 1=trust IMU)


@dataclass(frozen=True)
class LidarAvoidanceConfig:
    """
    Avoidance using ONLY a forward-sector minimum distance. This is conservative but robust.
    """
    slow_dist_m: float = 1.80           # start slowing down below this distance
    stop_dist_m: float = 1.20           # full stop below this distance
    # When stopped, rotate in place to try to clear the forward sector.
    stop_turn_rate_rps: float = 0.6
    # When slowing (but not stopped), we can add a small turning bias to "search" around obstacles.
    slow_turn_bias_rps: float = 0.25
    # Toggle turn direction if we stay stopped too long (prevents getting stuck turning one way).
    turn_toggle_s: float = 2.0
    # Hard clamp on output twist.
    max_v_mps: float = 0.35
    max_w_rps: float = 1.0


@dataclass(frozen=True)
class WaypointFollowerConfig:
    # Motion limits
    max_v_mps: float = 0.35
    max_w_rps: float = 1.0
    cruise_v_mps: float = 0.25
    min_v_mps: float = 0.05

    # Controller gains
    heading_kp: float = 1.2

    # Geometry / gating
    waypoint_radius_m: float = 0.45         # distance considered "arrived"
    slow_down_radius_m: float = 1.5         # start reducing speed within this distance
    heading_err_stop_rad: float = 0.65      # if error bigger, pivot instead of driving forward


@dataclass(frozen=True)
class BreadcrumbConfig:
    """
    Breadcrumb trail recording for return-to-home.

    record_min_step_m: how far you must move before adding a new crumb.
    max_points: safety cap to prevent unbounded growth.
    waypoint_radius_m: used by return-following step logic.
    """
    record_min_step_m: float = 0.60
    max_points: int = 20000
    waypoint_radius_m: float = 0.55
