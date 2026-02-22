"""
rover_learner.navigation.pose_tracker

Dead-reckoning pose tracking from differential drive encoders.

- Pure math, no ROS required.
- Accepts absolute tick counts (preferred) or wheel delta distances.
- Optional IMU yaw fusion (complementary filter).

NOTE: Dead reckoning drifts over long distances. For early phases, that's fine:
you can still do breadcrumb return and coarse waypoint following, and later add
visual markers (AprilTags) and/or SLAM as drop-in upgrades.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

from .config import PoseTrackerConfig
from .types import Pose2D, wrap_angle_rad, clamp


def ticks_to_meters(ticks: int, *, wheel_radius_m: float, ticks_per_rev: int) -> float:
    import math
    if ticks_per_rev <= 0:
        raise ValueError("ticks_per_rev must be positive")
    return float((ticks / float(ticks_per_rev)) * (2.0 * math.pi * wheel_radius_m))


class PoseTracker:
    def __init__(self, cfg: Optional[PoseTrackerConfig] = None) -> None:
        self.cfg = cfg or PoseTrackerConfig()
        self._pose = Pose2D()
        self._last_ticks: Optional[Tuple[int, int]] = None
        self._last_ts: Optional[float] = None
        self._imu_last_ts: Optional[float] = None

    @property
    def pose(self) -> Pose2D:
        return self._pose

    def reset(self, pose: Optional[Pose2D] = None) -> None:
        self._pose = pose or Pose2D()
        self._last_ticks = None
        self._last_ts = None
        self._imu_last_ts = None

    def update_from_ticks(
        self,
        *,
        left_ticks: int,
        right_ticks: int,
        timestamp_s: float,
        imu_yaw_rad: Optional[float] = None,
    ) -> Pose2D:
        """
        Update pose from absolute encoder tick counts.
        """
        if self._last_ticks is None or self._last_ts is None:
            self._last_ticks = (int(left_ticks), int(right_ticks))
            self._last_ts = float(timestamp_s)
            if imu_yaw_rad is not None:
                self._imu_last_ts = float(timestamp_s)
                if self.cfg.use_imu_yaw:
                    self._pose = replace(self._pose, yaw_rad=wrap_angle_rad(float(imu_yaw_rad)))
            return self._pose

        dt = float(timestamp_s) - float(self._last_ts)
        if dt <= 0.0 or dt > self.cfg.max_dt_s:
            # Stale or invalid — reset baseline but keep pose.
            self._last_ticks = (int(left_ticks), int(right_ticks))
            self._last_ts = float(timestamp_s)
            return self._pose

        dl_ticks = int(left_ticks) - int(self._last_ticks[0])
        dr_ticks = int(right_ticks) - int(self._last_ticks[1])

        p = self.cfg.drive
        dl_m = ticks_to_meters(dl_ticks, wheel_radius_m=p.wheel_radius_m, ticks_per_rev=p.ticks_per_rev)
        dr_m = ticks_to_meters(dr_ticks, wheel_radius_m=p.wheel_radius_m, ticks_per_rev=p.ticks_per_rev)

        self._pose = self.update_from_wheel_deltas(
            dl_m=dl_m,
            dr_m=dr_m,
            timestamp_s=float(timestamp_s),
            imu_yaw_rad=imu_yaw_rad,
        )

        self._last_ticks = (int(left_ticks), int(right_ticks))
        self._last_ts = float(timestamp_s)
        return self._pose

    def update_from_wheel_deltas(
        self,
        *,
        dl_m: float,
        dr_m: float,
        timestamp_s: float,
        imu_yaw_rad: Optional[float] = None,
    ) -> Pose2D:
        """
        Update pose from wheel delta distances in meters for left and right wheels.
        """
        import math

        p = self.cfg.drive
        if p.wheel_base_m <= 0:
            raise ValueError("wheel_base_m must be positive")

        ds = 0.5 * (float(dl_m) + float(dr_m))
        dtheta = (float(dr_m) - float(dl_m)) / float(p.wheel_base_m)

        # Midpoint integration (better than naive Euler for turns)
        theta0 = self._pose.yaw_rad
        theta_mid = theta0 + 0.5 * dtheta
        dx = ds * math.cos(theta_mid)
        dy = ds * math.sin(theta_mid)

        new_x = self._pose.x_m + dx
        new_y = self._pose.y_m + dy
        new_theta = wrap_angle_rad(theta0 + dtheta)

        # Optional IMU yaw fusion
        if imu_yaw_rad is not None and self.cfg.use_imu_yaw:
            imu_yaw = wrap_angle_rad(float(imu_yaw_rad))
            a = clamp(self.cfg.imu_yaw_alpha, 0.0, 1.0)
            # Blend angles carefully: blend the error, not raw values
            err = wrap_angle_rad(imu_yaw - new_theta)
            new_theta = wrap_angle_rad(new_theta + a * err)
            self._imu_last_ts = float(timestamp_s)

        # Degrade quality over time if we have no absolute correction.
        # (Simple heuristic; you can replace later with proper covariance.)
        q = float(self._pose.quality)
        if self._imu_last_ts is None:
            q = max(0.0, q - 0.002)
        else:
            age = float(timestamp_s) - float(self._imu_last_ts)
            if age > 2.0:
                q = max(0.0, q - 0.001 * (age - 2.0))

        self._pose = Pose2D(x_m=float(new_x), y_m=float(new_y), yaw_rad=float(new_theta), quality=float(min(1.0, q)))
        return self._pose
