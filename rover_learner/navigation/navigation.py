"""
rover_learner.navigation.navigation

Thin navigation orchestrator that:
- tracks pose (PoseTracker)
- records breadcrumbs (BreadcrumbTrail)
- selects an active behavior (SearchRoute / ReturnHome / GoToLaser / DockLaser)
- applies LiDAR avoidance as a modifier
- returns a NavProposal (Twist2D + status/reasons/debug)

This module is intentionally *not* tied to ROS. It expects you to supply:
- encoder ticks or wheel deltas
- lidar min_distance_m
- optional vision dict (e.g., marker bearing or harvest intent)

Integration pattern (typical):
  nav = Navigator(...)
  nav.update_sensors(...)
  proposal = nav.step(nav_mode=NavMode.SEARCH_ROUTE, timestamp_s=t, vision={...})
  # then pass proposal.twist to your safety stack and motor controller
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from .types import NavMode, NavObservation, NavProposal, NavStatus, Pose2D, Twist2D, Waypoint
from .config import (
    PoseTrackerConfig,
    BreadcrumbConfig,
    LidarAvoidanceConfig,
    WaypointFollowerConfig,
)
from .pose_tracker import PoseTracker
from .breadcrumb import BreadcrumbTrail
from .lidar_avoidance import LidarAvoidanceController
from .behaviors.search_route import SearchRouteBehavior, SearchRouteConfig
from .behaviors.return_home import ReturnHomeBehavior
from .behaviors.go_to_laser import GoToLaserBehavior
from .behaviors.dock_laser import DockLaserBehavior, DockLaserConfig


@dataclass
class NavigatorConfig:
    pose: PoseTrackerConfig = PoseTrackerConfig()
    breadcrumb: BreadcrumbConfig = BreadcrumbConfig()
    avoidance: LidarAvoidanceConfig = LidarAvoidanceConfig()
    follower: WaypointFollowerConfig = WaypointFollowerConfig()

    # Behavior configs
    search_route: SearchRouteConfig = SearchRouteConfig()
    dock_laser: DockLaserConfig = DockLaserConfig()

    # Safety: clamp returned twist as a last resort (still should be clamped downstream)
    max_v_mps: float = 0.35
    max_w_rps: float = 1.0


class Navigator:
    """
    One-stop nav orchestrator.

    Notes:
    - Call update_encoders(...) when you get new tick counts.
    - Call set_route(...) / set_laser_waypoint(...) to configure mission geometry.
    - Call step(...) each frame to get a motion proposal.
    """
    def __init__(self, cfg: Optional[NavigatorConfig] = None) -> None:
        self.cfg = cfg or NavigatorConfig()

        self.pose_tracker = PoseTracker(self.cfg.pose)
        self.trail = BreadcrumbTrail(self.cfg.breadcrumb)
        self.avoid = LidarAvoidanceController(self.cfg.avoidance)

        self._route: List[Waypoint] = []
        self._laser_wp: Optional[Waypoint] = None

        self._mode: NavMode = NavMode.IDLE
        self._behavior_name: str = "None"

        # Behaviors (constructed lazily / refreshed on reset)
        self._search: Optional[SearchRouteBehavior] = None
        self._return: Optional[ReturnHomeBehavior] = None
        self._go_laser: Optional[GoToLaserBehavior] = None
        self._dock: Optional[DockLaserBehavior] = None

        # Latest scalar sensor readings
        self._min_distance_m: Optional[float] = None
        self._last_ts: Optional[float] = None

    # -----------------
    # Configuration API
    # -----------------
    def set_route(self, route: List[Waypoint]) -> None:
        self._route = list(route)
        # Reset behavior next time mode enters SEARCH_ROUTE
        if self._search is not None:
            self._search.route = list(route)

    def set_laser_waypoint(self, wp: Waypoint) -> None:
        self._laser_wp = wp

    def reset(self, pose: Optional[Pose2D] = None) -> None:
        self.pose_tracker.reset(pose=pose)
        self.trail.reset()
        self.avoid.reset()
        self._mode = NavMode.IDLE
        self._behavior_name = "None"
        self._search = None
        self._return = None
        self._go_laser = None
        self._dock = None

    # -----------------
    # Sensor update API
    # -----------------
    def update_encoders_ticks(self, *, left_ticks: int, right_ticks: int, timestamp_s: float, imu_yaw_rad: Optional[float] = None) -> Pose2D:
        self._last_ts = float(timestamp_s)
        return self.pose_tracker.update_from_ticks(
            left_ticks=int(left_ticks),
            right_ticks=int(right_ticks),
            timestamp_s=float(timestamp_s),
            imu_yaw_rad=imu_yaw_rad,
        )

    def update_wheel_deltas(self, *, dl_m: float, dr_m: float, timestamp_s: float, imu_yaw_rad: Optional[float] = None) -> Pose2D:
        self._last_ts = float(timestamp_s)
        return self.pose_tracker.update_from_wheel_deltas(dl_m=float(dl_m), dr_m=float(dr_m), timestamp_s=float(timestamp_s), imu_yaw_rad=imu_yaw_rad)

    def update_lidar(self, *, min_distance_m: Optional[float]) -> None:
        self._min_distance_m = None if min_distance_m is None else float(min_distance_m)

    # -----------------
    # Orchestration
    # -----------------
    def _ensure_behavior(self, mode: NavMode) -> None:
        # Create / refresh behavior instance for the requested mode.
        if mode == NavMode.SEARCH_ROUTE:
            if self._search is None:
                self._search = SearchRouteBehavior(route=list(self._route), cfg=self.cfg.search_route)
            self._behavior_name = self._search.name
            return

        if mode == NavMode.GO_HOME:
            if self._return is None:
                self._return = ReturnHomeBehavior(trail=self.trail, follower=self.cfg.follower)
            self._behavior_name = self._return.name
            return

        if mode == NavMode.GO_LASER:
            if self._laser_wp is None:
                raise RuntimeError("Laser waypoint not set")
            if self._go_laser is None:
                self._go_laser = GoToLaserBehavior(laser_waypoint=self._laser_wp, follower=self.cfg.follower)
            self._behavior_name = self._go_laser.name
            return

        if mode == NavMode.DOCK_LASER:
            if self._dock is None:
                self._dock = DockLaserBehavior(cfg=self.cfg.dock_laser)
            self._behavior_name = self._dock.name
            return

        self._behavior_name = "Idle"

    def _get_behavior(self, mode: NavMode):
        if mode == NavMode.SEARCH_ROUTE:
            return self._search
        if mode == NavMode.GO_HOME:
            return self._return
        if mode == NavMode.GO_LASER:
            return self._go_laser
        if mode == NavMode.DOCK_LASER:
            return self._dock
        return None

    def step(
        self,
        *,
        nav_mode: NavMode,
        timestamp_s: float,
        vision: Optional[Dict[str, Any]] = None,
        record_breadcrumbs: bool = True,
    ) -> NavProposal:
        """
        Compute one navigation proposal.

        Parameters
        ----------
        nav_mode:
            Which nav behavior to run.
        timestamp_s:
            Time of this decision frame.
        vision:
            Optional dict for behavior hints (harvest intent, laser marker bearing, etc.)
        record_breadcrumbs:
            If True, record crumbs whenever we're in SEARCH_ROUTE / GO_LASER / DOCK_LASER.
            (Typically you record whenever you're not returning home.)
        """
        # Update mode and reset behavior on change
        if nav_mode != self._mode:
            self._mode = nav_mode
            self._ensure_behavior(nav_mode)
            b = self._get_behavior(nav_mode)
            if b is not None:
                b.reset()
            # If returning home, we want crumbs to already exist; do not reset trail.
            # If switching away from return-home, continue recording.

        # Record breadcrumb trail when exploring/moving (not when returning).
        if record_breadcrumbs and nav_mode != NavMode.GO_HOME:
            self.trail.record(self.pose_tracker.pose)

        obs = NavObservation(
            timestamp_s=float(timestamp_s),
            pose=self.pose_tracker.pose,
            min_distance_m=self._min_distance_m,
            vision=vision,
        )

        # Produce raw proposal from active behavior
        b = self._get_behavior(nav_mode)
        if b is None:
            return NavProposal(twist=Twist2D(0.0, 0.0), status=NavStatus.RUNNING, done=False, reasons=("IDLE",))

        try:
            raw = b.step(obs)
        except Exception as e:
            return NavProposal(twist=Twist2D(0.0, 0.0), status=NavStatus.ERROR, done=True, reasons=("BEHAVIOR_EXCEPTION", str(e)))

        # Apply avoidance modifier
        avoid_out = self.avoid.update(desired=raw.twist, min_distance_m=self._min_distance_m, timestamp_s=float(timestamp_s))

        # Compose final proposal
        reasons = list(raw.reasons)
        if avoid_out.active:
            reasons.extend(avoid_out.reasons)

        # Clamp output as a final check (downstream should clamp too)
        v = max(-self.cfg.max_v_mps, min(self.cfg.max_v_mps, float(avoid_out.twist.v_mps)))
        w = max(-self.cfg.max_w_rps, min(self.cfg.max_w_rps, float(avoid_out.twist.w_rps)))

        debug = dict(raw.debug or {})
        debug.update({
            "nav_mode": nav_mode.value,
            "behavior": self._behavior_name,
            "pose": {"x": self.pose_tracker.pose.x_m, "y": self.pose_tracker.pose.y_m, "yaw": self.pose_tracker.pose.yaw_rad, "q": self.pose_tracker.pose.quality},
            "min_distance_m": self._min_distance_m,
            "avoid_active": avoid_out.active,
        })

        return NavProposal(
            twist=Twist2D(v, w),
            status=raw.status,
            done=raw.done,
            reasons=tuple(reasons),
            debug=debug,
        )
