"""
rover_learner.navigation.breadcrumb

Breadcrumb recording + return-following for "return to home base" without SLAM.

Concept:
- While exploring, record a trail of poses spaced by record_min_step_m.
- When "return home" is requested, follow the trail in reverse order.
- Obstacle avoidance remains a separate module.

This gives you a reliable early navigation capability before you add:
- AprilTags on base/laser station
- global mapping/SLAM
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import BreadcrumbConfig
from .types import Pose2D, Waypoint, pose_distance_m


@dataclass
class BreadcrumbTrail:
    cfg: BreadcrumbConfig = BreadcrumbConfig()
    _poses: List[Pose2D] = None  # type: ignore
    _return_idx: Optional[int] = None

    def __post_init__(self) -> None:
        if self._poses is None:
            self._poses = []

    def reset(self) -> None:
        self._poses = []
        self._return_idx = None

    @property
    def poses(self) -> List[Pose2D]:
        # Return a copy to avoid accidental mutation by callers.
        return list(self._poses)

    def record(self, pose: Pose2D) -> bool:
        """
        Add a breadcrumb pose if we've moved enough since the last.
        Returns True if a new breadcrumb was added.
        """
        if not self._poses:
            self._poses.append(pose)
            self._trim()
            return True

        if pose_distance_m(self._poses[-1], pose) >= self.cfg.record_min_step_m:
            self._poses.append(pose)
            self._trim()
            return True
        return False

    def _trim(self) -> None:
        if len(self._poses) > self.cfg.max_points:
            # Drop oldest (rare; mostly for long missions)
            extra = len(self._poses) - self.cfg.max_points
            self._poses = self._poses[extra:]
            if self._return_idx is not None:
                self._return_idx = max(0, self._return_idx - extra)

    def begin_return(self) -> None:
        """
        Begin following breadcrumbs in reverse.
        """
        if not self._poses:
            self._return_idx = None
            return
        self._return_idx = len(self._poses) - 1

    def step_return(self, current_pose: Pose2D) -> Tuple[Optional[Waypoint], bool, Tuple[str, ...]]:
        """
        Returns (next_waypoint, done, reasons).
        """
        if self._return_idx is None:
            self.begin_return()

        if self._return_idx is None:
            return (None, True, ("NO_BREADCRUMBS",))

        # Skip crumbs already reached
        while self._return_idx >= 0:
            target_pose = self._poses[self._return_idx]
            if pose_distance_m(current_pose, target_pose) <= self.cfg.waypoint_radius_m:
                self._return_idx -= 1
                continue
            break

        if self._return_idx < 0:
            return (None, True, ("RETURN_COMPLETE",))

        t = self._poses[self._return_idx]
        wp = Waypoint(x_m=t.x_m, y_m=t.y_m, meta={"crumb_idx": self._return_idx})
        return (wp, False, ("FOLLOW_BREADCRUMB",))
