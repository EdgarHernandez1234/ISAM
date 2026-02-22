"""
rover_learner.navigation.behaviors.return_home

Return to home using a breadcrumb trail recorded during exploration.

This behavior expects a BreadcrumbTrail instance (from rover_learner.navigation.breadcrumb)
that is being recorded elsewhere (e.g., by the orchestrator).

Operation:
- begin_return() if not started
- each step: ask trail for next crumb waypoint in reverse
- use waypoint follower to drive toward that crumb
- done when trail exhausted

Obstacle avoidance should be applied by a higher layer (lidar_avoidance modifier).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..breadcrumb import BreadcrumbTrail
from ..config import WaypointFollowerConfig
from ..types import NavObservation, NavProposal, NavStatus, Twist2D
from ..waypoint_follower import compute_twist_to_waypoint


@dataclass
class ReturnHomeBehavior:
    trail: BreadcrumbTrail
    follower: WaypointFollowerConfig = WaypointFollowerConfig()

    _started: bool = False

    @property
    def name(self) -> str:
        return "ReturnHome"

    def reset(self) -> None:
        self._started = False

    def step(self, obs: NavObservation) -> NavProposal:
        if not self._started:
            self.trail.begin_return()
            self._started = True

        wp, done, reasons = self.trail.step_return(obs.pose)
        if done or wp is None:
            return NavProposal(twist=Twist2D(0.0, 0.0), status=NavStatus.ARRIVED, done=True, reasons=reasons)

        out = compute_twist_to_waypoint(pose=obs.pose, waypoint=wp, cfg=self.follower)

        # If we got "arrived" to this crumb, the next call will advance index
        return NavProposal(
            twist=out.twist,
            status=NavStatus.RUNNING,
            done=False,
            reasons=reasons + out.reasons,
            debug={"crumb_idx": wp.meta.get("crumb_idx", -1), **out.debug},
        )
