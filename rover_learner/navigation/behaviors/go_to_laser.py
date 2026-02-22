"""
rover_learner.navigation.behaviors.go_to_laser

Coarse approach to a known laser-station waypoint.

This is intentionally simple:
- drive to a configured waypoint (x,y)
- once inside waypoint_radius -> report ARRIVED (the orchestrator can switch to DockLaserBehavior)
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import WaypointFollowerConfig
from ..types import NavObservation, NavProposal, NavStatus, Twist2D, Waypoint
from ..waypoint_follower import compute_twist_to_waypoint


@dataclass
class GoToLaserBehavior:
    laser_waypoint: Waypoint
    follower: WaypointFollowerConfig = WaypointFollowerConfig()

    @property
    def name(self) -> str:
        return "GoToLaser"

    def reset(self) -> None:
        # stateless
        return

    def step(self, obs: NavObservation) -> NavProposal:
        out = compute_twist_to_waypoint(pose=obs.pose, waypoint=self.laser_waypoint, cfg=self.follower)
        if out.arrived:
            return NavProposal(twist=Twist2D(0.0, 0.0), status=NavStatus.ARRIVED, done=True, reasons=("LASER_WP_ARRIVED",))
        return NavProposal(
            twist=out.twist,
            status=NavStatus.RUNNING,
            done=False,
            reasons=tuple(out.reasons),
            debug={"laser_x": self.laser_waypoint.x_m, "laser_y": self.laser_waypoint.y_m, **out.debug},
        )
