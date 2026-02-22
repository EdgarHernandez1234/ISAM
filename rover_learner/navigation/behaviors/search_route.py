"""
rover_learner.navigation.behaviors.search_route

Follow a route of waypoints. Opportunistically pause/slow for harvesting based on a
caller-provided "harvest intent" signal (e.g., from SafetyStateController).

This behavior is intentionally decoupled from vision/ML details. It expects the
observation.vision dict (if present) to contain:

  vision["safety_state"]: str  (e.g., "Harvest", "Low_Hazard", ...)
or
  vision["harvest_intent"]: bool

Your orchestrator can inject those keys from whatever perception pipeline you use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..config import WaypointFollowerConfig
from ..types import NavObservation, NavProposal, NavStatus, Pose2D, Twist2D, Waypoint
from ..waypoint_follower import compute_twist_to_waypoint


@dataclass
class SearchRouteConfig:
    follower: WaypointFollowerConfig = WaypointFollowerConfig()
    # When harvest intent is true, reduce speed but keep steering stable.
    harvest_slow_v_mps: float = 0.06
    # Require harvest intent to persist for a couple frames before declaring BLOCKED (optional).
    harvest_hold_frames: int = 1


@dataclass
class SearchRouteBehavior:
    route: List[Waypoint]
    cfg: SearchRouteConfig = SearchRouteConfig()

    _idx: int = 0
    _harvest_hold: int = 0

    @property
    def name(self) -> str:
        return "SearchRoute"

    def reset(self) -> None:
        self._idx = 0
        self._harvest_hold = 0

    def _harvest_intent(self, obs: NavObservation) -> bool:
        v = obs.vision or {}
        if "harvest_intent" in v:
            return bool(v["harvest_intent"])
        # Allow string safety_state
        s = str(v.get("safety_state", "")).strip()
        return s.lower() == "harvest"

    def step(self, obs: NavObservation) -> NavProposal:
        if not self.route:
            return NavProposal(twist=Twist2D(0.0, 0.0), status=NavStatus.ERROR, done=True, reasons=("NO_ROUTE",))

        # Clamp index
        self._idx = max(0, min(self._idx, len(self.route) - 1))
        wp = self.route[self._idx]

        out = compute_twist_to_waypoint(pose=obs.pose, waypoint=wp, cfg=self.cfg.follower)

        if out.arrived:
            if self._idx >= len(self.route) - 1:
                return NavProposal(twist=Twist2D(0.0, 0.0), status=NavStatus.ARRIVED, done=True, reasons=("ROUTE_COMPLETE",))
            self._idx += 1
            wp = self.route[self._idx]
            out = compute_twist_to_waypoint(pose=obs.pose, waypoint=wp, cfg=self.cfg.follower)

        # Opportunistic harvest: slow down (or hold position) while the caller harvests.
        if self._harvest_intent(obs):
            self._harvest_hold += 1
            if self._harvest_hold >= max(1, self.cfg.harvest_hold_frames):
                # Slow creep forward, keep heading correction
                t = Twist2D(self.cfg.harvest_slow_v_mps, out.twist.w_rps)
                return NavProposal(
                    twist=t,
                    status=NavStatus.RUNNING,
                    done=False,
                    reasons=("HARVEST_INTENT_SLOW",),
                    debug={"route_idx": self._idx, "wp_x": wp.x_m, "wp_y": wp.y_m, **out.debug},
                )
        else:
            self._harvest_hold = 0

        return NavProposal(
            twist=out.twist,
            status=NavStatus.RUNNING,
            done=False,
            reasons=tuple(out.reasons),
            debug={"route_idx": self._idx, "wp_x": wp.x_m, "wp_y": wp.y_m, **out.debug},
        )
