"""
rover_learner.navigation.behaviors.dock_laser

Final approach docking behavior using *vision bearing* (recommended via AprilTag/ArUco)
or a model class detection injected into obs.vision.

Expected obs.vision schema (choose one path):

A) Fiducial path (best)
  vision["marker_seen"]: bool
  vision["marker_bearing_rad"]: float   # +right, -left
  vision["marker_range_m"]: float | None

B) Class-detection path (fallback)
  vision["laser_seen"]: bool
  vision["laser_bearing_rad"]: float

LiDAR min_distance_m is used to stop within a safe standoff distance.

This behavior does NOT itself manage arm/deposit actions — it only aligns and approaches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..types import NavObservation, NavProposal, NavStatus, Twist2D, clamp, wrap_angle_rad


@dataclass(frozen=True)
class DockLaserConfig:
    max_v_mps: float = 0.18
    max_w_rps: float = 1.0
    bearing_kp: float = 1.8

    # Standoff: stop when LiDAR says we're within this distance to anything ahead
    standoff_m: float = 0.90

    # If bearing error is large, pivot in place
    bearing_pivot_rad: float = 0.55

    # If marker range is available, creep slower near the station
    slow_range_m: float = 1.6
    creep_v_mps: float = 0.06


@dataclass
class DockLaserBehavior:
    cfg: DockLaserConfig = DockLaserConfig()

    @property
    def name(self) -> str:
        return "DockLaser"

    def reset(self) -> None:
        return

    def _get_bearing(self, obs: NavObservation) -> Tuple[bool, float, Tuple[str, ...], Dict[str, float]]:
        v = obs.vision or {}
        # Fiducial preferred
        if bool(v.get("marker_seen", False)) and ("marker_bearing_rad" in v):
            b = float(v["marker_bearing_rad"])
            rng = v.get("marker_range_m", None)
            dbg = {"bearing": b}
            if rng is not None:
                dbg["range_m"] = float(rng)
            return True, b, ("MARKER_TRACK",), dbg

        # Fallback
        if bool(v.get("laser_seen", False)) and ("laser_bearing_rad" in v):
            b = float(v["laser_bearing_rad"])
            return True, b, ("LASER_CLASS_TRACK",), {"bearing": b}

        return False, 0.0, ("NO_TARGET",), {}

    def step(self, obs: NavObservation) -> NavProposal:
        seen, bearing, reasons, dbg = self._get_bearing(obs)

        # Stop if we're too close (LiDAR)
        d = obs.min_distance_m
        if d is not None and float(d) > 0.0 and float(d) <= self.cfg.standoff_m:
            return NavProposal(
                twist=Twist2D(0.0, 0.0),
                status=NavStatus.ARRIVED,
                done=True,
                reasons=("DOCK_STANDOFF_REACHED",) + reasons,
                debug={"min_distance_m": float(d), **dbg},
            )

        if not seen:
            # If we can't see the station, rotate slowly to search
            return NavProposal(
                twist=Twist2D(0.0, 0.35),
                status=NavStatus.BLOCKED,
                done=False,
                reasons=reasons,
                debug=dbg,
            )

        # Controller: turn to minimize bearing
        berr = wrap_angle_rad(float(bearing))
        w = clamp(self.cfg.bearing_kp * berr, -self.cfg.max_w_rps, self.cfg.max_w_rps)

        # Speed: pivot if large error, else creep forward
        v = self.cfg.max_v_mps
        if abs(berr) >= self.cfg.bearing_pivot_rad:
            v = 0.0

        # If range is known, slow down near target
        rng = dbg.get("range_m", None)
        if rng is not None and rng <= self.cfg.slow_range_m:
            v = min(v, self.cfg.creep_v_mps)

        return NavProposal(
            twist=Twist2D(float(v), float(w)),
            status=NavStatus.RUNNING,
            done=False,
            reasons=reasons,
            debug={"bearing_err": berr, "min_distance_m": None if d is None else float(d), **dbg},
        )
