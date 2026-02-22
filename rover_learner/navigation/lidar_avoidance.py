"""
rover_learner.navigation.lidar_avoidance

Lightweight local avoidance using ONLY a forward-sector minimum distance.

Because we only have a scalar min_distance_m (no angle), this module is intentionally
conservative:
- far away: pass through the desired twist
- near: slow down and add a small "search turn"
- too close: stop forward motion and rotate in place (with a toggling direction)

This is meant to be a modifier that sits between a planner and your safety gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .config import LidarAvoidanceConfig
from .types import Twist2D, clamp


@dataclass
class AvoidanceOutput:
    twist: Twist2D
    active: bool
    reasons: Tuple[str, ...]


class LidarAvoidanceController:
    def __init__(self, cfg: Optional[LidarAvoidanceConfig] = None) -> None:
        self.cfg = cfg or LidarAvoidanceConfig()
        self._turn_sign: float = 1.0
        self._last_toggle_ts: Optional[float] = None
        self._stopped_since_ts: Optional[float] = None

    def reset(self) -> None:
        self._turn_sign = 1.0
        self._last_toggle_ts = None
        self._stopped_since_ts = None

    def update(
        self,
        *,
        desired: Twist2D,
        min_distance_m: Optional[float],
        timestamp_s: float,
    ) -> AvoidanceOutput:
        """
        Returns a modified twist command and an 'active' flag.
        """
        d = None if min_distance_m is None else float(min_distance_m)
        cfg = self.cfg

        # Clamp desired first (good hygiene)
        v0 = clamp(float(desired.v_mps), -cfg.max_v_mps, cfg.max_v_mps)
        w0 = clamp(float(desired.w_rps), -cfg.max_w_rps, cfg.max_w_rps)

        if d is None or d <= 0.0:
            return AvoidanceOutput(twist=Twist2D(v0, w0), active=False, reasons=())

        # Too close: stop and rotate
        if d <= cfg.stop_dist_m:
            if self._stopped_since_ts is None:
                self._stopped_since_ts = float(timestamp_s)

            # Toggle direction occasionally to avoid deadlocks
            if self._last_toggle_ts is None:
                self._last_toggle_ts = float(timestamp_s)
            else:
                if float(timestamp_s) - float(self._last_toggle_ts) >= cfg.turn_toggle_s:
                    self._turn_sign *= -1.0
                    self._last_toggle_ts = float(timestamp_s)

            v = 0.0
            w = clamp(self._turn_sign * cfg.stop_turn_rate_rps, -cfg.max_w_rps, cfg.max_w_rps)
            return AvoidanceOutput(
                twist=Twist2D(v, w),
                active=True,
                reasons=(f"LIDAR_STOP(d={d:.2f}<= {cfg.stop_dist_m:.2f})", "ROTATE_IN_PLACE"),
            )

        # Near: slow down linearly and add turn bias
        if d <= cfg.slow_dist_m:
            self._stopped_since_ts = None
            scale = (d - cfg.stop_dist_m) / max(1e-6, (cfg.slow_dist_m - cfg.stop_dist_m))
            scale = clamp(scale, 0.0, 1.0)

            v = clamp(v0 * scale, -cfg.max_v_mps, cfg.max_v_mps)

            # Add some turning bias toward the current sign; keep existing planner turn.
            w_bias = (1.0 - scale) * cfg.slow_turn_bias_rps * self._turn_sign
            w = clamp(w0 + w_bias, -cfg.max_w_rps, cfg.max_w_rps)

            return AvoidanceOutput(
                twist=Twist2D(v, w),
                active=True,
                reasons=(f"LIDAR_SLOW(scale={scale:.2f}, d={d:.2f})",),
            )

        # Clear
        self._stopped_since_ts = None
        return AvoidanceOutput(twist=Twist2D(v0, w0), active=False, reasons=())
