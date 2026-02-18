"""
conference_demo/power_saver_degradation_arduino.py

Parallel policy engine that extends power_saver_degradation.py with Arduino interlock inputs.

Behavior:
- If require_arduino=True:
    - Arduino unsafe (E-stop/watchdog/not armed) => CRITICAL (RETURN_HOME hint)
    - Arduino heartbeat lost => CRITICAL
- Otherwise, defer to base SystemHealthMonitor.
"""

from __future__ import annotations

import time
from typing import Optional, List

from rover_learner.conference_demo.power_saver_degradation import (
    SystemHealthMonitor,
    ResourcePlan,
    SystemMode,
)


ARDUINO_TIMEOUT_S = 1.5


class SystemHealthMonitorArduino(SystemHealthMonitor):
    def __init__(self, *args, require_arduino: bool = True, arduino_timeout_s: float = ARDUINO_TIMEOUT_S, **kwargs):
        super().__init__(*args, **kwargs)
        self.require_arduino = bool(require_arduino)
        self.arduino_timeout_s = float(arduino_timeout_s)

    def get_plan(
        self,
        lidar_last_ts: float,
        cam_last_ts: float,
        *,
        arduino_last_ts: float,
        arduino_seen_recently: Optional[bool] = None,
        arduino_safe: Optional[bool] = None,
        camera_count_ok: int = 2,
        lidar_seen_recently: Optional[bool] = None,
        cam_seen_recently: Optional[bool] = None,
    ) -> ResourcePlan:
        # First, compute baseline plan
        base = super().get_plan(
            lidar_last_ts,
            cam_last_ts,
            camera_count_ok=camera_count_ok,
            lidar_seen_recently=lidar_seen_recently,
            cam_seen_recently=cam_seen_recently,
        )

        if not self.require_arduino:
            return base

        now = time.time()

        # Determine Arduino lost / unsafe
        ar_lost = (arduino_last_ts > 0.0) and ((now - arduino_last_ts) > self.arduino_timeout_s)
        if arduino_seen_recently is not None:
            ar_lost = not bool(arduino_seen_recently)

        ar_unsafe = False
        if arduino_safe is not None:
            ar_unsafe = not bool(arduino_safe)

        if ar_lost or ar_unsafe:
            reasons: List[str] = list(getattr(base, "reasons", []))
            if ar_lost:
                reasons.append("ARDUINO_LOST")
            if ar_unsafe:
                reasons.append("ARDUINO_UNSAFE")

            # Escalate to CRITICAL by returning a new plan (keep metrics from base)
            plan = ResourcePlan(
                mode=SystemMode.CRITICAL,
                camera_count=0,
                lidar_on=False,
                res_scale=0.0,
                fps_target=0,
                action_hint="RETURN_HOME",
                reasons=reasons,
                cpu_temp_c=float(getattr(base, "cpu_temp_c", 0.0)),
                ram_free_mb=float(getattr(base, "ram_free_mb", 0.0)),
                cpu_load_per_core=float(getattr(base, "cpu_load_per_core", 0.0)),
                health_score=0.0,
            )
            # update hysteresis state
            self._update_mode(plan.mode, now)
            return plan

        return base
