"""
conference_demo/mode_manager_arduino.py

Parallel ModeManager that adds Arduino heartbeat caching (E-stop + watchdog) alongside LiDAR.

It does NOT change existing mode_manager.py; instead it:
- Inherits ModeManager for camera/LiDAR switching.
- Adds an Arduino provider + simple "alive" / "safe" probes.
- Provides timestamps used by an Arduino-aware health policy.

This module is intentionally conservative: if require_arduino=True and the Arduino is unsafe,
it forces OperatingMode.HALT (unless the caller chooses to ignore it / ghost mode).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from rover_learner.arduino_interlock import ArduinoInterlock, ArduinoStatus
from rover_learner.conference_demo.mode_manager import ModeManager, OperatingMode, _plan_is_critical


class ModeManagerArduino(ModeManager):
    def __init__(
        self,
        cam_provider: Any,
        lidar_provider: Any,
        arduino: ArduinoInterlock,
        *,
        require_arduino: bool = True,
    ):
        super().__init__(cam_provider, lidar_provider)
        self.arduino = arduino
        self.require_arduino = bool(require_arduino)

        # Arduino heartbeat caching
        self._arduino_last_ok_ts: float = 0.0
        self._arduino_last_status: Optional[ArduinoStatus] = None
        self._arduino_last_probe_ts: float = 0.0
        self._arduino_probe_interval_s: float = 0.25
        self._arduino_alive_window_s: float = 1.5

    def probe_arduino(self) -> None:
        now = time.time()
        if (now - self._arduino_last_probe_ts) < self._arduino_probe_interval_s:
            return
        self._arduino_last_probe_ts = now

        st = self.arduino.get_status()
        alive = self.arduino.is_alive(now=now)
        if st is not None and alive:
            self._arduino_last_status = st
            self._arduino_last_ok_ts = now

    def arduino_last_ok_ts(self) -> float:
        return float(self._arduino_last_ok_ts)

    def arduino_seen_recently(self) -> bool:
        now = time.time()
        return (self._arduino_last_ok_ts > 0.0) and ((now - self._arduino_last_ok_ts) <= self._arduino_alive_window_s)

    def arduino_is_safe(self) -> bool:
        st = self._arduino_last_status
        if st is None:
            return False
        return bool(st.is_interlock_safe)

    def apply_plan(self, plan: Any) -> None:
        # Update Arduino heartbeat snapshot before applying plan
        self.probe_arduino()

        # Respect any CRITICAL plan from policy first
        if _plan_is_critical(plan):
            return super().apply_plan(plan)

        if self.require_arduino:
            # If unsafe, halt regardless of camera/lidar plan.
            if (not self.arduino_seen_recently()) or (not self.arduino_is_safe()):
                self.clear_stabilization()
                self.set_mode(OperatingMode.HALT)
                return

        return super().apply_plan(plan)
