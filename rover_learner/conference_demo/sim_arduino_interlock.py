from __future__ import annotations

"""
conference_demo/sim_arduino_interlock.py

Software-only Arduino / watchdog / controller shim for Gazebo-based training.

Purpose
-------
This mimics the small subset of the real Arduino interlock API that
ModeManagerArduino currently expects:

- get_status()
- is_alive(now=...)
- close()

It also adds a lightweight Xbox-style controller layer so the same simulated
"Arduino" can later arbitrate between MANUAL / AUTO / HALT in demo runs.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    import pygame  # type: ignore
except Exception:
    pygame = None


@dataclass
class SimArduinoStatus:
    is_interlock_safe: bool = True
    estop: bool = False
    mode: str = "auto"
    controller_connected: bool = False
    manual_enabled: bool = False
    auto_enabled: bool = True
    throttle: float = 0.0
    steering: float = 0.0
    buttons: dict[str, bool] | None = None
    updated_ts: float = 0.0


class SimArduinoInterlock:
    MODE_MANUAL = "manual"
    MODE_AUTO = "auto"
    MODE_HALT = "halt"

    def __init__(
        self,
        *,
        autostart: bool = True,
        require_controller: bool = False,
        alive_window_s: float = 1.5,
        poll_hz: float = 30.0,
        deadzone: float = 0.12,
        max_throttle: float = 1.0,
        max_steering: float = 1.0,
        default_safe: bool = True,
        default_mode: str = MODE_AUTO,
    ) -> None:
        self.require_controller = bool(require_controller)
        self.alive_window_s = float(alive_window_s)
        self.poll_period_s = 1.0 / max(1.0, float(poll_hz))
        self.deadzone = float(deadzone)
        self.max_throttle = float(max_throttle)
        self.max_steering = float(max_steering)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._controller = None
        self._pygame_ready = False

        now = time.time()
        self._last_heartbeat_ts = now
        self._status = SimArduinoStatus(
            is_interlock_safe=bool(default_safe),
            estop=not bool(default_safe),
            mode=str(default_mode),
            controller_connected=False,
            manual_enabled=(default_mode == self.MODE_MANUAL),
            auto_enabled=(default_mode == self.MODE_AUTO),
            throttle=0.0,
            steering=0.0,
            buttons={},
            updated_ts=now,
        )

        if autostart:
            self.start()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._init_controller()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._shutdown_controller()

    def get_status(self) -> SimArduinoStatus:
        with self._lock:
            return self._copy_status_locked()

    def is_alive(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.time()
        with self._lock:
            last = float(self._last_heartbeat_ts)
            connected = bool(self._status.controller_connected)
        if self.require_controller and not connected:
            return False
        return (now - last) <= self.alive_window_s

    def set_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in {self.MODE_MANUAL, self.MODE_AUTO, self.MODE_HALT}:
            raise ValueError(f"Unsupported mode: {mode}")
        with self._lock:
            self._status.mode = mode
            self._status.manual_enabled = (mode == self.MODE_MANUAL)
            self._status.auto_enabled = (mode == self.MODE_AUTO)
            if mode == self.MODE_HALT:
                self._status.is_interlock_safe = False
                self._status.estop = True
                self._status.throttle = 0.0
                self._status.steering = 0.0
            self._status.updated_ts = time.time()

    def set_interlock_safe(self, safe: bool) -> None:
        with self._lock:
            self._status.is_interlock_safe = bool(safe)
            self._status.estop = not bool(safe)
            self._status.updated_ts = time.time()

    def clear_estop(self) -> None:
        with self._lock:
            self._status.estop = False
            self._status.is_interlock_safe = True
            if self._status.mode == self.MODE_HALT:
                self._status.mode = self.MODE_AUTO
                self._status.manual_enabled = False
                self._status.auto_enabled = True
            self._status.updated_ts = time.time()

    def manual_twist(self) -> tuple[float, float]:
        with self._lock:
            return float(self._status.throttle), float(self._status.steering)

    def controller_connected(self) -> bool:
        with self._lock:
            return bool(self._status.controller_connected)

    def _copy_status_locked(self) -> SimArduinoStatus:
        return SimArduinoStatus(
            is_interlock_safe=bool(self._status.is_interlock_safe),
            estop=bool(self._status.estop),
            mode=str(self._status.mode),
            controller_connected=bool(self._status.controller_connected),
            manual_enabled=bool(self._status.manual_enabled),
            auto_enabled=bool(self._status.auto_enabled),
            throttle=float(self._status.throttle),
            steering=float(self._status.steering),
            buttons=dict(self._status.buttons or {}),
            updated_ts=float(self._status.updated_ts),
        )

    def _init_controller(self) -> None:
        if pygame is None:
            return
        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                js = pygame.joystick.Joystick(0)
                js.init()
                self._controller = js
                self._pygame_ready = True
                with self._lock:
                    self._status.controller_connected = True
                    self._status.updated_ts = time.time()
        except Exception:
            self._controller = None
            self._pygame_ready = False

    def _shutdown_controller(self) -> None:
        if pygame is None:
            return
        try:
            if self._controller is not None:
                self._controller.quit()
        except Exception:
            pass
        try:
            pygame.joystick.quit()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass

    def _apply_deadzone(self, v: float) -> float:
        if abs(v) < self.deadzone:
            return 0.0
        return float(v)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()

            if self._pygame_ready and pygame is not None:
                try:
                    pygame.event.pump()
                    self._poll_pygame_controller(now)
                except Exception:
                    with self._lock:
                        self._status.controller_connected = False
                        self._status.updated_ts = now

            with self._lock:
                if (not self.require_controller) or self._status.controller_connected:
                    self._last_heartbeat_ts = now
                self._status.updated_ts = now

            time.sleep(self.poll_period_s)

    def _poll_pygame_controller(self, now: float) -> None:
        js = self._controller
        if js is None:
            with self._lock:
                self._status.controller_connected = False
            return

        try:
            left_x = self._apply_deadzone(float(js.get_axis(0)))
        except Exception:
            left_x = 0.0
        try:
            left_y = self._apply_deadzone(float(js.get_axis(1)))
        except Exception:
            left_y = 0.0

        def _btn(idx: int) -> bool:
            try:
                return bool(js.get_button(idx))
            except Exception:
                return False

        buttons = {
            "a": _btn(0),
            "b": _btn(1),
            "x": _btn(2),
            "y": _btn(3),
            "back": _btn(6),
            "start": _btn(7),
        }

        with self._lock:
            self._status.controller_connected = True
            self._status.buttons = buttons
            self._status.steering = max(-self.max_steering, min(self.max_steering, left_x))
            self._status.throttle = max(-self.max_throttle, min(self.max_throttle, -left_y))

            if buttons["back"]:
                self._status.mode = self.MODE_HALT
                self._status.estop = True
                self._status.is_interlock_safe = False
                self._status.manual_enabled = False
                self._status.auto_enabled = False
                self._status.throttle = 0.0
                self._status.steering = 0.0
            elif buttons["start"]:
                self._status.estop = False
                self._status.is_interlock_safe = True
                if self._status.mode == self.MODE_HALT:
                    self._status.mode = self.MODE_AUTO
                    self._status.manual_enabled = False
                    self._status.auto_enabled = True
            elif buttons["a"]:
                self._status.mode = self.MODE_MANUAL
                self._status.manual_enabled = True
                self._status.auto_enabled = False
                self._status.estop = False
                self._status.is_interlock_safe = True
            elif buttons["y"]:
                self._status.mode = self.MODE_AUTO
                self._status.manual_enabled = False
                self._status.auto_enabled = True
                self._status.estop = False
                self._status.is_interlock_safe = True
            elif buttons["b"]:
                self._status.mode = self.MODE_HALT
                self._status.manual_enabled = False
                self._status.auto_enabled = False
                self._status.estop = True
                self._status.is_interlock_safe = False
                self._status.throttle = 0.0
                self._status.steering = 0.0

            self._status.updated_ts = now


if __name__ == "__main__":
    sim = SimArduinoInterlock(autostart=True, require_controller=False)
    try:
        print("SimArduinoInterlock started. Press Ctrl+C to exit.")
        while True:
            st = sim.get_status()
            print(
                f"alive={sim.is_alive()} safe={st.is_interlock_safe} "
                f"mode={st.mode} connected={st.controller_connected} "
                f"throttle={st.throttle:.2f} steering={st.steering:.2f}"
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()
