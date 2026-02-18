"""
rover_trainer/scenario_logic_arduino.py

Parallel ScenarioManager that enforces Arduino interlock before any perception logic.

Policy:
- If Arduino is unsafe / missing (when required), return STOP immediately.
- Otherwise delegate to the base ScenarioManager.
"""

from __future__ import annotations

from rover_learner.rover_trainer.scenario_logic import ScenarioManager


class ScenarioManagerArduino(ScenarioManager):
    def __init__(self, *, arduino_required: bool = True):
        super().__init__()
        self.arduino_required = bool(arduino_required)

    def evaluate(self, packet, yolo_results) -> str:
        # Expect packet to carry arduino_* fields when using HardwareManagerArduino.
        arduino_safe = getattr(packet, "arduino_safe", None)
        arduino_alive = getattr(packet, "arduino_alive", True)
        arduino_estop = getattr(packet, "arduino_estop", False)
        arduino_wd = getattr(packet, "arduino_watchdog", False)

        if arduino_estop:
            self.sticky_state = "IDLE"
            return "STOP"

        if arduino_wd:
            self.sticky_state = "IDLE"
            return "STOP"

        if self.arduino_required and (not arduino_alive or not bool(arduino_safe)):
            self.sticky_state = "IDLE"
            return "STOP"

        return super().evaluate(packet, yolo_results)
