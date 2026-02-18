"""
rover_trainer/hardware_manager_arduino.py

Parallel HardwareManager that attaches an Arduino interlock (E-stop + watchdog)
without modifying rover_trainer/hardware_manager.py.

- Wraps the existing HardwareManager for cameras + lidar.
- Adds ArduinoInterlock status fields to the outgoing packet.

NOTE:
- The Arduino Uno typically enumerates as /dev/ttyACM0 on Jetson.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rover_learner.arduino_interlock import ArduinoInterlock, ArduinoStatus
from rover_learner.rover_trainer.hardware_manager import HardwareManager, SensorPacket


@dataclass
class SensorPacketArduino:
    # Original SensorPacket fields
    image_primary: Any
    image_secondary: Any
    dist_primary: float
    dist_secondary: float
    min_dist: float
    brightness: float

    # Arduino extras
    arduino_status: Optional[ArduinoStatus]
    arduino_alive: bool
    arduino_safe: bool
    arduino_estop: bool
    arduino_watchdog: bool
    arduino_armed: bool
    arduino_rx_age_s: float


class HardwareManagerArduino:
    """
    Composition wrapper around HardwareManager that augments packets with Arduino status.
    """

    def __init__(
        self,
        mode: int = 2,
        *,
        arduino_port: str = "/dev/ttyACM0",
        arduino_required: bool = True,
        auto_arm: bool = False,
        ping_interval_s: float = 0.2,
        stat_timeout_s: float = 1.0,
    ):
        self.mode = int(mode)
        self.hw = HardwareManager(mode=self.mode)

        self.arduino_required = bool(arduino_required)
        self.arduino = ArduinoInterlock(
            port=arduino_port,
            ping_interval_s=ping_interval_s,
            stat_timeout_s=stat_timeout_s,
            autostart=True,
        )

        if auto_arm:
            self.arduino.set_armed(True)

    def read(self) -> SensorPacketArduino:
        base: SensorPacket = self.hw.read()

        st = self.arduino.get_status()
        alive = self.arduino.is_alive()
        rx_age = self.arduino.rx_age_s()

        if st is None:
            estop = False
            wd = False
            armed = False
            safe = False
        else:
            estop = bool(st.estop)
            wd = bool(st.wd)
            armed = bool(st.armed)
            safe = bool(st.is_interlock_safe)

        # If Arduino is required and not alive, force safe False
        if self.arduino_required and not alive:
            safe = False

        return SensorPacketArduino(
            image_primary=base.image_primary,
            image_secondary=base.image_secondary,
            dist_primary=base.dist_primary,
            dist_secondary=base.dist_secondary,
            min_dist=base.min_dist,
            brightness=base.brightness,
            arduino_status=st,
            arduino_alive=alive,
            arduino_safe=safe,
            arduino_estop=estop,
            arduino_watchdog=wd,
            arduino_armed=armed,
            arduino_rx_age_s=rx_age,
        )

    def close(self) -> None:
        try:
            self.arduino.close()
        except Exception:
            pass
        try:
            self.hw.close()
        except Exception:
            pass
