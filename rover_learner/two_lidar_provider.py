#!/usr/bin/env python3
"""
two_lidar_provider.py (rover_learner)

DUAL LIDAR PROVIDER
- Wraps two LidarProvider instances (e.g., two SerialRPLidarProvider)
- Exposes:
    - get_distances_m() -> (Optional[float], Optional[float])
    - get_distance_m()  -> Optional[float]  # fused (min of valid distances)
    - close()
- Includes: Serial dual provider, ROS 2 dual provider (optional), and Mock provider.

Design goals:
- Minimal coupling: acts like a normal provider but can also return both distances.
- Backwards-friendly: get_distance_m() provides a single scalar for safety gating/logging.
- Conference-ready: defaults to /dev/ttyUSB0 + /dev/ttyUSB1 (you can swap later with udev symlinks).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple


# ---------------------------------------------------------------------
# Import single-lidar providers in a way that works for:
# 1) "python3 file.py" from rover_learner/
# 2) imports from conference_demo/ with sys.path injection
# ---------------------------------------------------------------------
try:
    # If used as a package module
    from .lidar_provider import (
        LidarProvider,
        MockLidarProvider,
        SerialRPLidarProvider,
        ROS2LaserScanProvider,
    )
except Exception:
    # If used as a plain script/module on PYTHONPATH
    from lidar_provider import (
        LidarProvider,
        MockLidarProvider,
        SerialRPLidarProvider,
        ROS2LaserScanProvider,
    )


class TwoLidarProvider(Protocol):
    """Protocol for dual LiDAR providers."""
    def get_distances_m(self) -> Tuple[Optional[float], Optional[float]]:
        ...

    def get_distance_m(self) -> Optional[float]:
        """Fused single metric (default: min of valid distances)."""
        ...

    def close(self) -> None:
        ...


def fuse_min_distance(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Return min of the valid (non-None) distances."""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a if a < b else b


@dataclass(frozen=True)
class TwoLidarReading:
    """Convenience structure for callers that want both + fused."""
    lidar0_m: Optional[float]
    lidar1_m: Optional[float]
    fused_min_m: Optional[float]


# =========================================================
# 1) GENERIC WRAPPER (compose any two LidarProvider instances)
# =========================================================
class DualLidarWrapper:
    """
    Wrap two single-lidar providers.

    By convention:
      - lidar0 = "front" (ttyUSB0)
      - lidar1 = "side"  (ttyUSB1)
    but the wrapper is agnostic.
    """
    def __init__(self, lidar0: LidarProvider, lidar1: LidarProvider, name0: str = "lidar0", name1: str = "lidar1"):
        self._lidar0 = lidar0
        self._lidar1 = lidar1
        self.name0 = name0
        self.name1 = name1

    def get_distances_m(self) -> Tuple[Optional[float], Optional[float]]:
        return (self._lidar0.get_distance_m(), self._lidar1.get_distance_m())

    def get_reading(self) -> TwoLidarReading:
        a, b = self.get_distances_m()
        return TwoLidarReading(lidar0_m=a, lidar1_m=b, fused_min_m=fuse_min_distance(a, b))

    def get_distance_m(self) -> Optional[float]:
        a, b = self.get_distances_m()
        return fuse_min_distance(a, b)

    def close(self) -> None:
        try:
            self._lidar0.close()
        finally:
            self._lidar1.close()


# =========================================================
# 2) MOCK DUAL PROVIDER (REQUIRED FOR TESTS)
# =========================================================
class MockTwoLidarProvider(DualLidarWrapper):
    """Dual fake lidars for testing without hardware."""
    def __init__(self, dist0: float = 1.0, dist1: float = 1.0):
        super().__init__(MockLidarProvider(dist0), MockLidarProvider(dist1), name0="mock0", name1="mock1")


# =========================================================
# 3) SERIAL DUAL PROVIDER (FOR JETSON / TWO C1s)
# =========================================================
class SerialTwoRPLidarProvider(DualLidarWrapper):
    """
    Dual SerialRPLidarProvider wrapper.

    Defaults:
      - lidar0: /dev/ttyUSB0 @ 460800
      - lidar1: /dev/ttyUSB1 @ 460800

    Tip: later you can swap ports to stable udev symlinks (e.g., /dev/rplidar_front).
    """
    def __init__(
        self,
        port0: str = "/dev/ttyUSB0",
        port1: str = "/dev/ttyUSB1",
        baudrate: int = 460800,
        name0: str = "front",
        name1: str = "side",
    ):
        lidar0 = SerialRPLidarProvider(port=port0, baudrate=baudrate)
        lidar1 = SerialRPLidarProvider(port=port1, baudrate=baudrate)
        super().__init__(lidar0, lidar1, name0=name0, name1=name1)


# =========================================================
# 4) ROS 2 DUAL PROVIDER (OPTIONAL)
# =========================================================
class ROS2TwoLaserScanProvider(DualLidarWrapper):
    """
    Dual ROS2LaserScanProvider wrapper.

    NOTE: Your current ROS2LaserScanProvider in lidar_provider.py prints the topic
    but subscribes to '/scan' internally; once that file uses the topic parameter
    properly, you can pass two distinct topics here.

    For now, this wrapper is still useful if you have one topic, or you fix the provider later.
    """
    def __init__(
        self,
        topic0: str = "/scan0",
        topic1: str = "/scan1",
        name0: str = "front",
        name1: str = "side",
    ):
        lidar0 = ROS2LaserScanProvider(topic=topic0)
        lidar1 = ROS2LaserScanProvider(topic=topic1)
        super().__init__(lidar0, lidar1, name0=name0, name1=name1)


# =========================================================
# 5) Quick manual test
# =========================================================
def _demo_loop(provider: TwoLidarProvider, hz: float = 5.0) -> None:
    dt = 1.0 / max(hz, 0.1)
    print("[TwoLidar] Press Ctrl+C to stop.")
    try:
        while True:
            if hasattr(provider, "get_reading"):
                r = provider.get_reading()  # type: ignore[attr-defined]
                print(f"[TwoLidar] {r.lidar0_m=} {r.lidar1_m=} {r.fused_min_m=}")
            else:
                a, b = provider.get_distances_m()
                fused = provider.get_distance_m()
                print(f"[TwoLidar] lidar0={a} lidar1={b} fused_min={fused}")
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        provider.close()


if __name__ == "__main__":
    # Default to serial dual provider on Jetson.
    p = SerialTwoRPLidarProvider(port0="/dev/ttyUSB0", port1="/dev/ttyUSB1")
    _demo_loop(p, hz=5.0)
