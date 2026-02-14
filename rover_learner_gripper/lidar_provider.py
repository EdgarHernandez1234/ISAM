#!/usr/bin/env python3
"""
lidar_provider.py (rover_learner)

UNIVERSAL PROVIDER
- Includes: Serial (C1), ROS 2, and Mock providers.
- Essential for running unit tests.
"""

from __future__ import annotations

import math
import time
import threading
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

# 1. ROS 2 Imports (Optional)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

# 2. Serial Imports (roboticia / rplidar)
try:
    from rplidar import RPLidar, RPLidarException
    HAS_RPLIDAR = True
except ImportError:
    HAS_RPLIDAR = False


class LidarProvider(Protocol):
    def get_distance_m(self) -> Optional[float]:
        ...
    def close(self): ...


@dataclass(frozen=True)
class LaserScanLike:
    """Helper for non-ROS usage to mimic ROS message structure."""
    angle_min: float
    angle_increment: float
    ranges: Sequence[float]
    range_min: float = 0.0
    range_max: float = 100.0


def min_distance_from_scan(scan: LaserScanLike, forward_half_angle_deg: float = 20.0) -> Optional[float]:
    """Scans the forward cone (+/- 20 deg) for the closest object."""
    min_dist = float('inf')
    found_valid = False
    half_angle_rad = math.radians(forward_half_angle_deg)
    
    current_angle = scan.angle_min
    for r in scan.ranges:
        # Normalize angle to -pi..pi
        norm_angle = math.atan2(math.sin(current_angle), math.cos(current_angle))
        
        if abs(norm_angle) < half_angle_rad:
            # Filter noise (0.001) and infinity
            if r > 0.001 and not math.isinf(r) and not math.isnan(r):
                if r < min_dist:
                    min_dist = r
                    found_valid = True
        current_angle += scan.angle_increment

    return min_dist if found_valid else None


# =========================================================
# 1. MOCK PROVIDER (REQUIRED FOR TESTS)
# =========================================================
class MockLidarProvider:
    """Fake Lidar for testing without hardware."""
    def __init__(self, static_dist: float = 1.0):
        self.d = static_dist
        
    def get_distance_m(self) -> Optional[float]:
        return self.d
        
    def close(self):
        pass


# =========================================================
# 2. SERIAL PROVIDER (FOR JETSON / C1)
# =========================================================
class SerialRPLidarProvider:
    """
    Direct Hardware Connection.
    Hardcoded for Slamtec C1: Port=/dev/ttyUSB0, Baud=460800
    """
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 460800):
        if not HAS_RPLIDAR:
            raise RuntimeError("Missing driver! Run: pip3 install rplidar-roboticia")
            
        self._latest_distance: Optional[float] = None
        self._running = True
        self.port = port
        self.baudrate = baudrate
        
        print(f"[Lidar] Connecting to {port} at {baudrate} baud...")
        self._thread = threading.Thread(target=self._serial_worker, daemon=True)
        self._thread.start()

    def _serial_worker(self):
        lidar = None
        while self._running:
            try:
                lidar = RPLidar(self.port, baudrate=self.baudrate)
                # max_buf_meas=500 keeps latency low
                for scan in lidar.iter_scans(max_buf_meas=500):
                    if not self._running: break
                    
                    min_d = float('inf')
                    found = False
                    
                    for (_, angle, dist_mm) in scan:
                        if dist_mm == 0: continue
                        
                        # RPLidar Angle: 0=Front, increases clockwise
                        # Check front cone: 0-20 degrees OR 340-360 degrees
                        if angle < 20 or angle > 340:
                            d_m = dist_mm / 1000.0
                            if d_m < min_d:
                                min_d = d_m
                                found = True
                                
                    if found:
                        self._latest_distance = min_d
                    else:
                        self._latest_distance = None

            except Exception as e:
                # Wait before retrying to prevent spamming logs
                time.sleep(1.0)
                try:
                    if lidar:
                        lidar.stop()
                        lidar.disconnect()
                except: pass
                lidar = None

    def get_distance_m(self) -> Optional[float]:
        return self._latest_distance

    def close(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


# =========================================================
# 3. ROS 2 PROVIDER (OPTIONAL)
# =========================================================
class ROS2LaserScanProvider:
    """Standard ROS 2 Subscriber"""
    def __init__(self, topic: str = "/scan"):
        if not HAS_ROS2: raise RuntimeError("ROS 2 not installed")
        self._latest_distance = None
        self._last_msg_time = 0.0
        self._lock = threading.Lock()
        
        threading.Thread(target=self._ros_worker, daemon=True).start()
        print(f"[Lidar] Listening to ROS 2 topic: {topic}")

    def _ros_worker(self):
        if not rclpy.ok(): rclpy.init()
        node = rclpy.create_node("lidar_listener_" + str(int(time.time())))
        node.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        try: rclpy.spin(node)
        except: pass
        finally: node.destroy_node()

    def _on_scan(self, msg):
        wrapper = LaserScanLike(msg.angle_min, msg.angle_increment, msg.ranges)
        d = min_distance_from_scan(wrapper)
        with self._lock:
            self._latest_distance = d
            self._last_msg_time = time.time()

    def get_distance_m(self):
        with self._lock:
            if time.time() - self._last_msg_time > 2.0: return None
            return self._latest_distance
    
    def close(self): pass