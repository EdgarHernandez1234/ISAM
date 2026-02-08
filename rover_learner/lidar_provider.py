#!/usr/bin/env python3
"""lidar_provider.py (rover_learner)

Live LiDAR support.
Includes a hybrid provider that supports:
  1. ROS2 /scan (preferred if available)
  2. Direct Serial/USB RPLidar (fallback with FAST Auto-Baud detection)

Unit-testable aspects:
  - min_distance_from_scan() pure function
  - MockLidarProvider
"""

from __future__ import annotations

import math
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Protocol, Any, Sequence, Tuple

# Try to import providers to detect what is available
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

try:
    from rplidar import RPLidar, RPLidarException
    HAS_RPLIDAR = True
except ImportError:
    HAS_RPLIDAR = False


class LidarProvider(Protocol):
    def get_distance_m(self) -> Optional[float]:
        """Return minimum forward distance in meters (or None if unavailable)."""
        ...


@dataclass(frozen=True)
class LaserScanLike:
    angle_min: float
    angle_increment: float
    ranges: Sequence[float]
    range_min: float = 0.0
    range_max: float = 100.0


def min_distance_from_scan(
    scan: LaserScanLike,
    forward_half_angle_deg: float = 15.0,
    invalid_values: Tuple[float, ...] = (0.0,),
) -> Optional[float]:
    """Compute min distance within a forward cone from a dense scan array."""
    half = math.radians(float(forward_half_angle_deg))
    best: Optional[float] = None

    for i, r in enumerate(scan.ranges):
        if r is None:
            continue
        try:
            r = float(r)
        except Exception:
            continue
        if math.isnan(r) or math.isinf(r):
            continue
        if r in invalid_values:
            continue
        if r < float(scan.range_min) or r > float(scan.range_max):
            continue

        angle = float(scan.angle_min) + i * float(scan.angle_increment)
        # normalize angle to [-pi, pi]
        angle = math.atan2(math.sin(angle), math.cos(angle))
        if abs(angle) > half:
            continue

        if best is None or r < best:
            best = r

    return best


class MockLidarProvider:
    def __init__(self, distance_m: Optional[float] = 2.0):
        self.distance_m = distance_m
    def get_distance_m(self) -> Optional[float]:
        return self.distance_m


class ROS2LaserScanProvider:
    """
    Hybrid Provider:
    - Attempts to use ROS2 ('rclpy') first.
    - If ROS2 is missing, falls back to direct 'rplidar' serial driver.
    """

    def __init__(self, topic: str = "/scan", forward_half_angle_deg: float = 15.0):
        self.topic = str(topic)
        self.forward_half_angle_deg = float(forward_half_angle_deg)
        self._latest_distance: Optional[float] = None
        self._started = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._mode = "none"  # 'ros2' or 'serial'

        self.serial_port = "/dev/ttyUSB0" 
        # C1 uses 460800. We prioritize it to save time during checks.
        self.baud_candidates = [460800, 256000, 115200]

    def start(self) -> None:
        if self._started:
            return

        # 1. Try ROS 2
        if HAS_ROS2:
            try:
                self._start_ros2()
                self._mode = "ros2"
                self._started = True
                print(f"[lidar] Started via ROS2 topic {self.topic}")
                return
            except Exception as e:
                print(f"[lidar] ROS2 failed to init: {e}. Trying fallback...")
        
        # 2. Try Direct Serial (RPLidar)
        if HAS_RPLIDAR:
            try:
                self._start_serial()
                self._mode = "serial"
                self._started = True
                print(f"[lidar] Starting Direct Serial ({self.serial_port})...")
                return
            except Exception as e:
                raise RuntimeError(f"Failed to start RPLidar serial driver: {e}") from e

        # 3. Failure
        raise RuntimeError(
            "Could not start LiDAR. \n"
            " - ROS2 not found (checked for 'rclpy').\n"
            " - RPLidar library not found (checked for 'rplidar').\n"
            "Fix: 'pip3 install rplidar-roboticia' OR install ROS2."
        )

    def _start_ros2(self):
        rclpy.init(args=None)
        self._rclpy = rclpy
        
        class _ScanNode(Node):
            def __init__(self, outer: "ROS2LaserScanProvider"):
                super().__init__("rover_learner_lidar_provider")
                self.outer = outer
                self.create_subscription(LaserScan, outer.topic, self._cb, 10)

            def _cb(self, msg):
                scan = LaserScanLike(
                    angle_min=float(msg.angle_min),
                    angle_increment=float(msg.angle_increment),
                    ranges=list(msg.ranges),
                    range_min=float(msg.range_min),
                    range_max=float(msg.range_max),
                )
                self.outer._latest_distance = min_distance_from_scan(scan, self.outer.forward_half_angle_deg)

        self._node = _ScanNode(self)
        self._thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._thread.start()

    def _start_serial(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._serial_worker, daemon=True)
        self._thread.start()

    def _serial_worker(self):
        lidar = None
        half_rad = math.radians(self.forward_half_angle_deg)
        
        # --- Connection / Auto-Baud Phase ---
        connected = False
        for baud in self.baud_candidates:
            if self._stop_event.is_set(): break
            try:
                # print(f"[lidar] Probe {baud} baud...")
                lidar = RPLidar(self.serial_port, baudrate=baud, timeout=1.0)
                info = lidar.get_info() 
                print(f"[lidar] Connected at {baud}! Info: {info}")
                connected = True
                break
            except Exception as e:
                if lidar:
                    try:
                        lidar.stop()
                        lidar.disconnect()
                    except:
                        pass
                lidar = None
                time.sleep(0.05)

        if not connected or lidar is None:
            print("[lidar] Could not connect to any supported baud rate within timeout.")
            return

        # --- Clean Start Sequence ---
        # Critical for C1/S1 high-speed sensors to prevent "Descriptor mismatch"
        try:
            lidar.stop()           # Stop any previous scan
            lidar.stop_motor()     # Spin down
            time.sleep(0.1)
            lidar.clean_input()    # Flush garbage
            lidar.start_motor()    # Spin up
            time.sleep(0.1)
        except Exception as e:
            print(f"[lidar] Warning during clean start: {e}")

        # --- Scanning Phase ---
        lidar.timeout = 2.0
        
        while not self._stop_event.is_set():
            try:
                # max_buf_meas=500 helps latency. 
                # If mismatch happens, we catch RPLidarException below.
                for scan in lidar.iter_scans(max_buf_meas=500, min_len=5):
                    if self._stop_event.is_set():
                        break
                    
                    min_d_mm = None
                    for (_, angle_deg, dist_mm) in scan:
                        if dist_mm <= 0: continue
                        
                        a = angle_deg
                        if a > 180: a -= 360
                        rad = math.radians(a)
                        
                        if abs(rad) <= half_rad:
                            if min_d_mm is None or dist_mm < min_d_mm:
                                min_d_mm = dist_mm
                    
                    if min_d_mm is not None:
                        self._latest_distance = min_d_mm / 1000.0
                        
            except RPLidarException as e:
                print(f"[lidar] Resyncing ({e})...")
                try:
                    lidar.stop()
                    lidar.clean_input()
                    # Don't sleep too long, just enough to clear buffer
                except:
                    pass
                continue 
                
            except Exception as e:
                print(f"[lidar] Critical serial worker error: {e}")
                # Try to reconnect or break? Breaking usually safer to avoid spam.
                break
        
        # Cleanup
        if lidar:
            try:
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
            except:
                pass

    def get_distance_m(self) -> Optional[float]:
        return self._latest_distance

    def close(self) -> None:
        if not self._started:
            return
            
        self._stop_event.set()
        
        if self._mode == "ros2":
            try:
                self._node.destroy_node()
                self._rclpy.shutdown()
            except Exception:
                pass
        
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            
        self._started = False