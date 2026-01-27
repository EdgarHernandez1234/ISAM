#!/usr/bin/env python3
"""
lidar_provider.py

A small, testable "distance provider" module with a single goal:
    get_distance_m() -> float | None

It supports three practical sources:
  1) Offline RELLIS-3D KITTI .bin point cloud frames (works today)
  2) ROS2 /scan (sensor_msgs/LaserScan)                  (2D LiDAR)
  3) ROS2 /points (sensor_msgs/PointCloud2)             (3D LiDAR)

Metric:
  min_distance_m in a forward sector (e.g., -15°..+15°), robustified by:
    - drop zeros/inf/NaN
    - median of last N samples (pure Python)

Run (examples):
  Offline (RELLIS KITTI bin directory or "virtual zip path"):
    python lidar_provider.py --mode rellis_bin --source "<path>"
    
    for us: 
      python lidar_provider.py --mode rellis_bin --source "G:\.shortcut-targets-by-id\1aZ1tJ3YYcWuL3oWKnrTIC5gq46zx1bMc\Rellis-3D_Release\Rellis_3D_os1_cloud_node_kitti_bin.zip\Rellis-3D\00000\os1_cloud_node_kitti_bin"

  ROS2 LaserScan:
    # in a terminal where you've sourced ROS2 and your LiDAR workspace
    python lidar_provider.py --mode ros2_scan --topic /scan
    
    ex) python lidar_provider.py --mode ros2_scan --topic /scan --rate-hz 10


  ROS2 PointCloud2:
    python lidar_provider.py --mode ros2_points --topic /points
    ex) python lidar_provider.py --mode ros2_points --topic /points --rate-hz 10 --stride 10


Unit tests (no ROS required):
  python lidar_provider.py --test

Notes:
- ROS2 providers require rclpy and message packages installed.
- PointCloud2 parsing uses sensor_msgs_py.point_cloud2 (no numpy).
- Reading from a large ZIP on a Google Drive mount may be slow; extraction is faster.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
import unittest
import zipfile
from array import array
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Deque, Iterable, List, Optional, Tuple, Union
from collections import deque


# -----------------------------
# Small utilities (pure Python)
# -----------------------------

def median(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return 0.5 * (vals[mid - 1] + vals[mid])


def is_finite_positive(x: float) -> bool:
    return x is not None and math.isfinite(x) and x > 0.0


# -----------------------------
# Core computations (testable)
# -----------------------------

def min_distance_from_scan_forward_sector(
    ranges: List[float],
    angle_min: float,
    angle_increment: float,
    fov_deg: float = 30.0,
    max_range_m: float = 10.0,
) -> Optional[float]:
    """
    Compute min distance from a LaserScan within a centered forward sector:
      angles in [-fov/2, +fov/2] relative to forward direction 0 rad.

    Inputs:
      ranges: LaserScan.ranges (meters)
      angle_min: LaserScan.angle_min (rad)
      angle_increment: LaserScan.angle_increment (rad)
    """
    if not ranges or angle_increment == 0.0:
        return None

    half = math.radians(float(fov_deg)) / 2.0
    best = None

    for i, r in enumerate(ranges):
        if not is_finite_positive(r):
            continue
        if r > max_range_m:
            continue
        ang = angle_min + i * angle_increment
        if abs(ang) > half:
            continue
        if best is None or r < best:
            best = r

    return float(best) if best is not None else None


def min_distance_from_points_forward_sector(
    points_xyz: Iterable[Tuple[float, float, float]],
    fov_deg: float = 30.0,
    max_range_m: float = 10.0,
    stride: int = 5,
) -> Optional[float]:
    """
    Compute min planar distance sqrt(x^2 + y^2) from a 3D point stream in meters,
    restricted to:
      - forward hemisphere (x > 0)
      - yaw angle within +- fov/2
      - planar distance <= max_range_m
    """
    half = math.radians(float(fov_deg)) / 2.0
    max_r2 = float(max_range_m) * float(max_range_m)

    best_r2 = None
    stride = max(1, int(stride))

    for idx, (x, y, z) in enumerate(points_xyz):
        if idx % stride != 0:
            continue
        if x is None or y is None:
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if x <= 0.0:
            continue
        ang = math.atan2(y, x)
        if abs(ang) > half:
            continue
        r2 = x * x + y * y
        if r2 <= 0.0 or r2 > max_r2:
            continue
        if best_r2 is None or r2 < best_r2:
            best_r2 = r2

    return math.sqrt(best_r2) if best_r2 is not None else None


# -----------------------------
# Provider interface
# -----------------------------

class LidarProvider:
    def get_distance_m(self) -> Optional[float]:
        raise NotImplementedError


@dataclass
class SmoothingConfig:
    window: int = 5
    timeout_s: float = 1.0  # None if no update within this duration


class SmoothedMetric:
    """Keeps a rolling history and returns median smoothing."""
    def __init__(self, cfg: SmoothingConfig):
        self.cfg = cfg
        self.hist: Deque[float] = deque(maxlen=max(1, int(cfg.window)))
        self.last_update_ts: float = 0.0

    def push(self, v: Optional[float]) -> None:
        if v is None:
            return
        self.hist.append(float(v))
        self.last_update_ts = time.time()

    def get(self) -> Optional[float]:
        if not self.hist:
            return None
        if self.cfg.timeout_s is not None and self.cfg.timeout_s > 0:
            if (time.time() - self.last_update_ts) > float(self.cfg.timeout_s):
                return None
        return median(list(self.hist))


# -----------------------------
# Offline: RELLIS KITTI bin (.bin float32 x,y,z,intensity)
# -----------------------------

def split_zip_virtual_path(path_str: str) -> Optional[Tuple[str, str]]:
    """
    Parse Windows Explorer "virtual" zip path:
      C:\\path\\file.zip\\inner\\folder
    -> (zip_path, inner_prefix_posix_ending_with_slash)

    If ".zip" is not present, return None.
    """
    if not path_str:
        return None
    lower = path_str.lower()
    if ".zip" not in lower:
        return None
    idx = lower.index(".zip") + 4
    zip_path = path_str[:idx]
    inner = path_str[idx:].lstrip("\\/").replace("\\", "/")
    if inner and not inner.endswith("/"):
        inner += "/"
    return zip_path, inner


def list_kitti_bins(source_path: str) -> List[Tuple[str, Union[Path, Tuple[str, str]]]]:
    """
    Returns references for reading:
      ("dir", Path(...)) or ("zip", (zip_path, member_name))
    """
    z = split_zip_virtual_path(source_path)
    if z is not None:
        zip_path, inner_prefix = z
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")
        refs = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if inner_prefix and not name.startswith(inner_prefix):
                    continue
                if name.lower().endswith(".bin"):
                    refs.append(("zip", (zip_path, name)))
        if not refs:
            raise FileNotFoundError(f"No .bin found under zip prefix '{inner_prefix}'")
        return refs

    d = Path(source_path)
    if not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {source_path}")
    bins = sorted(d.glob("*.bin"))
    if not bins:
        raise FileNotFoundError(f"No .bin files in directory: {source_path}")
    return [("dir", p) for p in bins]


def read_kitti_bin_floats(ref: Tuple[str, Union[Path, Tuple[str, str]]]) -> array:
    kind, payload = ref
    data = array("f")

    if kind == "dir":
        path = payload  # type: ignore
        assert isinstance(path, Path)
        n_floats = path.stat().st_size // 4
        with open(str(path), "rb") as f:
            data.fromfile(f, n_floats)
        return data

    if kind == "zip":
        zip_path, member = payload  # type: ignore
        assert isinstance(zip_path, str) and isinstance(member, str)
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(member, "r") as f:
                raw = f.read()
        data.frombytes(raw)
        return data

    raise ValueError(f"Unknown ref kind: {kind}")


def floats_to_xyz_iter(floats: array) -> Iterable[Tuple[float, float, float]]:
    """
    Convert float array (x,y,z,i repeated) to an iterator of (x,y,z).
    """
    n = len(floats)
    for i in range(0, n - 3, 4):
        yield floats[i + 0], floats[i + 1], floats[i + 2]


class RellisBinProvider(LidarProvider):
    def __init__(
        self,
        source_path: str,
        fov_deg: float = 30.0,
        max_range_m: float = 10.0,
        stride: int = 5,
        smoothing: SmoothingConfig = SmoothingConfig(),
    ):
        self.refs = list_kitti_bins(source_path)
        self.fov_deg = float(fov_deg)
        self.max_range_m = float(max_range_m)
        self.stride = int(stride)
        self.metric = SmoothedMetric(smoothing)

    def get_distance_m(self) -> Optional[float]:
        ref = random.choice(self.refs)
        floats = read_kitti_bin_floats(ref)
        d = min_distance_from_points_forward_sector(
            floats_to_xyz_iter(floats),
            fov_deg=self.fov_deg,
            max_range_m=self.max_range_m,
            stride=self.stride,
        )
        self.metric.push(d)
        return self.metric.get()


# -----------------------------
# ROS2 providers
# -----------------------------

class ROS2LaserScanProvider(LidarProvider):
    """
    ROS2 LaserScan provider using rclpy.
    Call start() once, then call get_distance_m() any time.
    """
    def __init__(
        self,
        topic: str = "/scan",
        fov_deg: float = 30.0,
        max_range_m: float = 10.0,
        smoothing: SmoothingConfig = SmoothingConfig(),
    ):
        self.topic = topic
        self.fov_deg = float(fov_deg)
        self.max_range_m = float(max_range_m)
        self.metric = SmoothedMetric(smoothing)
        self._node = None
        self._sub = None

    def start(self) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import LaserScan

        if not rclpy.ok():
            rclpy.init(args=None)

        class _ScanNode(Node):
            def __init__(self, outer: "ROS2LaserScanProvider"):
                super().__init__("lidar_provider_scan")
                self.outer = outer
                self.sub = self.create_subscription(LaserScan, outer.topic, self.cb, 10)

            def cb(self, msg: LaserScan):
                d = min_distance_from_scan_forward_sector(
                    list(msg.ranges),
                    msg.angle_min,
                    msg.angle_increment,
                    fov_deg=self.outer.fov_deg,
                    max_range_m=self.outer.max_range_m,
                )
                self.outer.metric.push(d)

        self._node = _ScanNode(self)
        self._sub = self._node.sub

    def spin_once(self, timeout_s: float = 0.1) -> None:
        import rclpy
        if self._node is None:
            raise RuntimeError("ROS2LaserScanProvider.start() must be called first.")
        rclpy.spin_once(self._node, timeout_sec=float(timeout_s))

    def get_distance_m(self) -> Optional[float]:
        return self.metric.get()

    def shutdown(self) -> None:
        import rclpy
        if self._node is not None:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self._node = None
        self._sub = None


class ROS2PointCloud2Provider(LidarProvider):
    """
    ROS2 PointCloud2 provider using rclpy + sensor_msgs_py.point_cloud2 (no numpy).
    """
    def __init__(
        self,
        topic: str = "/points",
        fov_deg: float = 30.0,
        max_range_m: float = 10.0,
        stride: int = 10,
        smoothing: SmoothingConfig = SmoothingConfig(),
    ):
        self.topic = topic
        self.fov_deg = float(fov_deg)
        self.max_range_m = float(max_range_m)
        self.stride = int(stride)
        self.metric = SmoothedMetric(smoothing)
        self._node = None
        self._sub = None

    def start(self) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs_py import point_cloud2

        if not rclpy.ok():
            rclpy.init(args=None)

        class _PCNode(Node):
            def __init__(self, outer: "ROS2PointCloud2Provider"):
                super().__init__("lidar_provider_points")
                self.outer = outer
                self.sub = self.create_subscription(PointCloud2, outer.topic, self.cb, 10)

            def cb(self, msg: PointCloud2):
                # read_points yields tuples of (x,y,z) if field_names specified.
                pts = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
                d = min_distance_from_points_forward_sector(
                    pts,
                    fov_deg=self.outer.fov_deg,
                    max_range_m=self.outer.max_range_m,
                    stride=self.outer.stride,
                )
                self.outer.metric.push(d)

        self._node = _PCNode(self)
        self._sub = self._node.sub

    def spin_once(self, timeout_s: float = 0.1) -> None:
        import rclpy
        if self._node is None:
            raise RuntimeError("ROS2PointCloud2Provider.start() must be called first.")
        rclpy.spin_once(self._node, timeout_sec=float(timeout_s))

    def get_distance_m(self) -> Optional[float]:
        return self.metric.get()

    def shutdown(self) -> None:
        import rclpy
        if self._node is not None:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self._node = None
        self._sub = None


# -----------------------------
# CLI runner
# -----------------------------

def run_cli() -> int:
    ap = argparse.ArgumentParser(description="LiDAR distance provider (offline RELLIS + ROS2 scan/points)")
    ap.add_argument("--test", action="store_true", help="Run unit tests and exit")

    ap.add_argument("--mode", choices=["rellis_bin", "ros2_scan", "ros2_points"], default="rellis_bin")
    ap.add_argument("--source", default="", help="(rellis_bin) Directory of .bin or virtual zip path ...zip\\inner\\folder")
    ap.add_argument("--topic", default="/scan", help="(ros2_*) Topic name (/scan or /points)")
    ap.add_argument("--fov-deg", type=float, default=30.0)
    ap.add_argument("--max-range-m", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=5, help="(points/bin) sample every N points")
    ap.add_argument("--window", type=int, default=5, help="Median window size")
    ap.add_argument("--timeout-s", type=float, default=1.0, help="If no update within this time, returns None")
    ap.add_argument("--rate-hz", type=float, default=10.0, help="Print rate for demo")

    args = ap.parse_args()

    if args.test:
        return run_tests()

    smoothing = SmoothingConfig(window=args.window, timeout_s=args.timeout_s)

    if args.mode == "rellis_bin":
        if not args.source:
            print("[ERROR] --source is required for mode=rellis_bin")
            return 2
        p = RellisBinProvider(
            args.source,
            fov_deg=args.fov_deg,
            max_range_m=args.max_range_m,
            stride=args.stride,
            smoothing=smoothing,
        )
        # Offline provider doesn't need spin; just poll
        dt = 1.0 / float(args.rate_hz)
        while True:
            d = p.get_distance_m()
            print(f"distance_m={d:.3f}" if d is not None else "distance_m=None")
            time.sleep(dt)

    if args.mode == "ros2_scan":
        p = ROS2LaserScanProvider(
            topic=args.topic,
            fov_deg=args.fov_deg,
            max_range_m=args.max_range_m,
            smoothing=smoothing,
        )
        p.start()
        dt = 1.0 / float(args.rate_hz)
        try:
            while True:
                p.spin_once(timeout_s=min(0.2, dt))
                d = p.get_distance_m()
                print(f"distance_m={d:.3f}" if d is not None else "distance_m=None")
                time.sleep(dt)
        finally:
            p.shutdown()

    if args.mode == "ros2_points":
        p = ROS2PointCloud2Provider(
            topic=args.topic,
            fov_deg=args.fov_deg,
            max_range_m=args.max_range_m,
            stride=args.stride,
            smoothing=smoothing,
        )
        p.start()
        dt = 1.0 / float(args.rate_hz)
        try:
            while True:
                p.spin_once(timeout_s=min(0.2, dt))
                d = p.get_distance_m()
                print(f"distance_m={d:.3f}" if d is not None else "distance_m=None")
                time.sleep(dt)
        finally:
            p.shutdown()

    return 0


# -----------------------------
# Unit tests (no ROS required)
# -----------------------------

def _floats_to_bin_bytes(points: List[Tuple[float, float, float, float]]) -> bytes:
    a = array("f")
    for x, y, z, i in points:
        a.extend([x, y, z, i])
    return a.tobytes()


class TestScanMetric(unittest.TestCase):
    def test_min_distance_scan_forward_sector(self):
        # angles: -0.2, 0.0, +0.2 rad (~-11.5°, 0°, +11.5°)
        ranges = [2.0, 1.0, 3.0]
        d = min_distance_from_scan_forward_sector(ranges, angle_min=-0.2, angle_increment=0.2, fov_deg=30, max_range_m=10)
        self.assertAlmostEqual(d or 0.0, 1.0, places=6)

    def test_scan_filters_inf_and_zero(self):
        ranges = [float("inf"), 0.0, 1.5]
        d = min_distance_from_scan_forward_sector(ranges, angle_min=-0.2, angle_increment=0.2, fov_deg=90, max_range_m=10)
        self.assertAlmostEqual(d or 0.0, 1.5, places=6)

    def test_scan_fov_excludes(self):
        # point only at 40 degrees should be excluded with fov 30
        # approximate: angle_min=0.7 rad (~40°), one sample at 0.7
        d = min_distance_from_scan_forward_sector([1.0], angle_min=0.7, angle_increment=0.0, fov_deg=30, max_range_m=10)
        self.assertIsNone(d)


class TestPointsMetric(unittest.TestCase):
    def test_min_distance_points_forward(self):
        pts = [(2.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
        d = min_distance_from_points_forward_sector(pts, fov_deg=60, max_range_m=10, stride=1)
        self.assertAlmostEqual(d or 0.0, 1.0, places=6)

    def test_points_fov_excludes(self):
        # 45 degree point excluded for fov 30 (half 15)
        pts = [(1.0, 1.0, 0.0)]
        d = min_distance_from_points_forward_sector(pts, fov_deg=30, max_range_m=10, stride=1)
        self.assertIsNone(d)

    def test_points_range_excludes(self):
        pts = [(20.0, 0.0, 0.0)]
        d = min_distance_from_points_forward_sector(pts, fov_deg=60, max_range_m=10, stride=1)
        self.assertIsNone(d)


class TestZipPathParsing(unittest.TestCase):
    def test_split_zip_virtual(self):
        p = r"C:\data\rellis.zip\Rellis-3D\00000\os1_cloud_node_kitti_bin"
        z = split_zip_virtual_path(p)
        self.assertIsNotNone(z)
        zip_path, inner = z  # type: ignore
        self.assertTrue(zip_path.lower().endswith(".zip"))
        self.assertIn("Rellis-3D/00000/os1_cloud_node_kitti_bin/", inner)

    def test_list_bins_from_zip_prefix(self):
        with TemporaryDirectory() as td:
            zip_path = Path(td) / "test.zip"
            prefix = "Rellis-3D/00000/os1_cloud_node_kitti_bin/"
            member1 = prefix + "000000.bin"
            member2 = prefix + "000001.bin"
            with zipfile.ZipFile(str(zip_path), "w") as zf:
                zf.writestr(member1, _floats_to_bin_bytes([(1.0, 0.0, 0.0, 0.0)]))
                zf.writestr(member2, _floats_to_bin_bytes([(2.0, 0.0, 0.0, 0.0)]))

            virtual = str(zip_path) + "\\" + prefix.replace("/", "\\").rstrip("\\")
            refs = list_kitti_bins(virtual)
            self.assertEqual(len(refs), 2)
            self.assertTrue(all(r[0] == "zip" for r in refs))


def run_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
