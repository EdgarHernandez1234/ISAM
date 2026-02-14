#!/usr/bin/env python3
"""rellis_lidar.py (rover_learner)

Compatibility wrapper.

In rover_decider/ (offline), rellis_lidar.py reads RELLIS-3D KITTI .bin frames.
In rover_learner/ (online), LiDAR is expected to be LIVE (ROS2 /scan).
We keep this file name so older imports keep working, but the implementation
delegates to lidar_provider.py.
"""

from __future__ import annotations

from .lidar_provider import (
    LaserScanLike,
    min_distance_from_scan,
    ROS2LaserScanProvider,
    MockLidarProvider,
)

__all__ = [
    "LaserScanLike",
    "min_distance_from_scan",
    "ROS2LaserScanProvider",
    "MockLidarProvider",
]
