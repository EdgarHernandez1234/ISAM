"""
Unit tests for waypoint_follower.

Run:
  pytest -q rover_learner/navigation/tests/test_waypoint_follower.py
"""
import math

from rover_learner.navigation.config import WaypointFollowerConfig
from rover_learner.navigation.types import Pose2D, Waypoint
from rover_learner.navigation.waypoint_follower import compute_twist_to_waypoint


def test_arrival_zero_twist():
    cfg = WaypointFollowerConfig(waypoint_radius_m=0.5)
    pose = Pose2D(0.0, 0.0, 0.0)
    wp = Waypoint(0.3, 0.4)  # dist=0.5
    out = compute_twist_to_waypoint(pose=pose, waypoint=wp, cfg=cfg)
    assert out.arrived is True
    assert out.twist.v_mps == 0.0
    assert out.twist.w_rps == 0.0


def test_forward_drive_small_heading_error():
    cfg = WaypointFollowerConfig(waypoint_radius_m=0.2, heading_err_stop_rad=0.65)
    pose = Pose2D(0.0, 0.0, 0.0)  # facing +x
    wp = Waypoint(3.0, 0.1)
    out = compute_twist_to_waypoint(pose=pose, waypoint=wp, cfg=cfg)
    assert out.arrived is False
    assert out.twist.v_mps > 0.0
    # heading error is small -> no pivot
    assert "PIVOT_TO_HEADING" not in out.reasons


def test_pivot_when_heading_error_large():
    cfg = WaypointFollowerConfig(heading_err_stop_rad=0.2, waypoint_radius_m=0.1)
    pose = Pose2D(0.0, 0.0, 0.0)
    wp = Waypoint(0.0, 2.0)  # 90 deg left
    out = compute_twist_to_waypoint(pose=pose, waypoint=wp, cfg=cfg)
    assert out.arrived is False
    assert out.twist.v_mps == 0.0
    assert "PIVOT_TO_HEADING" in out.reasons
    assert abs(out.twist.w_rps) > 0.0
