"""
Unit tests for lidar_avoidance.

Run:
  pytest -q rover_learner/navigation/tests/test_lidar_avoidance.py
"""
from rover_learner.navigation.lidar_avoidance import LidarAvoidanceController
from rover_learner.navigation.config import LidarAvoidanceConfig
from rover_learner.navigation.types import Twist2D


def test_no_lidar_pass_through():
    ctrl = LidarAvoidanceController(LidarAvoidanceConfig())
    out = ctrl.update(desired=Twist2D(0.2, 0.0), min_distance_m=None, timestamp_s=0.0)
    assert out.active is False
    assert out.twist.v_mps == 0.2


def test_slow_zone_reduces_speed():
    cfg = LidarAvoidanceConfig(slow_dist_m=2.0, stop_dist_m=1.0)
    ctrl = LidarAvoidanceController(cfg)
    out = ctrl.update(desired=Twist2D(0.3, 0.0), min_distance_m=1.5, timestamp_s=0.0)
    assert out.active is True
    assert 0.0 < out.twist.v_mps < 0.3


def test_stop_zone_stops_forward_and_turns():
    cfg = LidarAvoidanceConfig(slow_dist_m=2.0, stop_dist_m=1.0, stop_turn_rate_rps=0.6)
    ctrl = LidarAvoidanceController(cfg)
    out = ctrl.update(desired=Twist2D(0.3, 0.1), min_distance_m=0.8, timestamp_s=0.0)
    assert out.active is True
    assert out.twist.v_mps == 0.0
    assert abs(out.twist.w_rps) > 0.0
