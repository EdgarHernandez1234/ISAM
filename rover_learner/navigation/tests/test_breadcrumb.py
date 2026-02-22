"""
Unit tests for breadcrumb trail.

Run:
  pytest -q rover_learner/navigation/tests/test_breadcrumb.py
"""
from rover_learner.navigation.breadcrumb import BreadcrumbTrail
from rover_learner.navigation.config import BreadcrumbConfig
from rover_learner.navigation.types import Pose2D


def test_records_on_first_pose():
    trail = BreadcrumbTrail(BreadcrumbConfig(record_min_step_m=1.0))
    added = trail.record(Pose2D(0, 0, 0))
    assert added is True
    assert len(trail.poses) == 1


def test_record_min_step_spacing():
    trail = BreadcrumbTrail(BreadcrumbConfig(record_min_step_m=1.0))
    trail.record(Pose2D(0, 0, 0))
    # move less than 1m -> no new crumb
    assert trail.record(Pose2D(0.5, 0, 0)) is False
    assert len(trail.poses) == 1
    # move enough -> new crumb
    assert trail.record(Pose2D(1.01, 0, 0)) is True
    assert len(trail.poses) == 2


def test_return_steps_reverse_and_completes():
    trail = BreadcrumbTrail(BreadcrumbConfig(record_min_step_m=0.1, waypoint_radius_m=0.2))
    # make a simple line path
    for x in [0.0, 0.3, 0.6, 0.9]:
        trail.record(Pose2D(x, 0.0, 0.0))

    trail.begin_return()

    # start near last point; should advance toward earlier crumbs
    wp, done, reasons = trail.step_return(Pose2D(0.9, 0.0, 0.0))
    assert done is False
    assert wp is not None

    # simulate reaching crumbs in reverse
    while True:
        wp, done, reasons = trail.step_return(Pose2D(wp.x_m, wp.y_m, 0.0))  # jump to target
        if done:
            break
        assert wp is not None

    assert done is True
    assert "RETURN_COMPLETE" in reasons
