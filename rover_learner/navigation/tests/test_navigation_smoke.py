"""
Smoke test for navigation orchestrator.

Run:
  pytest -q rover_learner/navigation/tests/test_navigation_smoke.py
"""
import time

from rover_learner.navigation.navigation import Navigator
from rover_learner.navigation.types import NavMode, Waypoint


def test_navigator_search_route_smoke():
    nav = Navigator()
    nav.set_route([Waypoint(0.0, 0.0), Waypoint(2.0, 0.0), Waypoint(2.0, 2.0)])

    # Provide a few encoder updates and steps
    t0 = 1000.0
    nav.update_encoders_ticks(left_ticks=0, right_ticks=0, timestamp_s=t0)

    # Far from obstacles
    nav.update_lidar(min_distance_m=5.0)

    p = nav.step(nav_mode=NavMode.SEARCH_ROUTE, timestamp_s=t0, vision={"harvest_intent": False})
    assert p.status.value in ("RUNNING", "ARRIVED", "BLOCKED")
    assert "nav_mode" in p.debug
    assert p.debug["nav_mode"] == NavMode.SEARCH_ROUTE.value

    # Simulate motion via encoder ticks (small forward)
    nav.update_encoders_ticks(left_ticks=200, right_ticks=200, timestamp_s=t0 + 0.1)
    p2 = nav.step(nav_mode=NavMode.SEARCH_ROUTE, timestamp_s=t0 + 0.1, vision={"harvest_intent": False})
    assert p2.status.value in ("RUNNING", "ARRIVED", "BLOCKED")


def test_navigator_avoidance_engages():
    nav = Navigator()
    nav.set_route([Waypoint(5.0, 0.0)])

    t0 = 2000.0
    nav.update_encoders_ticks(left_ticks=0, right_ticks=0, timestamp_s=t0)
    # Obstacle very close -> avoidance should stop forward motion (v=0)
    nav.update_lidar(min_distance_m=0.6)

    p = nav.step(nav_mode=NavMode.SEARCH_ROUTE, timestamp_s=t0, vision={"harvest_intent": False})
    assert p.debug["avoid_active"] is True
    assert p.twist.v_mps == 0.0
