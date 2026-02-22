#!/usr/bin/env python3
"""
Minimal test runner (no pytest). Run from repo root:

  python3 -m rover_learner.navigation.tests.run_tests
"""
import traceback

from rover_learner.navigation.tests.test_waypoint_follower import (
    test_arrival_zero_twist,
    test_forward_drive_small_heading_error,
    test_pivot_when_heading_error_large,
)
from rover_learner.navigation.tests.test_breadcrumb import (
    test_records_on_first_pose,
    test_record_min_step_spacing,
    test_return_steps_reverse_and_completes,
)
from rover_learner.navigation.tests.test_lidar_avoidance import (
    test_no_lidar_pass_through,
    test_slow_zone_reduces_speed,
    test_stop_zone_stops_forward_and_turns,
)
from rover_learner.navigation.tests.test_navigation_smoke import (
    test_navigator_search_route_smoke,
    test_navigator_avoidance_engages,
)

TESTS = [
    test_arrival_zero_twist,
    test_forward_drive_small_heading_error,
    test_pivot_when_heading_error_large,
    test_records_on_first_pose,
    test_record_min_step_spacing,
    test_return_steps_reverse_and_completes,
    test_no_lidar_pass_through,
    test_slow_zone_reduces_speed,
    test_stop_zone_stops_forward_and_turns,
    test_navigator_search_route_smoke,
    test_navigator_avoidance_engages,
]


def main() -> int:
    passed = 0
    failed = 0

    for t in TESTS:
        name = t.__name__
        try:
            t()
            print(f"[PASS] {name}")
            passed += 1
        except Exception:
            print(f"[FAIL] {name}")
            traceback.print_exc()
            failed += 1

    print(f"\nSummary: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
