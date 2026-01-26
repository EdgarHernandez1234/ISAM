#!/usr/bin/env python3
"""
MyCobot 280-Pi Flex Demo (pymycobot + optional rospy) + Unit Tests

Features:
- Safe "flex" sequence of small joint deltas
- Returns to HOME pose
- Clean stop + release servos on Ctrl+C or error
- Optional ROS node:
  - Publishes /mycobot/angles (Float32MultiArray)
  - Services:
      /mycobot/run_flex_demo (std_srvs/Trigger)
      /mycobot/go_home       (std_srvs/Trigger)

Unit tests:
- Run without hardware using a FakeMyCobot simulator.
- To run tests:
    python3 cobot_flex_demo.py --test
- To run hardware smoke tests (requires arm):
    python3 cobot_flex_demo.py --hw-smoke

Conservative run:
python3 cobot_flex_demo.py --hw-smoke --hw-move --speed 20

Full Demo:
python3 cobot_flex_demo.py --port /dev/ttyAMA0 --baud 1000000 --speed 30 --repeat 2

Connection pattern matches your working script:
MyCobot('/dev/ttyAMA0', 1000000)
"""

import argparse
import sys
import time
import unittest
from typing import List, Optional, Tuple

# Import pymycobot only when needed for real hardware
try:
    from pymycobot.mycobot import MyCobot  # type: ignore
except Exception:
    MyCobot = None  # Allows tests to run on machines without pymycobot installed


# ----------------------------
# Utility helpers (no numpy)
# ----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def within_tol(a: List[float], b: List[float], tol: float = 2.0) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    return all(abs(ai - bi) <= tol for ai, bi in zip(a, b))

def median(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    vals = sorted(vals)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return 0.5 * (vals[mid - 1] + vals[mid])

def safe_get_angles(mc, retries: int = 3, delay_s: float = 0.2) -> Optional[List[float]]:
    for _ in range(retries):
        ang = mc.get_angles()
        if isinstance(ang, list) and len(ang) == 6 and all(x is not None for x in ang):
            return ang
        time.sleep(delay_s)
    return None

def wait_until_reached(mc, target: List[float], timeout_s: float = 10.0, tol_deg: float = 2.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        cur = safe_get_angles(mc, retries=1, delay_s=0.0)
        if cur and within_tol(cur, target, tol_deg):
            return True
        time.sleep(0.05)
    return False


# ----------------------------
# Demo motion planning
# ----------------------------

DEFAULT_SOFT_LIMITS: List[Tuple[float, float]] = [
    (-170, 170),  # J1
    (-120, 120),  # J2
    (-170, 170),  # J3
    (-170, 170),  # J4
    (-170, 170),  # J5
    (-180, 180),  # J6
]

def make_flex_poses(home: List[float], limits=DEFAULT_SOFT_LIMITS) -> List[List[float]]:
    """
    Generates a conservative "flex" routine:
    small deltas, alternating joints, always clamped to soft limits.
    """
    if home is None or len(home) != 6:
        raise ValueError("home must be a 6-element angle list")

    seq_deltas = [
        [ 10,   0,   0,   0,   0,  15],
        [-10,   0,   0,   0,   0, -15],
        [  0,  10, -10,   0,   0,   0],
        [  0, -10,  10,   0,   0,   0],
        [  0,   0,   0,  15, -10,   0],
        [  0,   0,   0, -15,  10,   0],
    ]

    poses: List[List[float]] = []
    for d in seq_deltas:
        p = []
        for i in range(6):
            lo, hi = limits[i]
            p.append(clamp(home[i] + d[i], lo, hi))
        poses.append(p)
    return poses


# ----------------------------
# Core demo
# ----------------------------

def connect_mycobot(port: str, baud: int):
    if MyCobot is None:
        raise RuntimeError("pymycobot is not available in this environment.")
    mc = MyCobot(port, baud)
    time.sleep(0.5)
    return mc

def ensure_ready(mc) -> None:
    try:
        mc.power_on()
        time.sleep(0.2)
    except Exception:
        pass

def safe_stop(mc) -> None:
    try:
        mc.stop()
    except Exception:
        pass
    try:
        mc.release_all_servos()
    except Exception:
        pass

def run_flex_demo(
    mc,
    speed: int = 30,
    hold_s: float = 0.5,
    repeat: int = 1,
    timeout_s: float = 10.0,
    tol_deg: float = 2.0,
    home_angles: Optional[List[float]] = None,
    verbose: bool = True,
) -> bool:
    """
    Executes flex poses then returns to home.
    Returns True if completed without timeouts.
    """
    ensure_ready(mc)

    if home_angles is None:
        home_angles = safe_get_angles(mc)
        if home_angles is None:
            if verbose:
                print("[ERROR] Could not read initial angles; aborting.")
            return False

    if verbose:
        print("[INFO] HOME angles:", home_angles)

    poses = make_flex_poses(home_angles)

    ok = True
    for r in range(repeat):
        if verbose:
            print(f"[INFO] Flex cycle {r+1}/{repeat}")
        for idx, p in enumerate(poses, start=1):
            if verbose:
                print(f"  -> Pose {idx}/{len(poses)}: {p}")
            mc.send_angles(p, speed)
            reached = wait_until_reached(mc, p, timeout_s=timeout_s, tol_deg=tol_deg)
            if not reached:
                if verbose:
                    print("  [WARN] Pose not reached within timeout.")
                ok = False
            time.sleep(hold_s)

    if verbose:
        print("[INFO] Returning to HOME…")
    mc.send_angles(home_angles, speed)
    reached_home = wait_until_reached(mc, home_angles, timeout_s=timeout_s, tol_deg=tol_deg)
    if not reached_home:
        if verbose:
            print("[WARN] HOME not reached within timeout.")
        ok = False

    return ok


# ----------------------------
# Optional ROS wrapper (rospy)
# ----------------------------

def ros_main(args):
    try:
        import rospy
        from std_srvs.srv import Trigger, TriggerResponse
        from std_msgs.msg import Float32MultiArray
    except Exception as e:
        print("[ERROR] rospy not available or ROS not sourced. Run without --ros. Details:", e)
        return 2

    rospy.init_node("mycobot_flex_demo", anonymous=False)

    mc = connect_mycobot(args.port, args.baud)
    ensure_ready(mc)

    pub = rospy.Publisher("/mycobot/angles", Float32MultiArray, queue_size=10)

    home = safe_get_angles(mc)
    if home is None:
        rospy.logerr("Could not read initial angles.")
        return 2

    def publish_loop(event):
        ang = safe_get_angles(mc, retries=1, delay_s=0.0)
        if ang:
            pub.publish(Float32MultiArray(data=[float(x) for x in ang]))

    timer = rospy.Timer(rospy.Duration(1.0 / float(args.ros_rate_hz)), publish_loop)

    def handle_run_demo(_req):
        try:
            ok = run_flex_demo(
                mc,
                speed=args.speed,
                hold_s=args.hold,
                repeat=args.repeat,
                timeout_s=args.timeout,
                tol_deg=args.tol,
                home_angles=home,
                verbose=False,
            )
            return TriggerResponse(success=bool(ok), message="Flex demo complete" if ok else "Flex demo warnings/timeouts")
        except Exception as e:
            safe_stop(mc)
            return TriggerResponse(success=False, message=f"Error: {e}")

    def handle_go_home(_req):
        try:
            mc.send_angles(home, args.speed)
            reached = wait_until_reached(mc, home, timeout_s=args.timeout, tol_deg=args.tol)
            return TriggerResponse(success=bool(reached), message="Reached HOME" if reached else "HOME timeout")
        except Exception as e:
            safe_stop(mc)
            return TriggerResponse(success=False, message=f"Error: {e}")

    rospy.Service("/mycobot/run_flex_demo", Trigger, handle_run_demo)
    rospy.Service("/mycobot/go_home", Trigger, handle_go_home)

    rospy.loginfo("mycobot_flex_demo node started.")
    rospy.loginfo("Services: /mycobot/run_flex_demo , /mycobot/go_home")
    rospy.loginfo("Publishing: /mycobot/angles")

    try:
        rospy.spin()
    finally:
        try:
            timer.shutdown()
        except Exception:
            pass
        safe_stop(mc)

    return 0


# ----------------------------
# Test doubles for unit tests
# ----------------------------

class FakeMyCobot:
    """
    A simple simulator:
    - get_angles returns current_angles
    - send_angles sets target and (optionally) steps there immediately
    """
    def __init__(self, start_angles=None, reach_immediately=True):
        self.current = start_angles[:] if start_angles else [0, 0, 0, 0, 0, 0]
        self.target = self.current[:]
        self.reach_immediately = reach_immediately
        self.powered = False
        self.released = False
        self.stopped = False
        self.sent = []

    def power_on(self):
        self.powered = True

    def get_angles(self):
        return self.current[:]

    def send_angles(self, angles, speed):
        self.sent.append((angles[:], speed))
        self.target = angles[:]
        if self.reach_immediately:
            self.current = self.target[:]

    def stop(self):
        self.stopped = True

    def release_all_servos(self):
        self.released = True


class FakeMyCobotNeverReaches(FakeMyCobot):
    """Simulates a robot that never reaches targets (timeout path)."""
    def __init__(self, start_angles=None):
        super().__init__(start_angles=start_angles, reach_immediately=False)

    def send_angles(self, angles, speed):
        self.sent.append((angles[:], speed))
        self.target = angles[:]  # but current never updates


# ----------------------------
# Unit Tests
# ----------------------------

class TestUtilities(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-1, 0, 10), 0)
        self.assertEqual(clamp(11, 0, 10), 10)

    def test_within_tol(self):
        self.assertTrue(within_tol([0,0,0,0,0,0], [1,1,1,1,1,1], tol=1.0))
        self.assertFalse(within_tol([0,0,0,0,0,0], [2,0,0,0,0,0], tol=1.0))

    def test_median(self):
        self.assertEqual(median([3,1,2]), 2.0)
        self.assertEqual(median([1,2,3,4]), 2.5)
        self.assertIsNone(median([]))


class TestPoseGeneration(unittest.TestCase):
    def test_make_flex_poses_shape(self):
        home = [0, 0, 0, 0, 0, 0]
        poses = make_flex_poses(home)
        self.assertEqual(len(poses), 6)
        for p in poses:
            self.assertEqual(len(p), 6)

    def test_make_flex_poses_clamps(self):
        # Put home near limits so deltas would exceed and must clamp
        home = [170, 120, 170, 170, 170, 180]
        poses = make_flex_poses(home)
        for p in poses:
            for i, val in enumerate(p):
                lo, hi = DEFAULT_SOFT_LIMITS[i]
                self.assertTrue(lo <= val <= hi)

    def test_make_flex_poses_invalid_home(self):
        with self.assertRaises(ValueError):
            make_flex_poses([0, 0, 0])  # wrong length


class TestDemoFlow(unittest.TestCase):
    def test_run_flex_demo_success(self):
        mc = FakeMyCobot(start_angles=[0, 0, 0, 0, 0, 0], reach_immediately=True)
        ok = run_flex_demo(mc, speed=20, hold_s=0.0, repeat=1, timeout_s=0.5, tol_deg=2.0, verbose=False)
        self.assertTrue(ok)
        # Should have sent poses + return home
        self.assertGreaterEqual(len(mc.sent), 7)

    def test_run_flex_demo_timeout_path(self):
        mc = FakeMyCobotNeverReaches(start_angles=[0, 0, 0, 0, 0, 0])
        ok = run_flex_demo(mc, speed=20, hold_s=0.0, repeat=1, timeout_s=0.2, tol_deg=2.0, verbose=False)
        self.assertFalse(ok)  # should report warnings/timeouts

    def test_safe_stop_flags(self):
        mc = FakeMyCobot()
        safe_stop(mc)
        self.assertTrue(mc.stopped)
        self.assertTrue(mc.released)


# ----------------------------
# Hardware smoke tests (optional)
# ----------------------------

def run_hw_smoke(args) -> int:
    """
    Minimal, non-destructive checks on real hardware.
    Only runs if explicitly invoked.
    """
    print("[HW-SMOKE] Connecting...")
    mc = connect_mycobot(args.port, args.baud)
    try:
        ensure_ready(mc)
        ang = safe_get_angles(mc, retries=5, delay_s=0.2)
        if ang is None:
            print("[HW-SMOKE][FAIL] Could not read angles.")
            return 2
        print("[HW-SMOKE] Angles:", ang)

        if args.hw_move:
            print("[HW-SMOKE] Performing one small move and return (very small deltas).")
            home = ang
            poses = make_flex_poses(home)
            p = poses[0]  # first small pose
            mc.send_angles(p, args.speed)
            time.sleep(1.0)
            mc.send_angles(home, args.speed)
            time.sleep(1.0)
            print("[HW-SMOKE] Move complete.")

        print("[HW-SMOKE][PASS]")
        return 0
    finally:
        safe_stop(mc)


# ----------------------------
# CLI entry
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="MyCobot 280-Pi Flex Demo + Unit Tests")
    ap.add_argument("--port", default="/dev/ttyAMA0", help="Serial port (default: /dev/ttyAMA0)")
    ap.add_argument("--baud", type=int, default=1000000, help="Baud rate (default: 1000000)")
    ap.add_argument("--speed", type=int, default=30, help="Move speed 1-100 (default: 30)")
    ap.add_argument("--hold", type=float, default=0.5, help="Hold time at each pose (s)")
    ap.add_argument("--repeat", type=int, default=1, help="Number of flex cycles")
    ap.add_argument("--timeout", type=float, default=10.0, help="Timeout waiting for each pose (s)")
    ap.add_argument("--tol", type=float, default=2.0, help="Angle tolerance to consider pose reached (deg)")
    ap.add_argument("--ros", action="store_true", help="Run as ROS node (requires rospy)")
    ap.add_argument("--ros-rate-hz", type=float, default=10.0, help="ROS publish rate for angles topic")

    # Test flags
    ap.add_argument("--test", action="store_true", help="Run unit tests (no hardware required)")
    ap.add_argument("--hw-smoke", action="store_true", help="Run hardware smoke tests (requires arm)")
    ap.add_argument("--hw-move", action="store_true", help="(HW smoke) do a single small move + return to home")

    args = ap.parse_args()

    if args.test:
        # Run unittest suite and exit
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1

    if args.hw_smoke:
        return run_hw_smoke(args)

    if args.ros:
        return ros_main(args)

    # Standalone mode
    mc = connect_mycobot(args.port, args.baud)
    try:
        ok = run_flex_demo(mc, speed=args.speed, hold_s=args.hold, repeat=args.repeat, timeout_s=args.timeout, tol_deg=args.tol)
        print("[RESULT]", "PASS" if ok else "DONE (with warnings)")
        return 0 if ok else 1
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Stopping safely...")
        safe_stop(mc)
        return 130
    except Exception as e:
        print("[ERROR]", e)
        safe_stop(mc)
        return 2
    finally:
        safe_stop(mc)

if __name__ == "__main__":
    raise SystemExit(main())