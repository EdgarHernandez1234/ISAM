#!/usr/bin/env python3
"""
Manual teleop for Gazebo rover + arm + gripper using only:
- Python standard library
- gz CLI through subprocess

Run inside container:
    python3 /root/ws/scripts/manual_rover_arm_gripper_teleop_no_rclpy.py
"""

import sys
import time
import tty
import termios
import select
import signal
import shutil
import subprocess

BURST_SECONDS = 0.18
GZ_TIMEOUT = 3.0

ROVER_LIN = 0.7
ROVER_ANG = 0.8

ARM_SPEED_SMALL = 0.15
GRIPPER_SPEED = 0.2

STOP_REQUESTED = False


def request_stop(signum=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)


def run_gz_topic(topic: str, msg_type: str, payload: str) -> bool:
    try:
        result = subprocess.run(
            ["gz", "topic", "-t", topic, "-m", msg_type, "-p", payload],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=GZ_TIMEOUT,
        )
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            print(f"[WARN] gz publish failed on {topic}: {err or 'nonzero exit'}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[WARN] gz publish timed out on {topic}")
        return False
    except FileNotFoundError:
        print("[ERROR] `gz` CLI was not found in PATH.")
        raise
    except Exception as exc:
        print(f"[WARN] gz publish exception on {topic}: {exc}")
        return False


def publish_double(topic: str, value: float) -> bool:
    return run_gz_topic(topic, "gz.msgs.Double", f"data: {value}")


def publish_twist(linear_x: float, angular_z: float) -> bool:
    payload = (
        "linear: {x: %.6f, y: 0.0, z: 0.0} "
        "angular: {x: 0.0, y: 0.0, z: %.6f}"
    ) % (linear_x, angular_z)
    return run_gz_topic("/cmd_vel", "gz.msgs.Twist", payload)


def safe_sleep(duration: float) -> None:
    end = time.time() + duration
    while time.time() < end:
        if STOP_REQUESTED:
            break
        time.sleep(0.01)


def burst_double(topic: str, value: float, duration: float = BURST_SECONDS) -> None:
    publish_double(topic, value)
    safe_sleep(duration)
    publish_double(topic, 0.0)


def zero_arm() -> None:
    for j in range(1, 7):
        publish_double(f"/arm/joint{j}/cmd_vel", 0.0)


def zero_gripper() -> None:
    publish_double("/gripper/left/cmd_vel", 0.0)
    publish_double("/gripper/right/cmd_vel", 0.0)


def zero_all() -> None:
    publish_twist(0.0, 0.0)
    zero_arm()
    zero_gripper()


def show_help() -> None:
    print(
        """
Manual rover + arm + gripper teleop
===================================

Rover:
  i  forward
  ,  reverse
  j  turn left
  l  turn right
  k  stop rover

Arm:
  q / a   joint1 + / -
  w / s   joint2 + / -
  e / d   joint3 + / -
  r / f   joint4 + / -
  t / g   joint5 + / -
  y / h   joint6 + / -

Gripper:
  o  open
  p  close

Safety / utility:
  x       zero arm + gripper
  space   stop rover + zero arm + gripper
  ?       show this help
  Ctrl+C  quit and send zeros
"""
    )


class CBreakKeyboard:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def getch(self, timeout: float = 0.1):
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return None


def handle_key(ch: str) -> None:
    global STOP_REQUESTED

    if ch == "\x03":
        STOP_REQUESTED = True
        return

    if ch == "i":
        publish_twist(ROVER_LIN, 0.0)
    elif ch == ",":
        publish_twist(-ROVER_LIN, 0.0)
    elif ch == "j":
        publish_twist(0.0, ROVER_ANG)
    elif ch == "l":
        publish_twist(0.0, -ROVER_ANG)
    elif ch == "k":
        publish_twist(0.0, 0.0)

    elif ch == "q":
        burst_double("/arm/joint1/cmd_vel", ARM_SPEED_SMALL)
    elif ch == "a":
        burst_double("/arm/joint1/cmd_vel", -ARM_SPEED_SMALL)
    elif ch == "w":
        burst_double("/arm/joint2/cmd_vel", ARM_SPEED_SMALL)
    elif ch == "s":
        burst_double("/arm/joint2/cmd_vel", -ARM_SPEED_SMALL)
    elif ch == "e":
        burst_double("/arm/joint3/cmd_vel", ARM_SPEED_SMALL)
    elif ch == "d":
        burst_double("/arm/joint3/cmd_vel", -ARM_SPEED_SMALL)
    elif ch == "r":
        burst_double("/arm/joint4/cmd_vel", ARM_SPEED_SMALL)
    elif ch == "f":
        burst_double("/arm/joint4/cmd_vel", -ARM_SPEED_SMALL)
    elif ch == "t":
        burst_double("/arm/joint5/cmd_vel", ARM_SPEED_SMALL)
    elif ch == "g":
        burst_double("/arm/joint5/cmd_vel", -ARM_SPEED_SMALL)
    elif ch == "y":
        burst_double("/arm/joint6/cmd_vel", ARM_SPEED_SMALL)
    elif ch == "h":
        burst_double("/arm/joint6/cmd_vel", -ARM_SPEED_SMALL)

    elif ch == "o":
        publish_double("/gripper/left/cmd_vel",  GRIPPER_SPEED)
        publish_double("/gripper/right/cmd_vel", -GRIPPER_SPEED)
        safe_sleep(BURST_SECONDS)
        zero_gripper()
    elif ch == "p":
        publish_double("/gripper/left/cmd_vel",  -GRIPPER_SPEED)
        publish_double("/gripper/right/cmd_vel", GRIPPER_SPEED)
        safe_sleep(BURST_SECONDS)
        zero_gripper()

    elif ch == "x":
        zero_arm()
        zero_gripper()
    elif ch == " ":
        zero_all()
    elif ch == "?":
        show_help()


def main() -> int:
    if not shutil.which("gz"):
        print("[ERROR] `gz` CLI was not found in PATH.")
        return 1

    if not sys.stdin.isatty():
        print("[ERROR] This script must be run in an interactive terminal.")
        return 1

    print("Starting teleop without rclpy. Press ? for help. Ctrl+C to quit.")
    zero_all()

    try:
        with CBreakKeyboard() as kb:
            while not STOP_REQUESTED:
                ch = kb.getch(timeout=0.1)
                if ch is None:
                    continue
                handle_key(ch)
    finally:
        zero_all()
        print("\nStopped. Sent zero commands.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
