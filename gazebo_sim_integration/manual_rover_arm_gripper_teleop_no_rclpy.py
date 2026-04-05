#!/usr/bin/env python3
"""
Manual teleop for Gazebo-only rover + arm + gripper workflow.

No rclpy required.
Uses `gz topic` CLI only.

Run inside the container shell:
    python3 /root/ws/scripts/manual_rover_arm_gripper_teleop_no_rclpy.py
"""

import sys
import time
import tty
import termios
import subprocess

BURST_SECONDS = 0.18

ROVER_LIN = 0.7
ROVER_ANG = 0.8

ARM_SPEED_SMALL = 0.15
GRIPPER_SPEED = 0.2


def run_gz_topic(topic: str, msg_type: str, payload: str) -> None:
    try:
        subprocess.run(
            ["gz", "topic", "-t", topic, "-m", msg_type, "-p", payload],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("Error: `gz` CLI was not found in PATH.")
        sys.exit(1)


def publish_double(topic: str, value: float) -> None:
    run_gz_topic(topic, "gz.msgs.Double", f"data: {value}")


def publish_twist(linear_x: float, angular_z: float) -> None:
    payload = (
        "linear: {x: %.6f, y: 0.0, z: 0.0} "
        "angular: {x: 0.0, y: 0.0, z: %.6f}"
    ) % (linear_x, angular_z)
    run_gz_topic("/cmd_vel", "gz.msgs.Twist", payload)


def burst_double(topic: str, value: float, duration: float = BURST_SECONDS) -> None:
    publish_double(topic, value)
    time.sleep(duration)
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

Notes:
  - Rover commands set /cmd_vel directly.
  - Arm and gripper commands are short bursts, then auto-zero.
"""
    )


class RawKeyboard:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def getch(self) -> str:
        return sys.stdin.read(1)


def handle_key(ch: str) -> None:
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
        time.sleep(BURST_SECONDS)
        zero_gripper()
    elif ch == "p":
        publish_double("/gripper/left/cmd_vel",  -GRIPPER_SPEED)
        publish_double("/gripper/right/cmd_vel", GRIPPER_SPEED)
        time.sleep(BURST_SECONDS)
        zero_gripper()

    elif ch == "x":
        zero_arm()
        zero_gripper()
    elif ch == " ":
        zero_all()
    elif ch == "?":
        show_help()


def main() -> int:
    if not sys.stdin.isatty():
        print("This script must be run in an interactive terminal.")
        return 1

    print("Starting teleop without rclpy. Press ? for help. Ctrl+C to quit.")
    zero_all()

    try:
        with RawKeyboard() as kb:
            while True:
                ch = kb.getch()
                handle_key(ch)
    except KeyboardInterrupt:
        pass
    finally:
        zero_all()
        print("\nStopped. Sent zero commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
