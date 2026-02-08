#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]

# A small helper to keep arrays consistent
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Keyframe:
    """A named joint pose (radians) and a hold duration (seconds)."""
    name: str
    positions: List[float]
    hold_s: float


class PickPlaceAnimator(Node):
    """
    Publishes /joint_states at a fixed rate and interpolates between keyframes
    to create a smooth pick -> lift -> place -> return loop.
    """

    def __init__(self):
        super().__init__("alam_pick_place_animator")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        # Tune these if motion looks too fast/slow
        self.rate_hz = 30.0
        self.dt = 1.0 / self.rate_hz

        # Keyframes: [base, shoulder, elbow, wrist_pitch, wrist_roll, tool_roll]
        # These are *illustrative* values; adjust for your preferred “scoop” posture.
        self.keyframes = [
            Keyframe(
                "home",
                [0.0, -0.3, 1.0, -0.7, -0.3, 0.0],
                hold_s=0.7,
            ),
            Keyframe(
                "approach_regolith",
                [0.2, 0.1, 1.25, -1.15, -0.4, 0.0],
                hold_s=0.5,
            ),
            Keyframe(
                "scoop_down",
                [0.2, 0.25, 1.45, -1.35, -0.4, 0.0],
                hold_s=0.8,
            ),
            Keyframe(
                "lift",
                [0.2, -0.15, 1.10, -0.95, -0.4, 0.0],
                hold_s=0.6,
            ),
            Keyframe(
                "move_to_drop_zone",
                [-0.6, -0.10, 1.05, -0.90, -0.4, 0.0],
                hold_s=0.5,
            ),
            Keyframe(
                "dump",
                [-0.6, 0.05, 0.95, -0.60, -0.4, 0.8],  # tool roll to "dump"
                hold_s=0.8,
            ),
            Keyframe(
                "return_home",
                [0.0, -0.3, 1.0, -0.7, -0.3, 0.0],
                hold_s=0.8,
            ),
        ]

        # Interpolation timing between keyframes (seconds)
        self.transition_s = 1.0

        self.k = 0  # current keyframe index
        self.phase = "hold"  # "hold" or "move"
        self.phase_t = 0.0   # time within current phase
        self.start_pose = self.keyframes[0].positions[:]
        self.target_pose = self.keyframes[0].positions[:]

        self.get_logger().info("Publishing /joint_states pick/place animation. Ctrl+C to stop.")
        self.timer = self.create_timer(self.dt, self.tick)

    def publish_pose(self, pose: List[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = pose
        self.pub.publish(msg)

    def smoothstep(self, u: float) -> float:
        # Smooth interpolation 0->1 with zero slope at endpoints
        u = clamp(u, 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    def lerp_pose(self, a: List[float], b: List[float], u: float) -> List[float]:
        s = self.smoothstep(u)
        return [(1.0 - s) * ai + s * bi for ai, bi in zip(a, b)]

    def tick(self) -> None:
        self.phase_t += self.dt

        current_kf = self.keyframes[self.k]

        if self.phase == "hold":
            # Hold at the current keyframe pose
            self.publish_pose(current_kf.positions)
            if self.phase_t >= current_kf.hold_s:
                # Begin transition to next keyframe
                self.phase = "move"
                self.phase_t = 0.0
                self.start_pose = current_kf.positions[:]
                self.k = (self.k + 1) % len(self.keyframes)
                self.target_pose = self.keyframes[self.k].positions[:]

        else:  # move
            u = self.phase_t / self.transition_s
            pose = self.lerp_pose(self.start_pose, self.target_pose, u)
            self.publish_pose(pose)
            if self.phase_t >= self.transition_s:
                # Finish transition; start holding next keyframe
                self.phase = "hold"
                self.phase_t = 0.0


def main():
    rclpy.init()
    node = PickPlaceAnimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
