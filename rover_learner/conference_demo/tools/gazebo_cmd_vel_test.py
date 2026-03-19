from __future__ import annotations

import argparse
import time

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
except Exception as e:
    raise RuntimeError("ROS 2 is required for gazebo_cmd_vel_test.py") from e


class CmdVelTestNode(Node):
    def __init__(self, topic_name: str):
        super().__init__("gazebo_cmd_vel_test")
        self.pub = self.create_publisher(Twist, topic_name, 10)
        self.topic_name = topic_name

    def publish_twist(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.0)


def parse_args():
    p = argparse.ArgumentParser(description="One-shot Gazebo cmd_vel test publisher.")
    p.add_argument("--topic", default="/model/vehicle_blue/cmd_vel",
                   help="ROS Twist topic bridged into Gazebo.")
    p.add_argument("--linear", type=float, default=1.0,
                   help="Linear X command in m/s.")
    p.add_argument("--angular", type=float, default=0.0,
                   help="Angular Z command in rad/s.")
    p.add_argument("--duration", type=float, default=2.0,
                   help="How long to publish the motion command.")
    p.add_argument("--rate", type=float, default=10.0,
                   help="Publish rate in Hz while command is active.")
    p.add_argument("--no-stop", action="store_true",
                   help="Do not publish a final zero Twist.")
    return p.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = CmdVelTestNode(args.topic)

    try:
        hz = max(1.0, float(args.rate))
        dt = 1.0 / hz
        end_t = time.time() + max(0.0, float(args.duration))

        node.get_logger().info(
            f"Publishing Twist to {args.topic}: "
            f"linear.x={args.linear:.3f}, angular.z={args.angular:.3f}, "
            f"duration={args.duration:.2f}s, rate={hz:.1f}Hz"
        )

        while time.time() < end_t:
            node.publish_twist(args.linear, args.angular)
            time.sleep(dt)

        if not args.no_stop:
            node.get_logger().info("Publishing final stop command.")
            for _ in range(3):
                node.publish_twist(0.0, 0.0)
                time.sleep(0.05)

        node.get_logger().info("Done.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
