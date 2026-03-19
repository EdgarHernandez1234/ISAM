from __future__ import annotations

from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    HAS_ROS2 = True
except Exception:
    HAS_ROS2 = False


class GazeboBaseCmdPublisher:
    """
    Minimal ROS 2 publisher for rover base Twist commands destined for Gazebo.
    """

    def __init__(self, topic_name: str = "/model/vehicle_blue/cmd_vel") -> None:
        if not HAS_ROS2:
            raise RuntimeError("ROS2 not available for GazeboBaseCmdPublisher.")
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("gazebo_alpha_base_cmd_pub")
        self.pub = self.node.create_publisher(Twist, str(topic_name), 10)
        self.topic_name = str(topic_name)

    def publish(self, v_mps: float, w_rps: float) -> None:
        msg = Twist()
        msg.linear.x = float(v_mps)
        msg.angular.z = float(w_rps)
        self.pub.publish(msg)
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def stop(self) -> None:
        self.publish(0.0, 0.0)

    def close(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:
            pass
