from __future__ import annotations

import argparse
import subprocess
import threading
import time
from dataclasses import dataclass

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
except Exception as exc:
    raise RuntimeError(
        "ROS 2 with rclpy and geometry_msgs is required to run ros_to_gz_cmd_proxy.py"
    ) from exc


@dataclass
class TwistState:
    linear_x: float = 0.0
    angular_z: float = 0.0
    stamp: float = 0.0


def _fmt(val: float) -> str:
    return f"{float(val):.6f}"


class RosToGzCmdProxy(Node):
    def __init__(
        self,
        ros_topic: str,
        gz_topic: str,
        container_name: str,
        rate_hz: float,
        hold_secs: float,
        min_delta: float,
        docker_timeout: float,
        verbose: bool,
    ) -> None:
        super().__init__("ros_to_gz_cmd_proxy")

        self.ros_topic = ros_topic
        self.gz_topic = gz_topic
        self.container_name = container_name
        self.rate_hz = max(0.5, float(rate_hz))
        self.hold_secs = max(0.05, float(hold_secs))
        self.min_delta = max(0.0, float(min_delta))
        self.docker_timeout = max(0.2, float(docker_timeout))
        self.verbose = bool(verbose)

        self._lock = threading.Lock()
        self._latest = TwistState()
        self._last_sent = TwistState()
        self._have_input = False
        self._zero_sent_after_timeout = True

        self.subscription = self.create_subscription(
            Twist,
            self.ros_topic,
            self._twist_cb,
            10,
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(
            f"Proxy listening on ROS topic {self.ros_topic} and forwarding to "
            f"Gazebo topic {self.gz_topic} in container {self.container_name} "
            f"at {self.rate_hz:.1f} Hz"
        )

    def _twist_cb(self, msg: Twist) -> None:
        with self._lock:
            self._latest = TwistState(
                linear_x=float(msg.linear.x),
                angular_z=float(msg.angular.z),
                stamp=time.time(),
            )
            self._have_input = True
            self._zero_sent_after_timeout = False

        if self.verbose:
            self.get_logger().info(
                f"ROS cmd_vel received: linear.x={msg.linear.x:.3f}, "
                f"angular.z={msg.angular.z:.3f}"
            )

    def _tick(self) -> None:
        now = time.time()
        with self._lock:
            latest = TwistState(
                linear_x=self._latest.linear_x,
                angular_z=self._latest.angular_z,
                stamp=self._latest.stamp,
            )
            have_input = self._have_input
            zero_sent_after_timeout = self._zero_sent_after_timeout

        if not have_input:
            return

        age = now - latest.stamp
        target = latest

        if age > self.hold_secs:
            target = TwistState(linear_x=0.0, angular_z=0.0, stamp=now)
            if zero_sent_after_timeout and self._is_zero(self._last_sent):
                return

        if self._should_send(target):
            self._send_to_gazebo(target)
            self._last_sent = TwistState(
                linear_x=target.linear_x,
                angular_z=target.angular_z,
                stamp=target.stamp,
            )

            if age > self.hold_secs and self._is_zero(target):
                with self._lock:
                    self._zero_sent_after_timeout = True

    def _is_zero(self, state: TwistState) -> bool:
        return abs(state.linear_x) <= self.min_delta and abs(state.angular_z) <= self.min_delta

    def _should_send(self, target: TwistState) -> bool:
        dl = abs(target.linear_x - self._last_sent.linear_x)
        da = abs(target.angular_z - self._last_sent.angular_z)

        if dl > self.min_delta or da > self.min_delta:
            return True

        if not self._is_zero(target):
            return True

        return False

    def _send_to_gazebo(self, target: TwistState) -> None:
        payload = (
            "linear: {"
            f"x: {_fmt(target.linear_x)}, y: 0.0, z: 0.0"
            "}, angular: {"
            f"x: 0.0, y: 0.0, z: {_fmt(target.angular_z)}"
            "}"
        )

        cmd = [
            "docker",
            "exec",
            self.container_name,
            "gz",
            "topic",
            "-t",
            self.gz_topic,
            "-m",
            "gz.msgs.Twist",
            "-p",
            payload,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.docker_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self.get_logger().warning("Timed out while forwarding cmd_vel to Gazebo.")
            return
        except Exception as exc:
            self.get_logger().error(f"Failed to invoke docker exec for Gazebo publish: {exc}")
            return

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            details = stderr if stderr else stdout
            self.get_logger().error(
                f"Gazebo publish failed (code {result.returncode}): {details}"
            )
            return

        if self.verbose:
            self.get_logger().info(
                f"Forwarded to Gazebo: linear.x={target.linear_x:.3f}, "
                f"angular.z={target.angular_z:.3f}"
            )

    def send_zero_once(self) -> None:
        zero = TwistState(0.0, 0.0, time.time())
        self._send_to_gazebo(zero)
        self._last_sent = zero


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Desktop-host ROS to Gazebo cmd_vel proxy using docker exec into the moon sim container."
    )
    p.add_argument("--ros-topic", default="/cmd_vel",
                   help="ROS Twist topic to subscribe to on the desktop host.")
    p.add_argument("--gz-topic", default="/cmd_vel",
                   help="Gazebo Transport topic to publish inside the sim container.")
    p.add_argument("--container-name", default="space_robotics_gz_envs",
                   help="Running moon sim container name.")
    p.add_argument("--rate", type=float, default=5.0,
                   help="Forwarding rate in Hz while commands are active.")
    p.add_argument("--hold-secs", type=float, default=0.5,
                   help="How long to keep forwarding the last command before forcing zero.")
    p.add_argument("--min-delta", type=float, default=1e-3,
                   help="Minimum change required to treat commands as different.")
    p.add_argument("--docker-timeout", type=float, default=2.0,
                   help="Timeout in seconds for each docker exec publish call.")
    p.add_argument("--verbose", action="store_true",
                   help="Print each received and forwarded command.")
    p.add_argument("--no-stop-on-exit", action="store_true",
                   help="Do not send a final zero command on shutdown.")
    return p.parse_args()


def ensure_container_running(container_name: str) -> None:
    cmd = ["docker", "inspect", "-f", "{{.State.Running}}", container_name]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Could not inspect Docker container '{container_name}'. {msg}"
        )
    if result.stdout.strip().lower() != "true":
        raise RuntimeError(f"Docker container '{container_name}' is not running.")


def main() -> None:
    args = parse_args()

    ensure_container_running(args.container_name)

    rclpy.init()
    node = RosToGzCmdProxy(
        ros_topic=args.ros_topic,
        gz_topic=args.gz_topic,
        container_name=args.container_name,
        rate_hz=args.rate,
        hold_secs=args.hold_secs,
        min_delta=args.min_delta,
        docker_timeout=args.docker_timeout,
        verbose=args.verbose,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_stop_on_exit:
            node.get_logger().info("Sending final zero command to Gazebo.")
            node.send_zero_once()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
