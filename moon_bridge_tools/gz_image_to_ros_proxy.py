from __future__ import annotations

import argparse
import codecs
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


@dataclass
class GzImageMsg:
    frame_id: str = ""
    width: int = 0
    height: int = 0
    step: int = 0
    pixel_format_type: str = ""
    data_bytes: bytes = b""


def decode_gz_escaped_bytes(payload: str) -> bytes:
    decoded = codecs.decode(payload, "unicode_escape")
    return decoded.encode("latin1", errors="ignore")


def pixel_format_to_encoding(fmt: str) -> str:
    fmt = (fmt or "").strip()
    mapping = {
        "RGB_INT8": "rgb8",
        "BGR_INT8": "bgr8",
        "L_INT8": "mono8",
        "R_FLOAT32": "32FC1",
    }
    return mapping.get(fmt, "rgb8")


class GzImageToRosProxy(Node):
    def __init__(self, container_name: str, gz_topic: str, ros_topic: str, verbose: bool = False) -> None:
        super().__init__("gz_image_to_ros_proxy")
        self.container_name = container_name
        self.gz_topic = gz_topic
        self.ros_topic = ros_topic
        self.verbose = verbose
        self.pub = self.create_publisher(Image, ros_topic, 10)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"Listening to Gazebo topic {self.gz_topic} in container {self.container_name} "
            f"and publishing ROS Image on {self.ros_topic}"
        )

    def destroy_node(self):
        self._stop.set()
        return super().destroy_node()

    def _reader_loop(self) -> None:
        cmd = ["docker", "exec", "-i", self.container_name, "gz", "topic", "-e", "-t", self.gz_topic]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to start gz topic reader: {exc}")
            return

        current: Optional[GzImageMsg] = None
        in_header = False
        collecting_data = False
        data_chunks: list[str] = []

        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                if self._stop.is_set():
                    break
                line = raw_line.rstrip("\n")
                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.startswith("header {"):
                    if current is not None and current.data_bytes:
                        self._publish_image(current)
                    current = GzImageMsg()
                    in_header = True
                    collecting_data = False
                    data_chunks = []
                    continue

                if current is None:
                    continue

                if collecting_data:
                    if stripped.endswith('"'):
                        data_chunks.append(stripped[:-1])
                        payload = "".join(data_chunks)
                        current.data_bytes = decode_gz_escaped_bytes(payload)
                        collecting_data = False
                        data_chunks = []
                    else:
                        data_chunks.append(stripped)
                    continue

                if stripped == "}":
                    if in_header:
                        in_header = False
                    continue

                if stripped.startswith('value: "') and "::" in stripped and in_header:
                    current.frame_id = stripped.split('"', 2)[1]
                elif stripped.startswith("width:"):
                    current.width = int(float(stripped.split(":", 1)[1].strip()))
                elif stripped.startswith("height:"):
                    current.height = int(float(stripped.split(":", 1)[1].strip()))
                elif stripped.startswith("step:"):
                    current.step = int(float(stripped.split(":", 1)[1].strip()))
                elif stripped.startswith("pixel_format_type:"):
                    current.pixel_format_type = stripped.split(":", 1)[1].strip()
                elif stripped.startswith('data: "'):
                    payload_start = stripped[len('data: "'):]
                    if payload_start.endswith('"'):
                        payload = payload_start[:-1]
                        current.data_bytes = decode_gz_escaped_bytes(payload)
                    else:
                        collecting_data = True
                        data_chunks = [payload_start]

            if current is not None and current.data_bytes:
                self._publish_image(current)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    proc.kill()

    def _publish_image(self, img: GzImageMsg) -> None:
        if img.width <= 0 or img.height <= 0 or not img.data_bytes:
            return

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = img.frame_id or "camera"
        msg.height = img.height
        msg.width = img.width
        msg.encoding = pixel_format_to_encoding(img.pixel_format_type)
        msg.is_bigendian = 0
        msg.step = img.step if img.step > 0 else len(img.data_bytes) // max(1, img.height)
        msg.data = img.data_bytes

        self.pub.publish(msg)

        if self.verbose:
            self.get_logger().info(
                f"Published {msg.width}x{msg.height} {msg.encoding} on {self.ros_topic} "
                f"(frame={msg.header.frame_id}, bytes={len(msg.data)})"
            )


def parse_args():
    p = argparse.ArgumentParser(
        description="Desktop-host proxy: Gazebo Image in the sim container -> ROS Image on the desktop host."
    )
    p.add_argument("--container-name", default="space_robotics_gz_envs", help="Running moon sim container name.")
    p.add_argument(
        "--gz-topic",
        default="/world/moon/model/explorer_r2_sensor_config_1/link/base_link/sensor/rs_front/image",
        help="Gazebo image topic inside the moon sim container.",
    )
    p.add_argument("--ros-topic", default="/sim/rs_front/image_raw", help="ROS image topic to publish.")
    p.add_argument("--verbose", action="store_true", help="Print a line for each published image.")
    return p.parse_args()


def ensure_container_running(name: str) -> None:
    cmd = ["docker", "inspect", "-f", "{{.State.Running}}", name]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect Docker container '{name}': {(result.stderr or result.stdout).strip()}")
    if result.stdout.strip().lower() != "true":
        raise RuntimeError(f"Docker container '{name}' is not running.")


def main():
    args = parse_args()
    ensure_container_running(args.container_name)

    rclpy.init()
    node = GzImageToRosProxy(
        container_name=args.container_name,
        gz_topic=args.gz_topic,
        ros_topic=args.ros_topic,
        verbose=args.verbose,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
