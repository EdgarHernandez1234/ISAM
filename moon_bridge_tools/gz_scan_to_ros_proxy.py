from __future__ import annotations

import argparse
import subprocess
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


@dataclass
class ScanMsg:
    frame_id: str = ""
    angle_min: float = 0.0
    angle_max: float = 0.0
    angle_step: float = 0.0
    range_min: float = 0.0
    range_max: float = 0.0
    count: int = 0
    vertical_count: int = 1
    ranges: List[float] = field(default_factory=list)


class GzScanToRosProxy(Node):
    def __init__(self, container_name: str, gz_topic: str, ros_topic: str, verbose: bool = False) -> None:
        super().__init__("gz_scan_to_ros_proxy")
        self.container_name = container_name
        self.gz_topic = gz_topic
        self.ros_topic = ros_topic
        self.verbose = verbose
        self.pub = self.create_publisher(LaserScan, ros_topic, 10)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"Listening to Gazebo topic {self.gz_topic} in container {self.container_name} "
            f"and publishing ROS LaserScan on {self.ros_topic}"
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

        current: Optional[ScanMsg] = None
        in_header = False
        in_world_pose = False

        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                if self._stop.is_set():
                    break
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("header {"):
                    if current is not None and current.ranges:
                        self._publish_scan(current)
                    current = ScanMsg()
                    in_header = True
                    in_world_pose = False
                    continue

                if current is None:
                    continue

                if line.startswith("world_pose {"):
                    in_world_pose = True
                    continue

                if line == "}":
                    if in_world_pose:
                        in_world_pose = False
                        continue
                    if in_header:
                        in_header = False
                        continue
                    continue

                if in_world_pose:
                    continue

                if line.startswith('frame: "'):
                    current.frame_id = line.split('"', 2)[1]
                elif line.startswith("angle_min:"):
                    current.angle_min = float(line.split(":", 1)[1].strip())
                elif line.startswith("angle_max:"):
                    current.angle_max = float(line.split(":", 1)[1].strip())
                elif line.startswith("angle_step:"):
                    current.angle_step = float(line.split(":", 1)[1].strip())
                elif line.startswith("range_min:"):
                    current.range_min = float(line.split(":", 1)[1].strip())
                elif line.startswith("range_max:"):
                    current.range_max = float(line.split(":", 1)[1].strip())
                elif line.startswith("count:"):
                    current.count = int(float(line.split(":", 1)[1].strip()))
                elif line.startswith("vertical_count:"):
                    current.vertical_count = int(float(line.split(":", 1)[1].strip()))
                elif line.startswith("ranges:"):
                    current.ranges.append(float(line.split(":", 1)[1].strip()))

            if current is not None and current.ranges:
                self._publish_scan(current)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    proc.kill()

    def _publish_scan(self, scan: ScanMsg) -> None:
        if scan.count <= 0 or not scan.ranges:
            return

        horiz = scan.count
        vert = max(1, scan.vertical_count)

        if len(scan.ranges) >= horiz * vert:
            mid = vert // 2
            start = mid * horiz
            end = start + horiz
            ranges = scan.ranges[start:end]
        else:
            ranges = scan.ranges[:horiz]

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = scan.frame_id or "front_laser"
        msg.angle_min = float(scan.angle_min)
        msg.angle_max = float(scan.angle_max)
        msg.angle_increment = float(scan.angle_step)
        msg.time_increment = 0.0
        msg.scan_time = 0.0
        msg.range_min = float(scan.range_min)
        msg.range_max = float(scan.range_max)
        msg.ranges = [float(x) for x in ranges]
        msg.intensities = []
        self.pub.publish(msg)

        if self.verbose:
            self.get_logger().info(
                f"Published {len(msg.ranges)} ranges on {self.ros_topic} "
                f"(vertical_count={vert}, frame={msg.header.frame_id})"
            )


def parse_args():
    p = argparse.ArgumentParser(
        description="Desktop-host proxy: Gazebo LaserScan in the sim container -> ROS LaserScan on the desktop host."
    )
    p.add_argument("--container-name", default="space_robotics_gz_envs",
                   help="Running moon sim container name.")
    p.add_argument("--gz-topic",
                   default="/world/moon/model/explorer_r2_sensor_config_1/link/base_link/sensor/front_laser/scan",
                   help="Gazebo LaserScan topic inside the moon sim container.")
    p.add_argument("--ros-topic", default="/scan",
                   help="ROS LaserScan topic to publish on the desktop host.")
    p.add_argument("--verbose", action="store_true",
                   help="Print a line for each published scan.")
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
    node = GzScanToRosProxy(
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
