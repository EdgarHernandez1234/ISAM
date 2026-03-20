from __future__ import annotations

import threading
import time
from typing import Any, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def _imgmsg_to_bgr(msg: Image) -> Optional[np.ndarray]:
    try:
        h = int(msg.height)
        w = int(msg.width)
        enc = str(msg.encoding).lower()
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

        if enc == "rgb8":
            arr = data.reshape((h, w, 3))
            return arr[:, :, ::-1].copy()  # RGB -> BGR
        if enc == "bgr8":
            return data.reshape((h, w, 3)).copy()
        if enc == "mono8":
            gray = data.reshape((h, w))
            return np.stack([gray, gray, gray], axis=-1).copy()

        return None
    except Exception:
        return None


class _RosTwoCamNode(Node):
    def __init__(self, topic_a: str, topic_b: str):
        super().__init__("ros_two_camera_provider")
        self.topic_a = topic_a
        self.topic_b = topic_b

        self.frame_a = None
        self.frame_b = None
        self.ts_a = 0.0
        self.ts_b = 0.0
        self.cam_a_ok = False
        self.cam_b_ok = False

        self._lock = threading.Lock()

        self.sub_a = self.create_subscription(Image, topic_a, self._cb_a, 10)
        self.sub_b = self.create_subscription(Image, topic_b, self._cb_b, 10)

    def _cb_a(self, msg: Image) -> None:
        img = _imgmsg_to_bgr(msg)
        with self._lock:
            self.frame_a = img
            self.ts_a = time.time()
            self.cam_a_ok = img is not None

    def _cb_b(self, msg: Image) -> None:
        img = _imgmsg_to_bgr(msg)
        with self._lock:
            self.frame_b = img
            self.ts_b = time.time()
            self.cam_b_ok = img is not None

    def snapshot(self) -> Tuple[Optional[Any], Optional[Any], float]:
        with self._lock:
            fa = None if self.frame_a is None else self.frame_a.copy()
            fb = None if self.frame_b is None else self.frame_b.copy()
            ts = max(self.ts_a, self.ts_b, time.time())
            return fa, fb, ts


class ROSTwoCameraProvider:
    def __init__(
        self,
        topic_a: str = "/sim/rs_front/image_raw",
        topic_b: str = "/sim/rs_back/image_raw",
    ):
        self.topic_a = topic_a
        self.topic_b = topic_b

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = _RosTwoCamNode(topic_a, topic_b)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._thread.start()

    @property
    def cam_a_ok(self) -> bool:
        return bool(self.node.cam_a_ok)

    @property
    def cam_b_ok(self) -> bool:
        return bool(self.node.cam_b_ok)

    def _spin_loop(self) -> None:
        while not self._stop.is_set():
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def read(self) -> Tuple[Optional[Any], Optional[Any], float]:
        return self.node.snapshot()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self.node.destroy_node()
        except Exception:
            pass
