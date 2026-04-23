
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
greedy_trial_common.py

Shared helpers for Gazebo-oriented epsilon-greedy rover trials.
These trials are designed for the current Jetson<->desktop Gazebo workflow where:
- the Jetson publishes commands
- the desktop proxies expose camera / LiDAR feeds
- /cmd_vel is forwarded into Gazebo

This module avoids CSI/USB camera startup by subscribing to proxy image topics instead.
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CompressedImage, LaserScan
    from geometry_msgs.msg import Twist
    HAS_ROS2 = True
except Exception:
    rclpy = None
    Node = object  # type: ignore
    Image = None  # type: ignore
    CompressedImage = None  # type: ignore
    LaserScan = None  # type: ignore
    Twist = None  # type: ignore
    HAS_ROS2 = False

from rl_safety_supervisor import SafetySupervisor, ShieldedController, RoverAction
from epsilon_greedy_policy import EpsilonGreedyMissionPolicy
from core import Perception, Telemetry, NavState, StepInputs, build_observation
from navigation.navigation import Navigator
from navigation.types import NavMode, Waypoint, Twist2D


DEFAULT_CLASS_NAMES = ["Bypass", "harvestable", "human hazard", "object hazards"]


@dataclass
class FramePacket:
    timestamp: float
    frame_a: Optional[np.ndarray] = None
    frame_b: Optional[np.ndarray] = None
    cam_a_ok: bool = False
    cam_b_ok: bool = False

    @property
    def frame(self) -> np.ndarray:
        if self.frame_a is not None:
            return self.frame_a
        if self.frame_b is not None:
            return self.frame_b
        return np.zeros((360, 640, 3), dtype=np.uint8)


class CsvFrameLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "ts", "rf_top_class", "ssc_state", "ssc_intent", "hard_stop", "soft_bypass",
                "nav_mode", "final_action", "cam_ok_count", "cam_expected", "dist_m",
                "label", "expected", "decision_ok", "proposed_action", "reward",
            ])

    def log(
        self,
        *,
        ts: float,
        rf_top_class: str,
        nav_mode: str,
        final_action: str,
        cam_ok_count: int,
        cam_expected: int,
        dist_m: Optional[float],
        label: str,
        expected: str,
        proposed_action: str,
        reward: float,
    ) -> None:
        decision_ok = int(str(final_action) == str(expected))
        with self.path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                float(ts),
                rf_top_class,
                "",
                "",
                0,
                0,
                nav_mode,
                final_action,
                int(cam_ok_count),
                int(cam_expected),
                "" if dist_m is None else float(dist_m),
                label,
                expected,
                decision_ok,
                proposed_action,
                float(reward),
            ])


def _try_load_names_from_sidecar(model_path: str) -> Optional[List[str]]:
    folder = os.path.dirname(os.path.abspath(model_path))
    candidates = [
        os.path.join(folder, "classes.cls"),
        os.path.join(folder, "classes.txt"),
        os.path.join(folder, "labels.txt"),
        os.path.join(folder, "dataset.yaml"),
        os.path.join(folder, "data.yaml"),
    ]
    for p in candidates:
        try:
            if not os.path.exists(p):
                continue
            if p.endswith((".cls", ".txt")):
                lines = [ln.strip() for ln in open(p, "r", encoding="utf-8").read().splitlines()]
                lines = [ln for ln in lines if ln and (not ln.startswith("#"))]
                if lines:
                    return lines
            if p.endswith((".yaml", ".yml")):
                raw = open(p, "r", encoding="utf-8").read().splitlines()
                names = []
                in_names = False
                for ln in raw:
                    s = ln.strip()
                    if not s:
                        continue
                    if s.startswith("names:"):
                        in_names = True
                        inline = s[len("names:"):].strip()
                        if inline.startswith("[") and inline.endswith("]"):
                            inner = inline[1:-1].strip()
                            if inner:
                                parts = [x.strip().strip("'\"") for x in inner.split(",")]
                                parts = [x for x in parts if x]
                                if parts:
                                    return parts
                        continue
                    if in_names:
                        if s.startswith("- "):
                            names.append(s[2:].strip().strip("'\""))
                        else:
                            if names:
                                return names
                            in_names = False
                if names:
                    return names
        except Exception:
            continue
    return None


class UltralyticsYOLOProvider:
    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: Optional[str] = None,
        max_det: int = 50,
        class_names: Optional[List[str]] = None,
    ):
        self.model_path = os.path.expanduser(model_path)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = None if (device is None or str(device).strip() == "") else str(device)
        self.max_det = int(max_det)

        names = class_names or _try_load_names_from_sidecar(self.model_path)
        self.names = list(names) if names else None

        from ultralytics import YOLO
        self.model = YOLO(self.model_path)

        if not self.names:
            try:
                n = getattr(self.model, "names", None)
                if isinstance(n, dict):
                    self.names = [n[i] for i in range(len(n))]
                elif isinstance(n, list):
                    self.names = list(n)
            except Exception:
                self.names = None

        if not self.names:
            self.names = list(DEFAULT_CLASS_NAMES)

    def infer(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or not hasattr(frame, "shape"):
            return {"predictions": []}
        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
                max_det=self.max_det,
            )
        except TypeError:
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
        if not results:
            return {"predictions": []}

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return {"predictions": []}

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            data = getattr(boxes, "data", None)
            if data is None:
                return {"predictions": []}
            try:
                arr = data.cpu().numpy()
            except Exception:
                arr = np.array(data)
            if arr.size == 0:
                return {"predictions": []}
            xyxy = arr[:, 0:4]
            confs = arr[:, 4] if arr.shape[1] > 4 else [1.0] * arr.shape[0]
            clss = arr[:, 5].astype(int) if arr.shape[1] > 5 else [0] * arr.shape[0]

        preds: List[Dict[str, Any]] = []
        for (x1, y1, x2, y2), c, ci in zip(xyxy, confs, clss):
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            xc = float(x1 + 0.5 * w)
            yc = float(y1 + 0.5 * h)
            cls_name = self.names[int(ci)] if int(ci) < len(self.names) else str(int(ci))
            preds.append({"class": cls_name, "confidence": float(c), "x": xc, "y": yc, "width": w, "height": h})
        return {"predictions": preds}


class _ProxyNode(Node):
    def __init__(self, front_topic: str, back_topic: str, scan_topic: str, compressed: bool = True):
        super().__init__("greedy_trial_proxy_hub")
        self.front_frame: Optional[np.ndarray] = None
        self.back_frame: Optional[np.ndarray] = None
        self.front_ts = 0.0
        self.back_ts = 0.0
        self.min_distance_m: Optional[float] = None
        self.scan_ts = 0.0
        self.compressed = bool(compressed)

        if self.compressed:
            self.create_subscription(CompressedImage, front_topic, self._on_front_compressed, 10)
            self.create_subscription(CompressedImage, back_topic, self._on_back_compressed, 10)
        else:
            self.create_subscription(Image, front_topic, self._on_front_image, 10)
            self.create_subscription(Image, back_topic, self._on_back_image, 10)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)

    def _decode_compressed(self, msg: CompressedImage) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def _decode_image(self, msg: Image) -> Optional[np.ndarray]:
        try:
            h = int(msg.height)
            w = int(msg.width)
            enc = str(msg.encoding).lower()
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            if enc in ("bgr8", "rgb8"):
                img = buf.reshape((h, w, 3))
                if enc == "rgb8":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img.copy()
            if enc in ("bgra8", "rgba8"):
                img = buf.reshape((h, w, 4))
                if enc == "rgba8":
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
            if enc == "mono8":
                img = buf.reshape((h, w))
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            step = int(msg.step)
            if step >= w * 3 and len(buf) >= h * step:
                row = buf.reshape((h, step))
                img = row[:, : w * 3].reshape((h, w, 3))
                return img.copy()
        except Exception:
            return None
        return None

    def _on_front_compressed(self, msg: CompressedImage) -> None:
        img = self._decode_compressed(msg)
        if img is not None:
            self.front_frame = img
            self.front_ts = time.time()

    def _on_back_compressed(self, msg: CompressedImage) -> None:
        img = self._decode_compressed(msg)
        if img is not None:
            self.back_frame = img
            self.back_ts = time.time()

    def _on_front_image(self, msg: Image) -> None:
        img = self._decode_image(msg)
        if img is not None:
            self.front_frame = img
            self.front_ts = time.time()

    def _on_back_image(self, msg: Image) -> None:
        img = self._decode_image(msg)
        if img is not None:
            self.back_frame = img
            self.back_ts = time.time()

    def _on_scan(self, msg: LaserScan) -> None:
        self.scan_ts = time.time()
        try:
            vals = [float(r) for r in msg.ranges if math.isfinite(float(r)) and float(r) > 0.0]
            self.min_distance_m = min(vals) if vals else None
        except Exception:
            self.min_distance_m = None


class ProxySensorHub:
    def __init__(self, front_topic: str, back_topic: str, scan_topic: str = "/scan", compressed: bool = True):
        if not HAS_ROS2:
            raise RuntimeError("ROS2 not available for proxy mode.")
        if not rclpy.ok():
            rclpy.init()
        self.node = _ProxyNode(front_topic, back_topic, scan_topic, compressed=compressed)

    def spin_once(self) -> None:
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def get_frame_packet(self, fresh_timeout_s: float = 1.5) -> FramePacket:
        now = time.time()
        cam_a_ok = self.node.front_frame is not None and (now - self.node.front_ts) <= fresh_timeout_s
        cam_b_ok = self.node.back_frame is not None and (now - self.node.back_ts) <= fresh_timeout_s
        ts = max(self.node.front_ts, self.node.back_ts, now)
        return FramePacket(
            timestamp=ts,
            frame_a=self.node.front_frame.copy() if self.node.front_frame is not None else None,
            frame_b=self.node.back_frame.copy() if self.node.back_frame is not None else None,
            cam_a_ok=cam_a_ok,
            cam_b_ok=cam_b_ok,
        )

    def get_distance(self) -> Optional[float]:
        return self.node.min_distance_m

    def close(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


class ReplayVideoHub:
    def __init__(
        self,
        video_path: Optional[str] = None,
        video_a: Optional[str] = None,
        video_b: Optional[str] = None,
        side_by_side: bool = False,
        loop: bool = True,
    ):
        self.video_path = os.path.expanduser(video_path) if video_path else None
        self.video_a = os.path.expanduser(video_a) if video_a else None
        self.video_b = os.path.expanduser(video_b) if video_b else None
        self.side_by_side = bool(side_by_side)
        self.loop = bool(loop)
        self.cap_main = None
        self.cap_a = None
        self.cap_b = None

        if self.video_a or self.video_b:
            if not (self.video_a and self.video_b):
                raise ValueError("Replay dual-cam requires both video_a and video_b")
            self.cap_a = cv2.VideoCapture(self.video_a)
            self.cap_b = cv2.VideoCapture(self.video_b)
            if not self.cap_a.isOpened() or not self.cap_b.isOpened():
                raise RuntimeError("Could not open replay video pair")
        else:
            if not self.video_path:
                raise ValueError("ReplayVideoHub requires a video path")
            self.cap_main = cv2.VideoCapture(self.video_path)
            if not self.cap_main.isOpened():
                raise RuntimeError(f"Could not open replay video: {self.video_path}")

    def spin_once(self) -> None:
        return

    def _rewind(self, cap) -> None:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            pass

    def _read_one(self, cap) -> Tuple[Optional[np.ndarray], bool]:
        if cap is None:
            return None, False
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame, True
        if self.loop:
            self._rewind(cap)
            ok2, frame2 = cap.read()
            if ok2 and frame2 is not None:
                return frame2, True
        return None, False

    def get_frame_packet(self, fresh_timeout_s: float = 1.5) -> FramePacket:
        ts = time.time()
        if self.cap_a is not None and self.cap_b is not None:
            fa, oka = self._read_one(self.cap_a)
            fb, okb = self._read_one(self.cap_b)
            return FramePacket(timestamp=ts, frame_a=fa, frame_b=fb, cam_a_ok=oka, cam_b_ok=okb)
        frame, ok = self._read_one(self.cap_main)
        if not ok or frame is None:
            return FramePacket(timestamp=ts)
        if self.side_by_side and frame.shape[1] >= 2:
            mid = frame.shape[1] // 2
            return FramePacket(timestamp=ts, frame_a=frame[:, :mid].copy(), frame_b=frame[:, mid:].copy(), cam_a_ok=True, cam_b_ok=True)
        return FramePacket(timestamp=ts, frame_a=frame, cam_a_ok=True, cam_b_ok=False)

    def get_distance(self) -> Optional[float]:
        return None

    def close(self) -> None:
        for cap in (self.cap_main, self.cap_a, self.cap_b):
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass


class CmdVelPublisher(Node):
    def __init__(self, topic: str = "/cmd_vel"):
        super().__init__("greedy_trial_cmd_vel_pub")
        self.pub = self.create_publisher(Twist, topic, 10)

    def send(self, v_mps: float, w_rps: float) -> None:
        msg = Twist()
        msg.linear.x = float(v_mps)
        msg.angular.z = float(w_rps)
        self.pub.publish(msg)


def ensure_ros() -> None:
    if not HAS_ROS2:
        raise RuntimeError("ROS2 not available.")
    if not rclpy.ok():
        rclpy.init()


def summarize_preds(inf: Any) -> Tuple[int, str, float]:
    preds = []
    if isinstance(inf, dict):
        preds = inf.get("predictions", []) or []
    elif isinstance(inf, list):
        preds = inf
    if not preds:
        return 0, "", 0.0
    top = max(preds, key=lambda p: float(p.get("confidence", 0.0)))
    cls = str(top.get("class", "") or "")
    conf = float(top.get("confidence", 0.0))
    return len(preds), cls, conf


def label_from_detection(top_cls: str, cam_ok_count: int, expected_cam: int) -> str:
    t = (top_cls or "").lower()
    if cam_ok_count <= 0:
        return "camera_compromised"
    if "human" in t or "person" in t:
        return "human_hazard"
    if "object" in t or "hazard" in t or "rock" in t or "tool" in t or "brick" in t:
        return "object_hazard"
    if "harvest" in t or "regolith" in t:
        return "harvestable"
    if cam_ok_count < expected_cam:
        return "camera_compromised"
    return "search"


def expected_action_for_label(label: str) -> str:
    if label == "object_hazard":
        return RoverAction.GO_HOME
    if label == "human_hazard":
        return RoverAction.HOLD
    if label == "camera_compromised":
        return RoverAction.GO_LASER
    return RoverAction.SEARCH_ROUTE


def seed_state_priors(policy: EpsilonGreedyMissionPolicy, obs) -> None:
    state_key = policy.state_key(obs)
    policy._ensure_state(state_key)
    q = policy.q_table[state_key]
    if any(abs(v) > 1e-9 for v in q.values()):
        return
    if obs.health_score is not None and obs.health_score <= 0.35:
        q[RoverAction.GO_HOME] = 1.0
        q[RoverAction.HOLD] = 0.2
        return
    if obs.distance_m is None or obs.lidar_fresh is False:
        q[RoverAction.HOLD] = 0.5
        q[RoverAction.SEARCH_ROUTE] = 0.2
        return
    if (obs.autonomy_phase or "").upper() in ("GO_HOME",):
        q[RoverAction.GO_HOME] = 0.9
    else:
        q[RoverAction.SEARCH_ROUTE] = 0.8
        q[RoverAction.GO_LASER] = 0.4


def compute_health_score(cam_ok_count: int, expected_cam: int, lidar_ok: bool) -> float:
    cam_part = 0.5 * (float(cam_ok_count) / max(1.0, float(expected_cam)))
    lidar_part = 0.5 if lidar_ok else 0.0
    return max(0.0, min(1.0, cam_part + lidar_part))


def compute_reward(label: str, dist_m: Optional[float], proposed_action: str, final_action: str, health_score: float) -> float:
    reward = 0.2 + 0.6 * float(health_score)
    if dist_m is None:
        reward -= 0.2
    else:
        d = float(dist_m)
        if d <= 0.35:
            reward -= 2.0
        elif d <= 0.60:
            reward -= 1.0
        elif d <= 1.50:
            reward -= 0.3
        else:
            reward += 0.1

    expected = expected_action_for_label(label)
    if str(final_action) == expected:
        reward += 0.8
    else:
        reward -= 0.4

    if proposed_action != final_action:
        reward -= 0.1
    if label == "camera_compromised" and final_action == RoverAction.GO_LASER:
        reward += 0.2
    if label == "object_hazard" and final_action == RoverAction.GO_HOME:
        reward += 0.3
    return float(reward)


def final_action_with_overrides(label: str, shield_action: str) -> str:
    if label == "object_hazard":
        return RoverAction.GO_HOME
    if label == "human_hazard":
        return RoverAction.HOLD
    if label == "camera_compromised" and shield_action == RoverAction.SEARCH_ROUTE:
        return RoverAction.GO_LASER
    return str(shield_action)


def action_to_nav_mode(final_action: str) -> Optional[NavMode]:
    if str(final_action) == RoverAction.GO_HOME:
        return NavMode.GO_HOME
    if str(final_action) == RoverAction.GO_LASER:
        return NavMode.GO_LASER
    if str(final_action) == RoverAction.SEARCH_ROUTE:
        return NavMode.SEARCH_ROUTE
    return None


def ui_model_name(model_path: str) -> str:
    base = os.path.basename(os.path.expanduser(model_path))
    low = base.lower()
    if low.endswith(".onnx") and ("v26" in low or "yolov26" in low):
        return "YOLOv26 (ONNX)"
    if low.endswith(".pt") and ("v11" in low or "yolov11" in low):
        return "YOLOv11 (PT)"
    if low.endswith(".engine"):
        return "TensorRT (ENGINE)"
    if low.endswith(".onnx"):
        return "ONNX"
    if low.endswith(".pt"):
        return "PT"
    return base or "model"


def render_overlay(
    frame: np.ndarray,
    *,
    model_name: str,
    top_cls: str,
    top_conf: float,
    label: str,
    proposed_action: str,
    final_action: str,
    nav_mode: str,
    dist_m: Optional[float],
    cam_ok_count: int,
    expected_cam: int,
    epsilon: float,
    reward: float,
    q_name: str,
) -> np.ndarray:
    disp = cv2.resize(frame, (960, 540))
    panel_w = 320
    canvas = np.zeros((540, 960 + panel_w, 3), dtype=np.uint8)
    canvas[:, :960] = disp
    canvas[:, 960:] = (20, 20, 20)

    action_color = (0, 255, 0)
    if str(final_action) in (RoverAction.HOLD, "STOP", "RETREAT", "DEGRADED"):
        action_color = (0, 165, 255)
    if str(final_action) in (RoverAction.GO_HOME, RoverAction.RETURN_HOME):
        action_color = (0, 0, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x0 = 970
    y = 28
    dy = 22
    dist_str = "--" if dist_m is None else f"{float(dist_m):.2f}m"
    conf_str = f"{float(top_conf):.2f}" if top_conf else "--"

    cv2.putText(canvas, f"MODEL: {model_name}", (x0, y), font, 0.55, (255,255,255), 1); y += dy
    cv2.putText(canvas, f"TOP: {top_cls or 'none'} @ {conf_str}", (x0, y), font, 0.52, (230,230,230), 1); y += dy
    cv2.putText(canvas, f"LABEL: {label}", (x0, y), font, 0.52, (230,230,230), 1); y += dy
    cv2.putText(canvas, f"PROPOSED: {proposed_action}", (x0, y), font, 0.52, (230,230,230), 1); y += dy
    cv2.putText(canvas, f"FINAL: {final_action}", (x0, y), font, 0.62, action_color, 2); y += dy
    cv2.putText(canvas, f"NAV: {nav_mode}", (x0, y), font, 0.52, (210,210,210), 1); y += dy
    cv2.putText(canvas, f"LiDAR min: {dist_str}", (x0, y), font, 0.52, (210,210,210), 1); y += dy
    cv2.putText(canvas, f"CAMS: {cam_ok_count}/{expected_cam}", (x0, y), font, 0.52, (210,210,210), 1); y += dy
    cv2.putText(canvas, f"eps={epsilon:.3f} reward={reward:+.2f}", (x0, y), font, 0.50, (200,200,200), 1); y += dy
    cv2.putText(canvas, f"Q: {q_name}", (x0, y), font, 0.45, (180,180,180), 1); y += dy

    cv2.rectangle(canvas, (0, 0), (960, 58), (0, 0, 0), -1)
    cv2.putText(canvas, f"DECISION={label}  FINAL={final_action}  NAV={nav_mode}", (10, 25), font, 0.60, (255,255,255), 1)
    cv2.putText(canvas, f"TOP={top_cls or 'none'}@{conf_str}  LiDAR={dist_str}  CAMS={cam_ok_count}/{expected_cam}", (10, 50), font, 0.48, (220,220,220), 1)
    return canvas


def run_greedy_trial(
    *,
    window_title: str,
    model_path: str,
    q_table_path: str,
    sensor_hub: Any,
    cmd_topic: str = "/cmd_vel",
    laser_x: float = 6.0,
    laser_y: float = 0.0,
    max_fps: int = 15,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "",
    max_det: int = 50,
    epsilon: float = 0.25,
    min_epsilon: float = 0.05,
    epsilon_decay: float = 0.995,
    alpha: float = 0.20,
    gamma: float = 0.95,
    eval_only: bool = False,
    rng_seed: int = 42,
) -> None:
    ensure_ros()
    cmd_node = CmdVelPublisher(topic=cmd_topic)

    detector = UltralyticsYOLOProvider(
        model_path=model_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=(device or "").strip() or None,
        max_det=max_det,
    )
    policy = EpsilonGreedyMissionPolicy(
        training=not eval_only,
        q_table_path=q_table_path,
        epsilon=epsilon,
        min_epsilon=min_epsilon,
        epsilon_decay=epsilon_decay,
        alpha=alpha,
        gamma=gamma,
        rng_seed=rng_seed,
    )
    controller = ShieldedController(policy=policy, supervisor=SafetySupervisor.default())
    nav = Navigator()
    nav.set_laser_waypoint(Waypoint(float(laser_x), float(laser_y), meta={"label": "LASER"}))

    q_name = Path(q_table_path).name
    logger = CsvFrameLogger(Path(q_table_path).with_suffix("_frames.csv"))

    pending_reward: Optional[float] = None
    last_nav_mode = NavMode.SEARCH_ROUTE
    model_name = ui_model_name(model_path)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    try:
        while True:
            loop_start = time.time()

            sensor_hub.spin_once()
            rclpy.spin_once(cmd_node, timeout_sec=0.0)
            pkt = sensor_hub.get_frame_packet()
            frame = pkt.frame
            cam_ok_count = int(bool(pkt.cam_a_ok)) + int(bool(pkt.cam_b_ok))
            expected_cam = 2 if (pkt.frame_b is not None or pkt.cam_b_ok) else 1
            dist_m = sensor_hub.get_distance()
            lidar_ok = dist_m is not None

            inf = detector.infer(frame)
            _, top_cls, top_conf = summarize_preds(inf)
            label = label_from_detection(top_cls, cam_ok_count, expected_cam)
            health_score = compute_health_score(cam_ok_count, expected_cam, lidar_ok)

            pose = nav.pose_tracker.pose
            dist_home = math.hypot(float(pose.x_m), float(pose.y_m))
            dist_laser = math.hypot(float(pose.x_m) - float(laser_x), float(pose.y_m) - float(laser_y))

            obs = build_observation(
                StepInputs(
                    perception=Perception(pred_class=(top_cls or label), pred_conf=float(top_conf or 0.0)),
                    distance_m=dist_m,
                    telemetry=Telemetry(health_score=health_score),
                    nav_state=NavState(
                        autonomy_phase=str(last_nav_mode.value),
                        pose_x_m=float(pose.x_m),
                        pose_y_m=float(pose.y_m),
                        yaw_rad=float(pose.yaw_rad),
                        dist_home_m=dist_home,
                        dist_laser_m=dist_laser,
                        route_idx=None,
                        dist_to_waypoint_m=None,
                        pose_fresh=bool(cam_ok_count > 0),
                        lidar_fresh=bool(lidar_ok),
                    ),
                )
            )

            if pending_reward is not None:
                policy.record_reward(reward=pending_reward, next_obs=obs, done=False)
                pending_reward = None

            seed_state_priors(policy, obs)
            decision = controller.step(obs)
            proposed_action = str(decision.proposed_action)
            final_action = final_action_with_overrides(label, str(decision.final_action))
            expected_action = expected_action_for_label(label)

            nav_mode = action_to_nav_mode(final_action)
            if nav_mode is None:
                twist = Twist2D(v_mps=0.0, w_rps=0.0)
                nav_mode_str = "HOLD"
            else:
                nav_prop = nav.step(nav_mode=nav_mode, timestamp_s=time.time(), vision={"harvest_intent": label == "harvestable"})
                twist = nav_prop.twist
                nav_mode_str = nav_mode.value
                last_nav_mode = nav_mode

            if final_action in (RoverAction.HOLD, RoverAction.STOP, RoverAction.RETREAT, RoverAction.DEGRADED):
                twist = Twist2D(v_mps=0.0, w_rps=0.0)

            cmd_node.send(float(getattr(twist, "v_mps", 0.0)), float(getattr(twist, "w_rps", 0.0)))
            reward = compute_reward(label, dist_m, proposed_action, final_action, health_score)
            pending_reward = reward

            logger.log(
                ts=time.time(),
                rf_top_class=str(top_cls or ""),
                nav_mode=nav_mode_str,
                final_action=str(final_action),
                cam_ok_count=cam_ok_count,
                cam_expected=expected_cam,
                dist_m=dist_m,
                label=label,
                expected=expected_action,
                proposed_action=proposed_action,
                reward=reward,
            )

            canvas = render_overlay(
                frame,
                model_name=model_name,
                top_cls=str(top_cls or ""),
                top_conf=float(top_conf or 0.0),
                label=label,
                proposed_action=proposed_action,
                final_action=str(final_action),
                nav_mode=nav_mode_str,
                dist_m=dist_m,
                cam_ok_count=cam_ok_count,
                expected_cam=expected_cam,
                epsilon=float(policy.epsilon),
                reward=reward,
                q_name=q_name,
            )
            cv2.imshow(window_title, canvas)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            period = 1.0 / float(max(1, int(max_fps)))
            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    finally:
        try:
            terminal_obs = build_observation(
                StepInputs(
                    perception=Perception(pred_class="terminal", pred_conf=1.0),
                    distance_m=None,
                    telemetry=Telemetry(health_score=0.0),
                    nav_state=NavState(autonomy_phase="TERMINAL"),
                )
            )
            if pending_reward is not None:
                policy.record_reward(reward=pending_reward, next_obs=terminal_obs, done=True)
        except Exception:
            pass

        try:
            policy.save(q_table_path)
            print(f"[RL] saved q-table -> {q_table_path}")
        except Exception as e:
            print(f"[RL][WARN] could not save q-table: {e}")

        try:
            cmd_node.send(0.0, 0.0)
        except Exception:
            pass
        try:
            cmd_node.destroy_node()
        except Exception:
            pass
        try:
            sensor_hub.close()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        cv2.destroyAllWindows()
