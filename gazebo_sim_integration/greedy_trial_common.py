
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
It also bootstraps sys.path explicitly because rover_learner root files are imported as top-level files, not installed modules.
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROVER_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(ROVER_ROOT, ".."))
for p in (THIS_DIR, ROVER_ROOT, WORKSPACE_ROOT):
    if p not in sys.path:
        sys.path.append(p)

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CompressedImage, LaserScan
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    HAS_ROS2 = True
except Exception:
    rclpy = None
    Node = object  # type: ignore
    Image = None  # type: ignore
    CompressedImage = None  # type: ignore
    LaserScan = None  # type: ignore
    Twist = None  # type: ignore
    String = None  # type: ignore
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
                "dist_home_m", "dist_laser_m", "label", "expected", "decision_ok",
                "proposed_action", "reward", "progress_tag", "no_progress_streak", "pose_fresh", "smoothed_action",
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
        dist_home_m: float,
        dist_laser_m: float,
        progress_tag: str,
        no_progress_streak: int,
        pose_fresh: bool,
        smoothed_action: str,
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
    def __init__(
        self,
        front_topic: str,
        back_topic: str,
        scan_topic: str,
        compressed: bool = True,
        pose_topic: Optional[str] = "/alam/rover_pose_json",
    ):
        super().__init__("greedy_trial_proxy_hub")
        self.front_frame: Optional[np.ndarray] = None
        self.back_frame: Optional[np.ndarray] = None
        self.front_ts = 0.0
        self.back_ts = 0.0
        self.min_distance_m: Optional[float] = None
        self.scan_ts = 0.0
        self.pose_xyz: Optional[Tuple[float, float, float]] = None
        self.pose_ts = 0.0
        self.compressed = bool(compressed)

        if self.compressed:
            self.create_subscription(CompressedImage, front_topic, self._on_front_compressed, 10)
            self.create_subscription(CompressedImage, back_topic, self._on_back_compressed, 10)
        else:
            self.create_subscription(Image, front_topic, self._on_front_image, 10)
            self.create_subscription(Image, back_topic, self._on_back_image, 10)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)

        if pose_topic:
            try:
                self.create_subscription(String, str(pose_topic), self._on_pose_json, 10)
            except Exception:
                pass

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

    def _on_pose_json(self, msg: String) -> None:
        try:
            import json as _json
            data = _json.loads(str(msg.data))
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
            yaw = float(data.get("yaw", 0.0))
            self.pose_xyz = (x, y, yaw)
            self.pose_ts = time.time()
        except Exception:
            return


class ProxySensorHub:
    def __init__(
        self,
        front_topic: str,
        back_topic: str,
        scan_topic: str = "/scan",
        image_transport: str = "raw",
        pose_topic: str = "/alam/rover_pose_json",
    ):
        if not HAS_ROS2:
            raise RuntimeError("ROS2 not available for proxy mode.")
        if not rclpy.ok():
            rclpy.init()
        transport = str(image_transport or "raw").strip().lower()
        compressed = (transport == "compressed")
        self.node = _ProxyNode(front_topic, back_topic, scan_topic, compressed=compressed, pose_topic=pose_topic)

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

    def get_pose(self, fresh_timeout_s: float = 1.5) -> Tuple[Optional[Tuple[float, float, float]], bool]:
        now = time.time()
        fresh = self.node.pose_xyz is not None and (now - self.node.pose_ts) <= fresh_timeout_s
        return (self.node.pose_xyz if fresh else None), bool(fresh)

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

    def get_pose(self, fresh_timeout_s: float = 1.5) -> Tuple[Optional[Tuple[float, float, float]], bool]:
        return None, False

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





@dataclass
class LabelState:
    object_seen_streak: int = 0
    human_seen_streak: int = 0
    harvest_seen_streak: int = 0
    partial_cam_streak: int = 0
    no_cam_streak: int = 0
    go_home_block_streak: int = 0


def label_from_detection(
    top_cls: str,
    top_conf: float,
    cam_ok_count: int,
    expected_cam: int,
    dist_m: Optional[float],
    state: LabelState,
) -> str:
    t = (top_cls or "").lower()
    conf = float(top_conf or 0.0)
    d = None if dist_m is None else float(dist_m)

    is_human = ("human" in t) or ("person" in t)
    is_object = ("object" in t) or ("hazard" in t) or ("rock" in t) or ("tool" in t) or ("brick" in t)
    is_harvest = ("harvest" in t) or ("regolith" in t)

    state.human_seen_streak = state.human_seen_streak + 1 if is_human else 0
    state.object_seen_streak = state.object_seen_streak + 1 if is_object else 0
    state.harvest_seen_streak = state.harvest_seen_streak + 1 if is_harvest else 0
    state.no_cam_streak = state.no_cam_streak + 1 if cam_ok_count <= 0 else 0
    state.partial_cam_streak = state.partial_cam_streak + 1 if (0 < cam_ok_count < expected_cam) else 0

    if state.no_cam_streak >= 2:
        return "camera_compromised"
    if state.partial_cam_streak >= 6:
        return "camera_compromised"

    if state.human_seen_streak >= 1:
        return "human_hazard"

    if state.harvest_seen_streak >= 1 and conf >= 0.22:
        return "harvestable"

    if state.object_seen_streak > 0:
        close_obj = (d is not None) and (d <= 0.55)
        medium_obj = (d is not None) and (d <= 1.10)
        persistent = state.object_seen_streak >= 4
        strong_conf = conf >= 0.70

        if close_obj and persistent and strong_conf:
            return "object_hazard"
        if medium_obj and state.object_seen_streak >= 3 and conf >= 0.45:
            return "object_caution"
        return "search"

    return "search"


def expected_action_for_label(
    label: str,
    dist_m: Optional[float] = None,
    progress_tag: str = "",
) -> str:
    if label == "object_hazard":
        return RoverAction.GO_HOME
    if label == "object_caution":
        if dist_m is not None and float(dist_m) <= 0.8:
            return RoverAction.HOLD
        return RoverAction.SEARCH_ROUTE
    if label == "human_hazard":
        return RoverAction.HOLD
    if label == "camera_compromised":
        return RoverAction.GO_LASER
    if progress_tag == "STUCK":
        return RoverAction.SEARCH_ROUTE
    return RoverAction.SEARCH_ROUTE


def compute_health_score(cam_ok_count: int, expected_cam: int, lidar_ok: bool) -> float:
    cam_part = 0.5 * (float(cam_ok_count) / max(1.0, float(expected_cam)))
    lidar_part = 0.5 if lidar_ok else 0.0
    return max(0.0, min(1.0, cam_part + lidar_part))


def seed_state_priors(policy: EpsilonGreedyMissionPolicy, obs) -> None:
    state_key = policy.state_key(obs)
    policy._ensure_state(state_key)
    q = policy.q_table[state_key]
    if any(abs(v) > 1e-9 for v in q.values()):
        return

    phase = (obs.autonomy_phase or "").upper()

    if obs.health_score is not None and obs.health_score <= 0.35:
        q[RoverAction.GO_HOME] = 0.55
        q[RoverAction.HOLD] = 0.20
        q[RoverAction.SEARCH_ROUTE] = 0.10
        return

    if "STUCK" in phase:
        q[RoverAction.SEARCH_ROUTE] = 0.85
        q[RoverAction.GO_LASER] = 0.20
        q[RoverAction.HOLD] = 0.05
        q[RoverAction.GO_HOME] = -0.10
        return

    if obs.distance_m is None or obs.lidar_fresh is False:
        q[RoverAction.SEARCH_ROUTE] = 0.65
        q[RoverAction.GO_LASER] = 0.20
        q[RoverAction.HOLD] = 0.05
        q[RoverAction.GO_HOME] = -0.15
        return

    if phase.startswith("GO_HOME"):
        q[RoverAction.GO_HOME] = 0.55
        q[RoverAction.SEARCH_ROUTE] = 0.20
        q[RoverAction.HOLD] = 0.05
        return
    if phase.startswith("GO_LASER"):
        q[RoverAction.GO_LASER] = 0.60
        q[RoverAction.SEARCH_ROUTE] = 0.30
        q[RoverAction.GO_HOME] = -0.10
        return

    q[RoverAction.SEARCH_ROUTE] = 0.95
    q[RoverAction.GO_LASER] = 0.20
    q[RoverAction.HOLD] = 0.00
    q[RoverAction.GO_HOME] = -0.20




@dataclass
class ActionSmoother:
    current_action: str = RoverAction.SEARCH_ROUTE
    last_switch_ts: float = 0.0
    stable_frames: int = 0


def smooth_action_choice(
    *,
    desired_action: str,
    q_values: Dict[str, float],
    smoother: ActionSmoother,
    now_ts: float,
    min_hold_frames: int = 5,
    switch_margin: float = 0.18,
    switch_cooldown_s: float = 0.40,
) -> str:
    desired = str(desired_action)
    current = str(smoother.current_action or RoverAction.SEARCH_ROUTE)

    if current not in q_values:
        smoother.current_action = desired
        smoother.last_switch_ts = float(now_ts)
        smoother.stable_frames = 1
        return desired

    if desired == current:
        smoother.stable_frames += 1
        return current

    current_q = float(q_values.get(current, -1e9))
    desired_q = float(q_values.get(desired, -1e9))
    q_gap = desired_q - current_q
    recent_switch = (float(now_ts) - float(smoother.last_switch_ts)) < float(switch_cooldown_s)

    if smoother.stable_frames < int(min_hold_frames) and q_gap < float(switch_margin):
        smoother.stable_frames += 1
        return current

    if recent_switch and q_gap < (float(switch_margin) * 1.5):
        smoother.stable_frames += 1
        return current

    smoother.current_action = desired
    smoother.last_switch_ts = float(now_ts)
    smoother.stable_frames = 1
    return desired


@dataclass
class RewardState:
    last_label: str = ""
    last_final_action: str = ""
    last_proposed_action: str = ""
    last_dist_m: Optional[float] = None
    last_home_dist_m: Optional[float] = None
    last_laser_dist_m: Optional[float] = None
    last_pose_x_m: Optional[float] = None
    last_pose_y_m: Optional[float] = None
    last_progress_tag: str = "INIT"
    hold_streak: int = 0
    action_switches: int = 0
    same_action_streak: int = 0
    no_progress_streak: int = 0
    frames_seen: int = 0
    go_home_streak: int = 0


def compute_progress_tag(
    *,
    pose_xy: Optional[Tuple[float, float]],
    dist_home_m: float,
    dist_laser_m: float,
    last_nav_mode: str,
    state: RewardState,
) -> str:
    if pose_xy is None:
        return "NOPOSE"

    moved = None
    if state.last_pose_x_m is not None and state.last_pose_y_m is not None:
        moved = math.hypot(float(pose_xy[0]) - float(state.last_pose_x_m), float(pose_xy[1]) - float(state.last_pose_y_m))

    toward_home = None
    toward_laser = None
    away_home = None
    if state.last_home_dist_m is not None:
        toward_home = float(state.last_home_dist_m) - float(dist_home_m)
        away_home = float(dist_home_m) - float(state.last_home_dist_m)
    if state.last_laser_dist_m is not None:
        toward_laser = float(state.last_laser_dist_m) - float(dist_laser_m)

    if moved is not None and moved < 0.02 and state.frames_seen >= 3:
        return "STUCK"

    if str(last_nav_mode) == str(NavMode.GO_HOME.value):
        if toward_home is not None and toward_home > 0.03:
            return "TOWARD_HOME"
        return "NO_PROGRESS"

    if str(last_nav_mode) == str(NavMode.GO_LASER.value):
        if toward_laser is not None and toward_laser > 0.03:
            return "TOWARD_LASER"
        return "NO_PROGRESS"

    if away_home is not None and away_home > 0.03:
        return "AWAY_HOME"
    if moved is not None and moved > 0.02:
        return "MOVING"
    return "NO_PROGRESS"


def compute_reward(
    *,
    label: str,
    dist_m: Optional[float],
    proposed_action: str,
    final_action: str,
    health_score: float,
    dist_home_m: float,
    dist_laser_m: float,
    progress_tag: str,
    pose_fresh: bool,
    state: RewardState,
) -> float:
    reward = 0.0
    reward += 0.20 + 0.70 * float(health_score)

    if not pose_fresh:
        reward -= 0.10

    if dist_m is None:
        reward -= 0.03
    else:
        d = float(dist_m)
        if d <= 0.35:
            reward -= 1.80
        elif d <= 0.60:
            reward -= 0.65
        elif d <= 1.50:
            reward -= 0.05
        elif d <= 4.0:
            reward += 0.08

    expected = expected_action_for_label(label, dist_m, progress_tag)
    if str(final_action) == expected:
        reward += 0.75
    else:
        reward -= 0.25

    if str(proposed_action) == expected:
        reward += 0.10
    if proposed_action != final_action:
        reward -= 0.06

    # Strongly discourage retreat in non-emergency states
    if label in ("search", "harvestable", "object_caution"):
        if str(final_action) == RoverAction.GO_HOME:
            reward -= 1.80
        if str(final_action) == RoverAction.HOLD and label != "object_caution":
            reward -= 0.75

    if label == "search":
        if str(final_action) == RoverAction.SEARCH_ROUTE:
            reward += 0.60
    elif label == "harvestable":
        if str(final_action) == RoverAction.SEARCH_ROUTE:
            reward += 0.70
        elif str(final_action) == RoverAction.GO_LASER:
            reward -= 0.25
    elif label == "camera_compromised":
        if str(final_action) == RoverAction.GO_LASER:
            reward += 0.55
        elif str(final_action) == RoverAction.GO_HOME:
            reward -= 0.70
    elif label == "human_hazard":
        if str(final_action) == RoverAction.HOLD:
            reward += 0.35
    elif label == "object_caution":
        if dist_m is not None and float(dist_m) <= 0.8:
            if str(final_action) == RoverAction.HOLD:
                reward += 0.35
            elif str(final_action) == RoverAction.SEARCH_ROUTE:
                reward -= 0.15
        else:
            if str(final_action) == RoverAction.SEARCH_ROUTE:
                reward += 0.45
    elif label == "object_hazard":
        if str(final_action) == RoverAction.GO_HOME:
            reward += 0.35

    # Progress shaping
    if str(final_action) == RoverAction.SEARCH_ROUTE:
        if progress_tag in ("AWAY_HOME", "MOVING", "TOWARD_LASER"):
            reward += 0.75
        elif progress_tag == "STUCK":
            reward -= 0.95
        elif progress_tag == "NO_PROGRESS":
            reward -= 0.35

    elif str(final_action) == RoverAction.GO_LASER:
        if progress_tag == "TOWARD_LASER":
            reward += 0.50
        elif progress_tag == "STUCK":
            reward -= 1.20
        elif progress_tag == "NO_PROGRESS":
            reward -= 0.55

    elif str(final_action) == RoverAction.GO_HOME:
        if label != "object_hazard":
            reward -= 0.50
        if progress_tag == "TOWARD_HOME":
            reward += 0.20
        elif progress_tag in ("STUCK", "NO_PROGRESS"):
            reward -= 0.40

    elif str(final_action) == RoverAction.HOLD:
        reward -= 0.18 * float(max(0, state.hold_streak))

    if state.last_final_action and state.last_final_action != str(final_action):
        reward -= 0.08

    return float(reward)


def update_reward_state(
    *,
    state: RewardState,
    label: str,
    proposed_action: str,
    final_action: str,
    dist_m: Optional[float],
    dist_home_m: float,
    dist_laser_m: float,
    pose_xy: Optional[Tuple[float, float]],
    progress_tag: str,
) -> None:
    state.frames_seen += 1
    if state.last_final_action and state.last_final_action != str(final_action):
        state.action_switches += 1

    if str(final_action) == RoverAction.HOLD:
        state.hold_streak += 1
    else:
        state.hold_streak = 0

    if str(final_action) == RoverAction.GO_HOME:
        state.go_home_streak += 1
    else:
        state.go_home_streak = 0

    if state.last_final_action == str(final_action):
        state.same_action_streak += 1
    else:
        state.same_action_streak = 1

    if progress_tag in ("STUCK", "NO_PROGRESS"):
        state.no_progress_streak += 1
    else:
        state.no_progress_streak = 0

    state.last_label = str(label)
    state.last_proposed_action = str(proposed_action)
    state.last_final_action = str(final_action)
    state.last_dist_m = None if dist_m is None else float(dist_m)
    state.last_home_dist_m = float(dist_home_m)
    state.last_laser_dist_m = float(dist_laser_m)
    state.last_progress_tag = str(progress_tag)
    if pose_xy is not None:
        state.last_pose_x_m = float(pose_xy[0])
        state.last_pose_y_m = float(pose_xy[1])


def final_action_with_overrides(
    label: str,
    shield_action: str,
    dist_m: Optional[float] = None,
    progress_tag: str = "",
    state: Optional[RewardState] = None,
) -> str:
    # Only allow direct GO_HOME on persistent close object hazards.
    if label == "object_hazard":
        return RoverAction.GO_HOME

    if label == "object_caution":
        if dist_m is not None and float(dist_m) <= 0.8:
            return RoverAction.HOLD if shield_action in (RoverAction.GO_HOME, RoverAction.SEARCH_ROUTE) else str(shield_action)
        return RoverAction.SEARCH_ROUTE if shield_action == RoverAction.GO_HOME else str(shield_action)

    if label == "human_hazard":
        return RoverAction.HOLD

    if label == "camera_compromised":
        return RoverAction.GO_LASER

    # In search/harvest states, explicitly block retreat unless we are deeply unhealthy.
    if label in ("search", "harvestable"):
        if shield_action == RoverAction.GO_HOME:
            return RoverAction.SEARCH_ROUTE
        if progress_tag == "STUCK":
            return RoverAction.SEARCH_ROUTE
        if shield_action == RoverAction.HOLD and (dist_m is None or float(dist_m) > 0.8):
            return RoverAction.SEARCH_ROUTE

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
    reset_q_table: bool = False,
    action_hold_frames: int = 5,
    action_switch_margin: float = 0.18,
    action_switch_cooldown_s: float = 0.40,
) -> None:
    ensure_ros()
    cmd_node = CmdVelPublisher(topic=cmd_topic)

    q_path_for_reset = Path(q_table_path)
    if bool(reset_q_table) and q_path_for_reset.exists():
        try:
            q_path_for_reset.unlink()
            print(f"[RL] reset q-table -> {q_path_for_reset}")
        except Exception as e:
            print(f"[RL][WARN] could not reset q-table: {e}")

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
    q_path = Path(q_table_path)
    logger = CsvFrameLogger(q_path.with_name(f"{q_path.stem}_frames.csv"))

    pending_reward: Optional[float] = None
    reward_state = RewardState()
    label_state = LabelState()
    action_smoother = ActionSmoother()
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
            label = label_from_detection(
                top_cls=top_cls,
                top_conf=top_conf,
                cam_ok_count=cam_ok_count,
                expected_cam=expected_cam,
                dist_m=dist_m,
                state=label_state,
            )
            health_score = compute_health_score(cam_ok_count, expected_cam, lidar_ok)

            pose_tuple = None
            pose_fresh = False
            if hasattr(sensor_hub, "get_pose"):
                try:
                    pose_tuple, pose_fresh = sensor_hub.get_pose()
                except Exception:
                    pose_tuple, pose_fresh = None, False

            if pose_tuple is not None:
                pose_x, pose_y, pose_yaw = pose_tuple
            else:
                pose = nav.pose_tracker.pose
                pose_x, pose_y, pose_yaw = float(pose.x_m), float(pose.y_m), float(pose.yaw_rad)

            dist_home = math.hypot(float(pose_x), float(pose_y))
            dist_laser = math.hypot(float(pose_x) - float(laser_x), float(pose_y) - float(laser_y))
            progress_tag = compute_progress_tag(
                pose_xy=((float(pose_x), float(pose_y)) if pose_fresh or pose_tuple is not None else None),
                dist_home_m=float(dist_home),
                dist_laser_m=float(dist_laser),
                last_nav_mode=str(last_nav_mode.value),
                state=reward_state,
            )
            phase_state = f"{last_nav_mode.value}|{label}|{progress_tag}"
            pseudo_route_idx = min(int(reward_state.no_progress_streak), 5)
            pseudo_target_dist = float(dist_laser if str(last_nav_mode.value) == str(NavMode.GO_LASER.value) else dist_home)

            obs = build_observation(
                StepInputs(
                    perception=Perception(pred_class=(top_cls or label), pred_conf=float(top_conf or 0.0)),
                    distance_m=dist_m,
                    telemetry=Telemetry(health_score=health_score),
                    nav_state=NavState(
                        autonomy_phase=str(phase_state),
                        pose_x_m=float(pose_x),
                        pose_y_m=float(pose_y),
                        yaw_rad=float(pose_yaw),
                        dist_home_m=dist_home,
                        dist_laser_m=dist_laser,
                        route_idx=pseudo_route_idx,
                        dist_to_waypoint_m=pseudo_target_dist,
                        pose_fresh=bool(pose_fresh or cam_ok_count > 0),
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
            raw_final_action = final_action_with_overrides(
                label,
                str(decision.final_action),
                dist_m=dist_m,
                progress_tag=str(progress_tag),
                state=reward_state,
            )
            try:
                state_key = controller.policy.state_key(obs)
                q_values = dict(getattr(controller.policy, "q_table", {}).get(state_key, {}))
            except Exception:
                q_values = {}
            final_action = smooth_action_choice(
                desired_action=str(raw_final_action),
                q_values=q_values,
                smoother=action_smoother,
                now_ts=time.time(),
                min_hold_frames=int(action_hold_frames),
                switch_margin=float(action_switch_margin),
                switch_cooldown_s=float(action_switch_cooldown_s),
            )
            expected_action = expected_action_for_label(label, dist_m, str(progress_tag))

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
            reward = compute_reward(
                label=label,
                dist_m=dist_m,
                proposed_action=proposed_action,
                final_action=final_action,
                health_score=health_score,
                dist_home_m=float(dist_home),
                dist_laser_m=float(dist_laser),
                progress_tag=str(progress_tag),
                pose_fresh=bool(pose_fresh),
                state=reward_state,
            )
            update_reward_state(
                state=reward_state,
                label=label,
                proposed_action=proposed_action,
                final_action=final_action,
                dist_m=dist_m,
                dist_home_m=float(dist_home),
                dist_laser_m=float(dist_laser),
                pose_xy=((float(pose_x), float(pose_y)) if (pose_fresh or pose_tuple is not None) else None),
                progress_tag=str(progress_tag),
            )
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
                dist_home_m=float(dist_home),
                dist_laser_m=float(dist_laser),
                progress_tag=str(progress_tag),
                no_progress_streak=int(reward_state.no_progress_streak),
                pose_fresh=bool(pose_fresh),
                smoothed_action=str(final_action),
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
