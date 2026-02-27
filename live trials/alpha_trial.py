#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
alpha_trial.py

Live, reactionary trial runner:
- Roboflow inference drives decisions (not scenario logic)
- Uses live_success_metric.py for success CSV logging

Priority decision labels:
1) object_hazard    -> CURL_UP + GO_HOME
2) human_hazard     -> DUMP_RETREAT + bypass (search) + brief reverse
3) camera_compromised -> GO_LASER + DUMP_AT_LASER + continue degraded
4) harvestable      -> SCOOP_DUMP (approach intent implied)
else: search

**run without lidar: python3 alpha_trial.py --lidar none
"""

from __future__ import annotations

import os, sys, time, math, argparse, dataclasses, json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROVER_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(ROVER_ROOT, ".."))
for p in (ROVER_ROOT, THIS_DIR, WORKSPACE_ROOT):
    if p not in sys.path:
        sys.path.append(p)

from power_saver_degradation import SystemMode
from failsafe_ui import FailsafeOverlayManager

from mode_manager_arduino_model import ModeManagerArduinoModel
from power_saver_degradation_arduino_model import SystemHealthMonitorArduinoModel

from rl_safety_supervisor import SafetySupervisor, ShieldedController, HeuristicPolicy, Observation
from arduino_interlock import ArduinoInterlock

from camera_provider import CSICameraProvider, USBCameraProvider
from two_camera_provider import TwoCameraProvider
from lidar_provider import SerialRPLidarProvider, ROS2LaserScanProvider
from two_lidar_provider import SerialTwoRPLidarProvider

try:
    from roboflow_provider import RoboflowProvider
except Exception:
    RoboflowProvider = None  # optional

from safety_state_controller import SafetyStateController, SafetyState
from live_success_metric import LiveSuccessMetric

from navigation.navigation import Navigator
from navigation.types import NavMode, Waypoint, Twist2D


# -----------------------
# Ultralytics ONNX/Engine inference
# -----------------------

DEFAULT_CLASS_NAMES = ["Bypass", "harvestable", "human hazard", "object hazards"]


def _try_load_names_from_sidecar(model_path: str) -> Optional[List[str]]:
    """Best-effort load of class names from files near the model (classes.txt/.cls/labels.txt/dataset.yaml)."""
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
                # light-weight YAML-ish parse for a 'names:' list block
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
    """
    Drop-in replacement for RoboflowProvider that runs Ultralytics inference on:
      - .onnx (ONNXRuntime)
      - .engine (TensorRT)
      - .pt (PyTorch)

    Returns a Roboflow-like dict:
      {"predictions": [{"class": str, "confidence": float, "x": float, "y": float, "width": float, "height": float}, ...]}
    """
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

        from ultralytics import YOLO  # deferred import
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
                import numpy as _np
                arr = _np.array(data)
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


# -----------------------
# Optional ROS2 publish
# -----------------------
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    HAS_ROS2 = True
except Exception:
    HAS_ROS2 = False


class ArmInterface:
    def publish(self, joints_deg: List[float], gripper_0_100: float) -> None:
        raise NotImplementedError
    def close(self) -> None:
        pass


class SimArmPublisher(ArmInterface):
    def __init__(self, joint_names: List[str], gripper_joint_name: str):
        if not HAS_ROS2:
            raise RuntimeError("ROS2 not available.")
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("alam_alpha_trial")
        self.pub = self.node.create_publisher(JointState, "joint_states", 10)
        self.joint_names = joint_names
        self.gripper_joint_name = gripper_joint_name

    def publish(self, joints_deg: List[float], gripper_0_100: float) -> None:
        joint_rads = [math.radians(float(x)) for x in joints_deg]
        gr = max(0.0, min(100.0, float(gripper_0_100)))
        urdf_grip = -0.7 + (gr / 100.0) * 0.85

        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(self.joint_names) + [self.gripper_joint_name]
        msg.position = list(joint_rads) + [urdf_grip]
        self.pub.publish(msg)
        rclpy.spin_once(self.node, timeout_sec=0.0)

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


class LiveArmStub(ArmInterface):
    def publish(self, joints_deg: List[float], gripper_0_100: float) -> None:
        return


@dataclass
class ClipStep:
    name: str
    joints: List[float]
    gripper: float
    duration_s: float


CLIPS: Dict[str, List[ClipStep]] = {
    "SCOOP_DUMP": [
        ClipStep("APPROACH_POSE", [0, 20, -20, 0, 0, 0], 0, 1.2),
        ClipStep("SCOOP",         [0, 45, -45, 0, 0, 0], 100, 1.3),
        ClipStep("LIFT",          [0, 10, 10, 0, 0, 0], 100, 1.0),
        ClipStep("DUMP",          [0, -45, 90, 0, 0, 0], 0, 1.2),
        ClipStep("HOME",          [0, 0, 0, 0, 0, 0], 0, 1.0),
    ],
    "DUMP_RETREAT": [
        ClipStep("DUMP", [0, -45, 90, 0, 0, 0], 0, 1.2),
        ClipStep("TUCK", [0, 0, 0, 0, 0, 0], 0, 1.0),
    ],
    "CURL_UP": [
        ClipStep("CURL", [0, -10, 60, 40, 0, 0], 0, 1.2),
        ClipStep("TUCK", [0, 0, 0, 0, 0, 0], 0, 1.0),
    ],
    "DUMP_AT_LASER": [
        ClipStep("DUMP", [0, -45, 90, 0, 0, 0], 0, 1.3),
        ClipStep("HOME", [0, 0, 0, 0, 0, 0], 0, 1.0),
    ],
}


class MotionInterpolator:
    def __init__(self, joint_speed_deg: float = 28.0, gripper_speed: float = 70.0):
        # slower + more “space-like”
        self.current_joints = [0.0] * 6
        self.current_gripper = 0.0
        self.joint_speed = float(joint_speed_deg)
        self.gripper_speed = float(gripper_speed)
        self.last_t = time.time()

    def update(self, target_joints: List[float], target_gripper: float) -> Tuple[List[float], float]:
        now = time.time()
        dt = max(1e-4, now - self.last_t)
        self.last_t = now

        out = []
        for curr, targ in zip(self.current_joints, target_joints):
            diff = targ - curr
            max_move = self.joint_speed * dt
            if abs(diff) < 0.15:
                out.append(float(targ))
            else:
                out.append(curr + math.copysign(min(abs(diff), max_move), diff))
        self.current_joints = out

        diff_g = float(target_gripper) - float(self.current_gripper)
        max_move_g = self.gripper_speed * dt
        if abs(diff_g) < 1.0:
            self.current_gripper = float(target_gripper)
        else:
            self.current_gripper += math.copysign(min(abs(diff_g), max_move_g), diff_g)

        return self.current_joints, self.current_gripper


class ClipPlayer:
    def __init__(self):
        self.active = "IDLE"
        self.idx = 0
        self.step_t0 = 0.0
        self.done = True

    def start(self, name: str):
        if name not in CLIPS:
            return
        self.active = name
        self.idx = 0
        self.step_t0 = time.time()
        self.done = False

    def tick(self) -> Tuple[List[float], float, str, bool]:
        if self.done or self.active not in CLIPS:
            return [0.0] * 6, 0.0, "IDLE", True
        now = time.time()
        steps = CLIPS[self.active]
        step = steps[self.idx]
        if (now - self.step_t0) >= step.duration_s:
            self.idx += 1
            if self.idx >= len(steps):
                self.done = True
                return [0.0] * 6, 0.0, f"{self.active}_DONE", True
            self.step_t0 = now
            step = steps[self.idx]
        return list(map(float, step.joints)), float(step.gripper), step.name, False


def _extract_distance_m(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    for attr in ("min_distance_m", "distance_m", "distance", "range_m", "value_m", "value"):
        if hasattr(x, attr):
            try:
                v = float(getattr(x, attr))
                if math.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass
    return None


def _extract_frame(packet: Any) -> Tuple[np.ndarray, int, float]:
    ts = float(getattr(packet, "timestamp", time.time()))
    if hasattr(packet, "frame") and getattr(packet, "frame") is not None:
        return getattr(packet, "frame"), 1, ts
    fa = getattr(packet, "frame_a", None)
    fb = getattr(packet, "frame_b", None)
    ok_a = getattr(packet, "cam_a_ok", fa is not None)
    ok_b = getattr(packet, "cam_b_ok", fb is not None)
    ok = int(bool(ok_a)) + int(bool(ok_b))
    if fa is not None:
        return fa, ok, ts
    if fb is not None:
        return fb, ok, ts
    return np.zeros((360, 480, 3), dtype=np.uint8), 0, ts


def _summarize_rf(inf: Any) -> Tuple[int, str, str, str]:
    """
    Summarize Roboflow output to:
      (num_preds, classes_csv, top_class, top_conf_str)

    Supports:
      - dict with 'predictions' list (classic)
      - list of dict predictions
      - RFInference-like object with:
          .detections (iterable of objects w/ .cls and .conf) and optional .raw
    """
    # RFInference-like path
    if inf is not None and hasattr(inf, "detections"):
        dets = getattr(inf, "detections") or []
        classes = []
        top_cls = ""
        top_conf = -1.0
        for d in dets:
            cls = getattr(d, "cls", "") or ""
            conf = getattr(d, "conf", 0.0)
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            if cls:
                classes.append(str(cls))
            if conf_f > top_conf:
                top_conf = conf_f
                top_cls = str(cls)
        classes_csv = ";".join(sorted(set(classes)))
        top_conf_str = "" if top_conf < 0 else f"{top_conf:.3f}"
        return len(dets), classes_csv, top_cls, top_conf_str

    # dict/list fallback (older providers)
    preds = None
    for key in ("predictions", "preds", "detections"):
        if isinstance(inf, dict) and key in inf and isinstance(inf[key], list):
            preds = inf[key]
            break
    if preds is None and isinstance(inf, list):
        preds = inf
    if preds is None:
        return 0, "", "", ""
    classes = []
    top_c = ""
    top_p = -1.0
    for p0 in preds:
        if not isinstance(p0, dict):
            continue
        c = p0.get("class") or p0.get("label") or ""
        conf = p0.get("confidence") or p0.get("conf") or p0.get("probability") or 0.0
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        if c:
            classes.append(str(c))
        if conf_f > top_p:
            top_p = conf_f
            top_c = str(c)
    classes_csv = ";".join(sorted(set(classes)))
    top_conf_str = "" if top_p < 0 else f"{top_p:.3f}"
    return len(preds), classes_csv, top_c, top_conf_str



def _rf_extract_preds(inf: Any) -> Optional[List[Dict[str, Any]]]:
    """Best-effort extraction of prediction list from Roboflow-like response."""
    if inf is None:
        return None

    # RFInference-like: use .raw if present (dict with 'predictions')
    if hasattr(inf, "raw") and isinstance(getattr(inf, "raw"), dict):
        raw = getattr(inf, "raw")
        preds = raw.get("predictions", None)
        if isinstance(preds, list):
            return [p for p in preds if isinstance(p, dict)]

    if isinstance(inf, list):
        return [p for p in inf if isinstance(p, dict)]

    if isinstance(inf, dict):
        for key in ("predictions", "preds", "detections", "outputs"):
            v = inf.get(key, None)
            if isinstance(v, list):
                return [p for p in v if isinstance(p, dict)]
    return None
    if isinstance(inf, list):
        return [p for p in inf if isinstance(p, dict)]
    if isinstance(inf, dict):
        for key in ("predictions", "preds", "detections", "outputs"):
            v = inf.get(key, None)
            if isinstance(v, list):
                return [p for p in v if isinstance(p, dict)]
    return None

def _rf_debug_line(inf: Any) -> str:
    """Returns short, UI-safe debug string about inference content."""
    try:
        tname = type(inf).__name__

        ok_str = ""
        err_str = ""
        if inf is not None and hasattr(inf, "ok"):
            try:
                ok_str = f" ok={bool(getattr(inf, 'ok'))}"
            except Exception:
                ok_str = ""
        if inf is not None and hasattr(inf, "error"):
            try:
                e = str(getattr(inf, "error") or "")
                if e:
                    err_str = " err=" + e[:60].replace("\n", " ")
            except Exception:
                err_str = ""

        keys = ""
        raw_keys = ""
        if isinstance(inf, dict):
            keys = ",".join(list(inf.keys())[:8])
        elif inf is not None and hasattr(inf, "raw") and isinstance(getattr(inf, "raw"), dict):
            raw = getattr(inf, "raw")
            raw_keys = ",".join(list(raw.keys())[:8])

        preds = _rf_extract_preds(inf)
        n = 0 if preds is None else len(preds)
        top = ""
        if preds:
            p0 = preds[0]
            top = str(p0.get("class") or p0.get("label") or "")
            conf = p0.get("confidence") or p0.get("conf") or p0.get("probability")
            try:
                conf = float(conf)
                top = f"{top}@{conf:.2f}"
            except Exception:
                if conf is not None:
                    top = f"{top}@{conf}"

        return f"RF[{tname}] preds={n} top0={top}{ok_str}{err_str} keys={keys or raw_keys}"
    except Exception as e:
        return f"RF[err] {e}"

def _pkt_debug_line(packet: Any) -> str:
    """Returns short debug string about packet fields that affect cam_ok_count."""
    try:
        fields = []
        for k in ("frame", "frame_a", "frame_b", "cam_a_ok", "cam_b_ok", "timestamp"):
            if hasattr(packet, k):
                v = getattr(packet, k)
                if k.startswith("frame"):
                    fields.append(f"{k}={'Y' if v is not None else 'N'}")
                else:
                    fields.append(f"{k}={v}")
        return "PKT " + " ".join(fields[:10])
    except Exception as e:
        return f"PKT[err] {e}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rf-model-id", default="object-detection-in-sand/4")
    p.add_argument("--rf-api-key", default="")
    p.add_argument("--vision-backend", choices=["ultralytics", "roboflow"], default="ultralytics",
                   help="Inference backend: 'ultralytics' runs local ONNX/engine/PT models; 'roboflow' uses hosted inference.")
    p.add_argument("--model-path", default=os.path.expanduser("~/Desktop/models/regolith/best_yolov26.onnx"),
                   help="Path to local model (.onnx/.engine/.pt) when --vision-backend=ultralytics.")
    p.add_argument("--model-imgsz", type=int, default=640, help="Ultralytics inference size (imgsz).")
    p.add_argument("--model-conf", type=float, default=0.25, help="Confidence threshold for detections.")
    p.add_argument("--model-iou", type=float, default=0.45, help="IoU threshold for NMS.")
    p.add_argument("--model-device", default="", help="Ultralytics device string (e.g., 'cpu', '0'). Empty = auto.")
    p.add_argument("--model-max-det", type=int, default=50, help="Max detections per frame.")
    p.add_argument("--class-names-file", default="", help="Optional path to classes file (one name per line). Overrides model names.")
    p.add_argument("--debug-rf", action="store_true", help="Print/overlay Roboflow inference diagnostics.")
    p.add_argument("--debug-cam", action="store_true", help="Print/overlay camera packet diagnostics.")
    p.add_argument("--debug-every", type=int, default=30, help="Print debug lines every N frames when debug flags enabled.")
    p.add_argument("--rf-dump-jsonl", default="", help="If set, append raw inference objects to this JSONL file (best-effort).")
    p.add_argument("--strict-cam-health", action="store_true", default=False,
                   help="If set, keep strict camera_degraded = cam_ok < expected_cam. Default: only degrade when cam_ok==0 for trials.")
    p.add_argument("--lidar", choices=["serial", "ros2", "none"], default="serial")
    p.add_argument("--lidar-port", default="/dev/ttyUSB0")
    p.add_argument("--lidar-port2", default="/dev/ttyUSB1")
    p.add_argument("--arduino-port", default="/dev/ttyACM0")
    p.add_argument("--sensor-mode", default=None,
                   choices=["1cam","1cam+lidar","2cam+lidar","2cam+2lidar","2cam+lidar+arduino","2cam+2lidar+arduino"])
    p.add_argument("--op-mode", default=None, choices=["interactive","ghost"])
    p.add_argument("--actuation", default=None, choices=["sim","live"])
    p.add_argument("--max-fps", type=int, default=20)
    p.add_argument("--trial-nohalt", action="store_true", default=True)
    p.add_argument("--clip-cooldown-s", type=float, default=4.0)
    p.add_argument("--reverse-seconds", type=float, default=1.0)
    p.add_argument("--metrics-out-dir", default=".")
    p.add_argument("--laser-x", type=float, default=6.0)
    p.add_argument("--laser-y", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    # minimal “friendly defaults” if launched from desktop icon:
    sensor_mode = args.sensor_mode or "2cam+lidar+arduino"
    op_mode = args.op_mode or "interactive"
    actuation = args.actuation or "sim"

    # Debug / diagnostics
    dbg_every = max(1, int(getattr(args, "debug_every", 30)))
    dbg_frame_i = 0
    rf_dump_f = None
    rf_dump_path = str(getattr(args, "rf_dump_jsonl", "") or "").strip()
    if rf_dump_path:
        try:
            rf_dump_f = open(rf_dump_path, "a", buffering=1)
            print(f"[DEBUG] Roboflow JSONL dump enabled -> {rf_dump_path}")
        except Exception as e:
            print(f"[WARN] Could not open rf dump file '{rf_dump_path}': {e}")
            rf_dump_f = None

    use_dual_cam = sensor_mode.startswith("2cam")
    use_dual_lidar = ("2lidar" in sensor_mode)
    use_lidar = (("+lidar" in sensor_mode) or use_dual_lidar) and (args.lidar != "none")
    use_arduino = ("arduino" in sensor_mode)
    is_ghost = (op_mode == "ghost")
    trial_nohalt = (not is_ghost) and bool(args.trial_nohalt)

    # providers
    lidar = None
    if use_lidar:
        if args.lidar == "serial":
            lidar = SerialTwoRPLidarProvider(port0=args.lidar_port, port1=args.lidar_port2) if use_dual_lidar \
                else SerialRPLidarProvider(port=args.lidar_port)
        elif args.lidar == "ros2":
            lidar = ROS2LaserScanProvider()

    cam_provider = TwoCameraProvider() if use_dual_cam else (CSICameraProvider(sensor_id=0) if os.path.exists("/dev/video0") else USBCameraProvider(0))

    arduino = None
    if use_arduino:
        arduino = ArduinoInterlock(port=args.arduino_port, autostart=True)

    mode_mgr = ModeManagerArduinoModel(
        cam_provider, lidar, arduino,
        require_arduino=(use_arduino and (not is_ghost) and (not trial_nohalt)),
        require_model_safety=((not is_ghost) and (not trial_nohalt)),
    )
    health_mon = SystemHealthMonitorArduinoModel(require_arduino=(use_arduino and (not is_ghost) and (not trial_nohalt)))

    # inference + SSC
    detector = None
    if getattr(args, "vision_backend", "ultralytics") == "roboflow":
        if RoboflowProvider is None:
            raise RuntimeError("Roboflow backend selected but roboflow_provider could not be imported.")
        api_key = args.rf_api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        detector = RoboflowProvider(model_id=args.rf_model_id, api_key=api_key)
    else:
        model_path = os.path.expanduser(getattr(args, "model_path", ""))
        names = None
        cnf = str(getattr(args, "class_names_file", "") or "").strip()
        if cnf:
            try:
                lines = [ln.strip() for ln in open(cnf, "r", encoding="utf-8").read().splitlines()]
                names = [ln for ln in lines if ln and (not ln.startswith("#"))]
            except Exception:
                names = None

        detector = UltralyticsYOLOProvider(
            model_path=model_path,
            imgsz=int(getattr(args, "model_imgsz", 640)),
            conf=float(getattr(args, "model_conf", 0.25)),
            iou=float(getattr(args, "model_iou", 0.45)),
            device=str(getattr(args, "model_device", "") or "").strip(),
            max_det=int(getattr(args, "model_max_det", 50)),
            class_names=names,
        )

    ssc = SafetyStateController()

    # nav + arm
    nav = Navigator()
    nav.set_laser_waypoint(Waypoint(args.laser_x, args.laser_y, meta={"label":"LASER"}))
    controller = ShieldedController(HeuristicPolicy(), SafetySupervisor.default())

    joint_names = ["joint2_to_joint1","joint3_to_joint2","joint4_to_joint3","joint5_to_joint4","joint6_to_joint5","joint6output_to_joint6"]
    gripper_joint_name = os.environ.get("ALAM_GRIPPER_JOINT", "gripper_controller")
    arm: ArmInterface
    if actuation == "sim" and HAS_ROS2:
        arm = SimArmPublisher(joint_names, gripper_joint_name)
    else:
        arm = LiveArmStub()

    interp = MotionInterpolator()
    clipper = ClipPlayer()
    last_clip_ts = 0.0
    reverse_until_ts = 0.0
    last_twist = Twist2D(0.0,0.0)

    # metrics
    run_tag = f"alpha_trial_{sensor_mode}_{op_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    metric = LiveSuccessMetric(run_tag=run_tag, out_dir=args.metrics_out_dir)

    expected_cam = 2 if use_dual_cam else 1
    expected_lidar_on = bool(use_lidar)

    win = f"ALPHA_TRIAL (LIVE) [{getattr(args, 'vision_backend', 'ultralytics')}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            now = time.time()
            packet = mode_mgr.get_frame()
            frame, cam_ok, cam_ts = _extract_frame(packet)

            dist_m = _extract_distance_m(mode_mgr.get_distance())
            inf = detector.infer(frame)

            # --- Roboflow diagnostics (best-effort, non-fatal) ---
            dbg_frame_i += 1
            if rf_dump_f is not None:
                try:
                    rf_dump_f.write(json.dumps({"t": now, "inf": inf}, default=str) + "\n")
                except Exception:
                    pass
            rf_dbg = _rf_debug_line(inf) if getattr(args, "debug_rf", False) else ""
            pkt_dbg = _pkt_debug_line(packet) if getattr(args, "debug_cam", False) else ""
            if getattr(args, "debug_rf", False) and hasattr(inf, 'ok') and (not bool(getattr(inf,'ok'))):
                try:
                    print(f"[WARN] Roboflow infer not ok: {getattr(inf,'error','')}")
                except Exception:
                    pass
            if (getattr(args, "debug_rf", False) or getattr(args, "debug_cam", False)) and (dbg_frame_i % dbg_every == 0):
                if rf_dbg:
                    print(rf_dbg)
                if pkt_dbg:
                    print(pkt_dbg)
            n_preds, classes_csv, top_cls, top_conf = _summarize_rf(inf)

            h,w = frame.shape[:2]
            ssc_out = ssc.update(inf=inf, frame_w=w, frame_h=h, min_distance_m=dist_m, timestamp=now)
            hard_stop = bool(ssc_out.hard_stop)
            soft_bypass = bool(ssc_out.soft_bypass)
            camera_degraded = (cam_ok < expected_cam) if getattr(args, "strict_cam_health", False) else (cam_ok == 0)

            top_l = (top_cls or "").lower()
            if hard_stop or (ssc_out.state == SafetyState.HIGH_HAZARD) or ("object" in top_l):
                label = "object_hazard"
            elif soft_bypass or (ssc_out.state == SafetyState.LOW_HAZARD) or ("human" in top_l) or ("bypass" in top_l):
                label = "human_hazard"
            elif camera_degraded:
                label = "camera_compromised"
            elif ("harvest" in top_l) or (ssc_out.task_intent in (SafetyState.APPROACH, SafetyState.HARVEST)):
                label = "harvestable"
            else:
                label = "search"

            # stop HALT flapping during trials
            mode_mgr.set_model_hazard(hard_stop and (not is_ghost) and (not trial_nohalt), ";".join(ssc_out.reasons) if hard_stop else "")

            plan = None
            if not is_ghost:
                plan = health_mon.get_plan(
                    lidar_last_ts=getattr(mode_mgr, "_lidar_last_ok_ts", now),
                    cam_last_ts=cam_ts,
                    arduino_last_ts=mode_mgr.arduino_last_ok_ts(),
                    arduino_safe=mode_mgr.arduino_is_safe(),
                    model_hazard_critical=(hard_stop and (not trial_nohalt)),
                    model_hazard_reason=";".join(ssc_out.reasons) if hard_stop else "",
                )
                if dataclasses.is_dataclass(plan):
                    plan = dataclasses.replace(plan, camera_count=expected_cam, lidar_on=expected_lidar_on)
                mode_mgr.apply_plan(plan)

            # reaction mapping
            wanted_clip = None
            if label == "object_hazard":
                nav_mode = NavMode.GO_HOME
                wanted_clip = "CURL_UP"
            elif label == "human_hazard":
                nav_mode = NavMode.SEARCH_ROUTE
                wanted_clip = "DUMP_RETREAT"
                reverse_until_ts = max(reverse_until_ts, now + float(args.reverse_seconds))
            elif label == "camera_compromised":
                nav_mode = NavMode.GO_LASER
            elif label == "harvestable":
                nav_mode = NavMode.SEARCH_ROUTE
                wanted_clip = "SCOOP_DUMP"
            else:
                nav_mode = NavMode.SEARCH_ROUTE

            # camera compromised: dump at laser when close (coarse)
            pose = nav.pose_tracker.pose
            dist_laser = math.hypot(pose.x_m - args.laser_x, pose.y_m - args.laser_y)
            if label == "camera_compromised" and dist_laser < 0.8:
                wanted_clip = "DUMP_AT_LASER"

            # debounce clip triggers
            if wanted_clip and (now - last_clip_ts) >= float(args.clip_cooldown_s):
                if clipper.done or (clipper.active != wanted_clip):
                    clipper.start(wanted_clip)
                    last_clip_ts = now

            tj, tg, clip_step, _ = clipper.tick()
            if clipper.done and wanted_clip is None:
                tj, tg, clip_step = [0,10,-10,0,0,0], 0.0, "IDLE"

            joints, grip = interp.update(tj, tg)
            arm.publish(joints, grip)

            nav_prop = nav.step(nav_mode=nav_mode, timestamp_s=now, vision={"harvest_intent": label=="harvestable"})
            last_twist = nav_prop.twist
            if now < reverse_until_ts:
                last_twist = Twist2D(v_mps=min(last_twist.v_mps, -0.10), w_rps=last_twist.w_rps)

            # log success metric
            metric.update(
                t=now,
                rf_top_class=top_cls,
                ssc_state=ssc_out.state.value,
                ssc_intent=ssc_out.task_intent.value,
                hard_stop=hard_stop,
                soft_bypass=soft_bypass,
                nav_mode=nav_mode.value,
                final_action=f"CLIP:{clipper.active}:{clip_step}",
                cam_ok_count=int(cam_ok),
                cam_expected=int(expected_cam if getattr(args, 'strict_cam_health', False) else 1),
                dist_m=dist_m,
            )

            # UI overlay
            disp = cv2.resize(frame, (960,540))
            cv2.rectangle(disp, (0,0), (960,170), (0,0,0), -1)
            cv2.putText(disp, f"DECISION={label}  NAV={nav_mode.value}  CLIP={clipper.active}:{clip_step}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(disp, f"RF top={top_cls} conf={top_conf} preds={n_preds} classes={classes_csv}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
            if getattr(args, "debug_rf", False):
                cv2.putText(disp, rf_dbg[:120], (10, 122),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
            if getattr(args, "debug_cam", False):
                cv2.putText(disp, pkt_dbg[:120], (10, 142),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
            cv2.putText(disp, f"SSC state={ssc_out.state.value} intent={ssc_out.task_intent.value} hard={int(hard_stop)} soft={int(soft_bypass)}", (10,75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
            cv2.putText(disp, f"cams={cam_ok}/{expected_cam} lidar={(0.0 if dist_m is None else dist_m):.2f}m laser={dist_laser:.2f}m", (10,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

            # loop pacing
            fps = max(5, min(int(args.max_fps), 60))
            time.sleep(1.0 / fps)

    finally:
        try:
            if 'rf_dump_f' in locals() and rf_dump_f is not None:
                rf_dump_f.close()
        except Exception:
            pass
        try: metric.close()
        except Exception: pass
        try: arm.close()
        except Exception: pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
