#!/usr/bin/env python3
"""
rover_learner/conference_demo/zeta_conf.py

Zeta conference demo = Alpha-style menu + Alpha-sized UI + Roboflow(v4) + SafetyStateController + Navigation preview.

Fixes vs prior glitches:
- Ghost mode does NOT apply dynamic health plans each frame (prevents HALT<->DUAL_CAM_LIDAR flapping).
  * In ghost: apply a fixed NOMINAL plan once (and re-assert if we ever drop to HALT).
  * Cameras + LiDAR stay on; hazards are logged for observability.
- Dual-camera FramePacket compatibility:
  * If packet has frame_a/frame_b, we use frame_a as primary (and optionally render both).
  * Avoids "black frame" + false camera failure feedback.
- Full scenario animations in ghost (SIM):
  * Uses ScenarioAnimator + MotionInterpolator (same as alpha_conf) and publishes JointState to RViz when ROS2 is available.

Hotkeys (while window focused):
  q   quit
  1   nav: SEARCH_ROUTE
  2   nav: GO_HOME
  3   nav: GO_LASER
  4   nav: DOCK_LASER
"""

from __future__ import annotations

import os
import sys
import time
import math
import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# sys.path setup (direct script execution)
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROVER_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))          # .../rover_learner
WORKSPACE_ROOT = os.path.abspath(os.path.join(ROVER_ROOT, ".."))    # parent of rover_learner
for p in (ROVER_ROOT, THIS_DIR, WORKSPACE_ROOT):
    if p not in sys.path:
        sys.path.append(p)

# ---------------------------------------------------------------------------
# Imports (local project)
# ---------------------------------------------------------------------------
from power_saver_degradation import SystemMode
from failsafe_ui import FailsafeOverlayManager

from mode_manager_arduino_model import ModeManagerArduinoModel
from power_saver_degradation_arduino_model import SystemHealthMonitorArduinoModel

from rl_safety_supervisor import SafetySupervisor, ShieldedController, HeuristicPolicy, Observation
from arduino_interlock import ArduinoInterlock, ArduinoStatus

from camera_provider import CSICameraProvider, USBCameraProvider
from two_camera_provider import TwoCameraProvider
from lidar_provider import SerialRPLidarProvider, ROS2LaserScanProvider
from two_lidar_provider import SerialTwoRPLidarProvider

from roboflow_provider import RoboflowProvider
from safety_state_controller import SafetyStateController, SafetyState

# Navigation (flat imports)
from navigation.navigation import Navigator
from navigation.types import NavMode, Waypoint, Twist2D

# ---------------------------------------------------------------------------
# ROS2 for RViz joint_states (optional) — copied from alpha_conf
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    HAS_ROS2 = True
except Exception:
    HAS_ROS2 = False

# ---------------------------------------------------------------------------
# Scenario keyframes + animation helpers (copied from alpha_conf)
# ---------------------------------------------------------------------------
SCENARIOS: Dict[str, List[Dict[str, Any]]] = {
    "UNBREAKABLE": [
        {"name": "TUCK", "joints": [0, 0, 0, 0, 0, 0], "gripper": 0, "duration": 2.0},
        {"name": "DUMP", "joints": [0, -45, 90, 0, 0, 0], "gripper": 100, "duration": 3.0},
        {"name": "HOME", "joints": [0, 0, 0, 0, 0, 0], "gripper": 0, "duration": 2.0},
    ],
    "STICKY": [
        {"name": "SHAKE_L", "joints": [30, 0, 0, 0, 0, 0], "gripper": 50, "duration": 0.5},
        {"name": "SHAKE_R", "joints": [-30, 0, 0, 0, 0, 0], "gripper": 50, "duration": 0.5},
        {"name": "SHAKE_L", "joints": [30, 0, 0, 0, 0, 0], "gripper": 50, "duration": 0.5},
        {"name": "HOME", "joints": [0, 0, 0, 0, 0, 0], "gripper": 0, "duration": 2.0},
    ],
    "ADAPTIVE": [
        {"name": "SEARCH", "joints": [0, 10, -10, 0, 0, 0], "gripper": 0, "duration": 4.0},
        {"name": "APPROACH", "joints": [0, 20, -20, 0, 0, 0], "gripper": 0, "duration": 3.0},
        {"name": "SCOOP", "joints": [0, 45, -45, 0, 0, 0], "gripper": 100, "duration": 2.0},
        {"name": "RETREAT", "joints": [0, 0, 0, 0, 0, 0], "gripper": 100, "duration": 2.0},
        {"name": "DEPOSIT", "joints": [0, -45, 90, 0, 0, 0], "gripper": 0, "duration": 2.0},
        {"name": "HOME", "joints": [0, 0, 0, 0, 0, 0], "gripper": 0, "duration": 2.0},
    ],
}

FAILSAFE_OVERHEAT = "FAILSAFE_OVERHEAT"
FAILSAFE_COMPROMISED_CAMERA = "FAILSAFE_COMPROMISED_CAMERA"

class MotionInterpolator:
    def __init__(self, joint_speed_deg: float = 60.0, gripper_speed: float = 150.0):
        self.current_joints = [0.0] * 6
        self.current_gripper = 0.0
        self.joint_speed = float(joint_speed_deg)
        self.gripper_speed = float(gripper_speed)
        self.last_update_time = time.time()

    def update(self, target_joints: List[float], target_gripper: float) -> Tuple[List[float], float]:
        now = time.time()
        dt = max(1e-4, now - self.last_update_time)
        self.last_update_time = now

        out = []
        for curr, targ in zip(self.current_joints, target_joints):
            diff = targ - curr
            max_move = self.joint_speed * dt
            if abs(diff) < 0.1:
                out.append(float(targ))
            else:
                step = math.copysign(min(abs(diff), max_move), diff)
                out.append(curr + step)
        self.current_joints = out

        diff_g = float(target_gripper) - float(self.current_gripper)
        max_move_g = self.gripper_speed * dt
        if abs(diff_g) < 1.0:
            self.current_gripper = float(target_gripper)
        else:
            step_g = math.copysign(min(abs(diff_g), max_move_g), diff_g)
            self.current_gripper += step_g

        return self.current_joints, self.current_gripper

class ScenarioAnimator:
    def __init__(self, scenario_name: str):
        self.steps = SCENARIOS.get(scenario_name, [])
        self.start_time = 0.0
        self.active = False
        self.total_duration = sum(float(s["duration"]) for s in self.steps) if self.steps else 0.0

    def start(self):
        self.start_time = time.time()
        self.active = True
        print(f"[Animator] Started scenario={len(self.steps)} steps cycle={self.total_duration:.1f}s")

    def get_target(self) -> Tuple[List[float], float, str]:
        if not self.active or not self.steps or self.total_duration <= 0.0:
            return [0.0] * 6, 0.0, "STOP"
        elapsed = (time.time() - self.start_time) % self.total_duration
        t = 0.0
        for step in self.steps:
            t += float(step["duration"])
            if elapsed < t:
                return list(map(float, step["joints"])), float(step["gripper"]), str(step["name"])
        step = self.steps[0]
        return list(map(float, step["joints"])), float(step["gripper"]), str(step["name"])

@dataclass
class FailsafeState:
    triggered: bool = False
    trigger_time_s: float = 10.0
    phase: str = "NORMAL"
    phase_start_ts: float = 0.0
    done: bool = False

def _drop_pose() -> Tuple[List[float], float, str]:
    return [0.0, -45.0, 90.0, 0.0, 0.0, 0.0], 0.0, "DROP"

def failsafe_overheat_tick(t0: float, fs: FailsafeState) -> Tuple[List[float], float, str, Dict[str, Any], str]:
    now = time.time()
    elapsed = now - t0
    overrides: Dict[str, Any] = {}
    thought = ""

    if (not fs.triggered) and elapsed >= fs.trigger_time_s:
        fs.triggered = True
        fs.phase = "DROP"
        fs.phase_start_ts = now

    if not fs.triggered:
        thought = "Monitoring thermals... nominal."
        return [0.0, 10.0, -10.0, 0.0, 0.0, 0.0], 0.0, "SEARCH", overrides, thought

    overrides.update({
        "mode": SystemMode.DEGRADED,
        "camera_count": 1,
        "lidar_on": True,
        "fps_target": 10,
        "action_hint": "RETURN_HOME",
    })

    if fs.phase == "DROP":
        thought = "FAILSAFE(OVERHEAT): drop payload now."
        joints, grip, name = _drop_pose()
        if (now - fs.phase_start_ts) > 2.0:
            fs.phase = "RETURN_HOME"
            fs.phase_start_ts = now
        return joints, grip, name, overrides, thought

    thought = "FAILSAFE(OVERHEAT): returning home."
    joints = [0.0] * 6
    grip = 0.0
    return joints, grip, "HOME", overrides, thought

def failsafe_compromised_tick(t0: float, fs: FailsafeState) -> Tuple[List[float], float, str, Dict[str, Any], str]:
    now = time.time()
    elapsed = now - t0
    overrides: Dict[str, Any] = {}
    thought = ""

    if (not fs.triggered) and elapsed >= fs.trigger_time_s:
        fs.triggered = True
        fs.phase = "TUCK"
        fs.phase_start_ts = now

    if not fs.triggered:
        thought = "Monitoring vision pipeline... nominal."
        return [0.0, 10.0, -10.0, 0.0, 0.0, 0.0], 0.0, "SEARCH", overrides, thought

    overrides.update({
        "mode": SystemMode.DEGRADED,
        "camera_count": 1,
        "lidar_on": True,
        "fps_target": 10,
        "action_hint": "SAFE_HOLD",
    })

    thought = "FAILSAFE(CAM): tucking and reducing dependency."
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0, "TUCK", overrides, thought


# ---------------------------------------------------------------------------
# Arm Interfaces (RViz publishing), copied from alpha_conf
# ---------------------------------------------------------------------------
class ArmInterface:
    def publish(self, joints_deg: List[float], gripper_0_100: float) -> None:
        raise NotImplementedError
    def close(self) -> None:
        pass

class SimArmPublisher(ArmInterface):
    def __init__(self, joint_names: List[str], gripper_joint_name: str):
        self.joint_names = joint_names
        self.gripper_joint_name = gripper_joint_name
        self.node: Optional[Node] = None
        self.pub = None

        if not HAS_ROS2:
            raise RuntimeError("ROS2 (rclpy) not available.")
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("alam_zeta_conf_demo")
        self.pub = self.node.create_publisher(JointState, "joint_states", 10)

    def publish(self, joints_deg: List[float], gripper_0_100: float) -> None:
        assert self.node is not None and self.pub is not None
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
            if self.node is not None:
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


# ---------------------------------------------------------------------------
# CSV Logger (zeta)
# ---------------------------------------------------------------------------
class ZetaLogger:
    def __init__(self, filename: str):
        self.filename = filename
        self.f = open(self.filename, "w", buffering=1)
        self.max_dist_home_m = 0.0
        self.min_hazard_dist_m: Optional[float] = None

        cols = [
            "timestamp","sensor_mode","scenario","op_mode","actuation",
            "sys_mode","health_score","plan_reasons","plan_camera_count","plan_lidar_on",
            "step_name",
            "ssc_state","ssc_intent","ssc_hard_stop","ssc_soft_bypass","ssc_reasons",
            "min_distance_m",
            "nav_mode","nav_behavior","nav_status","nav_done","avoid_active","v_cmd_mps","w_cmd_rps",
            "pose_x_m","pose_y_m","pose_yaw_rad","dist_home_m","max_dist_home_m","dist_laser_m",
            "hazard_dist_m","min_hazard_dist_m","hazard_class_hint",
            "ml_view","final_action",
        ]
        self.f.write(",".join(cols) + "\n")

    def _csv(self, x: Any) -> str:
        if x is None:
            return ""
        s = str(x)
        if any(c in s for c in [",", "\n", "\r", '"']):
            s = '"' + s.replace('"', '""') + '"'
        return s

    def update_max_home(self, d: float) -> float:
        self.max_dist_home_m = max(self.max_dist_home_m, float(d))
        return self.max_dist_home_m

    def update_min_hazard(self, d: Optional[float]) -> Optional[float]:
        if d is None:
            return self.min_hazard_dist_m
        if self.min_hazard_dist_m is None:
            self.min_hazard_dist_m = float(d)
        else:
            self.min_hazard_dist_m = min(self.min_hazard_dist_m, float(d))
        return self.min_hazard_dist_m

    def log_row(self, cols: List[Any]) -> None:
        self.f.write(",".join(self._csv(x) for x in cols) + "\n")

    def close(self) -> None:
        try: self.f.close()
        except Exception: pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _supports_path(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

def _extract_distance_m(x: Any) -> Optional[float]:
    if x is None:
        return None
    # already numeric?
    try:
        v = float(x)  # type: ignore[arg-type]
        if math.isfinite(v) and v > 0.0:
            return v
    except Exception:
        pass
    for attr in ("min_distance_m","distance_m","distance","range_m","value_m","value"):
        if hasattr(x, attr):
            try:
                v = float(getattr(x, attr))
                if math.isfinite(v) and v > 0.0:
                    return v
            except Exception:
                continue
    return None

def _action_to_str(x: Any) -> str:
    if x is None:
        return ""
    if hasattr(x, "name"):
        try: return str(getattr(x, "name"))
        except Exception: pass
    return str(x)

def _extract_primary_frame(packet: Any) -> Tuple[np.ndarray, int, float]:
    """
    Returns (frame, camera_count_ok, ts)
    Works for:
      - single-cam packet: packet.frame
      - dual-cam packet: packet.frame_a / packet.frame_b + cam_a_ok/cam_b_ok
    """
    ts = float(getattr(packet, "timestamp", time.time()))

    if hasattr(packet, "frame") and getattr(packet, "frame") is not None:
        fr = getattr(packet, "frame")
        return fr, 1, ts

    fa = getattr(packet, "frame_a", None)
    fb = getattr(packet, "frame_b", None)
    ok_a = getattr(packet, "cam_a_ok", fa is not None)
    ok_b = getattr(packet, "cam_b_ok", fb is not None)
    ok = int(bool(ok_a)) + int(bool(ok_b))

    if fa is not None:
        return fa, ok, ts
    if fb is not None:
        return fb, ok, ts

    # fallback black frame
    return np.zeros((360, 480, 3), dtype=np.uint8), 0, ts

def _default_route() -> List[Waypoint]:
    return [
        Waypoint(0.0, 0.0, meta={"label": "HOME"}),
        Waypoint(2.0, 0.0),
        Waypoint(4.0, 0.0),
        Waypoint(4.0, 2.0),
        Waypoint(2.0, 2.0),
        Waypoint(0.0, 2.0),
        Waypoint(0.0, 0.0, meta={"label": "HOME_BACK"}),
    ]


# ---------------------------------------------------------------------------
# CLI / Menu (alpha-style)
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ALAM Zeta Conf Demo")

    p.add_argument("--ui-width", type=int, default=480)
    p.add_argument("--ui-height", type=int, default=360)

    p.add_argument("--lidar", choices=["serial", "ros2", "none"], default="serial")
    p.add_argument("--lidar-port", default="/dev/ttyUSB0")
    p.add_argument("--lidar-port2", default="/dev/ttyUSB1")
    p.add_argument("--arduino-port", default="/dev/ttyACM0")

    p.add_argument("--sensor-mode",
                   choices=["1cam","1cam+lidar","2cam+lidar","2cam+2lidar","2cam+lidar+arduino","2cam+2lidar+arduino"],
                   default=None)
    p.add_argument("--scenario",
                   choices=["UNBREAKABLE","STICKY","ADAPTIVE",FAILSAFE_OVERHEAT,FAILSAFE_COMPROMISED_CAMERA],
                   default=None)
    p.add_argument("--op-mode", choices=["interactive","ghost"], default=None)
    p.add_argument("--actuation", choices=["sim","live"], default=None)

    p.add_argument("--max-fps", type=int, default=30)
    p.add_argument("--rf-model-id", type=str, default="object-detection-in-sand/4")
    p.add_argument("--rf-api-key", type=str, default="")

    p.add_argument("--laser-x", type=float, default=6.0)
    p.add_argument("--laser-y", type=float, default=0.0)
    p.add_argument("--nav-odom", type=str, default="sim", choices=["sim","none"])

    return p.parse_args()

def menu_select(args) -> Tuple[str, str, str, str]:
    # 1) sensor
    if args.sensor_mode:
        sensor_mode = args.sensor_mode
    else:
        while True:
            print("\n=== ZETA CONF DEMO CONFIG ===")
            print("Sensor Configuration:")
            print(" 1) 1 cam")
            print(" 2) 1 cam + 1 lidar")
            print(" 3) 2 cam + 1 lidar")
            print(" 4) 2 cam + 2 lidar")
            print(" 5) 2 cam + 1 lidar + Arduino interlock")
            print(" 6) 2 cam + 2 lidar + Arduino interlock")
            v = input("Selection (1-6) [5]: ").strip() or "5"
            sensor_mode = {
                "1":"1cam","2":"1cam+lidar","3":"2cam+lidar","4":"2cam+2lidar","5":"2cam+lidar+arduino","6":"2cam+2lidar+arduino"
            }.get(v, "2cam+lidar+arduino")

            need_lidar1 = (("+lidar" in sensor_mode) or ("2lidar" in sensor_mode)) and (args.lidar != "none")
            need_lidar2 = ("2lidar" in sensor_mode) and (args.lidar != "none")
            need_arduino = ("arduino" in sensor_mode)

            if need_lidar1 and (not _supports_path(args.lidar_port)):
                print(f"[Config] Missing LiDAR #1 at {args.lidar_port}")
                continue
            if need_lidar2 and (not _supports_path(args.lidar_port2)):
                print(f"[Config] Missing LiDAR #2 at {args.lidar_port2}")
                continue
            if need_arduino and (not _supports_path(args.arduino_port)):
                print(f"[Config] Missing Arduino at {args.arduino_port}")
                continue
            break

    # 2) scenario
    if args.scenario:
        scenario = args.scenario
    else:
        print("\nScenario:")
        print(" 1) ADAPTIVE")
        print(" 2) STICKY")
        print(" 3) UNBREAKABLE")
        print(" 4) FAILSAFE trigger - overheated")
        print(" 5) FAILSAFE trigger - compromised camera")
        v = input("Selection (1-5) [1]: ").strip() or "1"
        scenario = {"1":"ADAPTIVE","2":"STICKY","3":"UNBREAKABLE","4":FAILSAFE_OVERHEAT,"5":FAILSAFE_COMPROMISED_CAMERA}.get(v, "ADAPTIVE")

    # 3) op-mode
    if args.op_mode:
        op_mode = args.op_mode
    else:
        print("\nOperation Mode:")
        print(" 1) interactive (Safety active)")
        print(" 2) ghost (Safety muted; do not halt cams/lidar)")
        v = input("Selection (1-2) [2]: ").strip() or "2"
        op_mode = "interactive" if v == "1" else "ghost"

    # 4) actuation
    if args.actuation:
        actuation = args.actuation
    else:
        print("\nActuation Mode:")
        print(" 1) SIM (RViz joint_states)")
        print(" 2) LIVE (stub)")
        v = input("Selection (1-2) [1]: ").strip() or "1"
        actuation = "sim" if v == "1" else "live"

    return sensor_mode, scenario, op_mode, actuation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    sensor_mode, scenario, op_mode, actuation = menu_select(args)

    use_dual_cam = sensor_mode.startswith("2cam")
    use_dual_lidar = ("2lidar" in sensor_mode)
    use_lidar = (("+lidar" in sensor_mode) or use_dual_lidar) and (args.lidar != "none")
    use_arduino = ("arduino" in sensor_mode)
    is_ghost = (op_mode == "ghost")

    print(f"\n[Init] sensor_mode={sensor_mode} scenario={scenario} op_mode={op_mode} actuation={actuation}")

    # Providers
    lidar = None
    if use_lidar:
        try:
            if args.lidar == "serial":
                lidar = SerialTwoRPLidarProvider(port0=args.lidar_port, port1=args.lidar_port2) if use_dual_lidar else SerialRPLidarProvider(port=args.lidar_port)
            elif args.lidar == "ros2":
                lidar = ROS2LaserScanProvider()
        except Exception as e:
            print(f"[WARN] LiDAR init failed: {e}")
            lidar = None

    cam_provider: Any
    if use_dual_cam:
        cam_provider = TwoCameraProvider()
    else:
        try:
            cam_provider = CSICameraProvider(sensor_id=0)
        except Exception:
            cam_provider = USBCameraProvider(0)

    arduino = None
    if use_arduino:
        try:
            arduino = ArduinoInterlock(port=args.arduino_port, autostart=True)
            try:
                arduino.set_armed(True)
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] Arduino init failed: {e}")
            arduino = None

    # Mode manager + health monitor
    mode_mgr = ModeManagerArduinoModel(
        cam_provider, lidar, arduino,
        require_arduino=(use_arduino and (not is_ghost)),
        require_model_safety=(not is_ghost),
    )
    health_mon = SystemHealthMonitorArduinoModel(require_arduino=(use_arduino and (not is_ghost)))

    # Shield controller (used for logging; safety muted in ghost)
    controller = ShieldedController(HeuristicPolicy(), SafetySupervisor.default())

    # Roboflow + SSC
    api_key = args.rf_api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY", "rf_0fJm5jm2AXYfPISiCOuiZeRCj2p2")
    print(f"[INIT] Connecting Roboflow ({args.rf_model_id})...")
    rf = RoboflowProvider(model_id=args.rf_model_id, api_key=api_key)
    ssc = SafetyStateController()

    # Navigation
    nav = Navigator()
    nav.set_route(_default_route())
    laser_wp = Waypoint(float(args.laser_x), float(args.laser_y), meta={"label": "LASER"})
    nav.set_laser_waypoint(laser_wp)
    nav_mode: NavMode = NavMode.SEARCH_ROUTE
    last_nav_twist = Twist2D(0.0, 0.0)

    # Arm interface (RViz)
    joint_names = [
        "joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
        "joint5_to_joint4", "joint6_to_joint5", "joint6output_to_joint6",
    ]
    gripper_joint_name = os.environ.get("ALAM_GRIPPER_JOINT", "gripper_controller")

    if actuation == "sim" and HAS_ROS2:
        arm: ArmInterface = SimArmPublisher(joint_names, gripper_joint_name)
        print("[Actuation] SIM arm publisher active (ROS2).")
    else:
        arm = LiveArmStub()
        if actuation == "sim" and not HAS_ROS2:
            print("[Actuation] ROS2 not available; SIM publish disabled.")

    interp = MotionInterpolator(joint_speed_deg=60.0, gripper_speed=150.0)
    animator = ScenarioAnimator(scenario if scenario in SCENARIOS else "ADAPTIVE")
    animator.start()

    fs_over = FailsafeState(trigger_time_s=10.0)
    fs_cam = FailsafeState(trigger_time_s=10.0)

    # UI
    window = "ALAM Zeta Conference Demo"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, int(args.ui_width), int(args.ui_height))

    # Overlays
    overlay = FailsafeOverlayManager()

    # Logger
    ts = time.strftime("%Y%m%d_%H%M%S")
    log = ZetaLogger(f"zeta_conf_{scenario}_{sensor_mode}_{op_mode}_{ts}.csv")

    # Ghost fixed plan (prevents flapping)
    expected_cam_count = 2 if use_dual_cam else 1
    expected_lidar_on = bool(use_lidar)

    fixed_plan = None
    if is_ghost:
        # Build a nominal plan once and apply once; do not re-plan every frame.
        fixed_plan = health_mon.get_plan(
            lidar_last_ts=time.time(),
            cam_last_ts=time.time(),
            arduino_last_ts=mode_mgr.arduino_last_ok_ts(),
            arduino_safe=True,
            model_hazard_critical=False,
            model_hazard_reason="",
        )
        if dataclasses.is_dataclass(fixed_plan):
            fixed_plan = dataclasses.replace(
                fixed_plan,
                mode=SystemMode.NOMINAL,
                camera_count=expected_cam_count,
                lidar_on=expected_lidar_on,
                fps_target=int(args.max_fps),
                reasons=["GHOST_FIXED_PLAN"],
            )
        mode_mgr.set_model_hazard(False, "")
        mode_mgr.apply_plan(fixed_plan)

    last_reassert_ts = 0.0
    last_ts = time.time()
    t0 = time.time()

    print("\n[READY] Main Loop. Hotkeys: 1 SEARCH, 2 HOME, 3 LASER, 4 DOCK, q quit.\n")

    try:
        while True:
            now = time.time()
            dt = max(1e-3, now - last_ts)
            last_ts = now
            elapsed = now - t0

            # Sim odom from nav twist (presentation)
            if args.nav_odom == "sim":
                wheel_base = getattr(getattr(nav.pose_tracker, "cfg", None), "drive", None)
                wheel_base_m = getattr(wheel_base, "wheel_base_m", 0.40)
                v = float(last_nav_twist.v_mps)
                w = float(last_nav_twist.w_rps)
                dl = (v - w * wheel_base_m * 0.5) * dt
                dr = (v + w * wheel_base_m * 0.5) * dt
                nav.update_wheel_deltas(dl_m=dl, dr_m=dr, timestamp_s=now)

            # Reassert ghost plan if internal mode flips (rare, but protects demo)
            if is_ghost and fixed_plan is not None and (now - last_reassert_ts) > 1.0:
                last_reassert_ts = now
                try:
                    mode_mgr.apply_plan(fixed_plan)
                except Exception:
                    pass

            # Read sensors (via mode manager)
            packet = mode_mgr.get_frame()
            frame, cam_ok_count, cam_ts = _extract_primary_frame(packet)

            dist_raw = mode_mgr.get_distance()
            dist_m = _extract_distance_m(dist_raw)
            nav.update_lidar(min_distance_m=dist_m)

            # Roboflow inference + safety state controller
            inf = rf.infer(frame)
            h0, w0 = frame.shape[:2]
            state_out = ssc.update(inf=inf, frame_w=w0, frame_h=h0, min_distance_m=dist_m, timestamp=now)

            hard_stop = bool(state_out.hard_stop)
            soft_bypass = bool(state_out.soft_bypass)

            # In ghost: never let model hazard force HALT; just log.
            mode_mgr.set_model_hazard(hard_stop and (not is_ghost), ";".join(state_out.reasons) if hard_stop else "")

            # Health plan (interactive only)
            if not is_ghost:
                plan = health_mon.get_plan(
                    lidar_last_ts=getattr(mode_mgr, "_lidar_last_ok_ts", now),
                    cam_last_ts=cam_ts,
                    arduino_last_ts=mode_mgr.arduino_last_ok_ts(),
                    arduino_safe=mode_mgr.arduino_is_safe(),
                    model_hazard_critical=hard_stop,
                    model_hazard_reason=";".join(state_out.reasons) if hard_stop else "",
                )
                # Force the plan to match selected sensor mode (prevents accidental camera disable)
                if dataclasses.is_dataclass(plan):
                    plan = dataclasses.replace(plan, camera_count=expected_cam_count, lidar_on=expected_lidar_on)
                mode_mgr.apply_plan(plan)
            else:
                plan = fixed_plan  # type: ignore[assignment]

            # Scenario targets (full animation in ghost)
            overrides: Dict[str, Any] = {}
            thought = ""
            if scenario == FAILSAFE_OVERHEAT:
                tj, tg, step_name, overrides, thought = failsafe_overheat_tick(t0, fs_over)
            elif scenario == FAILSAFE_COMPROMISED_CAMERA:
                tj, tg, step_name, overrides, thought = failsafe_compromised_tick(t0, fs_cam)
            else:
                tj, tg, step_name = animator.get_target()

            joints_deg, grip = interp.update(tj, tg)
            arm.publish(joints_deg, grip)

            # Decision shield (for logging; muted in ghost)
            if state_out.state in (SafetyState.HIGH_HAZARD, SafetyState.LOW_HAZARD):
                pred_class = "hazard"
            else:
                pred_class = "clean" if state_out.task_intent in (SafetyState.APPROACH, SafetyState.HARVEST) else "hazard"

            obs = Observation.from_perception(pred_class, 1.0, float(dist_m or 0.0))
            decision = controller.step(obs)
            final_action_str = _action_to_str(getattr(decision, "final_action", decision))
            if is_ghost:
                # Keep scenario driving visible
                final_action_str = f"SCENARIO_{step_name}"

            # Navigation proposal
            vision_hint: Dict[str, Any] = {"harvest_intent": (state_out.state == SafetyState.HARVEST), "safety_state": state_out.state.value}
            nav_prop = nav.step(nav_mode=nav_mode, timestamp_s=now, vision=vision_hint, record_breadcrumbs=(nav_mode != NavMode.GO_HOME))
            last_nav_twist = nav_prop.twist

            # Distances
            pose = nav.pose_tracker.pose
            dist_home_m = float(math.hypot(pose.x_m, pose.y_m))
            max_home = log.update_max_home(dist_home_m)
            dist_laser_m = float(math.hypot(pose.x_m - laser_wp.x_m, pose.y_m - laser_wp.y_m))

            hazard_dist_m = None
            hazard_hint = ""
            if state_out.state in (SafetyState.HIGH_HAZARD, SafetyState.LOW_HAZARD):
                hazard_dist_m = dist_m
                hazard_hint = "object hazard" if state_out.state == SafetyState.HIGH_HAZARD else "human/bypass"
            log.update_min_hazard(hazard_dist_m)

            # Log row
            pmode = getattr(plan, "mode", SystemMode.NOMINAL) if plan is not None else SystemMode.NOMINAL
            preasons = getattr(plan, "reasons", []) if plan is not None else []
            pcam = getattr(plan, "camera_count", expected_cam_count) if plan is not None else expected_cam_count
            plidar = getattr(plan, "lidar_on", expected_lidar_on) if plan is not None else expected_lidar_on
            phealth = getattr(plan, "health_score", "") if plan is not None else ""

            log.log_row([
                now, sensor_mode, scenario, op_mode, actuation,
                getattr(pmode, "name", str(pmode)), phealth, ";".join(preasons or []), pcam, int(bool(plidar)),
                step_name,
                state_out.state.value, state_out.task_intent.value, int(hard_stop), int(soft_bypass), ";".join(state_out.reasons),
                "" if dist_m is None else f"{dist_m:.3f}",
                nav_mode.value, nav_prop.debug.get("behavior", ""), nav_prop.status.value, int(bool(nav_prop.done)),
                int(bool(nav_prop.debug.get("avoid_active", False))),
                f"{nav_prop.twist.v_mps:.3f}", f"{nav_prop.twist.w_rps:.3f}",
                f"{pose.x_m:.3f}", f"{pose.y_m:.3f}", f"{pose.yaw_rad:.3f}",
                f"{dist_home_m:.3f}", f"{max_home:.3f}", f"{dist_laser_m:.3f}",
                "" if hazard_dist_m is None else f"{hazard_dist_m:.3f}",
                "" if log.min_hazard_dist_m is None else f"{log.min_hazard_dist_m:.3f}",
                hazard_hint,
                pred_class.upper(), final_action_str,
            ])

            # Render window (alpha-sized)
            disp = frame.copy()
            disp = cv2.resize(disp, (int(args.ui_width), int(args.ui_height)), interpolation=cv2.INTER_AREA)

            # Header band
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 120), (0, 0, 0), -1)
            mode_color = (0, 255, 0)
            if pmode == SystemMode.DEGRADED:
                mode_color = (0, 165, 255)
            if pmode == SystemMode.CRITICAL:
                mode_color = (0, 0, 255)

            cv2.putText(disp, f"MODE: {getattr(pmode,'name',str(pmode))}  op={op_mode} sensor={sensor_mode}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)
            cv2.putText(disp, f"STEP: {step_name}   NAV: {nav_mode.value} v={nav_prop.twist.v_mps:.2f} w={nav_prop.twist.w_rps:.2f}",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
            cv2.putText(disp, f"SSC: {state_out.state.value} intent={state_out.task_intent.value} hard={int(hard_stop)} soft={int(soft_bypass)}",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            cv2.putText(disp, f"POSE: home={dist_home_m:.1f}m (max={max_home:.1f}) laser={dist_laser_m:.1f}m  lidar={0.0 if dist_m is None else dist_m:.2f}m",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            cv2.putText(disp, f"DECISION: {final_action_str}",
                        (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

            # Failsafe overlay animation (UI)
            overlay.update(now, scenario in (FAILSAFE_OVERHEAT, FAILSAFE_COMPROMISED_CAMERA) and elapsed > 10.0)
            overlay.draw(disp, sys_mode=getattr(pmode,'name',str(pmode)), metrics="", arm_joints=joints_deg, scale=0.32, color=(180,180,180), thickness=1)

            # heartbeat
            pulse = (0, 255, 0) if int(now * 2) % 2 == 0 else (40, 40, 40)
            cv2.circle(disp, (disp.shape[1] - 18, 18), 7, pulse, -1)

            cv2.imshow(window, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                nav_mode = NavMode.SEARCH_ROUTE
            elif key == ord("2"):
                nav_mode = NavMode.GO_HOME
            elif key == ord("3"):
                nav_mode = NavMode.GO_LASER
            elif key == ord("4"):
                nav_mode = NavMode.DOCK_LASER

            # pacing
            target_fps = max(5, min(int(args.max_fps), 60))
            dt_target = 1.0 / float(target_fps)
            loop_dur = time.time() - now
            if loop_dur < dt_target:
                time.sleep(dt_target - loop_dur)

    except KeyboardInterrupt:
        print("\n[Stopped] User interrupt.")
    finally:
        print("\n[Shutdown] Cleaning up...")
        try: log.close()
        except Exception: pass
        try: arm.close()
        except Exception: pass
        try:
            if hasattr(cam_provider, "close"):
                cam_provider.close()
        except Exception: pass
        try:
            if lidar is not None and hasattr(lidar, "close"):
                lidar.close()
        except Exception: pass
        try:
            if arduino is not None and hasattr(arduino, "close"):
                arduino.close()
        except Exception: pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
