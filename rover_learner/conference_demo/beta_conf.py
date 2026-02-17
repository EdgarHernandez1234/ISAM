#!/usr/bin/env python3
"""
beta_conf.py — ALAM Beta Conference Demo (Integrated)

Updates in this version:
  1) UNBREAKABLE-only stabilization:
     - While step == "TUCK", request a short stabilization window
     - Immediately clear stabilization once TUCK ends (revert to normal plan-driven mode)
  2) UI shows LiDAR 'alive' vs 'used' to avoid confusion during gating

NOTE:
  - Stabilization relies on ModeManager implementing:
      request_stabilization(), clear_stabilization(), is_stabilizing()
    and an OperatingMode enum (as provided in the patched mode_manager.py).
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
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Make imports work when executed as: python3 conference_demo/beta_conf.py
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROVER_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROVER_ROOT not in sys.path:
    sys.path.append(ROVER_ROOT)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

# conference_demo modules (local)
from power_saver_degradation import SystemHealthMonitor, SystemMode  # ResourcePlan type is returned by monitor

# ModeManager + OperatingMode (OperatingMode is optional fallback for older mode_manager)
try:
    from mode_manager import ModeManager, OperatingMode
except Exception:
    from mode_manager import ModeManager
    OperatingMode = None  # type: ignore

# rover_learner modules (parent)
from rl_safety_supervisor import (
    SafetySupervisor, ShieldedController, HeuristicPolicy, RoverAction, Observation
)
from logger import CSVDecisionLogger, DecisionFrame
from camera_provider import CSICameraProvider, USBCameraProvider
from two_camera_provider import TwoCameraProvider
from lidar_provider import SerialRPLidarProvider, ROS2LaserScanProvider

# ---------------------------------------------------------------------------
# ROS2 (optional) for RViz
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    HAS_ROS2 = True
except Exception:
    HAS_ROS2 = False


# ============================================================
# 0) Demo tuning knobs
# ============================================================

# UNBREAKABLE-only stabilization: force a lightweight mode while the arm "catches itself"
UNBREAKABLE_STABILIZE_STEP = "TUCK"
UNBREAKABLE_STABILIZE_DURATION_S = 2.4  # short window; we also clear immediately when TUCK ends


# ============================================================
# 1) Scenario keyframes (degrees) + gripper (0..100)
# ============================================================

SCENARIOS = {
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


# ============================================================
# 2) Motion interpolator (smooth RViz movement)
# ============================================================

class MotionInterpolator:
    """Smooth transitions between target poses for RViz."""
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


# ============================================================
# 3) Scenario animator (looping keyframes)
# ============================================================

class ScenarioAnimator:
    def __init__(self, scenario_name: str):
        self.steps = SCENARIOS.get(scenario_name, [])
        self.start_time = 0.0
        self.active = False
        self.total_duration = sum(s["duration"] for s in self.steps) if self.steps else 0.0

    def start(self):
        self.start_time = time.time()
        self.active = True
        print(f"[Animator] Started scenario steps={len(self.steps)} cycle={self.total_duration:.1f}s")

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


# ============================================================
# 4) Failsafe state machines
# ============================================================

@dataclass
class FailsafeState:
    triggered: bool = False
    trigger_time_s: float = 6.0
    phase: str = "NORMAL"
    phase_start_ts: float = 0.0
    done: bool = False


def _home_pose() -> Tuple[List[float], float, str]:
    return [0.0] * 6, 0.0, "HOME"


def _drop_pose() -> Tuple[List[float], float, str]:
    # dump orientation + open gripper
    return [0.0, -45.0, 90.0, 0.0, 0.0, 0.0], 0.0, "DROP"


def failsafe_overheat_tick(t0: float, fs: FailsafeState) -> Tuple[List[float], float, str, Dict[str, Any], str]:
    now = time.time()
    elapsed = now - t0
    overrides: Dict[str, Any] = {}
    thought = ""

    if not fs.triggered and elapsed >= fs.trigger_time_s:
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
        "res_scale": 0.5,
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

    if fs.phase == "RETURN_HOME":
        thought = "FAILSAFE(OVERHEAT): return home for recharge."
        joints, grip, name = _home_pose()
        if (now - fs.phase_start_ts) > 3.0:
            fs.phase = "RECHARGE"
            fs.phase_start_ts = now
        return joints, grip, name, overrides, thought

    if fs.phase == "RECHARGE":
        thought = "RECHARGE: safe mode enabled. End demo."
        fs.done = True
        joints, grip, _ = _home_pose()
        return joints, grip, "RECHARGE", overrides, thought

    joints, grip, _ = _home_pose()
    return joints, grip, "HOME", overrides, "FAILSAFE"


def failsafe_compromised_camera_tick(t0: float, fs: FailsafeState) -> Tuple[List[float], float, str, Dict[str, Any], str]:
    now = time.time()
    elapsed = now - t0
    overrides: Dict[str, Any] = {}
    thought = ""

    if not fs.triggered and elapsed >= fs.trigger_time_s:
        fs.triggered = True
        fs.phase = "FINISH_DEPOSIT"
        fs.phase_start_ts = now

    if not fs.triggered:
        thought = "Dual camera nominal. Searching for regolith."
        return [0.0, 10.0, -10.0, 0.0, 0.0, 0.0], 0.0, "SEARCH", overrides, thought

    overrides.update({
        "mode": SystemMode.DEGRADED,
        "camera_count": 1,
        "lidar_on": True,
        "res_scale": 0.6,
        "fps_target": 12,
        "action_hint": "CONTINUE_DEPOSIT_THEN_HOME",
    })

    thought = "FAILSAFE(CAM): camera compromised → continue with 1 cam + LiDAR."
    phase_elapsed = now - fs.phase_start_ts

    if fs.phase == "FINISH_DEPOSIT":
        if phase_elapsed < 2.0:
            return [0.0, 45.0, -45.0, 0.0, 0.0, 0.0], 100.0, "SCOOP", overrides, thought
        if phase_elapsed < 4.0:
            return [0.0, -45.0, 90.0, 0.0, 0.0, 0.0], 0.0, "DEPOSIT", overrides, thought
        fs.phase = "RETURN_HOME"
        fs.phase_start_ts = now
        joints, grip, _ = _home_pose()
        return joints, grip, "HOME", overrides, thought

    if fs.phase == "RETURN_HOME":
        thought = "Return to base for troubleshooting camera."
        if (now - fs.phase_start_ts) > 3.0:
            fs.done = True
        joints, grip, _ = _home_pose()
        return joints, grip, "RETURN_HOME", overrides, thought

    joints, grip, _ = _home_pose()
    return joints, grip, "HOME", overrides, "FAILSAFE"


# ============================================================
# 5) Plan override helper (robust across dataclass/normal class)
# ============================================================

def override_plan(plan: Any, **kwargs) -> Any:
    if dataclasses.is_dataclass(plan):
        field_names = {f.name for f in dataclasses.fields(plan)}
        clean = {k: v for k, v in kwargs.items() if k in field_names}
        try:
            return dataclasses.replace(plan, **clean)
        except Exception:
            return plan

    for k, v in kwargs.items():
        if hasattr(plan, k):
            try:
                setattr(plan, k, v)
            except Exception:
                pass
    return plan


# ============================================================
# 6) Arm interfaces
# ============================================================

class ArmInterface:
    def publish(self, joints_deg: List[float], gripper_0_100: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SimArmPublisher(ArmInterface):
    """Publishes JointState to /joint_states for RViz."""
    def __init__(self, joint_names: List[str], gripper_joint_name: str):
        self.joint_names = joint_names
        self.gripper_joint_name = gripper_joint_name
        self.node: Optional[Node] = None
        self.pub = None

        if not HAS_ROS2:
            raise RuntimeError("ROS2 (rclpy) not available.")

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("alam_beta_conf_demo")
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


# ============================================================
# 7) UI helper
# ============================================================

def put_text(frame: np.ndarray, text: str, xy: Tuple[int, int], scale: float, color: Tuple[int, int, int], thick: int = 1) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ============================================================
# 8) CLI + menu
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="ALAM Beta Conference Demo (Integrated)")
    p.add_argument("--ui-width", type=int, default=480)
    p.add_argument("--ui-height", type=int, default=360)

    p.add_argument("--lidar", choices=["serial", "ros2", "none"], default="serial")
    p.add_argument("--lidar-port", default="/dev/ttyUSB0")

    p.add_argument("--sensor-mode", choices=["1cam", "1cam+lidar", "2cam+lidar"], default=None)
    p.add_argument("--scenario", choices=["UNBREAKABLE", "STICKY", "ADAPTIVE", FAILSAFE_OVERHEAT, FAILSAFE_COMPROMISED_CAMERA], default=None)
    p.add_argument("--op-mode", choices=["interactive", "ghost"], default=None)
    p.add_argument("--actuation", choices=["sim", "live"], default=None)

    p.add_argument("--trigger-after", type=float, default=6.0)
    p.add_argument("--joint-speed", type=float, default=60.0)
    p.add_argument("--gripper-speed", type=float, default=150.0)
    p.add_argument("--max-fps", type=int, default=60)

    return p.parse_args()


def menu_select(args) -> Tuple[str, str, str, str]:
    if args.sensor_mode:
        sensor_mode = args.sensor_mode
    else:
        print("\n=== BETA CONF DEMO CONFIG ===")
        print("Sensor Configuration:")
        print(" 1) 1 cam")
        print(" 2) 1 cam + 1 lidar")
        print(" 3) 2 cam + 1 lidar")
        v = input("Selection (1-3) [3]: ").strip() or "3"
        sensor_mode = {"1": "1cam", "2": "1cam+lidar", "3": "2cam+lidar"}.get(v, "2cam+lidar")

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
        scenario = {
            "1": "ADAPTIVE",
            "2": "STICKY",
            "3": "UNBREAKABLE",
            "4": FAILSAFE_OVERHEAT,
            "5": FAILSAFE_COMPROMISED_CAMERA,
        }.get(v, "ADAPTIVE")

    if args.op_mode:
        op_mode = args.op_mode
    else:
        print("\nOperation Mode:")
        print(" 1) interactive (Safety shield active)")
        print(" 2) ghost (Safety muted / forced scripted)")
        v = input("Selection (1-2) [1]: ").strip() or "1"
        op_mode = "interactive" if v == "1" else "ghost"

    if args.actuation:
        actuation = args.actuation
    else:
        print("\nActuation Mode:")
        print(" 1) SIM (publish joint_states to RViz)")
        print(" 2) LIVE (stub, for future sand/bucket tests)")
        v = input("Selection (1-2) [1]: ").strip() or "1"
        actuation = "sim" if v == "1" else "live"

    return sensor_mode, scenario, op_mode, actuation


# ============================================================
# 9) Main loop
# ============================================================

def main():
    args = parse_args()
    sensor_mode, scenario, op_mode, actuation = menu_select(args)

    use_dual_cam = (sensor_mode == "2cam+lidar")
    use_lidar = ("lidar" in sensor_mode) and (args.lidar != "none")

    print(f"\n[Init] sensor_mode={sensor_mode} scenario={scenario} op_mode={op_mode} actuation={actuation}")

    # Providers
    lidar = None
    if use_lidar:
        try:
            if args.lidar == "serial":
                lidar = SerialRPLidarProvider(port=args.lidar_port)
            elif args.lidar == "ros2":
                lidar = ROS2LaserScanProvider()
        except Exception as e:
            print(f"[WARN] LiDAR init failed: {e}")
            lidar = None

    cam_provider = None
    if use_dual_cam:
        try:
            cam_provider = TwoCameraProvider()
        except Exception as e:
            print(f"[WARN] Dual cam failed -> fallback single: {e}")
            use_dual_cam = False

    if cam_provider is None:
        try:
            cam_provider = CSICameraProvider(sensor_id=0)
        except Exception:
            cam_provider = USBCameraProvider(0)

    # Policy + gating
    health_policy = SystemHealthMonitor()
    mode_mgr = ModeManager(cam_provider, lidar)

    # Safety controller
    controller = ShieldedController(HeuristicPolicy(), SafetySupervisor.default())

    # Arm interface (RViz)
    joint_names = [
        "joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
        "joint5_to_joint4", "joint6_to_joint5", "joint6output_to_joint6",
    ]
    gripper_joint_name = os.environ.get("ALAM_GRIPPER_JOINT", "gripper_controller")

    if actuation == "sim":
        if not HAS_ROS2:
            print("[ERROR] ROS2 not available (rclpy missing). Use --actuation live or install ROS2 Humble.")
            return
        arm: ArmInterface = SimArmPublisher(joint_names, gripper_joint_name)
    else:
        arm = LiveArmStub()

    # Scenario driver
    animator = ScenarioAnimator(scenario) if scenario in SCENARIOS else None
    if animator:
        animator.start()
    interp = MotionInterpolator(joint_speed_deg=args.joint_speed, gripper_speed=args.gripper_speed)

    # Failsafe state
    fs = FailsafeState(trigger_time_s=float(args.trigger_after))

    # Logger
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = Path.cwd() / f"beta_conf_{scenario}_{sensor_mode}_{op_mode}_{ts}.csv"
    log = CSVDecisionLogger(csv_path)

    # UI
    window = "ALAM Beta Conference Demo"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, int(args.ui_width), int(args.ui_height))

    last_cam_ts = 0.0
    last_lidar_ts = 0.0
    t0 = time.time()
    frame_id = 0

    print("\n=== DEMO RUNNING (press 'q' to quit) ===")

    try:
        while True:
            loop_start = time.time()

            # Read sensors (through ModeManager)
            fp = mode_mgr.get_frame()
            lp = mode_mgr.get_distance()

            now = time.time()
            if fp.cam_a_ok:
                last_cam_ts = now
            if lp.ok:
                last_lidar_ts = now

            # Plan from health monitor
            cam_count_ok = int(fp.cam_a_ok) + int(fp.cam_b_ok)
            plan = health_policy.get_plan(
                last_lidar_ts, last_cam_ts,
                camera_count_ok=cam_count_ok,
                lidar_seen_recently=lp.ok if use_lidar else None,
                cam_seen_recently=fp.cam_a_ok,
            )

            thought = ""
            step_name = ""
            overrides: Dict[str, Any] = {}

            # ----------------------------
            # Scenario target selection
            # ----------------------------
            if scenario == FAILSAFE_OVERHEAT:
                # ensure stabilization never interferes with forced failsafes
                if hasattr(mode_mgr, "clear_stabilization"):
                    mode_mgr.clear_stabilization()
                joints_t, grip_t, step_name, overrides, thought = failsafe_overheat_tick(t0, fs)
                plan = override_plan(plan, **overrides)

            elif scenario == FAILSAFE_COMPROMISED_CAMERA:
                if hasattr(mode_mgr, "clear_stabilization"):
                    mode_mgr.clear_stabilization()
                joints_t, grip_t, step_name, overrides, thought = failsafe_compromised_camera_tick(t0, fs)
                plan = override_plan(plan, **overrides)

            else:
                # Normal scenarios
                if scenario in SCENARIOS:
                    assert animator is not None
                    joints_t, grip_t, step_name = animator.get_target()
                    thought = f"Scenario keyframe={step_name}"
                else:
                    joints_t, grip_t, step_name = [0.0] * 6, 0.0, "IDLE"
                    thought = "No scenario selected"

                # ---------------------------------------------------------
                # UNBREAKABLE-only stabilization: TUCK -> stabilize -> revert
                # ---------------------------------------------------------
                if scenario == "UNBREAKABLE" and step_name == UNBREAKABLE_STABILIZE_STEP:
                    if OperatingMode is not None and hasattr(mode_mgr, "request_stabilization"):
                        if not getattr(mode_mgr, "is_stabilizing")():
                            mode_mgr.request_stabilization(
                                duration_s=UNBREAKABLE_STABILIZE_DURATION_S,
                                mode=OperatingMode.SINGLE_CAM_LIDAR,
                                reason="UNBREAKABLE_CATCH",
                            )
                else:
                    if hasattr(mode_mgr, "is_stabilizing") and getattr(mode_mgr, "is_stabilizing")():
                        if hasattr(mode_mgr, "clear_stabilization"):
                            mode_mgr.clear_stabilization()

            # Apply plan (this is where stabilization can override the applied mode)
            mode_mgr.apply_plan(plan)

            # Safety layer (muted in ghost mode)
            proposed_action = f"SCENARIO_{step_name}"
            final_action = proposed_action
            reason = "Ghost muted safety" if op_mode == "ghost" else "Safety active"

            # distance_m is "used distance" (may be None even if LiDAR is alive)
            distance_m = lp.distance_m if getattr(plan, "lidar_on", True) else None
            lidar_alive = bool(lp.ok) if use_lidar else False

            if op_mode == "interactive" and (scenario in SCENARIOS) and getattr(plan, "mode", SystemMode.NOMINAL) != SystemMode.CRITICAL:
                obs = Observation.from_perception("target", 0.90, distance_m)
                if hasattr(obs, "health_score") and hasattr(plan, "health_score"):
                    obs.health_score = plan.health_score
                decision = controller.step(obs)

                if decision.final_action == RoverAction.STOP:
                    final_action = "SAFETY_STOP"
                    reason = decision.reason
                    joints_t = interp.current_joints[:]
                    grip_t = interp.current_gripper
                elif decision.final_action == RoverAction.RETREAT:
                    final_action = "SAFETY_RETREAT"
                    reason = decision.reason
                    joints_t = [0.0] * 6
                    grip_t = 0.0
                elif decision.final_action == RoverAction.RETURN_HOME:
                    final_action = "RETURN_HOME"
                    reason = decision.reason
                    joints_t = [0.0] * 6
                    grip_t = 0.0
                else:
                    final_action = str(decision.final_action)
                    reason = decision.reason

            # If policy forces CRITICAL, override to return-home
            if getattr(plan, "mode", SystemMode.NOMINAL) == SystemMode.CRITICAL:
                final_action = "RETURN_HOME"
                reason = "Policy CRITICAL"
                joints_t = [0.0] * 6
                grip_t = 0.0

            # Smooth + publish (RViz)
            joints_s, grip_s = interp.update(joints_t, grip_t)
            arm.publish(joints_s, grip_s)

            # Log row
            log.log(DecisionFrame(
                ts=now - t0,
                frame_id=frame_id,
                pred_class="demo",
                pred_conf=0.90,
                distance_m=distance_m,
                proposed_action=proposed_action,
                final_action=final_action,
                reason=(
                    f"{reason} | {thought} | mode={getattr(plan,'mode',None)} "
                    f"cams={getattr(plan,'camera_count',None)} lidar={getattr(plan,'lidar_on',None)} alive={lidar_alive}"
                )
            ))

            # Render UI (resize first, then overlay)
            disp = cv2.resize(fp.frame.copy(), (int(args.ui_width), int(args.ui_height)), interpolation=cv2.INTER_AREA)

            cv2.rectangle(disp, (0, 0), (disp.shape[1], 108), (0, 0, 0), -1)

            mode_val = getattr(plan, "mode", SystemMode.NOMINAL)
            color = (0, 255, 0)
            if mode_val == SystemMode.DEGRADED:
                color = (0, 165, 255)
            if mode_val == SystemMode.CRITICAL:
                color = (0, 0, 255)

            put_text(disp, f"DECISION: {final_action}", (10, 28), 0.72, color, 2)
            put_text(disp, f"SCENARIO: {scenario} | STEP: {step_name} | {op_mode.upper()}", (10, 54), 0.44, (230, 230, 230), 1)

            lidar_used = getattr(plan, "lidar_on", True) and (distance_m is not None)
            d_txt = f"{distance_m:.2f}m" if distance_m is not None else "N/A"
            put_text(
                disp,
                f"PLAN: {getattr(plan,'mode',None)} cams={getattr(plan,'camera_count',None)} "
                f"lidar={'ON' if getattr(plan,'lidar_on',True) else 'OFF'} "
                f"alive={'Y' if lidar_alive else 'N'} used={'Y' if lidar_used else 'N'} d={d_txt}",
                (10, 78),
                0.42,
                (200, 200, 200),
                1
            )

            # Stabilization indicator
            stabilizing = False
            if hasattr(mode_mgr, "is_stabilizing"):
                stabilizing = bool(getattr(mode_mgr, "is_stabilizing")())
            put_text(
                disp,
                f"stabilize={'ON' if stabilizing else 'OFF'}",
                (10, 100),
                0.42,
                (180, 180, 180),
                1
            )

            pulse = color if int(time.time() * 2) % 2 == 0 else (40, 40, 40)
            cv2.circle(disp, (disp.shape[1] - 18, 18), 9, pulse, -1)

            cv2.rectangle(disp, (0, disp.shape[0] - 30), (disp.shape[1], disp.shape[0]), (0, 0, 0), -1)
            put_text(
                disp,
                f"grip={int(grip_s):3d}%  joints(deg)={','.join([str(int(x)) for x in joints_s[:3]])}...",
                (10, disp.shape[0]-10),
                0.42,
                (180, 180, 180),
                1
            )

            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if scenario in (FAILSAFE_OVERHEAT, FAILSAFE_COMPROMISED_CAMERA) and fs.done:
                time.sleep(1.0)
                break

            frame_id += 1

            # pacing
            fps_target = getattr(mode_mgr, "fps_target", args.max_fps)
            target_fps = max(5, min(int(args.max_fps), int(fps_target)))
            elapsed = time.time() - loop_start
            dt = 1.0 / float(target_fps)
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n[Stopped] User interrupt.")
    finally:
        print("\n[Shutdown] Cleaning up...")
        try:
            log.close()
        except Exception:
            pass
        try:
            if hasattr(cam_provider, "close"):
                cam_provider.close()
        except Exception:
            pass
        try:
            if lidar and hasattr(lidar, "close"):
                lidar.close()
        except Exception:
            pass
        try:
            arm.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

