#!/usr/bin/env python3
"""
gripper_scenario_demo.py (rover_learner_gripper)

ADVANCED GRIPPER DEMO - FIXED
-----------------------------
- Fixed: Relative imports (core.py)
- Fixed: Lidar NoneType crash
"""

from __future__ import annotations

import argparse
import math
import time
import sys
import os
from pathlib import Path
from dataclasses import dataclass

# --- ROS 2 Imports ---
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    print("[WARN] ROS 2 not found. Simulation will be skipped.")

# --- Hardware Imports ---
try:
    from pymycobot.mycobot import MyCobot
    HAS_HARDWARE = True
except ImportError:
    HAS_HARDWARE = False

# --- Rover Learner Modules ---
# Note: These imports must NOT use relative (.) syntax if running as a script
try:
    import cv2
    # Ensure core.py has also been fixed to remove "from .rl_safety_supervisor"
    from core import StepInputs, Perception, Telemetry, step_with_safety
    from rl_safety_supervisor import (
        HeuristicPolicy, SafetySupervisor, ShieldedController, RoverAction
    )
    from logger import CSVDecisionLogger, DecisionFrame
    from camera_provider import CSICameraProvider, USBCameraProvider
    from lidar_provider import SerialRPLidarProvider
except ImportError as e:
    print(f"[ERROR] Missing base module: {e}")
    print("Ensure core.py, logger.py, etc. are in this folder and do not use '.' imports.")
    sys.exit(1)


# ============================================================
# 1. CONFIGURATION
# ============================================================

JOINT_NAMES = [
    "joint2_to_joint1",       # J1
    "joint3_to_joint2",       # J2
    "joint4_to_joint3",       # J3
    "joint5_to_joint4",       # J4
    "joint6_to_joint5",       # J5
    "joint6output_to_joint6", # J6 (Flange)
    "gripper_controller"      # GRIPPER
]

# Path to your merged URDF
URDF_PATH = os.path.expanduser("~/Desktop/mycobot_attachments/mycobot_final.urdf")

# Gripper Limits (Sim)
GRIPPER_OPEN_SIM = -0.7
GRIPPER_CLOSED_SIM = 0.15

# Gripper Hardware State
GRIPPER_OPEN_HW = 0
GRIPPER_CLOSED_HW = 1

@dataclass
class RobotState:
    joints: list[float]  # 6 angles (degrees)
    gripper: float       # 0.0 (Open) to 1.0 (Closed)


# ============================================================
# 2. THE ROBOT DRIVER
# ============================================================

class RobotDriver:
    def __init__(self, shadow_mode: bool, port: str):
        self.shadow_mode = shadow_mode
        self.mc = None
        self.node = None
        self.pub = None
        
        # A. Setup ROS 2 Sim
        if HAS_ROS2:
            if not rclpy.ok(): rclpy.init()
            self.node = rclpy.create_node('gripper_demo_driver')
            self.pub = self.node.create_publisher(JointState, 'joint_states', 10)
            print(f"[Sim] Expecting URDF at: {URDF_PATH}")

        # B. Setup Hardware
        if self.shadow_mode:
            if not HAS_HARDWARE:
                print("[Error] pymycobot not installed. Cannot run shadow mode.")
                sys.exit(1)
            try:
                print(f"[Hardware] Connecting to {port}...")
                self.mc = MyCobot(port, 115200)
                time.sleep(0.5)
                self.mc.power_on()
                self.mc.send_angles([0,0,0,0,0,0], 50)
                time.sleep(1.5)
                print("[Hardware] Ready.")
            except Exception as e:
                print(f"[Error] Hardware connection failed: {e}")
                sys.exit(1)

    def publish(self, joints_deg: list[float], gripper_val: float):
        # 1. Update Sim (Radians + Mapped Gripper)
        if self.pub:
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.name = JOINT_NAMES
            
            rads = [math.radians(j) for j in joints_deg]
            sim_grip = GRIPPER_OPEN_SIM + (gripper_val * (GRIPPER_CLOSED_SIM - GRIPPER_OPEN_SIM))
            
            msg.position = rads + [sim_grip]
            self.pub.publish(msg)

        # 2. Update Hardware
        if self.mc:
            self.mc.send_angles(joints_deg, 80)
            if gripper_val > 0.5:
                self.mc.set_gripper_state(GRIPPER_CLOSED_HW, 80)
            else:
                self.mc.set_gripper_state(GRIPPER_OPEN_HW, 80)


# ============================================================
# 3. SCENARIO LOGIC
# ============================================================

def get_scenario_move(t: float, scenario: str, dist_m: float) -> RobotState:
    # Base "Breathing"
    sway = math.sin(t) * 10.0
    joints = [0.0, 0.0, 0.0, 0.0, 0.0, sway]
    gripper = 0.0 # Default Open

    if scenario == "Dance":
        joints[0] = math.sin(t * 3) * 20
        joints[1] = math.sin(t * 6) * 10
        joints[5] = (t * 100) % 360 - 180
        if (int(t * 2) % 2) == 0:
            gripper = 1.0

    elif scenario == "Sticky":
        # GRIPPER REACTIVE MODE
        joints[1] = -10 # Look down
        if dist_m is not None and dist_m < 0.20:
            gripper = 1.0 # CLOSE
        else:
            gripper = 0.0 # OPEN

    return RobotState(joints, gripper)


# ============================================================
# 4. MAIN LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow", action="store_true", help="Enable hardware")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--scenario", default="Sticky", choices=["Dance", "Sticky"])
    args = parser.parse_args()

    # Setup Logging
    log_dir = Path("logs") / f"gripper_{int(time.time())}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVDecisionLogger(log_dir / "decisions.csv")
    
    # Setup Sensors
    print("[Init] Starting Camera...")
    try:
        cam = CSICameraProvider(width=640, height=480)
    except Exception:
        print("[Warn] CSI failed, using USB Camera.")
        cam = USBCameraProvider(device_id=0)
        
    print("[Init] Starting Lidar...")
    lidar = SerialRPLidarProvider(port='/dev/ttyUSB1') 

    # Setup Robot
    bot = RobotDriver(args.shadow, args.port)

    # Loop
    print(f"--- RUNNING SCENARIO: {args.scenario} ---")
    start_time = time.time()
    frame_idx = 0

    try:
        while True:
            iter_start = time.time()
            elapsed = iter_start - start_time
            
            # 1. Perception
            frame_bgr, cam_ts = cam.read()
            dist_m = lidar.get_distance_m()
            
            pred_class = "object" if (dist_m and dist_m < 0.3) else "clear"
            pred_conf = 0.95

            # 2. Decision
            target_state = get_scenario_move(elapsed, args.scenario, dist_m)
            
            # 3. Action
            bot.publish(target_state.joints, target_state.gripper)

            # 4. Visualization
            if frame_bgr is not None:
                color = (0, 255, 0) if target_state.gripper < 0.5 else (0, 0, 255)
                status = "GRIPPER: OPEN" if target_state.gripper < 0.5 else "GRIPPER: CLOSED"
                
                # --- FIXED DISPLAY LOGIC ---
                dist_str = f"{dist_m:.2f}m" if dist_m is not None else "Searching..."
                
                cv2.putText(frame_bgr, f"Mode: {args.scenario}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame_bgr, f"Lidar: {dist_str}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame_bgr, status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                
                cv2.imshow("Gripper Cam", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Log
            logger.log(DecisionFrame(
                ts=elapsed,
                frame_id=frame_idx,
                pred_class=pred_class,
                pred_conf=pred_conf,
                distance_m=dist_m,
                proposed_action=args.scenario,
                final_action="GRIP" if target_state.gripper > 0.5 else "RELEASE",
                reason=f"Lidar={dist_m}"
            ))
            frame_idx += 1
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("[Stop] User interrupted.")
