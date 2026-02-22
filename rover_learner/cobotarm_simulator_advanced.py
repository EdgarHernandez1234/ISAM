#!/usr/bin/env python3
"""
gripper_basic_demo.py

BASIC ARM & SENSOR DEMO
-----------------------
- Menu: Select Camera/Lidar configuration.
- Action: Cycles through SCAN -> SCOOP -> RETREAT -> STOP.
- No Safety Layer.
"""

from __future__ import annotations

import time
import sys
import os
import math
import cv2
import numpy as np
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

# --- Rover Modules ---
try:
    # We only need basic providers, no safety supervisor
    from camera_provider import CSICameraProvider, USBCameraProvider
    from lidar_provider import SerialRPLidarProvider
except ImportError as e:
    print(f"[ERROR] Missing base module: {e}")
    sys.exit(1)


# ============================================================
# 1. CONFIGURATION & POSES
# ============================================================

JOINT_NAMES = [
    "joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
    "joint5_to_joint4", "joint6_to_joint5", "joint6output_to_joint6",
    "gripper_controller"
]

# Standard MyCobot 280 Joint Angles [J1, J2, J3, J4, J5, J6]
POSES = {
    "HOME":    [0, 0, 0, 0, 0, 0],
    "SCAN":    [0, -10, -10, -10, 0, 0],       # Slight lean forward
    "SCOOP":   [0, -50, -40, -30, 0, 90],      # Down and scoop
    "RETREAT": [0, 10, -20, 10, 0, 0],         # Pull back
}

# Gripper States (1 = Closed/Scoop, 0 = Open)
GRIPPER_STATES = {
    "HOME": 0, "SCAN": 0, "SCOOP": 1, "RETREAT": 1
}

@dataclass
class RobotState:
    joints: list[float]
    gripper: float

# ============================================================
# 2. ROBOT DRIVER (Simplified)
# ============================================================

class RobotDriver:
    def __init__(self, use_hardware: bool, port: str):
        self.use_hardware = use_hardware
        self.mc = None
        self.node = None
        self.pub = None
        
        # ROS 2 Sim Setup
        if HAS_ROS2:
            if not rclpy.ok(): rclpy.init()
            self.node = rclpy.create_node('gripper_basic_demo')
            self.pub = self.node.create_publisher(JointState, 'joint_states', 10)

        # Hardware Setup
        if self.use_hardware and HAS_HARDWARE:
            try:
                print(f"[Hardware] Connecting to {port}...")
                self.mc = MyCobot(port, 115200)
                time.sleep(0.5)
                self.mc.power_on()
                self.mc.send_angles([0,0,0,0,0,0], 50)
                time.sleep(1.0)
                print("[Hardware] Robot Ready.")
            except Exception as e:
                print(f"[Error] Connection failed: {e}")

    def move(self, joints: list[float], gripper_val: int):
        # 1. Update Sim
        if self.pub:
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.name = JOINT_NAMES
            rads = [math.radians(j) for j in joints]
            # Map 0/1 to sim gripper range -0.7 to 0.15
            sim_grip = -0.7 + (gripper_val * (0.15 - (-0.7)))
            msg.position = rads + [sim_grip]
            self.pub.publish(msg)

        # 2. Update Hardware
        if self.mc:
            self.mc.send_angles(joints, 40)
            # 0 is Open, 1 is Closed
            self.mc.set_gripper_state(gripper_val, 80)

# ============================================================
# 3. MAIN APPLICATION
# ============================================================

def main():
    # --- PROMPTS ---
    print("\n=== GRIPPER DEMO SETUP ===")
    arm_cycle_input = input("Display arm cycle? (y/n): ").strip().lower()
    enable_arm = arm_cycle_input == 'y'

    print("\nSelect Mode:")
    print("1) One Camera")
    print("2) One Camera + One Lidar")
    print("3) Two Cameras + One Lidar")
    
    try:
        mode = int(input("Selection (1-3): ").strip())
    except ValueError:
        mode = 1

    # --- SENSOR INIT ---
    cam1 = None
    cam2 = None
    lidar = None

    # Init Camera 1 (Always on)
    print("[Init] Starting Primary Camera...")
    try:
        cam1 = CSICameraProvider(width=640, height=480)
    except Exception:
        print("  -> CSI not found, trying USB(0).")
        cam1 = USBCameraProvider(device_id=0)

    # Init Lidar (Mode 2 or 3)
    if mode >= 2:
        print("[Init] Starting Lidar...")
        try:
            lidar = SerialRPLidarProvider(port='/dev/ttyUSB1')
        except Exception as e:
            print(f"  -> Lidar failed: {e}")

    # Init Camera 2 (Mode 3)
    if mode == 3:
        print("[Init] Starting Secondary Camera...")
        try:
            # Assuming USB for second cam
            cam2 = USBCameraProvider(device_id=1) 
        except Exception as e:
             print(f"  -> Cam2 failed: {e}")

    # --- ROBOT INIT ---
    bot = None
    if enable_arm:
        bot = RobotDriver(use_hardware=True, port="/dev/ttyUSB0")
    
    # --- LOOP VARIABLES ---
    print("\n=== STARTING LOOP (Press 'q' to quit) ===")
    start_time = time.time()
    current_pose_name = "HOME"
    
    try:
        while True:
            # 1. TIME KEEPER (4 Second Cycle Steps)
            elapsed = time.time() - start_time
            cycle_idx = int(elapsed / 3.0) % 4 
            
            # 2. DETERMINE ACTION
            # Map time steps to actions
            if cycle_idx == 0: action = "SCAN"
            elif cycle_idx == 1: action = "SCOOP"
            elif cycle_idx == 2: action = "RETREAT"
            else: action = "STOP"

            # 3. SET POSE (Logic Requested)
            target_joints = POSES["HOME"]
            
            if action == "SCOOP": 
                target_joints = POSES["SCOOP"]
            elif action == "RETREAT": 
                target_joints = POSES["RETREAT"]
            elif action == "STOP": 
                target_joints = POSES["HOME"]
            else: 
                target_joints = POSES["SCAN"]

            current_pose_name = action
            
            # 4. EXECUTE ROBOT MOVEMENT
            if bot:
                grip_state = GRIPPER_STATES.get(action, 0)
                bot.move(target_joints, grip_state)

            # 5. READ SENSORS & VISUALIZE
            frame1, _ = cam1.read()
            frame2, _ = cam2.read() if cam2 else (None, None)
            dist_m = lidar.get_distance_m() if lidar else None

            # Prepare Display
            if frame1 is not None:
                # Resize Cam2 to match Cam1 if necessary for stacking
                if frame2 is not None:
                    if frame1.shape != frame2.shape:
                        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
                    # Horizontal Stack
                    display_frame = cv2.hconcat([frame1, frame2])
                else:
                    display_frame = frame1

                # Overlays
                lidar_text = f"Lidar: {dist_m:.3f}m" if dist_m is not None else "Lidar: N/A"
                cv2.putText(display_frame, f"Action: {current_pose_name}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, lidar_text, (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                cv2.imshow("Rover Demo", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # If no camera, just sleep briefly
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[Stopped] User interrupt.")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        if HAS_ROS2 and rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
