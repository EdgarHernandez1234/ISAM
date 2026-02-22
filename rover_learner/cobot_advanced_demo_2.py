#!/usr/bin/env python3
"""
cobot_advanced_demo_2.py (rover_learner)

SCENARIO SHOWCASE & DATA LOGGER
-------------------------------
1. Select a Scenario (Unbreakable, Sticky, Adaptive).
2. Runs the scenario logic (Hybrid: Interactive or Scripted).
3. INTAKES LIVE DATA (Lidar + Camera) and logs it to CSV + Video.
4. "Show, Don't Just Tell": Even if the arm is dancing, we log what the Lidar sees.
"""

from __future__ import annotations

import argparse
import math
import time
import threading
import cv2
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- ROS 2 Imports ---
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

# --- Rover Learner Imports ---
from .core import StepInputs, Perception, Telemetry, step_with_safety
from .rl_safety_supervisor import (
    HeuristicPolicy, SafetySupervisor, ShieldedController, RoverAction
)
from .logger import CSVDecisionLogger, DecisionFrame
from .camera_provider import CSICameraProvider, USBCameraProvider
from .lidar_provider import SerialRPLidarProvider # Forced Robust Serial


# ============================================================
# 1. ANIMATION ENGINE
# ============================================================

JOINT_NAMES = [
    "joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
    "joint5_to_joint4", "joint6_to_joint5", "joint6output_to_joint6"
]

POSES = {
    "HOME":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "SEARCH":  [0.0, -0.5, 0.5, 0.0, 0.0, 0.0],
    "SCOOP":   [0.0, -1.2, 1.5, 0.5, 0.0, 0.0],
    "DUMP":    [3.14, -0.5, 0.5, 0.0, 0.0, 0.0],
    "RETREAT": [0.0, 0.5, -0.5, 0.0, 0.0, 0.0],
    "DANCE_1": [0.5, -0.5, 0.5, 0.0, -1.5, 0.0],
    "DANCE_2": [-0.5, -0.5, 0.5, 0.0, -1.5, 0.0]
}

def lerp_pose(current: List[float], target: List[float], alpha: float) -> List[float]:
    return [c + (t - c) * alpha for c, t in zip(current, target)]

class AsyncArmAnimator(threading.Thread):
    def __init__(self, node: Optional[Node]):
        super().__init__(daemon=True)
        self.node = node
        self.publisher = None
        if HAS_ROS2 and node:
            self.publisher = node.create_publisher(JointState, 'joint_states', 10)
        
        self.current_pose = list(POSES["HOME"])
        self.target_pose = list(POSES["HOME"])
        self.status_text = "INIT"
        self.running = True
        self.hz = 30.0
        
        # Scenario State
        self.scenario_mode = "NORMAL" # NORMAL, WIGGLE, DANCE
        self.anim_phase = 0.0

    def set_target(self, action: str, mode: str = "NORMAL"):
        self.scenario_mode = mode
        mapping = {
            "SCOOP": "SCOOP", "FORWARD": "SEARCH", "STOP": "HOME",
            "RETREAT": "RETREAT", "DUMP": "DUMP"
        }
        
        # If we are in a special mode (like DANCE), we ignore the action input
        # unless it is a safety override (RETREAT)
        if action == "RETREAT":
            self.target_pose = POSES["RETREAT"]
            self.status_text = "SAFETY RETREAT!"
            self.scenario_mode = "NORMAL" # Override dance
            return

        if self.scenario_mode == "NORMAL":
            pose_key = mapping.get(action, "HOME")
            if pose_key in POSES:
                self.target_pose = POSES[pose_key]
                self.status_text = action

    def run(self):
        period = 1.0 / self.hz
        while self.running:
            start_t = time.time()
            self.anim_phase += 0.1
            
            # --- SCENARIO ANIMATION LOGIC ---
            if self.scenario_mode == "WIGGLE":
                # "Sticky Regolith": Shake the wrist (Joint 6)
                base = list(POSES["DUMP"])
                shake = math.sin(self.anim_phase * 3.0) * 0.8 # Fast shake
                base[5] = shake
                self.target_pose = base
                self.status_text = "SHAKING BUCKET"

            elif self.scenario_mode == "DANCE":
                # "Adaptive": Smooth sine wave between two poses
                factor = (math.sin(self.anim_phase) + 1.0) / 2.0
                p1 = POSES["DANCE_1"]
                p2 = POSES["DANCE_2"]
                self.target_pose = [a + (b - a) * factor for a, b in zip(p1, p2)]
                self.status_text = "ADAPTIVE DANCE"

            # Smooth Interpolation
            self.current_pose = lerp_pose(self.current_pose, self.target_pose, 0.15)
            
            # Publish
            if self.publisher:
                msg = JointState()
                msg.header = Header()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.name = JOINT_NAMES
                msg.position = self.current_pose
                self.publisher.publish(msg)
            
            elapsed = time.time() - start_t
            if elapsed < period:
                time.sleep(period - elapsed)

    def stop(self):
        self.running = False
        if self.is_alive(): self.join()


# ============================================================
# 2. UTILS
# ============================================================

def select_scenario() -> str:
    print("\n" + "="*40)
    print("      SCENARIO DEMO SELECTION")
    print("="*40)
    print("1) UNBREAKABLE (Defensive)")
    print("   - Reacts to Lidar < 0.5m by retreating.")
    print("2) STICKY (Mechanical Stress)")
    print("   - Simulates shaking a stuck bucket.")
    print("3) ADAPTIVE (Behavioral)")
    print("   - Continuous 'Searching/Dancing' motion.")
    print("="*40)
    
    while True:
        choice = input("Select Scenario [1-3]: ").strip()
        if choice == '1': return "UNBREAKABLE"
        elif choice == '2': return "STICKY"
        elif choice == '3': return "ADAPTIVE"
        print("Invalid selection.")

# ============================================================
# 3. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best.pt")
    args = parser.parse_args()

    # 1. Select Scenario
    scenario = select_scenario()
    print(f"\n[System] Initializing Scenario: {scenario}")
    print("[System] Logging enabled -> logs/demo_data.csv & logs/demo_video.mp4")

    # 2. Init ROS 2
    node = None
    if HAS_ROS2:
        try:
            rclpy.init()
            node = rclpy.create_node("demo_logger_node")
        except: pass

    # 3. Start Animator
    animator = AsyncArmAnimator(node)
    animator.start()

    # 4. Init Hardware (Live Intake)
    cam = None
    try:
        cam = CSICameraProvider(sensor_id=0, width=640, height=480)
    except:
        print("[Warn] CSI Camera failed, using USB.")
        cam = USBCameraProvider(0)

    print("[Init] Opening LiDAR (C1 Serial)...")
    lidar = SerialRPLidarProvider(port="/dev/ttyUSB0", baudrate=460800)

    # 5. Logging Setup
    logger = CSVDecisionLogger(Path("logs/demo_data.csv"))
    vid_writer = None
    
    # YOLO (Optional for this demo, but good for logging)
    model = None
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
    except: pass

    print("\n[READY] Running Scenario. Press 'q' to quit.\n")

    try:
        frame_idx = 0
        start_time = time.time()
        
        while True:
            iter_start = time.time()
            elapsed = iter_start - start_time
            
            # --- A. SENSE (Live Data) ---
            frame, ts = cam.read()
            dist_m = lidar.get_distance_m()
            
            # --- B. PERCEIVE ---
            pred_class = "none"
            pred_conf = 0.0
            if model and frame is not None:
                res = model(frame, verbose=False, imgsz=320)
                if res and res[0].boxes:
                    box = res[0].boxes[0]
                    pred_class = model.names[int(box.cls[0])]
                    pred_conf = float(box.conf[0])

            # --- C. SCENARIO LOGIC ---
            action = "WAIT"
            anim_mode = "NORMAL"

            if scenario == "UNBREAKABLE":
                # Interactive: If Lidar sees threat, RETREAT. Else SCOOP.
                if dist_m is not None and dist_m < 0.40: # 40cm threat zone
                    action = "RETREAT"
                else:
                    # Simple cycle: Search -> Scoop
                    action = "SCOOP" if (int(elapsed) % 10 > 5) else "FORWARD"
            
            elif scenario == "STICKY":
                # Scripted: Cycle to DUMP, then WIGGLE
                cycle = int(elapsed) % 10
                if cycle < 5: action = "FORWARD"
                elif cycle < 8: action = "SCOOP"
                else: 
                    action = "DUMP"
                    anim_mode = "WIGGLE" # Trigger shake

            elif scenario == "ADAPTIVE":
                # Scripted: Just Dance
                action = "FORWARD"
                anim_mode = "DANCE"

            # --- D. ACT ---
            animator.set_target(action, mode=anim_mode)

            # --- E. DISPLAY & LOG ---
            if frame is not None:
                # Resize for display
                disp_frame = cv2.resize(frame, (640, 480))
                
                # Setup Video Writer on first frame
                if vid_writer is None:
                    h, w = disp_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vid_writer = cv2.VideoWriter('logs/demo_video.mp4', fourcc, 15.0, (w, h))
                
                # HUD Overlay
                lines = [
                    f"SCENARIO: {scenario}",
                    f"ACTION: {action} ({animator.status_text})",
                    f"LIDAR: {dist_m:.3f}m" if dist_m else "LIDAR: --",
                    f"OBJ: {pred_class}"
                ]
                
                # Color code based on Lidar safety
                color = (0, 255, 0) # Green
                if dist_m and dist_m < 0.4: color = (0, 0, 255) # Red Alert
                
                y = 30
                for line in lines:
                    cv2.putText(disp_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                    cv2.putText(disp_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y += 30

                # Show & Save
                cv2.imshow("Scenario Demo", disp_frame)
                vid_writer.write(disp_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # CSV Log
            logger.log(DecisionFrame(
                ts=ts, frame_id=frame_idx,
                pred_class=pred_class, pred_conf=pred_conf,
                distance_m=dist_m,
                proposed_action=scenario, # Log which scenario we are running
                final_action=action,
                reason=f"Scenario: {scenario} | Lidar: {dist_m}"
            ))
            frame_idx += 1
            
            # Rate Limit (15Hz)
            dt = time.time() - iter_start
            if dt < 0.066:
                time.sleep(0.066 - dt)

    except KeyboardInterrupt:
        print("[Stop] Demo interrupted.")
    finally:
        animator.stop()
        if cam: cam.close()
        if lidar: lidar.close()
        if vid_writer: vid_writer.release()
        logger.close()
        cv2.destroyAllWindows()
        if node: node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()