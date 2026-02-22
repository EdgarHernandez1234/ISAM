#!/usr/bin/env python3
"""
twocam_demo2.py (rover_learner)

BLINDNESS PANIC & SCENARIO PLAYER (With Complex Narratives)
-----------------------------------------------------------
1. SCENARIOS:
   - ADAPTIVE: "Catch & Recover" (Fall -> Catch -> Recover -> Home)
   - STICKY:   "Full Body Shake" (Multi-joint regolith removal)
   - UNBREAKABLE: "Smart Retreat" (Tuck -> Dump -> Stabilize -> Home)
2. SAFETY:
   - Blindness or Lidar triggers panic.
   - Ghost Mode allows scenarios to finish even if sensors scream.

Run:
  python3 -m rover_learner.twocam_demo2
"""

import time
import threading
import cv2
import numpy as np
import math
import sys
import os
import random

# --- HARDWARE IMPORTS ---
from rover_learner.two_camera_provider import TwoCameraProvider
from rover_learner.lidar_provider import SerialRPLidarProvider

# --- BRAIN IMPORTS ---
from rover_learner.rl_safety_supervisor import (
    ShieldedController, 
    HeuristicPolicy, 
    SafetySupervisor, 
    Observation,
    RoverAction
)
from rover_learner.logger import CSVDecisionLogger, DecisionFrame

# --- ROS 2 / RVIZ IMPORTS ---
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

# ============================================================
# 0. UI MENU
# ============================================================

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_menu():
    clear_screen()
    print("==========================================")
    print("      ROVER SCENARIO PLAYER (DEMO 2)      ")
    print("==========================================")
    print("Select Operational Scenario:")
    print("  [1] UNBREAKABLE (Smart Dump & Retreat)")
    print("  [2] STICKY      (Full Body Shake)")
    print("  [3] ADAPTIVE    (Catch & Recover)")
    print("==========================================")
    
    scenario_choice = "ADAPTIVE"
    while True:
        choice = input("Enter Selection (1-3): ").strip()
        if choice == '1': 
            scenario_choice = "UNBREAKABLE"
            break
        elif choice == '2': 
            scenario_choice = "STICKY"
            break
        elif choice == '3': 
            scenario_choice = "ADAPTIVE"
            break
        print("Invalid selection. Try again.")
    
    print("\n--- SAFETY CONFIGURATION ---")
    print("If cameras are blinded/covered, should the robot PANIC STOP?")
    print("  [Y] Yes (Real Safety)")
    print("  [n] No  (Ghost Mode/Mute)")
    stop_choice = input("Enable Panic Stop? [Y/n]: ").strip().lower()
    
    panic_enabled = True
    if stop_choice == 'n' or stop_choice == 'no':
        panic_enabled = False
        print(f"\n[WARNING] PANIC STOP DISABLED. GHOST MODE ACTIVE.")
        
    return scenario_choice, panic_enabled

# ============================================================
# 1. SCENARIO ENGINE
# ============================================================

JOINT_NAMES = [
    "joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
    "joint5_to_joint4", "joint6_to_joint5", "joint6output_to_joint6"
]

# --- KEYFRAME POSES ---
POSES = {
    "HOME":         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    
    # ADAPTIVE
    "FALL":         [0.0, -1.0, 1.5, 0.5, -1.0, 0.0],   
    "CATCH":        [0.0, -0.8, 1.2, 0.0, 0.5, 0.0],    
    "RECOVER":      [0.0, -0.2, 0.2, 0.0, 0.0, 0.0],    
    
    # STICKY (Now with multi-joint shake targets)
    "PRE_DUMP":     [1.57, -0.5, 0.5, 0.0, 0.0, 0.0],   
    "DUMP":         [1.57, -0.5, 0.5, 0.0, 1.5, 0.0],   
    "SHAKE_A":      [1.60, -0.4, 0.6, 0.2, 1.2, 0.2],   # Jolt all joints one way
    "SHAKE_B":      [1.54, -0.6, 0.4, -0.2, 1.8, -0.2], # Jolt all joints the other
    
    # UNBREAKABLE (Smart Narrative)
    "TUCK":         [0.0, 1.2, -1.2, 0.0, 0.0, 0.0],    # Protective defensive tuck
    "DUMP_CARGO":   [0.0, 0.8, -0.5, 0.0, 1.5, 0.0],    # Lighten load for safety
}

def lerp_pose(current, target, alpha):
    return [c + (t - c) * alpha for c, t in zip(current, target)]

def dist_sq(p1, p2):
    return sum((a - b)**2 for a, b in zip(p1, p2))

class ScenarioArmAnimator(threading.Thread):
    def __init__(self, node):
        super().__init__(daemon=True)
        self.node = node
        self.publisher = None
        if HAS_ROS2 and node:
            self.publisher = node.create_publisher(JointState, 'joint_states', 10)
        
        self.current_pose = list(POSES["HOME"])
        self.target_pose = list(POSES["HOME"])
        self.running = True
        
        self.scenario_mode = "NONE" 
        self.current_sequence = []
        self.seq_idx = 0
        self.wait_timer = 0
        self.shake_counter = 0
        
        self.override_panic = False
        self.speed_factor = 0.1

    def set_panic(self, state: bool):
        self.override_panic = state
        if state:
            self.target_pose = POSES["HOME"] 

    def set_scenario(self, mode: str):
        self.scenario_mode = mode
        self.seq_idx = 0
        self.wait_timer = 0
        
        if mode == "ADAPTIVE":
            self.current_sequence = ["HOME", "FALL", "CATCH", "RECOVER", "HOME"]
            self.speed_factor = 0.10
        elif mode == "STICKY":
            self.current_sequence = ["HOME", "PRE_DUMP", "DUMP", "FULL_SHAKE", "HOME"]
            self.speed_factor = 0.10
        elif mode == "UNBREAKABLE":
            self.current_sequence = ["HOME", "TUCK", "DUMP_CARGO", "STABILIZE", "HOME"]
            self.speed_factor = 0.05 
        elif mode == "RETREAT":
            self.current_sequence = ["TUCK"] 
            self.speed_factor = 0.2

    def run(self):
        hz = 30.0
        period = 1.0 / hz
        
        while self.running:
            start_t = time.time()
            
            if not self.override_panic:
                current_phase_name = "DONE"
                if self.current_sequence and self.seq_idx < len(self.current_sequence):
                    current_phase_name = self.current_sequence[self.seq_idx]

                if current_phase_name == "FULL_SHAKE":
                    # Rapidly oscillate ALL joints
                    if self.shake_counter % 6 < 3: self.target_pose = POSES["SHAKE_A"]
                    else: self.target_pose = POSES["SHAKE_B"]
                    self.shake_counter += 1
                    if self.shake_counter > 40: # Shake for ~1.3s
                        self.shake_counter = 0
                        self.seq_idx += 1 
                
                elif current_phase_name == "STABILIZE":
                    # High-torque safety crawl vibration
                    self.shake_counter += 1
                    lean = math.sin(self.shake_counter * 0.2) * 0.08
                    base_pose = list(POSES["TUCK"])
                    base_pose[0] = lean 
                    self.target_pose = base_pose
                    if self.shake_counter > 60: 
                        self.shake_counter = 0
                        self.seq_idx += 1
                
                elif current_phase_name != "DONE":
                    self.target_pose = POSES[current_phase_name]
                    if dist_sq(self.current_pose, self.target_pose) < 0.01:
                        self.wait_timer += 1
                        wait_limit = 10 
                        if current_phase_name == "DUMP_CARGO": wait_limit = 25
                        if self.wait_timer > wait_limit:
                            self.wait_timer = 0
                            self.seq_idx = (self.seq_idx + 1) % len(self.current_sequence)

            self.current_pose = lerp_pose(self.current_pose, self.target_pose, self.speed_factor)
            
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

    def get_status(self):
        if self.override_panic: return "PANIC HALT"
        if self.current_sequence and self.seq_idx < len(self.current_sequence):
            return self.current_sequence[self.seq_idx]
        return "IDLE"

# ============================================================
# 2. MAIN LOGIC
# ============================================================

def get_brightness(frame):
    if frame is None: return 0
    return np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

def main():
    selected_scenario, panic_enabled = show_menu()
    
    node = None
    if HAS_ROS2:
        try:
            rclpy.init()
            node = rclpy.create_node("twocam_scenario_node")
        except: pass

    animator = ScenarioArmAnimator(node)
    animator.set_scenario(selected_scenario) 
    animator.start()
    
    cameras = TwoCameraProvider()
    try:
        lidar = SerialRPLidarProvider(port="/dev/ttyUSB0")
    except:
        lidar = None

    brain = ShieldedController(HeuristicPolicy(), SafetySupervisor.default())
    logger = CSVDecisionLogger("twocam_demo2_final")
    
    try:
        frame_idx = 0
        while True:
            step_start = time.time()
            frame_a, frame_b, ts = cameras.read()
            dist = lidar.get_distance_m() if lidar else 2.0
            
            b_a = get_brightness(frame_a)
            b_b = get_brightness(frame_b)
            is_blind = (b_a < 15) or (b_b < 15)
            
            obs = Observation.from_perception(selected_scenario, 0.9, dist)
            decision = brain.step(obs)
            
            if is_blind:
                if panic_enabled:
                    animator.set_panic(True)
                else:
                    animator.set_panic(False)
                    if animator.scenario_mode != selected_scenario: animator.set_scenario(selected_scenario)
            elif decision.final_action in [RoverAction.STOP, RoverAction.RETREAT]:
                if panic_enabled:
                    animator.set_panic(False)
                    animator.set_scenario("RETREAT")
                else:
                    animator.set_panic(False)
                    if animator.scenario_mode != selected_scenario: animator.set_scenario(selected_scenario)
            else:
                animator.set_panic(False)
                if animator.scenario_mode != selected_scenario: animator.set_scenario(selected_scenario)
                
            # Visualization
            if frame_a is None: frame_a = np.zeros((360,640,3), np.uint8)
            if frame_b is None: frame_b = np.zeros((360,640,3), np.uint8)
            combined = np.vstack((frame_a, frame_b))
            
            cv2.putText(combined, f"SCENARIO: {selected_scenario}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(combined, f"PHASE: {animator.get_status()}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            if is_blind:
                if panic_enabled:
                    cv2.rectangle(combined, (0, 350), (640, 450), (0,0,255), -1)
                    cv2.putText(combined, "PANIC: BLIND!", (100, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
                else:
                    cv2.putText(combined, "BLIND (GHOST MODE)", (300, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,255), 2)

            cv2.imshow("Blindness Scenario", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            frame_idx += 1
            elapsed = time.time() - step_start
            if elapsed < 0.066: time.sleep(0.066 - elapsed)

    except KeyboardInterrupt: pass
    finally:
        animator.stop(); cameras.close()
        if lidar: lidar.close()
        cv2.destroyAllWindows()
        if node: node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
