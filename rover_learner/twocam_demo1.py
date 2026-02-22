#!/usr/bin/env python3
"""
twocam_demo1.py (rover_learner)

THE COMPLETE SAFETY SUPERVISOR (With "Ghost Mode" & Full Mission Cycle)
-----------------------------------------------------------------------
1. MENU: Select Speed/Safety Level.
2. LOGIC: 
   - 'Active Safety': Stops/Retreats arm when Lidar < Threshold.
   - 'Ghost Mode': Warnings appear, but arm continues mission.
3. MOTION: Cycles through Approach -> Harvest -> Deposit -> Home.

Run:
  python3 -m rover_learner.twocam_demo1
"""

import time
import threading
import cv2
import numpy as np
import math
import sys
import os

# --- HARDWARE IMPORTS ---
from rover_learner.two_camera_provider import TwoCameraProvider
from rover_learner.lidar_provider import SerialRPLidarProvider

# --- BRAIN IMPORTS ---
from rover_learner.rl_safety_supervisor import (
    ShieldedController, 
    HeuristicPolicy, 
    SafetySupervisor, 
    Observation
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
    print("   ROVER SAFETY SUPERVISOR (DEMO 1)       ")
    print("==========================================")
    print("Select Operation Mode (Speed & Safety):")
    print("  [1] PRECISE (Slow Speed, High Safety Dist)")
    print("  [2] NORMAL  (Standard Speed, Med Safety Dist)")
    print("  [3] RAPID   (Fast Speed, Low Safety Dist)")
    print("==========================================")
    
    dist_threshold = 0.5
    speed_factor = 0.1 # Default alpha (lerp speed)
    mode_name = "NORMAL"

    while True:
        choice = input("Enter Selection (1-3): ").strip()
        if choice == '1': 
            dist_threshold = 1.0
            speed_factor = 0.02 # Very Slow
            mode_name = "PRECISE"
            break
        elif choice == '2': 
            dist_threshold = 0.5
            speed_factor = 0.08 # Normal
            mode_name = "NORMAL"
            break
        elif choice == '3': 
            dist_threshold = 0.2
            speed_factor = 0.25 # Fast
            mode_name = "RAPID"
            break
        print("Invalid selection. Try again.")

    print("\n--- SAFETY CONFIGURATION ---")
    print("Do you want the Safety Stop to physically HALT the arm?")
    print("  [Y] Yes, enable brakes (Real Safety)")
    print("  [n] No, just show warning (Ghost Mode/Mute)")
    stop_choice = input("Enable Physical Stop? [Y/n]: ").strip().lower()
    
    physical_stop_enabled = True
    if stop_choice == 'n' or stop_choice == 'no':
        physical_stop_enabled = False
        print(f"\n[WARNING] PHYSICAL SAFETY DISABLED. GHOST MODE ACTIVE.")
    
    return dist_threshold, speed_factor, mode_name, physical_stop_enabled

# ============================================================
# 1. ARM ANIMATION ENGINE (Mission Cycle)
# ============================================================

JOINT_NAMES = [
    "joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
    "joint5_to_joint4", "joint6_to_joint5", "joint6output_to_joint6"
]

# Key Waypoints for the Mission
POSES = {
    "HOME":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "APPROACH": [0.0, -0.5, 0.5, 0.0, 0.0, 0.0],   # Reaching out
    "HARVEST":  [0.0, -1.0, 1.2, 0.5, 1.5, 0.0],   # Scooping down
    "CARRY":    [0.0, -0.5, 0.5, 0.0, 1.5, 0.0],   # Lifting up
    "DEPOSIT":  [3.14, -0.5, 0.5, 0.0, 1.5, 0.0],  # Dump behind (180 deg)
    "RETREAT":  [0.0, 0.5, -0.5, 0.0, 0.0, 0.0],   # Defensive tuck
}

def lerp_pose(current, target, alpha):
    """Linear interpolation between poses for smooth movement."""
    return [c + (t - c) * alpha for c, t in zip(current, target)]

def dist_sq(p1, p2):
    """Squared distance between two poses to check if we arrived."""
    return sum((a - b)**2 for a, b in zip(p1, p2))

class MissionArmAnimator(threading.Thread):
    def __init__(self, node, speed_factor=0.1):
        super().__init__(daemon=True)
        self.node = node
        self.speed_factor = speed_factor
        self.publisher = None
        if HAS_ROS2 and node:
            self.publisher = node.create_publisher(JointState, 'joint_states', 10)
        
        self.current_pose = list(POSES["HOME"])
        self.target_pose = list(POSES["HOME"])
        
        # Mission State Machine
        self.mission_phases = ["HOME", "APPROACH", "HARVEST", "CARRY", "DEPOSIT", "HOME"]
        self.phase_idx = 0
        self.wait_timer = 0
        self.running = True
        self.safety_override = False # If True, ignores mission and holds position/retreats

    def set_safety_override(self, active: bool, action: str = "STOP"):
        self.safety_override = active
        if active:
            if action == "RETREAT":
                self.target_pose = POSES["RETREAT"]
            else:
                # STOP means hold current target (or freeze)
                # For visual clarity, we'll just stop updating the phase
                pass 

    def run(self):
        hz = 30.0
        period = 1.0 / hz
        
        while self.running:
            start_t = time.time()
            
            # --- MISSION LOGIC ---
            if not self.safety_override:
                # 1. Check if we reached the current target
                target_name = self.mission_phases[self.phase_idx]
                self.target_pose = POSES[target_name]
                
                # If close enough to target, wait a bit, then move to next phase
                if dist_sq(self.current_pose, self.target_pose) < 0.01:
                    self.wait_timer += 1
                    if self.wait_timer > 30: # Wait ~1 second at each waypoint
                        self.wait_timer = 0
                        self.phase_idx = (self.phase_idx + 1) % len(self.mission_phases)
            
            # --- MOVEMENT LOGIC ---
            # Interpolate towards target
            self.current_pose = lerp_pose(self.current_pose, self.target_pose, self.speed_factor)
            
            # --- ROS PUBLISH ---
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
        
    def get_current_phase(self):
        return self.mission_phases[self.phase_idx]

# ============================================================
# 2. MAIN LOGIC
# ============================================================

def stack_images(img1, img2):
    if img1 is None: img1 = np.zeros((360, 640, 3), dtype=np.uint8)
    if img2 is None: img2 = np.zeros((360, 640, 3), dtype=np.uint8)
    if img1.shape[1] != img2.shape[1]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return np.vstack((img1, img2))

def main():
    # 1. SHOW MENU
    safety_dist, speed_factor, mode_name, safety_enabled = show_menu()
    print(f"\n[INFO] Starting {mode_name} MODE")
    print(f"       Speed Factor: {speed_factor}")
    print(f"       Safety Dist:  {safety_dist}m")
    print(f"       Physical Stop: {'ENABLED' if safety_enabled else 'MUTED'}")
    time.sleep(2)

    # 2. INIT ROS
    node = None
    if HAS_ROS2:
        try:
            rclpy.init()
            node = rclpy.create_node("twocam_supervisor_node")
        except: pass

    # 3. INIT HARDWARE
    # Pass the selected speed factor to the animator
    animator = MissionArmAnimator(node, speed_factor=speed_factor)
    animator.start()
    
    try:
        lidar = SerialRPLidarProvider(port="/dev/ttyUSB0") 
    except Exception as e:
        print(f"[ERROR] Lidar not found: {e}")
        return
    cameras = TwoCameraProvider()
    
    # 4. INIT BRAIN
    try:
        policy = HeuristicPolicy()
        supervisor = SafetySupervisor.default()
        brain = ShieldedController(policy, supervisor)
    except Exception as e:
        print(f"[FATAL] Brain Init Failed: {e}")
        return
        
    logger = CSVDecisionLogger("twocam_demo1_log")

    print("System Active. Press 'q' to quit.")
    
    try:
        frame_idx = 0
        while True:
            step_start = time.time()
            
            # SENSE
            dist = lidar.get_distance_m()
            dist_val = dist if dist else None 
            frame_a, frame_b, ts = cameras.read()
            
            # THINK
            obs = Observation.from_perception("clean", 0.5, dist_val)
            decision = brain.step(obs)
            
            # --- GHOST MODE & OVERRIDE LOGIC ---
            
            # 1. Check if we crossed the user's manual threshold
            is_unsafe = False
            if dist_val and dist_val < safety_dist:
                is_unsafe = True

            # 2. Decide Physical Action
            if is_unsafe:
                if safety_enabled:
                    # REAL STOP: Tell animator to pause/retreat
                    animator.set_safety_override(True, "RETREAT")
                    final_action = "RETREAT"
                else:
                    # GHOST MODE: Tell animator to IGNORE safety (keep running mission)
                    animator.set_safety_override(False) 
                    final_action = animator.get_current_phase() # Show what it's doing
            else:
                # SAFE: Let mission proceed
                animator.set_safety_override(False)
                final_action = animator.get_current_phase()

            # VISUALIZE
            combined_view = stack_images(frame_a, frame_b)
            
            # Text Colors
            status_color = (0, 255, 0) # Green
            if is_unsafe and safety_enabled:
                status_color = (0, 0, 255) # Red (Stopped)
            elif is_unsafe and not safety_enabled:
                status_color = (0, 165, 255) # Orange (Ghost Warning)

            cv2.putText(combined_view, f"MODE: {mode_name} ({safety_dist}m)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            cv2.putText(combined_view, f"ACTION: {final_action}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            dist_str = f"{dist_val:.2f}m" if dist_val else "---"
            cv2.putText(combined_view, f"LIDAR: {dist_str}", (20, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Ghost Mode Warning Label
            if is_unsafe and not safety_enabled:
                cv2.putText(combined_view, "SAFETY TRIGGERED (MUTED)", (20, 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            cv2.imshow("Dual Cam Supervisor", combined_view)
            
            logger.log(DecisionFrame(ts, frame_idx, "clean", 0.5, dist_val, decision.proposed_action, final_action, decision.reason))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_idx += 1
            elapsed = time.time() - step_start
            if elapsed < 0.066: time.sleep(0.066 - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] Shutting down...")
        animator.stop()
        cameras.close()
        lidar.close()
        cv2.destroyAllWindows()
        if node: node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
