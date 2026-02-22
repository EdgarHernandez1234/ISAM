#!/usr/bin/env python3
"""
demo_dance.py (Standalone)

SHADOW MODE DANCE ROUTINE
-------------------------
A 20-second dance routine for the ALAM/ISAM cobot arm.
- Default: Runs in Simulator (RViz) only.
- Shadow Mode: Drives the physical MyCobot 280 synchronously.

Usage:
  1. Sim Only:    python3 demo_dance.py
  2. Shadow Mode: python3 demo_dance.py --shadow --port /dev/ttyUSB0
"""

import time
import math
import argparse
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

# --- ROS 2 Imports ---
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    print("[WARN] ROS 2 not found. Sim visualization will be disabled.")

# --- Hardware Imports ---
try:
    from pymycobot.mycobot import MyCobot
    HAS_HARDWARE = True
except ImportError:
    HAS_HARDWARE = False


# ============================================================
# 1. ARM CONFIGURATION (Copied from Simulator)
# ============================================================

JOINT_NAMES = [
    "joint2_to_joint1",  # Base
    "joint3_to_joint2",  # Shoulder
    "joint4_to_joint3",  # Elbow
    "joint5_to_joint4",  # Vertical Wrist
    "joint6_to_joint5",  # Twist
    "joint6output_to_joint6" # Flange
]

@dataclass
class ArmConfig:
    """Joint limits in Radians."""
    j1_min: float = -2.8
    j1_max: float = 2.8
    j2_min: float = -2.5
    j2_max: float = 2.5
    j3_min: float = -2.5
    j3_max: float = 2.5
    j4_min: float = -2.5
    j4_max: float = 2.5
    j5_min: float = -2.5
    j5_max: float = 2.5
    j6_min: float = -3.0
    j6_max: float = 3.0

def clamp_pose(cfg: ArmConfig, pose: List[float]) -> Tuple[List[float], bool]:
    """Clamps a 6-DOF pose to valid joint limits."""
    clamped = []
    was_clamped = False
    
    # Limits list matching index order
    limits = [
        (cfg.j1_min, cfg.j1_max),
        (cfg.j2_min, cfg.j2_max),
        (cfg.j3_min, cfg.j3_max),
        (cfg.j4_min, cfg.j4_max),
        (cfg.j5_min, cfg.j5_max),
        (cfg.j6_min, cfg.j6_max),
    ]
    
    for i, val in enumerate(pose):
        mn, mx = limits[i]
        if val < mn:
            clamped.append(mn)
            was_clamped = True
        elif val > mx:
            clamped.append(mx)
            was_clamped = True
        else:
            clamped.append(val)
            
    return clamped, was_clamped

class ArmJointStatePublisher:
    """
    Publishes /joint_states so RViz can visualize the arm.
    Emulates the behavior of robot_state_publisher's source.
    """
    def __init__(self, cfg: ArmConfig):
        if not HAS_ROS2:
            self.node = None
            return
            
        # Initialize ROS 2 context only if not already initialized
        if not rclpy.ok():
            rclpy.init()
            
        self.node = rclpy.create_node('cobot_dance_animator')
        self.pub = self.node.create_publisher(JointState, 'joint_states', 10)
        self.cfg = cfg

    def publish(self, pose: List[float]):
        if not self.node:
            return

        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = pose
        msg.velocity = []
        msg.effort = []
        
        self.pub.publish(msg)


# ============================================================
# 2. DANCE LOGIC
# ============================================================

BPM = 92.0
BEAT_SEC = 60.0 / BPM

class DanceAnimator:
    def __init__(self, cfg: ArmConfig):
        self.cfg = cfg
        self.home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    def get_pose_at_time(self, t: float) -> Tuple[List[float], str]:
        """Returns (joint_angles_rad, move_name) based on time t."""
        
        # 0-5s: Intro (Slow Wakeup)
        if t < 5.0:
            progress = t / 5.0
            # Slowly lift head (Joint 2 and 3)
            p = [0.0, -1.0 * progress, 1.0 * progress, 0.0, 0.0, 0.0]
            return p, "WAKE_UP"

        # 5-15s: The Carlton (Side to Side swing)
        elif t < 15.0:
            # Base Swing: sin wave
            swing = math.sin(t * (math.pi)) * 0.8  # Left/Right
            
            # Head Bob: faster sin wave (2x speed of swing)
            bob = math.sin(t * (math.pi * 2)) * 0.3
            
            # Wrist twirl (J6)
            twirl = (t * 3.0) % (math.pi * 2) - math.pi
            
            p = [
                swing,          # J1 Base
                -0.5 + bob,     # J2 Shoulder
                0.5 + bob,      # J3 Elbow
                0.0,            # J4
                -1.5,           # J5 (Wrist tilt down)
                twirl           # J6 (Spin)
            ]
            return p, "THE_CARLTON"

        # 15-20s: Big Finish
        else:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "BIG_FINISH"


# ============================================================
# 3. MAIN RUNNER
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Dance Demo with Shadow Mode")
    parser.add_argument("--shadow", action="store_true", help="Enable physical robot movement")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port for MyCobot")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    args = parser.parse_args()

    print(f"--- STARTING DANCE ROUTINE ({BPM} BPM) ---")
    print("Song: It's Not Unusual")
    print("Duration: 20 seconds")
    
    # 1. Setup Simulator Publisher
    cfg = ArmConfig()
    pub = ArmJointStatePublisher(cfg)
    if pub.node:
        print("[Sim] ROS 2 Publisher started.")
    else:
        print("[Sim] Running without RViz visualization.")

    # 2. Setup Physical Hardware (Shadow Mode)
    mc = None
    if args.shadow:
        if not HAS_HARDWARE:
            print("[Error] 'pymycobot' library not found. Cannot run Shadow Mode.")
            sys.exit(1)
        
        print(f"[Shadow] Connecting to MyCobot at {args.port}...")
        try:
            mc = MyCobot(args.port, args.baud)
            time.sleep(0.5)
            mc.power_on()
            time.sleep(0.5)
            print("[Shadow] Connection successful. Moving to HOME...")
            mc.send_angles([0, 0, 0, 0, 0, 0], 50)
            time.sleep(2.0)
        except Exception as e:
            print(f"[Shadow] Failed to connect: {e}")
            sys.exit(1)

    # 3. Execution Loop
    animator = DanceAnimator(cfg)
    start_t = time.time()
    
    # Run at 20Hz (Good balance for Serial bus and Animation smoothness)
    rate_hz = 20.0
    period = 1.0 / rate_hz 

    try:
        while True:
            iter_start = time.time()
            elapsed = iter_start - start_t
            
            if elapsed > 20.0:
                break
            
            # A. Calculate Pose (Radians)
            raw_pose_rad, move_name = animator.get_pose_at_time(elapsed)
            clamped_pose_rad, _ = clamp_pose(cfg, raw_pose_rad)
            
            # B. Publish to Simulator (RViz)
            if pub:
                pub.publish(clamped_pose_rad)
            
            # C. Drive Physical Robot (Shadow Mode)
            if mc:
                # Convert Radians to Degrees
                pose_deg = [math.degrees(angle) for angle in clamped_pose_rad]
                # Send to robot (speed=80 is responsive for dancing)
                mc.send_angles(pose_deg, 80)

            # D. Feedback & Timing
            if int(elapsed * 10) % 10 == 0:
                 print(f"[{elapsed:04.1f}s] Move: {move_name}")

            # Smart Sleep
            dt = time.time() - iter_start
            if dt < period:
                time.sleep(period - dt)

    except KeyboardInterrupt:
        print("\n[Stopped] User interrupted.")
        
    finally:
        print("Dance Complete.")
        if mc:
            print("[Shadow] Relaxing servos...")
            mc.release_all_servos()
        if pub and pub.node:
            pub.node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
