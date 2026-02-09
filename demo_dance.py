#!/usr/bin/env python3
"""
demo_dance.py (rover_learner)

A 20-second dance routine for the ALAM cobot arm, synchronized to 
the beat of "It's Not Unusual" (~92 BPM / 0.65s per beat).

Usage:
  1. Open Tab A: ros2 launch mycobot_280 simple_gui.launch.py
  2. Open Tab B: python3 -m rover_learner.demo_dance
"""

import time
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple

# Re-using your existing infrastructure
# Ensure you are in the folder above 'rover_learner' to run this as a module
try:
    from rover_learner.cobotarm_simulator_advanced import (
        ArmConfig, ArmJointStatePublisher, JOINT_NAMES, clamp_pose
    )
except ImportError:
    # Fallback if running directly inside the folder
    from cobotarm_simulator_advanced import (
        ArmConfig, ArmJointStatePublisher, JOINT_NAMES, clamp_pose
    )

# ============================================================
# Dance Configuration
# ============================================================

BPM = 92.0  # Approximate beat of "It's Not Unusual"
BEAT_SEC = 60.0 / BPM  # ~0.652 seconds per beat

class DanceAnimator:
    def __init__(self, cfg: ArmConfig):
        self.cfg = cfg
        self.start_time = time.time()
        
        # Base Pose (Neutral)
        self.home = [0.0, -0.3, 1.0, -0.7, -0.3, 0.0]

    def get_pose_at_time(self, t: float) -> Tuple[List[float], str]:
        """
        Returns (pose_list, move_name) based on the music timing.
        """
        beat_idx = int(t / BEAT_SEC)
        phase = (t % BEAT_SEC) / BEAT_SEC  # 0.0 to 1.0 within a beat
        
        # 0s - 4s: THE INTRO SWAY (Side to side)
        if t < 4.0:
            # Joint 0 (Base) sways sine wave
            sway = 0.5 * math.sin(t * math.pi * 2.0) # Fast sway
            pose = list(self.home)
            pose[0] = sway
            pose[1] = -0.3 + 0.1 * math.cos(t * math.pi * 2.0) # Bob head
            return pose, "INTRO_SWAY"

        # 4s - 12s: THE CARLTON (Arm pumps up and down, hip sway)
        elif t < 12.0:
            # Snap motion on the beat
            # Joint 2 (Elbow) and Joint 3 (Wrist) pump opposite
            pump = 0.4 * math.sin(phase * math.pi * 2.0)
            
            pose = list(self.home)
            pose[0] = 0.3 * math.sin(t * 4.0) # Slow hip rotation
            pose[2] = 1.0 + pump              # Elbow up/down
            pose[3] = -0.7 - pump             # Wrist flex
            pose[5] = 2.0 * phase             # Continuous tool spin
            return pose, "THE_CARLTON"

        # 12s - 18s: THE SPIN (Full base rotation with arm extended)
        elif t < 18.0:
            # Full 360 spins using base joint limits
            spin_phase = (t - 12.0) / 2.0 # One spin every 2 seconds
            angle = -2.8 + (spin_phase * 5.6) % 5.6
            
            pose = [0.0] * 6
            pose[0] = angle
            pose[1] = 0.0  # Upright
            pose[2] = 0.0  # Straight
            pose[3] = 0.0  # Straight
            pose[5] = math.sin(t * 10) # Jazz hands tool shake
            return pose, "SPIN_CYCLE"

        # 18s - 20s: BIG FINISH (Pose)
        else:
            pose = [0.0, -0.5, -1.0, 0.5, 0.0, 0.0] # Dramatic bow/lean back
            return pose, "BIG_FINISH"

# ============================================================
# Main Runner
# ============================================================

def main():
    print(f"--- STARTING DANCE ROUTINE ({BPM} BPM) ---")
    print("Song: It's Not Unusual")
    print("Duration: 20 seconds")
    
    cfg = ArmConfig()
    try:
        pub = ArmJointStatePublisher(cfg)
    except Exception as e:
        print(f"Could not start ROS2 publisher: {e}")
        print("Running in 'dry run' mode (printing only).")
        pub = None

    animator = DanceAnimator(cfg)
    start_t = time.time()
    
    # Run at 30Hz for smooth motion
    rate = 1.0 / 30.0 

    try:
        while True:
            now = time.time()
            elapsed = now - start_t
            
            if elapsed > 20.0:
                break
            
            raw_pose, move_name = animator.get_pose_at_time(elapsed)
            clamped_pose, _ = clamp_pose(cfg, raw_pose)
            
            if pub:
                pub.publish(clamped_pose)
            
            # Print status every ~0.5s
            if int(elapsed * 10) % 5 == 0:
                 print(f"[{elapsed:04.1f}s] Move: {move_name}")

            time.sleep(rate)

    except KeyboardInterrupt:
        pass
    finally:
        if pub:
            pub.shutdown()
        print("--- DANCE COMPLETE ---")

if __name__ == "__main__":
    main()