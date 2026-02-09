#!/usr/bin/env python3
"""
cobotarm_simulator_advanced.py (rover_learner)

Desktop-only "arm simulator bridge" for ALAM/ISAM demos.

UPDATED FEATURE: "Scenario Roulette"
Randomly selects one of three impressive mechanical demos:
  1. UNBREAKABLE (Stall Recovery): Simulates hitting a rock, triggering safety stop/retreat.
  2. STICKY (Bucket Shake): Simulates shaking sticky regolith off the tool.
  3. ADAPTIVE (Dynamic Targeting): Randomizes scoop locations to show IK flexibility.
"""

from __future__ import annotations

import argparse
import math
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# --- Existing rover_learner modules ---
from rover_learner.core import StepInputs, Perception, Telemetry, step_with_safety
from rover_learner.rl_safety_supervisor import HeuristicPolicy, SafetySupervisor, ShieldedController, RoverAction
from rover_learner.logger import CSVDecisionLogger, DecisionFrame
from rover_learner.camera_provider import CSICameraProvider, USBCameraProvider, MockCameraProvider
from rover_learner.lidar_provider import ROS2LaserScanProvider, MockLidarProvider

# --- ROS2 (for desktop sim publishing) ---
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    HAS_ROS2 = True
except Exception:
    HAS_ROS2 = False


# ============================================================
# 1) Arm kinematics & Configuration
# ============================================================

JOINT_NAMES = [
    "joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
    "joint5_to_joint4", "joint6_to_joint5", "joint6output_to_joint6",
]

@dataclass(frozen=True)
class JointLimits:
    lo: float
    hi: float

@dataclass
class ArmConfig:
    publish_hz: float = 50.0
    transition_s: float = 1.0
    limits: Dict[str, JointLimits] = None

    def __post_init__(self):
        if self.limits is None:
            # Conservative limits (radians)
            self.limits = {
                JOINT_NAMES[0]: JointLimits(lo=-2.8, hi=2.8),
                JOINT_NAMES[1]: JointLimits(lo=-1.8, hi=1.8),
                JOINT_NAMES[2]: JointLimits(lo=-1.8, hi=1.8),
                JOINT_NAMES[3]: JointLimits(lo=-2.2, hi=2.2),
                JOINT_NAMES[4]: JointLimits(lo=-2.2, hi=2.2),
                JOINT_NAMES[5]: JointLimits(lo=-3.2, hi=3.2),
            }

@dataclass
class Keyframe:
    name: str
    positions: List[float]
    hold_s: float

def clamp_pose(cfg: ArmConfig, pose: List[float]) -> Tuple[List[float], float]:
    clamped = []
    sq = 0.0
    for name, val in zip(JOINT_NAMES, pose):
        lim = cfg.limits[name]
        v2 = max(lim.lo, min(lim.hi, float(val)))
        clamped.append(v2)
        sq += (v2 - float(val)) ** 2
    return clamped, math.sqrt(sq)


# ============================================================
# 2) Scenario Definitions
# ============================================================

SCENARIO_UNBREAKABLE = "UNBREAKABLE (Stall Recovery)"
SCENARIO_STICKY      = "STICKY (Bucket Shake)"
SCENARIO_ADAPTIVE    = "ADAPTIVE (Dynamic Targeting)"

def pick_scenario(force_name: Optional[str] = None) -> str:
    if force_name:
        if "stall" in force_name.lower(): return SCENARIO_UNBREAKABLE
        if "sticky" in force_name.lower(): return SCENARIO_STICKY
        if "adapt" in force_name.lower(): return SCENARIO_ADAPTIVE
    
    # Roulette!
    return random.choice([SCENARIO_UNBREAKABLE, SCENARIO_STICKY, SCENARIO_ADAPTIVE])


# ============================================================
# 3) Advanced Animator (Handles Shake & Randomization)
# ============================================================

class PickPlaceAnimator:
    def __init__(self, cfg: ArmConfig, scenario: str):
        self.cfg = cfg
        self.scenario = scenario
        self.dt = 1.0 / max(1.0, float(cfg.publish_hz))
        
        self.phase = "hold"
        self.phase_t = 0.0
        self.k = 0
        
        # Shake state
        self.shake_timer = 0.0

        # Base Keyframes
        # [base, shoulder, elbow, wrist_pitch, wrist_roll, tool_roll]
        self._base_keyframes = [
            Keyframe("home",              [ 0.0, -0.3,  1.0, -0.7, -0.3,  0.0], 0.5),
            Keyframe("approach",          [ 0.2,  0.1,  1.25, -1.15,-0.4,  0.0], 0.4),
            Keyframe("scoop",             [ 0.2,  0.25, 1.45, -1.35,-0.4,  0.0], 0.8),
            Keyframe("lift",              [ 0.2, -0.15, 1.10, -0.95,-0.4,  0.0], 0.5),
            Keyframe("move_drop",         [-0.6, -0.10, 1.05, -0.90,-0.4,  0.0], 0.4),
            Keyframe("dump",              [-0.6,  0.05, 0.95, -0.60,-0.4,  0.9], 0.8), # Target for shake
            Keyframe("return",            [ 0.0, -0.3,  1.0, -0.7, -0.3,  0.0], 0.5),
        ]

        self.keyframes = self._generate_cycle_keyframes()
        
        self.start_pose = self.keyframes[0].positions[:]
        self.target_pose = self.keyframes[0].positions[:]
        self.current_pose = self.keyframes[0].positions[:]

    def _generate_cycle_keyframes(self) -> List[Keyframe]:
        """
        Regenerates the keyframe list. 
        If ADAPTIVE scenario is active, adds random noise to approach/scoop.
        """
        frames = []
        
        # Random offsets for Adaptive Scenario
        # We modify the Base (Joint 0) and Shoulder (Joint 1) slightly
        offset_base = 0.0
        offset_reach = 0.0
        
        if self.scenario == SCENARIO_ADAPTIVE:
            offset_base = random.uniform(-0.15, 0.15)  # Left/Right
            offset_reach = random.uniform(-0.05, 0.05) # Near/Far

        for k in self._base_keyframes:
            pos = list(k.positions)
            if k.name in ["approach", "scoop", "lift"]:
                pos[0] += offset_base # Base pan
                pos[1] += offset_reach # Shoulder reach
            frames.append(Keyframe(k.name, pos, k.hold_s))
            
        return frames

    def _apply_shake(self, pose: List[float]) -> List[float]:
        """Adds a high-frequency sine wave to the tool joint (index 5)."""
        # 15Hz shake, +/- 0.3 radians
        shake_offset = 0.3 * math.sin(self.shake_timer * 15.0 * 2 * math.pi)
        new_pose = list(pose)
        new_pose[5] += shake_offset
        return new_pose

    def _smoothstep(self, u: float) -> float:
        u = max(0.0, min(1.0, float(u)))
        return u * u * (3.0 - 2.0 * u)

    def _lerp(self, a: List[float], b: List[float], u: float) -> List[float]:
        s = self._smoothstep(u)
        return [(1.0 - s) * ai + s * bi for ai, bi in zip(a, b)]

    def tick(self, force_retreat: bool = False) -> Tuple[str, List[float]]:
        self.phase_t += self.dt
        self.shake_timer += self.dt

        # 1. HANDLE SAFETY RETREAT (Unbreakable Mode)
        if force_retreat:
            # Interrupt animation. Pull 'Shoulder' back and 'Elbow' up.
            # We don't advance the keyframe index. We just output a safe pose.
            retreat_pose = list(self.current_pose)
            retreat_pose[1] -= 0.05 * self.dt * 10.0 # Pull shoulder back
            retreat_pose[2] -= 0.05 * self.dt * 10.0 # Lift elbow up
            self.current_pose = retreat_pose
            self.phase = "retreating"
            return ("RETREAT_INTERRUPT", self.current_pose[:])

        # 2. NORMAL ANIMATION
        cur = self.keyframes[self.k]
        
        # Sticky Mode: Shake only during 'dump' hold phase
        is_shaking = (self.scenario == SCENARIO_STICKY and cur.name == "dump" and self.phase == "hold")

        if self.phase == "hold":
            # Holding position
            base_pose = cur.positions[:]
            self.current_pose = self._apply_shake(base_pose) if is_shaking else base_pose
            
            if self.phase_t >= cur.hold_s:
                self.phase = "move"
                self.phase_t = 0.0
                self.start_pose = self.current_pose[:] # Start from where we left off (even if shaken)
                
                # Advance Frame
                self.k = (self.k + 1) % len(self.keyframes)
                
                # If we wrapped around to 0, regenerate random targets for Adaptive Mode
                if self.k == 0:
                    self.keyframes = self._generate_cycle_keyframes()
                
                self.target_pose = self.keyframes[self.k].positions[:]
        
        else: # Moving
            u = self.phase_t / max(1e-6, float(self.cfg.transition_s))
            self.current_pose = self._lerp(self.start_pose, self.target_pose, u)
            
            if self.phase_t >= self.cfg.transition_s:
                self.phase = "hold"
                self.phase_t = 0.0

        state_name = cur.name if self.phase == "hold" else f"moving_to_{self.keyframes[self.k].name}"
        if is_shaking: state_name += "_SHAKING"
        
        return (state_name, self.current_pose[:])


# ============================================================
# 4) Scenario Inputs Generator
# ============================================================

def generate_scenario_inputs(elapsed_s: float, scenario: str) -> Tuple[str, float, float, bool]:
    """
    Returns: (pred_class, pred_conf, distance_m, STALL_FLAG)
    """
    stall_flag = False
    
    # UNBREAKABLE SCENARIO: Trigger a stall at t=12s for 2 seconds
    if scenario == SCENARIO_UNBREAKABLE:
        if 12.0 < elapsed_s < 14.5:
            stall_flag = True

    # Perception is always "good" to keep the arm moving, 
    # unless stall triggers the safety stop.
    return ("clean_regolith", 0.98, 0.8, stall_flag)


# ============================================================
# 5) Main Runtime
# ============================================================

class ArmJointStatePublisher:
    """ROS2 Publisher (same as before)"""
    def __init__(self, cfg: ArmConfig):
        if not HAS_ROS2: raise RuntimeError("ROS2 not available.")
        self.cfg = cfg
        rclpy.init(args=None)
        self.node = rclpy.create_node("alam_arm_demo")
        self.pub = self.node.create_publisher(JointState, "/joint_states", 10)
    def publish(self, pose: List[float]):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = list(pose)
        self.pub.publish(msg)
    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force-scenario", type=str, default=None, help="stall, sticky, or adaptive")
    p.add_argument("--no-arm", action="store_true")
    p.add_argument("--hz", type=float, default=30.0)
    return p.parse_args()

def main():
    args = parse_args()
    
    # 1. Pick Scenario
    scenario = pick_scenario(args.force_scenario)
    print("\n=======================================================")
    print(f"   SELECTED SCENARIO:  {scenario}")
    print("=======================================================")
    print(" Check RViz for visual confirmation.")
    print(" Check console for 'SAFETY OVERRIDE' logs.")
    print("=======================================================\n")
    time.sleep(2.0) # Let user read

    # 2. Setup Components
    ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())
    
    log_path = Path("/mnt/c/Users/kaddo/Desktop/rover_learner/logs") / f"demo_{scenario.split()[0]}_{int(time.time())}.csv"
    logger = CSVDecisionLogger(log_path)
    
    cfg = ArmConfig()
    animator = PickPlaceAnimator(cfg, scenario)
    arm_pub = None if args.no_arm else ArmJointStatePublisher(cfg)

    # 3. Main Loop
    start_t = time.time()
    period = 1.0 / args.hz
    
    try:
        while True:
            iter_start = time.time()
            elapsed = iter_start - start_t
            if elapsed > 30.0: break # Demo runs for 30s

            # A. Generate Inputs
            s_class, s_conf, s_dist, s_stall = generate_scenario_inputs(elapsed, scenario)
            
            # B. Decision Step
            # We treat 'stall' as valid telemetry for the Supervisor
            inp = StepInputs(
                perception=Perception(s_class, s_conf),
                distance_m=s_dist,
                telemetry=Telemetry(stall_flag=s_stall)
            )
            out = step_with_safety(ctrl, inp)

            # C. Map to Arm Animation
            force_retreat = (out.final_action in [RoverAction.RETREAT, RoverAction.STOP])
            
            # If "Safety Stop" is active, we tell animator to RETREAT/FREEZE
            # If "Scoop" is active, we tell animator to tick normally
            arm_status, raw_pose = animator.tick(force_retreat=force_retreat)
            
            pose, _ = clamp_pose(cfg, raw_pose)

            # D. Publish
            if arm_pub: arm_pub.publish(pose)

            # E. Logging & Display
            print(f"[{elapsed:04.1f}s] Act: {out.final_action:<10} | Arm: {arm_status:<20} | Stall: {s_stall}")
            
            logger.log(DecisionFrame(
                ts=iter_start,
                frame_id=int(elapsed*args.hz),
                pred_class=s_class, pred_conf=s_conf, distance_m=s_dist,
                proposed_action=out.proposed_action, final_action=out.final_action,
                reason=out.reason, safety_json=CSVDecisionLogger.safety_to_json({"scenario": scenario})
            ))

            dt = time.time() - iter_start
            if dt < period: time.sleep(period - dt)

    except KeyboardInterrupt:
        pass
    finally:
        if arm_pub: arm_pub.shutdown()
        logger.close()
        print("\nDemo Complete.")

if __name__ == "__main__":
    main()