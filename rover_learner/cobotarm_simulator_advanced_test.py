#!/usr/bin/env python3
"""
cobotarm_simulator_advanced.py (rover_learner)

HYBRID SIMULATOR & LIVE DEMO BRIDGE

Modes:
  1. SIMULATION (--force-scenario X): Runs deterministic scenarios.
  2. LIVE (--live): Uses REAL Camera + REAL LiDAR + REAL YOLO to drive the simulated arm in RViz.
"""

from __future__ import annotations

import argparse
import math
import time
import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Imports for AI & Display ---
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None 

try:
    import cv2
except ImportError:
    cv2 = None

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

# These MUST match the joint names in your ~/my_robot.urdf file
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
    publish_hz: float = 30.0
    transition_s: float = 1.0
    limits: Dict[str, JointLimits] = None

    def __post_init__(self):
        if self.limits is None:
            self.limits = {
                JOINT_NAMES[0]: JointLimits(lo=-3.14, hi=3.14),
                JOINT_NAMES[1]: JointLimits(lo=-3.14, hi=3.14),
                JOINT_NAMES[2]: JointLimits(lo=-3.14, hi=3.14),
                JOINT_NAMES[3]: JointLimits(lo=-3.14, hi=3.14),
                JOINT_NAMES[4]: JointLimits(lo=-3.14, hi=3.14),
                JOINT_NAMES[5]: JointLimits(lo=-3.14, hi=3.14),
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
# 2) Scenario Definitions (Simulated Mode)
# ============================================================

SCENARIO_UNBREAKABLE = "UNBREAKABLE"
SCENARIO_STICKY      = "STICKY"
SCENARIO_ADAPTIVE    = "ADAPTIVE"

def pick_scenario(force_name: Optional[str] = None) -> str:
    if force_name:
        if "stall" in force_name.lower(): return SCENARIO_UNBREAKABLE
        if "sticky" in force_name.lower(): return SCENARIO_STICKY
        if "adapt" in force_name.lower(): return SCENARIO_ADAPTIVE
    return random.choice([SCENARIO_UNBREAKABLE, SCENARIO_STICKY, SCENARIO_ADAPTIVE])


# ============================================================
# 3) Advanced Animator
# ============================================================

class PickPlaceAnimator:
    def __init__(self, cfg: ArmConfig, scenario: str):
        self.cfg = cfg
        self.scenario = scenario
        self.dt = 1.0 / max(1.0, float(cfg.publish_hz))
        
        self.phase = "hold"
        self.phase_t = 0.0
        self.k = 0
        self.shake_timer = 0.0

        # Base Keyframes (Updated for better visibility)
        self._base_keyframes = [
            Keyframe("home",      [ 0.0, -0.3,  1.0, -0.7, -0.3,  0.0], 0.5),
            Keyframe("approach",  [ 0.2,  0.1,  1.25, -1.15,-0.4,  0.0], 0.4),
            Keyframe("scoop",     [ 0.2,  0.25, 1.45, -1.35,-0.4,  0.0], 0.8),
            Keyframe("lift",      [ 0.2, -0.15, 1.10, -0.95,-0.4,  0.0], 0.5),
            Keyframe("dump",      [-0.6,  0.05, 0.95, -0.60,-0.4,  0.9], 0.8), 
            Keyframe("return",    [ 0.0, -0.3,  1.0, -0.7, -0.3,  0.0], 0.5),
        ]

        self.keyframes = self._generate_cycle_keyframes()
        self.start_pose = self.keyframes[0].positions[:]
        self.target_pose = self.keyframes[0].positions[:]
        self.current_pose = self.keyframes[0].positions[:]

    def _generate_cycle_keyframes(self) -> List[Keyframe]:
        frames = []
        for k in self._base_keyframes:
            frames.append(Keyframe(k.name, list(k.positions), k.hold_s))
        return frames

    def _smoothstep(self, u: float) -> float:
        u = max(0.0, min(1.0, float(u)))
        return u * u * (3.0 - 2.0 * u)

    def _lerp(self, a: List[float], b: List[float], u: float) -> List[float]:
        s = self._smoothstep(u)
        return [(1.0 - s) * ai + s * bi for ai, bi in zip(a, b)]

    def tick(self, force_retreat: bool = False, force_home: bool = False) -> Tuple[str, List[float]]:
        self.phase_t += self.dt
        
        # A. Safety Interrupts
        if force_retreat:
            # Simple retreat logic: pull back joint 2 and 3
            retreat_pose = list(self.current_pose)
            retreat_pose[1] -= 0.1 * self.dt # Pull back
            self.current_pose = retreat_pose
            return ("RETREAT_INTERRUPT", self.current_pose[:])

        if force_home:
            # Move towards home index 0
            self.target_pose = self._base_keyframes[0].positions[:]
            u = 0.1
            self.current_pose = self._lerp(self.current_pose, self.target_pose, u)
            return ("RETURNING_HOME", self.current_pose[:])

        # B. Normal Cycle
        cur = self.keyframes[self.k]
        if self.phase == "hold":
            self.current_pose = cur.positions[:]
            if self.phase_t >= cur.hold_s:
                self.phase = "move"
                self.phase_t = 0.0
                self.start_pose = self.current_pose[:]
                self.k = (self.k + 1) % len(self.keyframes)
                self.target_pose = self.keyframes[self.k].positions[:]
        else:
            u = self.phase_t / max(1e-6, float(self.cfg.transition_s))
            self.current_pose = self._lerp(self.start_pose, self.target_pose, u)
            if self.phase_t >= self.cfg.transition_s:
                self.phase = "hold"
                self.phase_t = 0.0

        return (cur.name, self.current_pose[:])

def generate_scenario_inputs(elapsed_s: float, scenario: str):
    return ("clean_regolith", 0.98, 0.8, False)


# ============================================================
# 5) Main Runtime
# ============================================================

class ArmJointStatePublisher:
    def __init__(self, cfg: ArmConfig):
        if not HAS_ROS2: raise RuntimeError("ROS2 not available.")
        self.cfg = cfg
        
        # --- CRITICAL FIX: CHECK IF ROS IS ALREADY RUNNING ---
        if not rclpy.ok():
            rclpy.init(args=None)
        
        self.node = rclpy.create_node("alam_arm_demo_pub")
        self.pub = self.node.create_publisher(JointState, "/joint_states", 10)

    def publish(self, pose: List[float]):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = list(pose)
        self.pub.publish(msg)

    def shutdown(self):
        try:
            self.node.destroy_node()
        except:
            pass
        # Do not call rclpy.shutdown() here if other parts of the script need it

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force-scenario", type=str, default=None)
    p.add_argument("--no-arm", action="store_true")
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--live", action="store_true", help="Use REAL Sensors")
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument("--camera-id", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    
    # 1. Logging Path
    if "WSL_DISTRO_NAME" in os.environ:
        base_dir = Path("/mnt/c/Users/kaddo/Desktop/rover_learner/logs")
    else:
        base_dir = Path.home() / "rover_learner" / "logs"
    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / f"demo_run_{int(time.time())}.csv"
    logger = CSVDecisionLogger(log_path)

    # 2. Setup Sensors
    camera = None
    lidar = None
    model = None
    
    if args.live:
        print("\n[SYSTEM] STARTING LIVE MODE...")
        
        # Init Camera (With Failsafe)
        try:
            print("[init] Opening CSI Camera...")
            camera = CSICameraProvider(sensor_id=args.camera_id, width=1280, height=720)
            # Force a read to ensure it works
            camera.read()
            print("[init] Camera OK.")
        except Exception as e:
            print(f"\n[WARN] Camera Init Failed: {e}")
            print("[TIP] Try running: sudo systemctl restart nvargus-daemon")
            print("[WARN] Switching to MOCK Camera to keep demo alive.\n")
            camera = MockCameraProvider()

        # Init LiDAR
        try:
            print("[init] Opening LiDAR...")
            lidar = ROS2LaserScanProvider() 
            lidar.start()
        except Exception as e:
             print(f"[WARN] LiDAR Failed ({e}), using Mock.")
             lidar = MockLidarProvider()

        # Init Model
        if YOLO:
            print(f"[init] Loading Model: {args.model}")
            model = YOLO(args.model)
        
        scenario = "LIVE_DECISION"
    else:
        scenario = pick_scenario(args.force_scenario)
        print(f"\n[SYSTEM] STARTING SIMULATION: {scenario}")

    # 3. Setup Arm & Controller
    ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())
    cfg = ArmConfig()
    animator = PickPlaceAnimator(cfg, scenario)
    arm_pub = None if args.no_arm else ArmJointStatePublisher(cfg)

    print(f"[INFO] Log: {log_path}")
    print("[INFO] Running... Check RViz!")

    # 4. Loop
    start_t = time.time()
    period = 1.0 / args.hz
    
    try:
        while True:
            iter_start = time.time()
            elapsed = iter_start - start_t
            
            # A. Input
            if args.live:
                frame, _ = camera.read()
                
                s_class, s_conf = "unknown", 0.0
                if model and frame is not None:
                    try:
                        res = model(frame, verbose=False)[0]
                        if res.probs:
                            s_class = res.names[int(res.probs.top1)]
                            s_conf = float(res.probs.top1conf)
                        elif res.boxes:
                             s_class = res.names[int(res.boxes[0].cls)]
                             s_conf = float(res.boxes[0].conf)
                    except:
                        pass
                
                s_dist = lidar.get_distance_m()
                s_stall = False 
            else:
                frame = None
                s_class, s_conf, s_dist, s_stall = generate_scenario_inputs(elapsed, scenario)
                if elapsed > 40.0: break

            # B. Decide
            inp = StepInputs(Perception(str(s_class), float(s_conf)), s_dist, Telemetry(stall_flag=s_stall))
            out = step_with_safety(ctrl, inp)

            # C. Animate
            force_retreat = (out.final_action in [RoverAction.RETREAT, RoverAction.STOP])
            force_home = (out.final_action in [RoverAction.BYPASS, RoverAction.RETURN_HOME])
            
            status, raw_pose = animator.tick(force_retreat, force_home)
            pose, _ = clamp_pose(cfg, raw_pose)

            # D. Publish
            if arm_pub: arm_pub.publish(pose)
            
            # E. Log & Display
            logger.log(DecisionFrame(iter_start, int(elapsed*args.hz), str(s_class), float(s_conf), s_dist, out.proposed_action, out.final_action, out.reason))
            
            print(f"[{elapsed:04.1f}s] {s_class} ({s_conf:.2f}) | Dist: {s_dist} | Act: {out.final_action}")

            if args.live and frame is not None and cv2:
                cv2.putText(frame, f"ACT: {out.final_action}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("Rover Eye", frame)
                if cv2.waitKey(1) == ord('q'): break

            dt = time.time() - iter_start
            if dt < period: time.sleep(period - dt)

    except KeyboardInterrupt:
        pass
    finally:
        if arm_pub: arm_pub.shutdown()
        logger.close()
        if hasattr(lidar, 'close'): lidar.close()
        if hasattr(camera, 'close'): camera.close()
        if cv2: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()