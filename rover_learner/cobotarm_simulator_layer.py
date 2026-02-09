#!/usr/bin/env python3
"""
cobotarm_simulator_layer.py (rover_learner)

Desktop-only "arm simulator bridge" for ALAM/ISAM demos.

UPDATED FIX:
  - Replaced random frame toggling with a structured "Scenario Generator."
  - Sequence: Search (15s) -> Harvest (20s) -> Return Home (Safety Trigger).
  - Ensures continuous 'SCOOP' signals so the arm animation can actually complete.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# 1) Arm kinematics interface for RViz simulation
# ============================================================

JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
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
            self.limits = {
                JOINT_NAMES[0]: JointLimits(lo=-2.2, hi=2.2),
                JOINT_NAMES[1]: JointLimits(lo=-1.6, hi=1.6),
                JOINT_NAMES[2]: JointLimits(lo=-1.6, hi=1.9),
                JOINT_NAMES[3]: JointLimits(lo=-2.2, hi=2.2),
                JOINT_NAMES[4]: JointLimits(lo=-2.2, hi=2.2),
                JOINT_NAMES[5]: JointLimits(lo=-2.8, hi=2.8),
            }


@dataclass(frozen=True)
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
# 2) ROS2 publisher
# ============================================================

class ArmJointStatePublisher:
    def __init__(self, cfg: ArmConfig):
        if not HAS_ROS2:
            raise RuntimeError("ROS2 (rclpy) not available. Install ROS2 Humble on this machine.")
        self.cfg = cfg
        rclpy.init(args=None)

        class _Node(Node):
            def __init__(self):
                super().__init__("alam_arm_jointstate_publisher")
                self.pub = self.create_publisher(JointState, "/joint_states", 10)

        self.node = _Node()
        self.last_pose = [0.0] * len(JOINT_NAMES)

    def publish(self, pose: List[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = list(pose)
        self.node.pub.publish(msg)
        self.last_pose = list(pose)

    def spin_once(self, timeout_s: float = 0.0) -> None:
        rclpy.spin_once(self.node, timeout_sec=timeout_s)

    def shutdown(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


# ============================================================
# 3) Arm "motion script"
# ============================================================

class PickPlaceAnimator:
    def __init__(self, cfg: ArmConfig):
        self.cfg = cfg
        self.dt = 1.0 / max(1.0, float(cfg.publish_hz))
        self.phase = "hold"
        self.phase_t = 0.0
        self.k = 0

        # Animation Sequence
        self.keyframes = [
            Keyframe("home",              [ 0.0, -0.3,  1.0, -0.7, -0.3,  0.0], 0.6),
            Keyframe("approach_regolith", [ 0.2,  0.1,  1.25, -1.15,-0.4,  0.0], 0.4),
            Keyframe("scoop_down",        [ 0.2,  0.25, 1.45, -1.35,-0.4,  0.0], 0.7),
            Keyframe("lift",              [ 0.2, -0.15, 1.10, -0.95,-0.4,  0.0], 0.5),
            Keyframe("move_to_drop",      [-0.6, -0.10, 1.05, -0.90,-0.4,  0.0], 0.4),
            Keyframe("dump",              [-0.6,  0.05, 0.95, -0.60,-0.4,  0.9], 0.7),
            Keyframe("return_home",       [ 0.0, -0.3,  1.0, -0.7, -0.3,  0.0], 0.6),
        ]

        self.start_pose = self.keyframes[0].positions[:]
        self.target_pose = self.keyframes[0].positions[:]
        self.current_pose = self.keyframes[0].positions[:]

    def _smoothstep(self, u: float) -> float:
        u = max(0.0, min(1.0, float(u)))
        return u * u * (3.0 - 2.0 * u)

    def _lerp(self, a: List[float], b: List[float], u: float) -> List[float]:
        s = self._smoothstep(u)
        return [(1.0 - s) * ai + s * bi for ai, bi in zip(a, b)]

    def tick(self) -> Tuple[str, List[float]]:
        self.phase_t += self.dt
        cur = self.keyframes[self.k]

        if self.phase == "hold":
            self.current_pose = cur.positions[:]
            if self.phase_t >= cur.hold_s:
                self.phase = "move"
                self.phase_t = 0.0
                self.start_pose = cur.positions[:]
                self.k = (self.k + 1) % len(self.keyframes)
                self.target_pose = self.keyframes[self.k].positions[:]
        else:
            u = self.phase_t / max(1e-6, float(self.cfg.transition_s))
            self.current_pose = self._lerp(self.start_pose, self.target_pose, u)
            if self.phase_t >= self.cfg.transition_s:
                self.phase = "hold"
                self.phase_t = 0.0

        return (cur.name if self.phase == "hold" else f"move_to_{self.keyframes[self.k].name}",
                self.current_pose[:])


# ============================================================
# 4) Scenario Generator (The Script)
# ============================================================

def generate_scenario_inputs(elapsed_s: float) -> Tuple[str, float, float, float]:
    """
    Returns: (pred_class, pred_conf, distance_m, health_score)
    """
    # PHASE 1: SEARCHING (0s - 15s)
    # Rover is looking around. Seeing mostly 'dirt' or 'unknown'. 
    # Distance is far.
    if elapsed_s < 15.0:
        return ("dirty", 0.85, 4.0, 1.0)

    # PHASE 2: HARVESTING (15s - 35s)
    # Rover found 'safe regolith'. 
    # Class is 'clean', confidence high, distance is perfect for scooping (0.8m).
    # This continuous 20s window allows the arm to loop fully.
    if elapsed_s < 35.0:
        return ("clean_regolith", 0.98, 0.8, 0.95)

    # PHASE 3: RETURN HOME (35s+)
    # Mission complete or battery low.
    # We simulate a health score drop to force the SafetySupervisor to trigger 'RETURN_HOME'.
    return ("clean_regolith", 0.98, 0.8, 0.25) # Health=0.25 triggers Critical Return


# ============================================================
# 5) Main Demo Runner
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ALAM cobot arm simulator bridge demo")
    p.add_argument("--camera", choices=["mock", "usb", "csi"], default="mock")
    p.add_argument("--lidar", choices=["mock", "ros2"], default="mock")
    p.add_argument("--hz", type=float, default=20.0, help="Simulation Hz (higher is smoother)")
    p.add_argument("--log-dir", type=str, default=str(Path(__file__).resolve().parent / "logs"))
    p.add_argument("--no-arm", action="store_true", help="Run without publishing /joint_states")
    return p.parse_args()


def make_camera(args):
    if args.camera == "mock": return MockCameraProvider()
    if args.camera == "usb": return USBCameraProvider()
    return CSICameraProvider()


def make_lidar(args):
    if args.lidar == "mock": return MockLidarProvider(distance_m=2.0)
    p = ROS2LaserScanProvider()
    p.start()
    return p


def main() -> None:
    args = parse_args()
    cam = make_camera(args)
    lidar = make_lidar(args)

    # Controller Setup
    ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())

    # Logger Setup
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_csv = log_dir / f"cobotarm_sim_demo_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    logger = CSVDecisionLogger(out_csv)

    # Arm Setup
    cfg = ArmConfig()
    arm_pub = None if args.no_arm else ArmJointStatePublisher(cfg)
    animator = PickPlaceAnimator(cfg)

    # Initial Pose
    home_pose, _ = clamp_pose(cfg, animator.keyframes[0].positions)
    if arm_pub:
        arm_pub.publish(home_pose)

    print(f"\n[DEMO STARTED] Logging to {out_csv}")
    print("---------------------------------------------------------------")
    print(" 0s - 15s: SEARCHING (Arm Home)")
    print("15s - 35s: HARVESTING (Arm Moving)")
    print("35s +    : RETURN HOME (Safety Trigger)")
    print("---------------------------------------------------------------\n")

    start_time = time.time()
    period = 1.0 / max(0.1, float(args.hz))

    try:
        while True:
            iter_start = time.time()
            elapsed = iter_start - start_time
            
            # End demo after 40 seconds
            if elapsed > 45.0:
                break

            # 1. GENERATE SCENARIO INPUTS
            # We ignore the actual camera/lidar for the narrative logic, 
            # but we still read them to keep the drivers alive if they were real.
            _ = cam.read() 
            
            s_class, s_conf, s_dist, s_health = generate_scenario_inputs(elapsed)

            # 2. RUN DECISION STACK
            inp = StepInputs(
                perception=Perception(pred_class=s_class, pred_conf=s_conf),
                distance_m=s_dist,
                telemetry=Telemetry(health_score=s_health)
            )

            out = step_with_safety(ctrl, inp)

            # 3. MAP DECISION TO ARM MOTION
            if out.final_action == RoverAction.SCOOP:
                arm_phase, raw_pose = animator.tick()
                pose, clamp_err = clamp_pose(cfg, raw_pose)
                arm_mode = "PICK_PLACE_LOOP"
            else:
                # If we are BYPASS (Searching) or RETURN_HOME, go to Home Pose.
                pose = home_pose
                clamp_err = 0.0
                arm_mode = "HOME_HOLD"
                arm_phase = "idle"

            # 4. PUBLISH TO RVIZ
            if arm_pub:
                arm_pub.publish(pose)
                arm_pub.spin_once(0.0)

            # 5. LOGGING
            safety_meta = {
                "decision": out.signals,
                "arm": {
                    "mode": arm_mode, 
                    "phase": arm_phase,
                    "pose": [round(x, 2) for x in pose]
                }
            }
            
            logger.log(DecisionFrame(
                ts=iter_start,
                frame_id=int(elapsed * args.hz),
                pred_class=s_class,
                pred_conf=s_conf,
                distance_m=s_dist,
                proposed_action=out.proposed_action,
                final_action=out.final_action,
                reason=out.reason,
                safety_json=CSVDecisionLogger.safety_to_json(safety_meta),
            ))

            # Console Feedback (throttled)
            if int(elapsed * 10) % 10 == 0:
                print(f"[{elapsed:04.1f}s] Input: {s_class}/{s_dist}m | Dec: {out.final_action} | Arm: {arm_phase}")

            # Rate Limit
            dt = time.time() - iter_start
            if dt < period:
                time.sleep(period - dt)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[DONE] Simulation ended.")
        logger.close()
        if arm_pub:
            arm_pub.shutdown()
        if hasattr(lidar, "close"):
            lidar.close()
        if hasattr(cam, "close"):
            cam.close()

if __name__ == "__main__":
    main()