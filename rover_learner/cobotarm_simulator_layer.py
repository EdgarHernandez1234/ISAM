#!/usr/bin/env python3
"""
cobotarm_simulator_layer.py (rover_learner)

Desktop-only "arm simulator bridge" for ALAM/ISAM demos.

WHY THIS EXISTS
---------------
You can run live perception (camera + lidar + ML + RL safety) on Jetson,
but run RViz myCobot arm simulation on a Windows desktop (WSL2) instead.

This file provides a modular "simulator layer" that:
  1) Checks camera_provider + lidar_provider health/online status
  2) Runs the existing core step_with_safety() decision pipeline
  3) Applies an "arm safety layer" (joint limit clamp + hazard veto behavior)
  4) Publishes /joint_states for RViz to visualize smooth pick/place motion
  5) Logs decisions + arm-phase meta into logger CSV (without changing logger schema)

Design principles:
  - Keep business logic pure / unit-testable where possible.
  - Isolate ROS2 publishing in a small class (ArmJointStatePublisher).
  - Make it runnable as a standalone demo script.

IMPORTANT OPERATIONAL RULE
--------------------------
Only ONE publisher should publish /joint_states at a time:
  - Either joint_state_publisher_gui
  - Or this script
  - Or a slider_control stack

If two publishers run simultaneously, RViz will "snap"/jitter.

REFERENCES (existing rover_learner modules):
  - core.step_with_safety(): policy + safety supervisor pipeline
  - rl_safety_supervisor.py: Safety supervisor + Policy placeholder
  - camera_provider.py / lidar_provider.py: live inputs
  - logger.py: CSV logging

"""

from __future__ import annotations

import argparse
import json
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

# These names MUST match what your RViz robot description expects.
# You already confirmed these via `ros2 topic echo /joint_states --once`.
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
    """
    Simple joint limit structure.

    NOTE:
    - These limits are conservative placeholders. Update them to match the myCobot URDF
      or your real joint soft-limits once you confirm them.
    - Values are radians.
    """
    lo: float
    hi: float


@dataclass
class ArmConfig:
    """
    Configuration for the simulator layer.
    """
    publish_hz: float = 50.0           # smoother RViz motion
    transition_s: float = 1.0          # seconds between keyframes
    # Conservative generic limits. Adjust later to your robot specs.
    limits: Dict[str, JointLimits] = None  # filled in __post_init__

    def __post_init__(self):
        if self.limits is None:
            # These are intentionally conservative "won't fold into itself" ranges.
            self.limits = {
                JOINT_NAMES[0]: JointLimits(lo=-2.2, hi=2.2),  # base
                JOINT_NAMES[1]: JointLimits(lo=-1.6, hi=1.6),  # shoulder
                JOINT_NAMES[2]: JointLimits(lo=-1.6, hi=1.9),  # elbow
                JOINT_NAMES[3]: JointLimits(lo=-2.2, hi=2.2),  # wrist pitch
                JOINT_NAMES[4]: JointLimits(lo=-2.2, hi=2.2),  # wrist roll
                JOINT_NAMES[5]: JointLimits(lo=-2.8, hi=2.8),  # tool roll
            }


@dataclass(frozen=True)
class Keyframe:
    """
    A named joint pose and a hold duration.
    """
    name: str
    positions: List[float]  # length == len(JOINT_NAMES)
    hold_s: float


def clamp_pose(cfg: ArmConfig, pose: List[float]) -> Tuple[List[float], float]:
    """
    Clamp a pose to joint limits and compute a simple 'joint error norm'
    (how much we had to clamp).

    Returns:
      (clamped_pose, clamp_error_norm)
    """
    clamped = []
    sq = 0.0
    for name, val in zip(JOINT_NAMES, pose):
        lim = cfg.limits[name]
        v2 = max(lim.lo, min(lim.hi, float(val)))
        clamped.append(v2)
        sq += (v2 - float(val)) ** 2
    return clamped, math.sqrt(sq)


# ============================================================
# 2) ROS2 publisher for /joint_states (desktop sim)
# ============================================================

class ArmJointStatePublisher:
    """
    ROS2 publisher that drives RViz by publishing JointState messages to /joint_states.

    This is NOT a motor controller. It is visualization-level simulation.
    """

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
# 3) Arm "motion script" (pick + place animation)
# ============================================================

class PickPlaceAnimator:
    """
    Generates a smooth pick/place loop from keyframes.

    How it works:
      - "hold" at a keyframe pose for hold_s
      - "move" to the next keyframe pose over transition_s using smoothstep interpolation
    """

    def __init__(self, cfg: ArmConfig):
        self.cfg = cfg
        self.dt = 1.0 / max(1.0, float(cfg.publish_hz))
        self.phase = "hold"
        self.phase_t = 0.0
        self.k = 0

        # Keyframes: [base, shoulder, elbow, wrist_pitch, wrist_roll, tool_roll]
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
        """
        Advances the animation by one time step.

        Returns:
          (phase_name, pose)
        """
        self.phase_t += self.dt
        cur = self.keyframes[self.k]

        if self.phase == "hold":
            self.current_pose = cur.positions[:]
            if self.phase_t >= cur.hold_s:
                # transition to next keyframe
                self.phase = "move"
                self.phase_t = 0.0
                self.start_pose = cur.positions[:]
                self.k = (self.k + 1) % len(self.keyframes)
                self.target_pose = self.keyframes[self.k].positions[:]

        else:  # move
            u = self.phase_t / max(1e-6, float(self.cfg.transition_s))
            self.current_pose = self._lerp(self.start_pose, self.target_pose, u)
            if self.phase_t >= self.cfg.transition_s:
                self.phase = "hold"
                self.phase_t = 0.0

        return (cur.name if self.phase == "hold" else f"move_to_{self.keyframes[self.k].name}",
                self.current_pose[:])


# ============================================================
# 4) Hypothetical rover_learner ↔ arm simulator integration
# ============================================================

def check_camera_online(cam) -> None:
    """
    camera_provider.py integration point:
      - verifies one camera is online and producing frames
    """
    frame, ts = cam.read()
    # We don't do any heavy work here; just sanity check.
    print(f"[OK] camera online: ts={ts:.3f} frame_type={type(frame)}")


def check_lidar_online(lidar) -> None:
    """
    lidar_provider.py integration point:
      - verifies lidar is online and producing a distance estimate
    """
    t0 = time.time()
    while time.time() - t0 < 3.0:
        d = lidar.get_distance_m()
        if d is not None:
            print(f"[OK] lidar online: distance_m={d:.3f}")
            return
        time.sleep(0.1)
    raise RuntimeError("LiDAR did not produce distance within 3 seconds.")


def build_arm_telemetry_from_clamp(clamp_error_norm: float) -> Telemetry:
    """
    rl_safety_supervisor.py integration point:
      - In real hardware, telemetry would come from encoders / controller.
      - In desktop sim, we approximate a 'joint_error_norm' using clamp magnitude.

    The Safety Supervisor can learn over time / log these conditions,
    but in this demo we just feed it in deterministically each step.
    """
    return Telemetry(
        joint_error_norm=clamp_error_norm if clamp_error_norm > 0.0 else 0.0,
        motor_current_a=None,
        stall_flag=None,
        health_score=None,
    )


def action_to_arm_mode(final_action: str) -> str:
    """
    Maps rover_learner final action -> what the arm simulator should do.

    You can extend this mapping as your policy evolves.
    """
    a = str(final_action).upper()
    if a == RoverAction.SCOOP:
        return "PICK_PLACE_LOOP"
    if a in (RoverAction.BYPASS, RoverAction.HOLD, RoverAction.APPROACH):
        return "HOLD_HOME"
    if a in (RoverAction.STOP, RoverAction.RETREAT, RoverAction.RETURN_HOME, RoverAction.DEGRADED):
        return "SAFE_HOME"
    return "SAFE_HOME"


# ============================================================
# 5) Main demo runner
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ALAM cobot arm simulator bridge demo")

    # Desktop demo defaults: mock camera + mock lidar
    p.add_argument("--camera", choices=["mock", "usb", "csi"], default="mock")
    p.add_argument("--usb-index", type=int, default=0)
    p.add_argument("--sensor-id", type=int, default=0)

    p.add_argument("--lidar", choices=["mock", "ros2"], default="mock")
    p.add_argument("--lidar-topic", type=str, default="/scan")

    p.add_argument("--hz", type=float, default=2.0, help="Decision loop rate (Hz)")
    p.add_argument("--num-steps", type=int, default=60)

    p.add_argument("--log-dir", type=str, default=str(Path(__file__).resolve().parent / "logs"))
    p.add_argument("--check-only", action="store_true", help="Only run camera/lidar checks, then exit.")
    p.add_argument("--no-arm", action="store_true", help="Run decision+logging without publishing /joint_states.")

    return p.parse_args()


def make_camera(args):
    if args.camera == "mock":
        return MockCameraProvider()
    if args.camera == "usb":
        return USBCameraProvider(index=args.usb_index)
    return CSICameraProvider(sensor_id=args.sensor_id)


def make_lidar(args):
    if args.lidar == "mock":
        return MockLidarProvider(distance_m=2.0)
    p = ROS2LaserScanProvider(topic=args.lidar_topic)
    p.start()
    return p


def main() -> None:
    args = parse_args()

    # 1) Bring up providers (camera/lidar)
    cam = make_camera(args)
    lidar = make_lidar(args)

    try:
        # 2) Online checks
        check_camera_online(cam)
        check_lidar_online(lidar)

        if args.check_only:
            print("[DONE] check-only mode.")
            return

        # 3) RL controller (policy + supervisor)
        ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())

        # 4) Logger (reuse existing schema; store arm meta in safety_json)
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        out_csv = log_dir / f"cobotarm_sim_demo_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        logger = CSVDecisionLogger(out_csv)

        # 5) Arm publisher + animator
        cfg = ArmConfig()
        arm_pub = None if args.no_arm else ArmJointStatePublisher(cfg)
        animator = PickPlaceAnimator(cfg)

        # Start at home pose
        home_pose, clamp_err = clamp_pose(cfg, animator.keyframes[0].positions)
        if arm_pub is not None:
            arm_pub.publish(home_pose)

        period = 1.0 / max(0.1, float(args.hz))

        print("\n[RUN] Starting decision loop. "
              "RViz should already be open with Fixed Frame = g_base.\n"
              "Rule: do NOT run joint_state_publisher_gui while this script runs.\n")

        for frame_id in range(int(args.num_steps)):
            t0 = time.time()

            # In a real integration:
            # - pred_class/pred_conf comes from your model inference on cam frame.
            # For this desktop demo, we mock a simple alternating perception.
            frame, ts = cam.read()

            # Minimal mock perception (replace with real ML output when wired)
            looks_clean = (frame_id % 2 == 0)
            pred_class = "clean" if looks_clean else "dirty"
            pred_conf = 0.90 if looks_clean else 0.92

            distance_m = lidar.get_distance_m()

            # 6) Decide using the rover_learner safety pipeline
            inp = StepInputs(
                perception=Perception(pred_class=pred_class, pred_conf=float(pred_conf)),
                distance_m=distance_m,
                telemetry=Telemetry(),  # will be replaced below once we compute clamp telemetry
            )

            out = step_with_safety(ctrl, inp)

            # 7) Map final action -> arm mode
            arm_mode = action_to_arm_mode(out.final_action)

            # 8) Step the arm animation (only when SCOOP is allowed)
            arm_phase = "idle"
            pose = home_pose

            if arm_mode == "PICK_PLACE_LOOP":
                arm_phase, raw_pose = animator.tick()
                pose, clamp_err = clamp_pose(cfg, raw_pose)

            elif arm_mode in ("SAFE_HOME", "HOLD_HOME"):
                # deterministically go/stay home
                arm_phase = "home_hold"
                pose = home_pose
                clamp_err = 0.0

            # 9) Hypothetical "arm safety telemetry" back into supervisor logging
            # (If you want: you can rerun step_with_safety with telemetry here.
            # For now we embed it in the log meta to keep the demo deterministic.)
            arm_telemetry = build_arm_telemetry_from_clamp(clamp_err)

            # 10) Publish to RViz
            if arm_pub is not None:
                arm_pub.publish(pose)
                arm_pub.spin_once(0.0)

            # 11) Log (arm meta is embedded into safety_json)
            safety_meta = {
                "decision": out.signals,  # existing safety supervisor signals dict
                "arm": {
                    "mode": arm_mode,
                    "phase": arm_phase,
                    "joint_names": JOINT_NAMES,
                    "joint_positions_rad": pose,
                    "clamp_error_norm": arm_telemetry.joint_error_norm,
                },
            }
            logger.log(DecisionFrame(
                ts=ts,
                frame_id=frame_id,
                pred_class=pred_class,
                pred_conf=float(pred_conf),
                distance_m=None if distance_m is None else float(distance_m),
                proposed_action=out.proposed_action,
                final_action=out.final_action,
                reason=out.reason,
                safety_json=CSVDecisionLogger.safety_to_json(safety_meta),
            ))

            print(f"[step={frame_id:03d}] class={pred_class} conf={pred_conf:.2f} dist={distance_m} "
                  f"proposed={out.proposed_action} final={out.final_action} arm={arm_mode}/{arm_phase}")

            # Rate control
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

        print(f"\n[DONE] log written to: {out_csv}")

    finally:
        # Cleanup
        try:
            if hasattr(lidar, "close"):
                lidar.close()
        except Exception:
            pass
        try:
            if hasattr(cam, "close"):
                cam.close()
        except Exception:
            pass
        try:
            if "logger" in locals():
                logger.close()  # type: ignore
        except Exception:
            pass
        try:
            if "arm_pub" in locals() and arm_pub is not None:
                arm_pub.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
