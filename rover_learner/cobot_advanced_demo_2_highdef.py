#!/usr/bin/env python3
"""
cobot_advanced_demo_2.py (rover_learner)

Integrated Lab Demo with "Active Idle" animations.
- SCENARIOS: UNBREAKABLE (Defensive), STICKY (Wiggle), ADAPTIVE (Dance).
- UI: Real-time HUD + MP4 Recording.
"""

from __future__ import annotations
import argparse
import math
import time
import random
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# rover_learner core modules
from .core import StepInputs, Perception, Telemetry, step_with_safety
from .rl_safety_supervisor import (
    HeuristicPolicy, SafetySupervisor, ShieldedController, RoverAction
)
from .logger import CSVDecisionLogger, DecisionFrame
from .camera_provider import CSICameraProvider, USBCameraProvider, MockCameraProvider
from .lidar_provider import ROS2LaserScanProvider, MockLidarProvider
from .demo_decider import make_model, overlay_text

JOINT_NAMES = [
    "joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
    "joint5_to_joint4", "joint6_to_joint5", "joint6output_to_joint6",
]

SCENARIO_UNBREAKABLE = "UNBREAKABLE (Active Defense)"
SCENARIO_STICKY      = "STICKY (Continuous Wiggle)"
SCENARIO_ADAPTIVE    = "ADAPTIVE (Discovery Dance)"

class ActiveLabAnimator:
    """Animates the arm with scenario-specific 'dances' to ensure constant movement."""
    def __init__(self, scenario: str, hz: float = 30.0):
        self.scenario = scenario
        self.dt = 1.0 / hz
        self.t = 0.0  # Master clock for sine waves
       
        # Base Poses [base, shoulder, elbow, wrist_p, wrist_r, tool]
        self.home_pose = [0.0, -0.3, 1.0, -0.7, -0.3, 0.0]
        self.current_pose = list(self.home_pose)

    def tick(self, force_retreat: bool = False) -> Tuple[str, List[float]]:
        self.t += self.dt
       
        # 1. PRIORITY: Safety Retreat (LiDAR or Stall trigger)
        if force_retreat:
            # Move shoulder back and elbow up significantly
            self.current_pose[1] = -0.8 + (0.1 * math.sin(self.t * 10)) # Defensive oscillation
            self.current_pose[2] = 0.5
            return ("SAFETY_RETREAT_ACTIVE", self.current_pose)

        # 2. SCENARIO-SPECIFIC IDLE MOTIONS
        new_pose = list(self.home_pose)

        if self.scenario == SCENARIO_STICKY:
            # Wiggle joints 4 and 5 (Wrist) rapidly
            status = "WIGGLE_IDLE"
            new_pose[4] += 0.4 * math.sin(self.t * 12.0)
            new_pose[5] += 0.8 * math.sin(self.t * 20.0)

        elif self.scenario == SCENARIO_ADAPTIVE:
            # "Discovery Dance": Slow sweeping of the base and shoulder
            status = "DANCE_IDLE"
            new_pose[0] += 0.6 * math.sin(self.t * 1.5) # Pan base
            new_pose[1] += 0.2 * math.cos(self.t * 2.0) # Bob shoulder
            new_pose[5] += 1.5 * math.sin(self.t * 5.0) # Spin tool

        else: # UNBREAKABLE
            # "Breathing" motion: Subtle movement of the whole arm
            status = "BREATHING_IDLE"
            new_pose[1] += 0.05 * math.sin(self.t * 1.0)
            new_pose[2] += 0.05 * math.cos(self.t * 1.0)

        self.current_pose = new_pose
        return (status, self.current_pose)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mock")
    parser.add_argument("--camera", choices=["csi", "usb", "mock"], default="csi")
    parser.add_argument("--lidar", choices=["ros2", "mock"], default="ros2")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    if not rclpy.ok(): rclpy.init()
    node = rclpy.create_node("cobot_advanced_demo_2")
    pub = node.create_publisher(JointState, "/joint_states", 10)
   
    scenario = random.choice([SCENARIO_UNBREAKABLE, SCENARIO_STICKY, SCENARIO_ADAPTIVE])
   
    # Init Hardware with safety checks
    try:
        if args.camera == "csi": cam = CSICameraProvider()
        elif args.camera == "usb": cam = USBCameraProvider()
        else: cam = MockCameraProvider()
       
        lidar = ROS2LaserScanProvider() if args.lidar == "ros2" else MockLidarProvider()
        if hasattr(lidar, "start"): lidar.start()
    except Exception as e:
        print(f"[ERROR] Hardware Init Failed: {e}. Falling back to mocks.")
        cam = MockCameraProvider()
        lidar = MockLidarProvider()

    model = make_model(args)
    ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())
    animator = ActiveLabAnimator(scenario)
   
    # Desktop Logging
    ts_str = time.strftime('%Y%m%d_%H%M%S')
    csv_path = Path("~/Desktop/").expanduser() / f"active_lab_log_{ts_str}.csv"
    vid_path = Path("~/Desktop/").expanduser() / f"active_lab_vid_{ts_str}.mp4"
    logger = CSVDecisionLogger(csv_path)

    video_writer = None
    start_t = time.time()
   
    try:
        print(f"\n[ACTIVE DEMO] Scenario: {scenario}\n")
        while rclpy.ok():
            elapsed = time.time() - start_t
            if elapsed > 60.0: break

            frame, ts = cam.read()
            pred_class, pred_conf = model.predict(frame)
            distance = lidar.get_distance_m()
           
            # Simulate a stall event for UNBREAKABLE scenario to show safety response
            is_stall = (scenario == SCENARIO_UNBREAKABLE and 15.0 < elapsed < 18.0)
           
            inp = StepInputs(
                perception=Perception(pred_class, pred_conf),
                distance_m=distance,
                telemetry=Telemetry(stall_flag=is_stall)
            )
            out = step_with_safety(ctrl, inp)

            # Determine if arm should enter 'Safety Retreat'
            hazard = (out.final_action in [RoverAction.RETREAT, RoverAction.STOP])
            status, pose = animator.tick(force_retreat=hazard)

            # Publish to RViz
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = JOINT_NAMES
            msg.position = [float(p) for p in pose]
            pub.publish(msg)

            # UI HUD
            hud = [
                f"LAB MODE: {scenario}",
                f"DIST: {f'{distance:.2f}m' if distance else 'SEARCHING...'}",
                f"SAFETY: {out.final_action}",
                f"ARM STATUS: {status}"
            ]
            overlay_text(frame, hud)

            if args.record:
                if video_writer is None:
                    h, w = frame.shape[:2]
                    video_writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*'XVID'), 20.0, (w, h))
                video_writer.write(frame)

            cv2.imshow("Active Lab Demo", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

            logger.log(DecisionFrame(ts=ts, frame_id=0, pred_class=pred_class, pred_conf=pred_conf, distance_m=distance,
                                   proposed_action=out.proposed_action, final_action=out.final_action,
                                   reason=out.reason, safety_json=f'{{"scenario": "{scenario}", "arm": "{status}"}}'))
            time.sleep(0.033)
           
    finally:
        logger.close()
        if video_writer: video_writer.release()
        if hasattr(lidar, "close"): lidar.close()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
