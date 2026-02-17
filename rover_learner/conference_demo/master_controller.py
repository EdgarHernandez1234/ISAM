#!/usr/bin/env python3
"""
master_controller.py
====================

Conference Demo Master Controller (Integrated):
  - power_saver_degradation.SystemHealthMonitor (policy)
  - mode_manager.ModeManager (gating/executor)
  - rl_safety_supervisor.ShieldedController (decision + safety)
  - Unified CSV + on-screen scrolling log

Folder intent (per your layout):
  rover_learner/
    conference_demo/
      master_controller.py   <-- THIS FILE
      mode_manager.py
      power_saver_degradation.py

Your provider modules live in rover_learner/ root (parent folder):
  rover_learner/lidar_provider.py
  rover_learner/camera_provider.py
  rover_learner/two_camera_provider.py
  rover_learner/rl_safety_supervisor.py

So we add the parent folder to sys.path before importing providers.
"""

from __future__ import annotations

import os
import sys
import time
import csv
import argparse
from datetime import datetime
from collections import deque
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import trick: allow conference_demo/ scripts to import rover_learner root
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Local conference_demo modules (policy + gating)
from power_saver_degradation import SystemHealthMonitor, SystemMode
from mode_manager import ModeManager

# Providers + safety supervisor (from rover_learner root)
try:
    from lidar_provider import SerialRPLidarProvider, ROS2LaserScanProvider
    from camera_provider import CSICameraProvider
    from two_camera_provider import TwoCameraProvider
    from rl_safety_supervisor import (
        ShieldedController, HeuristicPolicy, SafetySupervisor, Observation, RoverAction
    )
except ImportError as e:
    print(f"[ERROR] Could not import rover_learner modules: {e}")
    raise


# ============================================================
# Logging
# ============================================================

class SystemLogger:
    """
    Writes one CSV row per tick and maintains a short scrolling UI log.
    """
    def __init__(self, folder_path: str, max_ui_lines: int = 6):
        self.log_file = os.path.join(folder_path, "conference_master_log.csv")
        self.ui_lines = deque(maxlen=max_ui_lines)

        with open(self.log_file, mode="w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp",
                "policy_mode",
                "op_mode",
                "plan_cams",
                "plan_lidar_on",
                "res_scale",
                "fps_target",
                "cpu_temp_c",
                "ram_free_mb",
                "cpu_load_per_core",
                "cam_hb_ok",
                "lidar_hb_ok",
                "final_action",
                "reasons",
            ])

    def add_event(
        self,
        *,
        policy_mode: str,
        op_mode: str,
        plan_cams: int,
        plan_lidar_on: bool,
        res_scale: float,
        fps_target: int,
        cpu_temp_c: float,
        ram_free_mb: float,
        cpu_load_per_core: float,
        cam_hb_ok: bool,
        lidar_hb_ok: bool,
        final_action: str,
        reasons: str,
    ) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # UI line
        c = "CAM:OK" if cam_hb_ok else "CAM:LAG"
        l = "LIDAR:OK" if lidar_hb_ok else "LIDAR:LAG"
        plan = f"{plan_cams}C+{'L' if plan_lidar_on else 'noL'}"
        msg = f"[{ts}] {policy_mode}/{op_mode} {plan} -> {final_action} ({reasons or 'OK'})"
        self.ui_lines.append(msg)

        # CSV
        with open(self.log_file, mode="a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                ts,
                policy_mode,
                op_mode,
                plan_cams,
                plan_lidar_on,
                f"{res_scale:.2f}",
                fps_target,
                f"{cpu_temp_c:.1f}",
                f"{ram_free_mb:.0f}",
                f"{cpu_load_per_core:.2f}",
                cam_hb_ok,
                lidar_hb_ok,
                final_action,
                reasons,
            ])

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        # background strip
        cv2.rectangle(frame, (0, h - 130), (w, h), (0, 0, 0), -1)
        for i, line in enumerate(self.ui_lines):
            color = (0, 255, 0)
            if "DEGRADED" in line or "LAG" in line:
                color = (0, 165, 255)
            if "CRITICAL" in line or "RETURN_HOME" in line or "STOP" in line:
                color = (0, 0, 255)
            y = (h - 110) + (i * 20)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


# ============================================================
# Demo perception stub
# ============================================================

def infer_regolith_stub(frame: np.ndarray) -> tuple[str, float]:
    """
    Replace later with your ML model inference.

    For conference stability:
      - always returns a high-confidence "clean" classification
      - lets the safety supervisor decide SCOOP vs BYPASS based on distance + health
    """
    return "clean", 0.95


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="ALAM Conference Demo Master Controller")
    p.add_argument("--cam", choices=["single", "dual"], default="dual", help="Camera provider mode.")
    p.add_argument("--lidar", choices=["serial", "ros2", "none"], default="serial", help="LiDAR provider mode.")
    p.add_argument("--lidar-port", default="/dev/ttyUSB0", help="Serial port for LiDAR.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--window", default="ALAM Conference Demo Master")
    p.add_argument("--max-fps", type=int, default=30, help="Hard cap; policy may throttle lower.")
    p.add_argument("--ui-width", type=int, default=480, help="UI window width (pixels).")
    p.add_argument("--ui-height", type=int, default=360, help="UI window height (pixels).")
    return p.parse_args()


def main():
    args = parse_args()

    # Create a resizable window and force a consistent UI size.
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window, int(args.ui_width), int(args.ui_height))

    # -------------------- Init providers --------------------
    lidar = None
    if args.lidar == "serial":
        lidar = SerialRPLidarProvider(port=args.lidar_port)
    elif args.lidar == "ros2":
        lidar = ROS2LaserScanProvider()
    else:
        lidar = None

    if args.cam == "dual":
        cam_provider = TwoCameraProvider()
    else:
        cam_provider = CSICameraProvider(width=args.width, height=args.height)

    # -------------------- Init policy + gating + safety --------------------
    health_policy = SystemHealthMonitor()
    mode_mgr = ModeManager(cam_provider, lidar)

    brain = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())

    # Logger stores inside conference_demo/ for easy sharing after demos
    logger = SystemLogger(current_dir)

    # Heartbeat timestamps (0 means "not yet seen" → don't fail immediately)
    last_cam_ts = 0.0
    last_lidar_ts = 0.0

    # Initial plan (conservative until first sensor data arrives)
    plan = health_policy.get_plan(last_lidar_ts, last_cam_ts, camera_count_ok=2)
    mode_mgr.apply_plan(plan)

    try:
        while True:
            loop_start = time.time()

            # 1) Acquire sensor data (gated)
            fp = mode_mgr.get_frame()
            lp = mode_mgr.get_distance()

            # Update heartbeat timestamps when valid data arrives
            now = time.time()
            if fp.cam_a_ok:
                last_cam_ts = now
            if lp.ok:
                last_lidar_ts = now

            # 2) Policy: decide resource plan
            cam_count_ok = int(fp.cam_a_ok) + int(fp.cam_b_ok)
            plan = health_policy.get_plan(
                last_lidar_ts,
                last_cam_ts,
                camera_count_ok=cam_count_ok,
                lidar_seen_recently=lp.ok if args.lidar != "none" else None,
                cam_seen_recently=fp.cam_a_ok,
            )

            # 3) Apply plan to gating layer (will affect next read)
            mode_mgr.apply_plan(plan)

            # 4) Create observation + run safety supervisor
            pred_class, pred_conf = infer_regolith_stub(fp.frame)
            distance_m = lp.distance_m if plan.lidar_on else None
            obs = Observation.from_perception(pred_class, pred_conf, distance_m)
            obs.health_score = plan.health_score

            # If CRITICAL, override: return home/end demo
            if plan.mode == SystemMode.CRITICAL:
                final_action = RoverAction.RETURN_HOME
                reasons = "|".join(plan.reasons) if plan.reasons else "CRITICAL"
            else:
                decision = brain.step(obs)
                final_action = decision.final_action
                reasons = "|".join(plan.reasons) if plan.reasons else decision.reason

            # 5) Heartbeat flags for logging/UI (using policy timeouts)
            cam_hb_ok = (last_cam_ts > 0.0) and ((now - last_cam_ts) <= health_policy.cam_timeout_s)
            lidar_hb_ok = (args.lidar != "none") and (last_lidar_ts > 0.0) and ((now - last_lidar_ts) <= health_policy.lidar_timeout_s)

            logger.add_event(
                policy_mode=plan.mode.name,
                op_mode=mode_mgr.current_mode.name,
                plan_cams=plan.camera_count,
                plan_lidar_on=plan.lidar_on,
                res_scale=plan.res_scale,
                fps_target=plan.fps_target,
                cpu_temp_c=plan.cpu_temp_c,
                ram_free_mb=plan.ram_free_mb,
                cpu_load_per_core=plan.cpu_load_per_core,
                cam_hb_ok=cam_hb_ok,
                lidar_hb_ok=lidar_hb_ok,
                final_action=str(final_action),
                reasons=reasons,
            )

            # 6) Visual dashboard
            # Resize FIRST so overlays are drawn at final resolution (crisp text).
            disp_show = cv2.resize(fp.frame.copy(), (int(args.ui_width), int(args.ui_height)), interpolation=cv2.INTER_AREA)

            # Header banner
            cv2.rectangle(disp_show, (0, 0), (disp_show.shape[1], 95), (0, 0, 0), -1)

            color = (0, 255, 0)
            if plan.mode == SystemMode.DEGRADED:
                color = (0, 165, 255)
            if plan.mode == SystemMode.CRITICAL:
                color = (0, 0, 255)

            cv2.putText(disp_show, f"DECISION: {final_action}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

            plan_str = f"POLICY: {plan.mode.name} | MODE: {mode_mgr.current_mode.name} | PLAN: {plan.camera_count}C + {'LIDAR' if plan.lidar_on else 'NO_LIDAR'}"
            cv2.putText(disp_show, plan_str, (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

            stat_str = f"T={plan.cpu_temp_c:.1f}C  RAM={plan.ram_free_mb:.0f}MB  LOAD={plan.cpu_load_per_core:.2f}/core  HS={plan.health_score:.2f}"
            cv2.putText(disp_show, stat_str, (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

            # Heartbeat pulse dot (blinks)
            pulse = color if int(time.time() * 2) % 2 == 0 else (50, 50, 50)
            cv2.circle(disp_show, (disp_show.shape[1] - 20, 22), 10, pulse, -1)

            # Scrolling log
            logger.draw(disp_show)

            cv2.imshow(args.window, disp_show)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # 7) Loop pacing (policy fps_target, capped by user max)
            target_fps = min(int(args.max_fps), int(mode_mgr.fps_target))
            if target_fps > 0:
                period = 1.0 / float(target_fps)
                elapsed = time.time() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)

            # 8) If we commanded RETURN_HOME from CRITICAL, end demo shortly
            if plan.mode == SystemMode.CRITICAL:
                time.sleep(1.0)
                break

    finally:
        # Cleanup
        try:
            if hasattr(lidar, "close"):
                lidar.close()
        except Exception:
            pass
        try:
            if hasattr(cam_provider, "close"):
                cam_provider.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
