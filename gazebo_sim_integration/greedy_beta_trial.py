
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
greedy_beta_trial.py

Gazebo-oriented epsilon-greedy beta trial.

Compared with greedy_alpha_trial.py, this starts from the beta-side model defaults
and keeps the same proxy-fed Gazebo /cmd_vel workflow.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from greedy_trial_common import ProxySensorHub, run_greedy_trial


def parse_args():
    p = argparse.ArgumentParser(description="Greedy Gazebo beta trial using proxy image feeds")
    p.add_argument("--front-image-topic", default="/sim/rs_front/image_raw")
    p.add_argument("--back-image-topic", default="/sim/rs_back/image_raw")
    p.add_argument("--scan-topic", default="/scan")
    p.add_argument("--pose-topic", default="/alam/rover_pose_json")
    p.add_argument("--image-transport", choices=["raw", "compressed"], default="raw")
    p.add_argument("--cmd-vel-topic", default="/cmd_vel")

    p.add_argument("--model-path", default=os.path.expanduser("~/Desktop/models/regolith/best_v11.pt"))
    p.add_argument("--model-imgsz", type=int, default=640)
    p.add_argument("--model-conf", type=float, default=0.25)
    p.add_argument("--model-iou", type=float, default=0.45)
    p.add_argument("--model-device", default="")
    p.add_argument("--model-max-det", type=int, default=50)

    p.add_argument("--q-table", default="q_table_greedy_beta_trial.json")
    p.add_argument("--epsilon", type=float, default=0.45)
    p.add_argument("--min-epsilon", type=float, default=0.15)
    p.add_argument("--epsilon-decay", type=float, default=0.999)
    p.add_argument("--alpha", type=float, default=0.20)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--reset-q-table", action="store_true")
    p.add_argument("--action-hold-frames", type=int, default=5)
    p.add_argument("--action-switch-margin", type=float, default=0.18)
    p.add_argument("--action-switch-cooldown-s", type=float, default=0.40)
    p.add_argument("--rng-seed", type=int, default=42)

    p.add_argument("--laser-x", type=float, default=6.0)
    p.add_argument("--laser-y", type=float, default=0.0)
    p.add_argument("--max-fps", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    hub = ProxySensorHub(
        front_topic=str(args.front_image_topic),
        back_topic=str(args.back_image_topic),
        scan_topic=str(args.scan_topic),
        image_transport=str(args.image_transport),
        pose_topic=str(args.pose_topic),
    )
    q_path = str(Path(__file__).resolve().parent / args.q_table)
    run_greedy_trial(
        window_title="GREEDY BETA TRIAL (GAZEBO PROXIES)",
        model_path=str(args.model_path),
        q_table_path=q_path,
        sensor_hub=hub,
        cmd_topic=str(args.cmd_vel_topic),
        laser_x=float(args.laser_x),
        laser_y=float(args.laser_y),
        max_fps=int(args.max_fps),
        imgsz=int(args.model_imgsz),
        conf=float(args.model_conf),
        iou=float(args.model_iou),
        device=str(args.model_device),
        max_det=int(args.model_max_det),
        epsilon=float(args.epsilon),
        min_epsilon=float(args.min_epsilon),
        epsilon_decay=float(args.epsilon_decay),
        alpha=float(args.alpha),
        gamma=float(args.gamma),
        eval_only=bool(args.eval_only),
        rng_seed=int(args.rng_seed),
        reset_q_table=bool(args.reset_q_table),
        action_hold_frames=int(args.action_hold_frames),
        action_switch_margin=float(args.action_switch_margin),
        action_switch_cooldown_s=float(args.action_switch_cooldown_s),
    )


if __name__ == "__main__":
    main()
