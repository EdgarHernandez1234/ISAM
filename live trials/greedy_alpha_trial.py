
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
greedy_alpha_trial.py

Gazebo-oriented epsilon-greedy alpha trial.

Intent:
- Use Gazebo camera / LiDAR proxies instead of local CSI/USB cameras
- Publish /cmd_vel for the current validated desktop proxy workflow
- Run ONNX/PT/engine inference locally on proxied Gazebo images
- Learn a mission-level action policy with epsilon-greedy Q-learning

This file is a new trial fork based on alpha_trial.py, but retargeted for proxy-fed Gazebo use.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from greedy_trial_common import ProxySensorHub, run_greedy_trial


def parse_args():
    p = argparse.ArgumentParser(description="Greedy Gazebo alpha trial using proxy image feeds")
    p.add_argument("--front-image-topic", default="/alam/front_camera/image_raw/compressed")
    p.add_argument("--back-image-topic", default="/alam/back_camera/image_raw/compressed")
    p.add_argument("--scan-topic", default="/scan")
    p.add_argument("--image-compressed", action="store_true", default=True)
    p.add_argument("--cmd-vel-topic", default="/cmd_vel")

    p.add_argument("--model-path", default=os.path.expanduser("~/Desktop/models/regolith/best_yolov26.onnx"))
    p.add_argument("--model-imgsz", type=int, default=640)
    p.add_argument("--model-conf", type=float, default=0.25)
    p.add_argument("--model-iou", type=float, default=0.45)
    p.add_argument("--model-device", default="")
    p.add_argument("--model-max-det", type=int, default=50)

    p.add_argument("--q-table", default="q_table_greedy_alpha_trial.json")
    p.add_argument("--epsilon", type=float, default=0.25)
    p.add_argument("--min-epsilon", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--alpha", type=float, default=0.20)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eval-only", action="store_true")
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
        compressed=bool(args.image_compressed),
    )
    q_path = str(Path(__file__).resolve().parent / args.q_table)
    run_greedy_trial(
        window_title="GREEDY ALPHA TRIAL (GAZEBO PROXIES)",
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
    )


if __name__ == "__main__":
    main()
