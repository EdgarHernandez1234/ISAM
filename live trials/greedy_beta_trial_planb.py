
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
greedy_beta_trial_planb.py

Replay-based epsilon-greedy beta trial for Gazebo/demo fallback use.

Use this when you want the same greedy trial logic but do not want live sensors at all.
This is the plan-B counterpart to greedy_beta_trial.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from greedy_trial_common import ReplayVideoHub, run_greedy_trial


def parse_args():
    p = argparse.ArgumentParser(description="Greedy beta plan-B trial (replay video)")
    p.add_argument("--replay-video", default="")
    p.add_argument("--replay-video-a", default="")
    p.add_argument("--replay-video-b", default="")
    p.add_argument("--replay-side-by-side", action="store_true", default=False)
    p.add_argument("--replay-loop", action="store_true", default=True)

    p.add_argument("--cmd-vel-topic", default="/cmd_vel")
    p.add_argument("--model-path", default=os.path.expanduser("~/Desktop/models/regolith/best_v11.pt"))
    p.add_argument("--model-imgsz", type=int, default=640)
    p.add_argument("--model-conf", type=float, default=0.25)
    p.add_argument("--model-iou", type=float, default=0.45)
    p.add_argument("--model-device", default="")
    p.add_argument("--model-max-det", type=int, default=50)

    p.add_argument("--q-table", default="q_table_greedy_beta_trial_planb.json")
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
    hub = ReplayVideoHub(
        video_path=(args.replay_video or None),
        video_a=(args.replay_video_a or None),
        video_b=(args.replay_video_b or None),
        side_by_side=bool(args.replay_side_by_side),
        loop=bool(args.replay_loop),
    )
    q_path = str(Path(__file__).resolve().parent / args.q_table)
    run_greedy_trial(
        window_title="GREEDY BETA PLANB TRIAL",
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
