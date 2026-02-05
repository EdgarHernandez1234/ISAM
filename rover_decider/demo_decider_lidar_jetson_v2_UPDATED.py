#!/usr/bin/env python3
"""
demo_decider_lidar_jetson_v2.py

Jetson-friendly demo decider with:
- YOLOv8 classification (clean vs dirty/contaminated)
- RELLIS-3D LiDAR min-forward distance from KITTI .bin frames
- Per-decision CSV logging (one row per decision frame)
- RL-style safety supervisor (shield layer) enforcing failsafes
- Optional live labeling via hotkeys (0/1/u) when display is enabled

Termination behavior:
- By default, runs for --num-decisions (default: 1) then exits.
- If display is enabled, you can press 'q' or ESC to stop early.
- If you set --display-ms 0, it will wait for a key each decision (interactive).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import unittest
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore

# Local modules (support both packaged and flat-file layouts)
try:
    import rover_decider  # triggers rover_decider/__init__.py
    from rover_decider import core
    from rover_decider import rellis_lidar
    from rover_decider.logger import CSVDecisionLogger, DecisionFrame
except Exception:  # pragma: no cover
    rover_decider = None  # type: ignore
    import core  # type: ignore
    import rellis_lidar  # type: ignore
    from logger import CSVDecisionLogger, DecisionFrame  # type: ignore

# Safety shield layer (policy proposes, supervisor can override)
from rl_safety_supervisor import (  # stdlib-only module
    Observation,
    HeuristicPolicy,
    SafetySupervisor,
    ShieldedController,
)


# --------------------------------------------------
# Defaults (override via CLI)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models/sand_cls_clean_vs_dirt.pt"
CLEAN_DIR = BASE_DIR / "demo_images/Clean"
DIRTY_DIR = BASE_DIR / "demo_images/Dirt"

NUM_PER_CLASS = 5
TILE_SIZE = (240, 240)


# Some repos name this constant differently; keep demo robust.
DEFAULT_MAX_SCOOP_DIST_M = getattr(core, "DEFAULT_MAX_SCOOP_DIST_M", 2.5)


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not installed or not importable.")


# --------------------------------------------------
# Image / YOLO helpers
# --------------------------------------------------
def get_image_paths(folder: Path, n: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    imgs = [p for p in folder.glob("*") if p.suffix.lower() in exts]
    if not imgs:
        raise FileNotFoundError(f"No images found in: {folder}")
    imgs.sort()
    if len(imgs) <= n:
        return imgs
    return random.sample(imgs, n)


def infer_true_label(path: Path) -> str:
    s = str(path).lower()
    if "dirt" in s or "dirty" in s:
        return "dirty"
    return "clean"


def load_yolo_model(model_path: Path):
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed or not importable.")
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    return YOLO(str(model_path))


def yolo_infer_top1(model, bgr_img) -> Tuple[str, float]:
    """
    Returns (pred_class_name, pred_conf).
    """
    # ultralytics classification returns a Results list
    results = model.predict(source=bgr_img, verbose=False)
    r0 = results[0]
    # probs.top1 gives integer index, probs.top1conf gives tensor/float
    top1_idx = int(r0.probs.top1)
    top1_conf = float(r0.probs.top1conf)
    class_name = r0.names.get(top1_idx, str(top1_idx))
    return str(class_name), float(top1_conf)


# --------------------------------------------------
# Shielded decision helper (Policy + Safety Supervisor)
# --------------------------------------------------
def decide_with_safety(
    pred_class: str,
    pred_conf: float,
    distance_m: Optional[float],
    conf_thresh: float,
    max_scoop_dist: float,
    *,
    use_rl_safety: bool = True,
    joint_error_norm: Optional[float] = None,
    motor_current_a: Optional[float] = None,
    stall_flag: Optional[bool] = None,
    health_score: Optional[float] = None,
):
    """Return (final_action, reason, safety_state, looks_dirty, decision_or_none).

    - If RL safety is enabled:
        policy proposes SCOOP/BYPASS, supervisor may override to STOP/RETREAT/RETURN_HOME/DEGRADED.
    - safety_state is kept compatible with existing logs:
        * if core distance state != NORMAL -> that state wins (SAFE_HOLD/STOP)
        * else RL_OVERRIDE / RL_ADVISORY / NORMAL
    """
    looks_dirty = core.looks_dirty_from_class(pred_class)

    if not use_rl_safety:
        action, reason, safety_state = core.choose_action(
            looks_dirty=looks_dirty,
            pred_conf=pred_conf,
            min_distance_m=distance_m,
            conf_thresh=float(conf_thresh),
            max_scoop_dist=float(max_scoop_dist),
        )
        return action, reason, safety_state, looks_dirty, None

    obs = Observation(
        pred_class=str(pred_class),
        pred_conf=float(pred_conf),
        distance_m=None if distance_m is None else float(distance_m),
        joint_error_norm=joint_error_norm,
        motor_current_a=motor_current_a,
        stall_flag=stall_flag,
        health_score=health_score,
    )

    ctrl = ShieldedController(
        policy=HeuristicPolicy(conf_thresh=float(conf_thresh), max_scoop_dist=float(max_scoop_dist)),
        supervisor=SafetySupervisor.default(),
    )
    decision = ctrl.step(obs)

    base_state = core.compute_safety_state(distance_m)
    if base_state != "NORMAL":
        safety_state = base_state
    elif decision.final_action != decision.proposed_action:
        safety_state = "RL_OVERRIDE"
    elif decision.signals:
        safety_state = "RL_ADVISORY"
    else:
        safety_state = "NORMAL"

    return str(decision.final_action), str(decision.reason), str(safety_state), looks_dirty, decision


# --------------------------------------------------
# OpenCV visualization (optional)
# --------------------------------------------------
def load_and_resize(path: Path, tile_size=TILE_SIZE):
    _require_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.resize(img, tile_size)


def make_gallery_image(clean_paths: List[Path], dirty_paths: List[Path], tile_size=TILE_SIZE):
    _require_cv2()
    tiles = []
    for p in clean_paths:
        tiles.append(load_and_resize(p, tile_size))
    for p in dirty_paths:
        tiles.append(load_and_resize(p, tile_size))
    cols = max(len(clean_paths), len(dirty_paths))
    # build 2 rows: clean row then dirty row
    clean_tiles = tiles[:len(clean_paths)]
    dirty_tiles = tiles[len(clean_paths):]

    # pad rows
    blank = 255 * (clean_tiles[0] * 0 + 1)
    while len(clean_tiles) < cols:
        clean_tiles.append(blank.copy())
    while len(dirty_tiles) < cols:
        dirty_tiles.append(blank.copy())

    row1 = cv2.hconcat(clean_tiles)
    row2 = cv2.hconcat(dirty_tiles)
    return cv2.vconcat([row1, row2])


def show_gallery_cv(clean_paths: List[Path], dirty_paths: List[Path], *, hold_s: float = 300.0) -> int:
    """Show the 2-row gallery.

    By default, the gallery remains visible for 5 minutes (300s). You may end early
    by pressing Enter/Space (continue) or 'q'/ESC (quit).

    Returns:
        Pressed key (0-255) or -1 if timed out.
    """
    _require_cv2()
    gallery = make_gallery_image(clean_paths, dirty_paths)

    # Add a minimal instruction banner for safe demos.
    banner = gallery.copy()
    cv2.putText(
        banner,
        "Gallery: top=clean, bottom=dirty | Enter/Space=continue | q/ESC=quit",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    win = "Gallery (top=clean, bottom=dirty)"
    cv2.imshow(win, banner)

    wait_ms = 0 if hold_s <= 0 else int(float(hold_s) * 1000)
    k = cv2.waitKey(wait_ms)
    key = (int(k) & 0xFF) if k is not None and int(k) != -1 else -1
    cv2.destroyWindow(win)
    return key


def show_single_decision_cv(
    img_path: Path,
    pred_class: str,
    pred_conf: float,
    distance_m: Optional[float],
    action: str,
    reason: str,
    safety_state: str,
    proposed_action: str = "",
    top_safety: str = "",
    true_label: Optional[str] = None,
    lidar_note: str = "",
    wait_ms: int = 5000,
) -> int:
    """
    Draw overlay and return pressed key (0-255) or -1 if timed out.
    If wait_ms == 0 => wait forever for a key (interactive).
    """
    _require_cv2()
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    display = cv2.resize(img, (640, 480))
    if true_label is None:
        true_label = infer_true_label(img_path)

    dist_str = "None" if distance_m is None else f"{float(distance_m):.2f} m"

    lines = [
        f"File: {img_path.name}",
        f"True: {true_label.upper()}",
        f"Pred: {str(pred_class).upper()} ({pred_conf:.2f})",
        f"LiDAR min: {dist_str}",
        f"Safety: {safety_state}",
    ]
    if proposed_action:
        lines.append(f"Policy: {proposed_action}")
    lines.append(f"Action: {action}")
    lines.append(f"Reason: {reason}")
    lines.append("Label keys: 1=good scoop, 0=bad scoop, u=clear | Enter/Space=next | q/ESC=quit")

    if top_safety:
        lines.append(top_safety)
    if lidar_note:
        lines.append(lidar_note)

    y = 25
    for line in lines:
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y += 24

    cv2.imshow("Rover Decision", display)

    # With a long dwell time, you can still advance early by pressing any key
    # (e.g., Enter/Space). Label keys still work the same.
    k = cv2.waitKey(0 if wait_ms == 0 else int(wait_ms))
    key = (int(k) & 0xFF) if k is not None and int(k) != -1 else -1
    cv2.destroyWindow("Rover Decision")
    return key


def label_from_keypress(key: int) -> Optional[int]:
    if key == ord("1"):
        return 1
    if key == ord("0"):
        return 0
    if key in (ord("u"), ord("U")):
        return None
    return None


def default_log_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return BASE_DIR / "logs" / f"decision_log_{stamp}.csv"


def run_tests() -> int:
    """
    Discover and run unit tests under ./tests (relative to this file).
    """
    root = str(BASE_DIR)
    suite = unittest.defaultTestLoader.discover(start_dir=root, pattern="test_*.py")
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Jetson demo decider: YOLO + RELLIS LiDAR + CSV logging")
    ap.add_argument("--model", default=str(MODEL_PATH), help="Path to YOLOv8 classification model .pt")
    ap.add_argument("--clean-dir", default=str(CLEAN_DIR), help="Directory of clean images")
    ap.add_argument("--dirty-dir", default=str(DIRTY_DIR), help="Directory of dirty images")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    ap.add_argument("--no-gallery", action="store_true", help="Skip gallery window")
    ap.add_argument("--gallery-hold-s", type=float, default=300.0,
                    help="How long to show the gallery in seconds (default: 300 = 5 minutes). 0 = wait for key.")
    ap.add_argument("--no-display", action="store_true", help="Run headless: do not open any OpenCV windows")
    ap.add_argument("--display-ms", type=int, default=5000,
                    help="Decision window dwell time in ms (default: 5000). 0 = wait for key (interactive).")

    ap.add_argument("--num-decisions", type=int, default=1,
                    help="How many decision frames to run/log (default: 1)")
    ap.add_argument("--duration-s", type=float, default=0.0,
                    help="Optional max duration in seconds (0 disables). Stops when reached.")

    ap.add_argument("--log-csv", default="", help="CSV output path. Default: ./logs/decision_log_*.csv")
    ap.add_argument("--test", action="store_true", help="Run unit tests and exit")

    # LiDAR
    ap.add_argument("--rellis-kitti-path", default="",
                    help="Directory of *.bin OR a virtual zip path like C:\\x.zip\\inner\\folder")
    ap.add_argument("--fov-deg", type=float, default=rellis_lidar.DEFAULT_FOV_DEG)
    ap.add_argument("--max-range-m", type=float, default=rellis_lidar.DEFAULT_MAX_RANGE_M)
    ap.add_argument("--stride", type=int, default=rellis_lidar.DEFAULT_STRIDE)

    # Decision tunables
    ap.add_argument("--conf-thresh", type=float, default=core.DEFAULT_CONF_THRESH)
    ap.add_argument("--max-scoop-dist", type=float, default=DEFAULT_MAX_SCOOP_DIST_M)

    # Safety supervisor / failsafe inputs (optional telemetry or demo knobs)
    ap.add_argument("--no-rl-safety", action="store_true",
                    help="Disable rl_safety_supervisor (legacy core.choose_action only)")
    ap.add_argument("--health-score", type=float, default=None,
                    help="0..1 rover health score (lower triggers DEGRADED / RETURN_HOME)")
    ap.add_argument("--stall-flag", action="store_true",
                    help="Simulate arm stall condition (forces safety reaction)")
    ap.add_argument("--joint-error-norm", type=float, default=None,
                    help="Optional joint tracking error norm (>= threshold triggers STOP)")
    ap.add_argument("--motor-current-a", type=float, default=None,
                    help="Optional motor current (A) (>= threshold triggers DEGRADED)")

    args = ap.parse_args()

    if args.test:
        return run_tests()

    random.seed(int(args.seed))

    if args.no_display:
        # OK to be headless; still need cv2+YOLO for inference unless you replace imread
        pass

    if cv2 is None or YOLO is None:
        raise RuntimeError("Runtime requires cv2 + ultralytics. Use --test to run unit tests without them.")

    clean_dir = Path(args.clean_dir)
    dirty_dir = Path(args.dirty_dir)
    model_path = Path(args.model)

    clean_paths = get_image_paths(clean_dir, NUM_PER_CLASS)
    dirty_paths = get_image_paths(dirty_dir, NUM_PER_CLASS)

    if (not args.no_display) and (not args.no_gallery):
        gkey = show_gallery_cv(clean_paths, dirty_paths, hold_s=float(args.gallery_hold_s))
        # q/ESC exits immediately; Enter/Space (or timeout) simply continues.
        if gkey in (ord("q"), 27):
            return 0

    model = load_yolo_model(model_path)

    # Logging
    csv_path = Path(args.log_csv) if args.log_csv else default_log_path()

    start_t = time.time()
    current_label: Optional[int] = None

    with CSVDecisionLogger(csv_path) as logger:
        for i in range(max(0, int(args.num_decisions))):
            if args.duration_s and (time.time() - start_t) >= float(args.duration_s):
                break

            # pick one image per decision
            chosen_path = random.choice(clean_paths + dirty_paths)
            true_label = infer_true_label(chosen_path)

            # LiDAR distance (optional)
            distance_m: Optional[float] = None
            lidar_note = ""
            if args.rellis_kitti_path:
                try:
                    d, lidar_note = rellis_lidar.get_rellis_distance_m(
                        args.rellis_kitti_path,
                        fov_deg=args.fov_deg,
                        max_range_m=args.max_range_m,
                        stride=args.stride,
                    )
                    distance_m = float(d)
                except Exception as e:
                    distance_m = None
                    lidar_note = f"LiDAR unavailable: {e}"

            # YOLO inference
            frame = cv2.imread(str(chosen_path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[warn] Failed to read image: {chosen_path}", file=sys.stderr)
                continue

            pred_class, pred_conf = yolo_infer_top1(model, frame)

            action, reason, safety_state, looks_dirty, shield_decision = decide_with_safety(
                pred_class=pred_class,
                pred_conf=pred_conf,
                distance_m=distance_m,
                conf_thresh=float(args.conf_thresh),
                max_scoop_dist=float(args.max_scoop_dist),
                use_rl_safety=not bool(args.no_rl_safety),
                joint_error_norm=args.joint_error_norm,
                motor_current_a=args.motor_current_a,
                stall_flag=(True if args.stall_flag else None),
                health_score=args.health_score,
            )

            proposed_action = ""
            top_safety = ""
            if shield_decision is not None:
                proposed_action = str(getattr(shield_decision, "proposed_action", ""))
                sigs = getattr(shield_decision, "signals", ())
                if sigs:
                    top = sigs[0]
                    top_safety = f"TopSafety: {top.source}:{top.level.name} override={top.override_action or '-'}"

            # Optional live labeling BEFORE logging (so label lands in the row)
            if not args.no_display:
                key = show_single_decision_cv(
                    chosen_path,
                    pred_class=pred_class,
                    pred_conf=pred_conf,
                    distance_m=distance_m,
                    action=action,
                    reason=reason,
                    safety_state=safety_state,
                    proposed_action=proposed_action,
                    top_safety=top_safety,
                    true_label=true_label,
                    lidar_note=lidar_note,
                    wait_ms=int(args.display_ms),
                )

                if key in (ord("q"), 27):  # q or ESC
                    # still log this frame, then exit
                    pass

                if key in (ord("0"), ord("1"), ord("u"), ord("U")):
                    current_label = label_from_keypress(key)

            # Log row
            row = DecisionFrame.build(
                min_distance_m=distance_m,
                pred_conf=pred_conf,
                looks_dirty=looks_dirty,
                class_id=pred_class,
                action=action,
                safety_state=safety_state,
                label=current_label,
            )
            logger.log(row)

            if "shield_decision" in locals() and shield_decision is not None:
                print(f"[{i+1}/{args.num_decisions}] proposed={proposed_action} final={action} safety={safety_state} "
                      f"conf={pred_conf:.3f} class={pred_class} lidar={distance_m}")
            else:
                print(f"[{i+1}/{args.num_decisions}] action={action} safety={safety_state} conf={pred_conf:.3f} "
                      f"class={pred_class} lidar={distance_m}")

            # early stop if quit pressed
            if not args.no_display:
                if 'key' in locals() and key in (ord("q"), 27):
                    break

    if not args.no_display and cv2 is not None:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print(f"\nDone. Logged to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
