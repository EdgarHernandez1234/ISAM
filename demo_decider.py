#!/usr/bin/env python3
"""
demo_decider_cv_only.py

Numpy-free *in your code* demo of the regolith decision process:

- Loads your YOLOv8 classification model (same one used before).
- Collects up to 5 "clean" and 5 "dirty" images.
- Builds a 2xN gallery using ONLY OpenCV (no matplotlib/seaborn).
- Randomly picks 1 of those images, runs YOLO, and decides:
      SCOOP  vs  BYPASS
  based on predicted class, confidence, and distance.

Visualization:
  - Uses cv2.imshow() to display:
      * "Regolith Gallery"  (top: clean, bottom: dirty)
      * "Rover Decision"    (single chosen image with overlay text)
"""

import random
from pathlib import Path

import cv2
from ultralytics import YOLO

# -----------------------------------------------------------------------------#
# CONFIG
# -----------------------------------------------------------------------------#

# Where your YOLO model lives.
# Change this to match whatever you use in regolith_decider.py
MODEL_PATH = Path("/mnt/model_sd/models/regolith/cls/sand_cls_clean_vs_dirt.pt")

# Where your demo images live
CLEAN_DIR = Path("images/clean")
DIRTY_DIR = Path("images/dirty")

NUM_PER_CLASS = 5                     # up to 5 clean + 5 dirty
DEFAULT_DISTANCE_RANGE = (0.5, 3.0)   # meters (for demo randomization)
TILE_SIZE = (240, 240)               # (width, height) for gallery tiles

random.seed(42)  # reproducible demo (optional)


# -----------------------------------------------------------------------------#
# Helpers: load images + YOLO model
# -----------------------------------------------------------------------------#

def get_image_paths(directory: Path, max_images: int):
    """Return up to max_images image paths from a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    paths = [p for p in directory.iterdir() if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")

    random.shuffle(paths)
    return paths[:max_images]


def load_yolo_model():
    """Load the YOLOv8 classifier model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    return model


def infer_true_label(img_path: Path) -> str:
    """
    Ground-truth label based on folder/filename:
      .../clean/... -> 'clean'
      .../dirty/... -> 'dirty'
    """
    parent = img_path.parent.name.lower()
    if "clean" in parent:
        return "clean"
    if "dirty" in parent:
        return "dirty"

    name = img_path.name.lower()
    if "clean" in name:
        return "clean"
    if "dirty" in name:
        return "dirty"
    return "unknown"


# -----------------------------------------------------------------------------#
# Demo decision logic (no bandit, YOLO-only)
# -----------------------------------------------------------------------------#

def decide_action_yolo_only(model, frame, distance_m: float):
    """
    Run YOLO on a frame and decide SCOOP or BYPASS.

    Heuristic:
      - If predicted class name contains 'dirty', 'dirt', 'trash', or 'plastic' -> BYPASS
      - Else if confidence is too low or object is too far away -> BYPASS
      - Otherwise -> SCOOP
    """
    results = model(frame, imgsz=224, verbose=False)[0]

    probs = results.probs
    top_idx = int(probs.top1)
    pred_conf = float(probs.top1conf)
    pred_class = results.names[top_idx]

    cls_lower = pred_class.lower()

    # Class-based decision (dirty vs clean)
    looks_dirty = any(word in cls_lower for word in ["dirty", "dirt", "trash", "plastic"])

    # Simple thresholds
    CONF_THRESH = 0.6
    MAX_SCOOP_DIST = 2.5  # m

    if looks_dirty:
        action = "BYPASS"
        reason = "looks dirty/contaminated"
    elif pred_conf < CONF_THRESH:
        action = "BYPASS"
        reason = f"low confidence ({pred_conf:.2f} < {CONF_THRESH})"
    elif distance_m > MAX_SCOOP_DIST:
        action = "BYPASS"
        reason = f"too far away ({distance_m:.2f} m > {MAX_SCOOP_DIST} m)"
    else:
        action = "SCOOP"
        reason = "clean and confident within safe distance"

    return {
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "distance_m": distance_m,
        "action": action,
        "reason": reason,
    }


# -----------------------------------------------------------------------------#
# Visualization with OpenCV only
# -----------------------------------------------------------------------------#

def load_and_resize(path: Path, tile_size=TILE_SIZE):
    """Read an image and resize it to tile_size for gallery use."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.resize(img, tile_size)


def make_gallery_image(clean_paths, dirty_paths):
    """
    Build a 2xN gallery image with OpenCV:

    Top row  : clean images
    Bottom row: dirty images

    We use up to the same number of clean & dirty images so rows align.
    """
    if not clean_paths or not dirty_paths:
        raise ValueError("Need at least one clean and one dirty image to build gallery.")

    n = min(len(clean_paths), len(dirty_paths), NUM_PER_CLASS)

    clean_tiles = [load_and_resize(p) for p in clean_paths[:n]]
    dirty_tiles = [load_and_resize(p) for p in dirty_paths[:n]]

    top_row = cv2.hconcat(clean_tiles)
    bottom_row = cv2.hconcat(dirty_tiles)
    gallery = cv2.vconcat([top_row, bottom_row])

    # Add simple labels on the left side of each row
    # (draw text on the first tile's area in each row)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        gallery,
        "CLEAN (top row)",
        (10, 30),
        font,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        gallery,
        "DIRTY (bottom row)",
        (10, gallery.shape[0] // 2 + 30),
        font,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return gallery


def show_gallery_cv(clean_paths, dirty_paths):
    """Create and display the gallery image with OpenCV."""
    gallery = make_gallery_image(clean_paths, dirty_paths)
    cv2.imshow("Regolith Gallery (CLEAN top, DIRTY bottom)", gallery)
    cv2.waitKey(0)  # waits for a keypress
    cv2.destroyWindow("Regolith Gallery (CLEAN top, DIRTY bottom)")


def show_single_decision_cv(img_path: Path, decision: dict, true_label=None):
    """Show one image with the rover's prediction + action overlaid using OpenCV."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    # Resize for nicer display (optional)
    display = cv2.resize(img, (640, 480))

    if true_label is None:
        true_label = infer_true_label(img_path)

    lines = [
        f"File: {img_path.name}",
        f"True: {true_label.upper()}",
        f"Pred: {decision['pred_class'].upper()} ({decision['pred_conf']:.2f})",
        f"Dist: {decision['distance_m']:.2f} m",
        f"Action: {decision['action']} ({decision['reason']})",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 25
    for line in lines:
        cv2.putText(display, line, (10, y), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y += 25

    cv2.imshow("Rover Decision", display)
    cv2.waitKey(0)
    cv2.destroyWindow("Rover Decision")


# -----------------------------------------------------------------------------#
# MAIN
# -----------------------------------------------------------------------------#

def main():
    # 1. Collect images
    clean_paths = get_image_paths(CLEAN_DIR, NUM_PER_CLASS)
    dirty_paths = get_image_paths(DIRTY_DIR, NUM_PER_CLASS)

    # 2. Show gallery of candidates (OpenCV only)
    show_gallery_cv(clean_paths, dirty_paths)

    # 3. Load YOLO model
    model = load_yolo_model()

    # 4. Randomly pick one of the images
    all_paths = clean_paths + dirty_paths
    chosen_path = random.choice(all_paths)
    true_label = infer_true_label(chosen_path)

    # Random demo distance
    distance_m = round(random.uniform(*DEFAULT_DISTANCE_RANGE), 2)

    print("\n========== ROVER DECISION DEMO (YOLO only, OpenCV display) ==========")
    print(f"Selected image : {chosen_path}")
    print(f"True label     : {true_label}")
    print(f"Distance (m)   : {distance_m}")

    # 5. Run decision logic
    frame = cv2.imread(str(chosen_path), cv2.IMREAD_COLOR)
    decision = decide_action_yolo_only(model, frame, distance_m)

    print("\n=== Decision ===")
    print(f"Predicted class : {decision['pred_class']}")
    print(f"Confidence      : {decision['pred_conf']:.4f}")
    print(f"Action          : {decision['action']}")
    print(f"Reason          : {decision['reason']}")
    print("======================================================================")

    # 6. Visualize selected image + decision overlay (OpenCV only)
    show_single_decision_cv(chosen_path, decision, true_label=true_label)


if __name__ == "__main__":
    main()
