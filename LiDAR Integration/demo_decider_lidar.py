
#!/usr/bin/env python3
"""
demo_decider.py (CV + LiDAR distance)

What this demo does:
- Shows a 2xN OpenCV gallery of "clean" + "dirty" sand images.
- Randomly selects one image, runs your YOLOv8 classifier, and produces:
    SCOOP vs BYPASS
  based on:
    (1) predicted class name
    (2) prediction confidence
    (3) LiDAR-derived distance (RELLIS-3D KITTI .bin frames)
    
    
    
LiDAR integration:
- Reads RELLIS-3D KITTI point cloud frames (.bin, float32 x,y,z,intensity)
- Computes a robust scalar distance metric:
    min planar distance in a forward sector (FOV cone) within a max range
- This distance replaces the old randomized distance

Supports:
- RELLIS directory of .bin files
- RELLIS ZIP "virtual path" (Windows Explorer-style path inside zip)
  Example you provided:
    G:\...\Rellis_3D_os1_cloud_node_kitti_bin.zip\Rellis-3D\00000\os1_cloud_node_kitti_bin

Unit tests:    
- Run:  python demo_decider.py --test
- Tests do NOT require cv2/ultralytics or GPU; they validate:
  - zip virtual path parsing
  - KITTI .bin parsing on synthetic point clouds
  - distance computation (FOV/range filtering)
  - decision gating logic (clean/dirty/confidence/distance)
  
  
Regular run: python demo_decider_lidar.py
No image gallery run: python demo_decider_lidar.py --no-gallery


 Notes:
- Reading directly from a large zip on Google Drive can be slow.
  If performance is an issue, extract the zip once to a local folder and
  pass the extracted directory path instead. 
  
  
"""

import argparse
import math
import os
import random
import sys
import time
import unittest
import zipfile
from array import array
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
from tempfile import TemporaryDirectory

import cv2
from ultralytics import YOLO

# -----------------------------------------------------------------------------#
# CONFIG
# -----------------------------------------------------------------------------#

# YOLO model
MODEL_PATH = Path("C:/Users/Miles/Desktop/ALAM work/sand_cls_clean_vs_dirt.pt")

# demo images
CLEAN_DIR = Path("C:/Users/Miles/Desktop/ALAM work/MoonSand_SRC_merged/Clean")
DIRTY_DIR = Path("C:/Users/Miles/Desktop/ALAM work/MoonSand_SRC_merged/Dirt")

NUM_PER_CLASS = 5
DEFAULT_DISTANCE_RANGE = (0.5, 3.0)  # meters (fallback)
TILE_SIZE = (240, 240)

# If LiDAR fails, fallback distance forces BYPASS due to distance threshold.
LIDAR_FALLBACK_DISTANCE_M = 999.0

# Default LiDAR metric parameters
DEFAULT_FOV_DEG = 30.0         # forward sector total width
DEFAULT_MAX_RANGE_M = 10.0     # ignore points beyond this
DEFAULT_STRIDE = 5             # 1 = every point, 5 = sample every 5 points for speed

# RELLIS 3D KittiBin path from drive
DEFAULT_RELLIS_KITTI_PATH = r"G:\.shortcut-targets-by-id\1aZ1tJ3YYcWuL3oWKnrTIC5gq46zx1bMc\Rellis-3D_Release\Rellis_3D_os1_cloud_node_kitti_bin.zip\Rellis-3D\00000\os1_cloud_node_kitti_bin"

# -----------------------------------------------------------------------------
# Helpers: images + YOLO
# -----------------------------------------------------------------------------

def get_image_paths(directory: Path, max_images: int) -> List[Path]:
    """Return up to max_images image paths from a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    paths = [p for p in directory.iterdir() if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")

    random.shuffle(paths)
    return paths[:max_images]


def load_yolo_model(model_path: Path):
    """Load the YOLOv8 classifier model."""
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed or not importable in this environment.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"[INFO] Loading YOLO model from: {model_path}")
    return YOLO(str(model_path))


def infer_true_label(img_path: Path) -> str:
    """Ground-truth label based on folder/filename."""
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


def yolo_infer_top1(model, frame) -> Tuple[str, float]:
    """
    Returns (pred_class_name, confidence) using YOLOv8 classification outputs.
    """
    results = model(frame, imgsz=224, verbose=False)[0]
    probs = results.probs
    top_idx = int(probs.top1)
    pred_conf = float(probs.top1conf)
    pred_class = results.names[top_idx]
    return pred_class, pred_conf
    
    
    
# -----------------------------------------------------------------------------
# Decision logic (testable, independent of YOLO/cv2)
# -----------------------------------------------------------------------------

def decide_action(pred_class: str, pred_conf: float, distance_m: float,
                  conf_thresh: float = 0.6, max_scoop_dist: float = 2.5) -> dict:
    """
    Decide SCOOP or BYPASS based on predicted class, confidence, and distance.

    Heuristic:
      - If predicted class name contains dirt/trash/plastic -> BYPASS
      - Else if confidence too low -> BYPASS
      - Else if too far away -> BYPASS
      - Else -> SCOOP
    """
    cls_lower = (pred_class or "").lower()
    looks_dirty = any(word in cls_lower for word in ["dirty", "dirt", "trash", "plastic"])

    if looks_dirty:
        action = "BYPASS"
        reason = "looks dirty/contaminated"
    elif pred_conf < conf_thresh:
        action = "BYPASS"
        reason = f"low confidence ({pred_conf:.2f} < {conf_thresh})"
    elif distance_m > max_scoop_dist:
        action = "BYPASS"
        reason = f"too far away ({distance_m:.2f} m > {max_scoop_dist} m)"
    else:
        action = "SCOOP"
        reason = "clean and confident within safe distance"

    return {
        "pred_class": pred_class,
        "pred_conf": float(pred_conf),
        "distance_m": float(distance_m),
        "action": action,
        "reason": reason,
    }
    
    
    
# -----------------------------------------------------------------------------
# LiDAR: RELLIS KITTI .bin integration 
# -----------------------------------------------------------------------------

def split_zip_virtual_path(path_str: str) -> Optional[Tuple[str, str]]:
    """
    Given a Windows Explorer "virtual path" like:
      C:\\path\\file.zip\\inner\\folder
    return:
      (zip_path, inner_prefix_posix)

    If path_str does not contain ".zip", return None.
    """
    if path_str is None:
        return None

    lower = path_str.lower()
    if ".zip" not in lower:
        return None

    # Find the first occurrence of ".zip" in a case-insensitive way
    idx = lower.index(".zip") + 4
    zip_path = path_str[:idx]
    inner = path_str[idx:]  # includes separator + inner path
    inner = inner.lstrip("\\/")
    # Zip members use forward slashes
    inner_posix = inner.replace("\\", "/")
    # Normalize: remove leading "./"
    if inner_posix.startswith("./"):
        inner_posix = inner_posix[2:]
    # Ensure it ends with "/" for prefix matching, unless empty
    if inner_posix and not inner_posix.endswith("/"):
        inner_posix += "/"
    return zip_path, inner_posix


def list_kitti_bins(source_path: str) -> List[Tuple[str, Union[Path, Tuple[str, str]]]]:
    """
    Return a list of LiDAR bin "references" from either:
      - a directory containing *.bin files   -> ("dir", Path(...))
      - a zip virtual path                  -> ("zip", (zip_path, member_name))

    The returned list elements are normalized references and can be fed into
    read_kitti_bin_floats(...).
    """
    # If it's a virtual zip path, parse it and scan the zip contents
    z = split_zip_virtual_path(source_path)
    if z is not None:
        zip_path, inner_prefix = z
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")
        refs: List[Tuple[str, Union[Path, Tuple[str, str]]]] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                # filter to inner prefix and .bin
                if inner_prefix and not name.startswith(inner_prefix):
                    continue
                if name.lower().endswith(".bin"):
                    refs.append(("zip", (zip_path, name)))
        if not refs:
            raise FileNotFoundError(f"No .bin files found in zip under prefix '{inner_prefix}'")
        return refs

    # Otherwise treat as directory
    d = Path(source_path)
    if not d.is_dir():
        raise FileNotFoundError(f"RELLIS bin directory not found: {source_path}")
    bins = sorted(d.glob("*.bin"))
    if not bins:
        raise FileNotFoundError(f"No .bin files found in directory: {source_path}")
    return [("dir", p) for p in bins]


def read_kitti_bin_floats(ref: Tuple[str, Union[Path, Tuple[str, str]]]) -> array:
    """
    Read KITTI .bin point cloud into array('f') containing float32s:
      x, y, z, intensity repeated.
    """
    kind, payload = ref
    data = array("f")

    if kind == "dir":
        path = payload  # type: ignore
        assert isinstance(path, Path)
        n_floats = path.stat().st_size // 4
        with open(str(path), "rb") as f:
            data.fromfile(f, n_floats)
        return data

    if kind == "zip":
        zip_path, member = payload  # type: ignore
        assert isinstance(zip_path, str) and isinstance(member, str)
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(member, "r") as f:
                raw = f.read()
        # raw is little-endian float32 stream; array('f') assumes native endianness.
        # On typical x86/ARM little-endian platforms this is correct.
        data.frombytes(raw)
        return data

    raise ValueError(f"Unknown ref kind: {kind}")


def min_forward_distance_from_floats(
    floats: array,
    fov_deg: float = DEFAULT_FOV_DEG,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    stride: int = DEFAULT_STRIDE,
) -> Optional[float]:
    """
    Compute min planar distance sqrt(x^2 + y^2) within:
      - forward hemisphere (x > 0)
      - +- fov/2 radians
      - range <= max_range_m
    Uses stride sampling for speed.
    """
    if floats is None or len(floats) < 4:
        return None

    fov = math.radians(float(fov_deg))
    half = fov / 2.0
    max_r2 = float(max_range_m) * float(max_range_m)

    best_r2 = None
    step = 4 * max(1, int(stride))
    n = len(floats)

    for idx in range(0, n - 3, step):
        x = floats[idx + 0]
        y = floats[idx + 1]
        # z = floats[idx + 2]
        # i = floats[idx + 3]

        if x <= 0.0:
            continue

        ang = math.atan2(y, x)
        if abs(ang) > half:
            continue

        r2 = x * x + y * y
        if r2 <= 0.0 or r2 > max_r2:
            continue

        if best_r2 is None or r2 < best_r2:
            best_r2 = r2

    return math.sqrt(best_r2) if best_r2 is not None else None


def get_rellis_distance_m(
    rellis_source_path: str,
    fov_deg: float = DEFAULT_FOV_DEG,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    stride: int = DEFAULT_STRIDE,
) -> Tuple[float, str]:
    """
    Picks a random .bin frame from the RELLIS source and returns:
      (distance_m, provenance_str)

    If LiDAR data yields no valid points, returns (fallback_distance, reason).
    """
    refs = list_kitti_bins(rellis_source_path)
    chosen = random.choice(refs)
    floats = read_kitti_bin_floats(chosen)
    dist = min_forward_distance_from_floats(floats, fov_deg=fov_deg, max_range_m=max_range_m, stride=stride)

    if dist is None:
        return LIDAR_FALLBACK_DISTANCE_M, "RELLIS LiDAR: no valid points (fallback)"
    return float(dist), "RELLIS LiDAR (KITTI .bin)"
    
# -----------------------------------------------------------------------------
# Visualization with OpenCV only
# -----------------------------------------------------------------------------

def _require_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not installed or not importable in this environment.")

def load_and_resize(path: Path, tile_size=TILE_SIZE):
    _require_cv2()
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.resize(img, tile_size)


def make_gallery_image(clean_paths, dirty_paths):
    _require_cv2()
    if not clean_paths or not dirty_paths:
        raise ValueError("Need at least one clean and one dirty image to build gallery.")

    n = min(len(clean_paths), len(dirty_paths), NUM_PER_CLASS)

    clean_tiles = [load_and_resize(p) for p in clean_paths[:n]]
    dirty_tiles = [load_and_resize(p) for p in dirty_paths[:n]]

    top_row = cv2.hconcat(clean_tiles)
    bottom_row = cv2.hconcat(dirty_tiles)
    gallery = cv2.vconcat([top_row, bottom_row])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(gallery, "CLEAN (top row)", (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(gallery, "DIRTY (bottom row)", (10, gallery.shape[0] // 2 + 30),
                font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return gallery


def show_gallery_cv(clean_paths, dirty_paths):
    _require_cv2()
    gallery = make_gallery_image(clean_paths, dirty_paths)
    cv2.imshow("Regolith Gallery (CLEAN top, DIRTY bottom)", gallery)
    cv2.waitKey(0)
    cv2.destroyWindow("Regolith Gallery (CLEAN top, DIRTY bottom)")


def show_single_decision_cv(img_path: Path, decision: dict, true_label=None, lidar_source_note: str = ""):
    _require_cv2()
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

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
    if lidar_source_note:
        lines.append(lidar_source_note)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 25
    for line in lines:
        cv2.putText(display, line, (10, y), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y += 25

    cv2.imshow("Rover Decision", display)
    cv2.waitKey(0)
    cv2.destroyWindow("Rover Decision")



# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------

def _floats_to_bin_bytes(points: List[Tuple[float, float, float, float]]) -> bytes:
    """Helper for tests: build KITTI .bin bytes from a list of (x,y,z,i)."""
    a = array("f")
    for x, y, z, i in points:
        a.extend([x, y, z, i])
    return a.tobytes()


class TestZipVirtualPath(unittest.TestCase):
    def test_split_zip_virtual_path(self):
        p = r"C:\data\rellis.zip\Rellis-3D\00000\os1_cloud_node_kitti_bin"
        z = split_zip_virtual_path(p)
        self.assertIsNotNone(z)
        zip_path, inner = z  # type: ignore
        self.assertTrue(zip_path.lower().endswith(".zip"))
        self.assertIn("Rellis-3D/00000/os1_cloud_node_kitti_bin/", inner)

    def test_split_zip_virtual_path_none(self):
        self.assertIsNone(split_zip_virtual_path(r"C:\data\nozip\folder"))


class TestDistanceComputation(unittest.TestCase):
    def test_min_forward_distance_basic(self):
        # One point at 1m ahead, one at 2m, one behind
        raw = _floats_to_bin_bytes([
            (1.0, 0.0, 0.0, 0.0),
            (2.0, 0.1, 0.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0),
        ])
        f = array("f")
        f.frombytes(raw)
        d = min_forward_distance_from_floats(f, fov_deg=60, max_range_m=10, stride=1)
        self.assertIsNotNone(d)
        self.assertAlmostEqual(d or 0.0, 1.0, places=3)

    def test_fov_filter_excludes(self):
        # Point at 45 degrees should be excluded when fov=30 (half=15)
        raw = _floats_to_bin_bytes([(1.0, 1.0, 0.0, 0.0)])  # ~45 deg
        f = array("f"); f.frombytes(raw)
        d = min_forward_distance_from_floats(f, fov_deg=30, max_range_m=10, stride=1)
        self.assertIsNone(d)

    def test_range_filter_excludes(self):
        raw = _floats_to_bin_bytes([(20.0, 0.0, 0.0, 0.0)])
        f = array("f"); f.frombytes(raw)
        d = min_forward_distance_from_floats(f, fov_deg=60, max_range_m=10, stride=1)
        self.assertIsNone(d)

    def test_stride_still_finds_near_point(self):
        # Build many points; near point first; stride shouldn't skip it if stride>1 but still starts at 0
        pts = [(1.2, 0.0, 0.0, 0.0)] + [(5.0, 0.0, 0.0, 0.0)] * 100
        raw = _floats_to_bin_bytes(pts)
        f = array("f"); f.frombytes(raw)
        d = min_forward_distance_from_floats(f, fov_deg=60, max_range_m=10, stride=10)
        self.assertIsNotNone(d)
        self.assertAlmostEqual(d or 0.0, 1.2, places=3)


class TestDecisionLogic(unittest.TestCase):
    def test_dirty_class_bypass(self):
        out = decide_action("dirty", 0.99, 0.5)
        self.assertEqual(out["action"], "BYPASS")

    def test_low_conf_bypass(self):
        out = decide_action("clean", 0.3, 0.5, conf_thresh=0.6)
        self.assertEqual(out["action"], "BYPASS")

    def test_far_bypass(self):
        out = decide_action("clean", 0.99, 3.0, max_scoop_dist=2.5)
        self.assertEqual(out["action"], "BYPASS")

    def test_scoop(self):
        out = decide_action("clean", 0.99, 1.0, conf_thresh=0.6, max_scoop_dist=2.5)
        self.assertEqual(out["action"], "SCOOP")


class TestZipListing(unittest.TestCase):
    def test_list_kitti_bins_from_zip_prefix(self):
        with TemporaryDirectory() as td:
            zip_path = Path(td) / "test.zip"
            inner_prefix = "Rellis-3D/00000/os1_cloud_node_kitti_bin/"
            member1 = inner_prefix + "000000.bin"
            member2 = inner_prefix + "000001.bin"
            other = "other/skip.bin"

            with zipfile.ZipFile(str(zip_path), "w") as zf:
                zf.writestr(member1, _floats_to_bin_bytes([(1.0, 0.0, 0.0, 0.0)]))
                zf.writestr(member2, _floats_to_bin_bytes([(2.0, 0.0, 0.0, 0.0)]))
                zf.writestr(other, _floats_to_bin_bytes([(0.5, 0.0, 0.0, 0.0)]))

            virtual = str(zip_path) + "\\" + inner_prefix.replace("/", "\\").rstrip("\\")
            refs = list_kitti_bins(virtual)
            self.assertEqual(len(refs), 2)
            self.assertTrue(all(r[0] == "zip" for r in refs))


def run_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="YOLO regolith demo decider + RELLIS LiDAR distance (KITTI .bin)")
    ap.add_argument("--model", default=str(MODEL_PATH), help="Path to YOLOv8 classification model")
    ap.add_argument("--clean-dir", default=str(CLEAN_DIR), help="Directory of clean images")
    ap.add_argument("--dirty-dir", default=str(DIRTY_DIR), help="Directory of dirty images")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducible selection")
    ap.add_argument("--no-gallery", action="store_true", help="Skip gallery window")
    ap.add_argument("--test", action="store_true", help="Run unit tests and exit")

    # LiDAR options
    ap.add_argument("--rellis-kitti-path", default=DEFAULT_RELLIS_KITTI_PATH,
                    help="Directory of *.bin files OR a 'virtual zip path' ending in ...zip\\inner\\folder")
    ap.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG, help="Forward sector FOV degrees")
    ap.add_argument("--max-range-m", type=float, default=DEFAULT_MAX_RANGE_M, help="Max LiDAR range considered (m)")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Point sampling stride (>=1)")

    args = ap.parse_args()

    if args.test:
        return run_tests()

    random.seed(int(args.seed))

    if cv2 is None or YOLO is None:
        raise RuntimeError("This demo requires cv2 and ultralytics. Install them or run with --test for unit tests.")

    clean_dir = Path(args.clean_dir)
    dirty_dir = Path(args.dirty_dir)
    model_path = Path(args.model)

    # 1) Collect images
    clean_paths = get_image_paths(clean_dir, NUM_PER_CLASS)
    dirty_paths = get_image_paths(dirty_dir, NUM_PER_CLASS)

    # 2) Show gallery (OpenCV only)
    if not args.no_gallery:
        show_gallery_cv(clean_paths, dirty_paths)

    # 3) Load YOLO model
    model = load_yolo_model(model_path)

    # 4) Pick one image
    all_paths = clean_paths + dirty_paths
    chosen_path = random.choice(all_paths)
    true_label = infer_true_label(chosen_path)

    # 5) Get LiDAR distance from RELLIS KITTI bins
    try:
        distance_m, lidar_note = get_rellis_distance_m(
            args.rellis_kitti_path,
            fov_deg=args.fov_deg,
            max_range_m=args.max_range_m,
            stride=args.stride,
        )
    except Exception as e:
        distance_m = LIDAR_FALLBACK_DISTANCE_M
        lidar_note = f"RELLIS LiDAR unavailable: {e} (fallback)"

    distance_m = round(float(distance_m), 2)

    print("\n========== ROVER DECISION DEMO (YOLO + RELLIS LiDAR distance) ==========")
    print(f"Selected image : {chosen_path}")
    print(f"True label     : {true_label}")
    print(f"Distance (m)   : {distance_m}  |  {lidar_note}")
    print("=======================================================================")

    # 6) Run YOLO inference + decision logic
    frame = cv2.imread(str(chosen_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to read image: {chosen_path}")

    pred_class, pred_conf = yolo_infer_top1(model, frame)
    decision = decide_action(pred_class, pred_conf, distance_m)

    print("\n=== Decision ===")
    print(f"Predicted class : {decision['pred_class']}")
    print(f"Confidence      : {decision['pred_conf']:.4f}")
    print(f"Action          : {decision['action']}")
    print(f"Reason          : {decision['reason']}")
    print("=======================================================================")

    # 7) Visualize selected image + decision overlay (OpenCV only)
    show_single_decision_cv(chosen_path, decision, true_label=true_label, lidar_source_note=lidar_note)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
