"""rover_decider.rellis_lidar

RELLIS-3D KITTI .bin support (stdlib only).

- Supports directory trees or Windows Explorer-style "virtual zip path":
    C:\path\dataset.zip\Rellis-3D\00000\os1_cloud_node_kitti_bin

- Reads float32 point clouds (x, y, z, intensity repeating)
- Computes a robust scalar distance: min planar distance within a forward FOV sector

These helpers are designed to be unit-tested without cv2/YOLO.
"""

from __future__ import annotations

import math
import os
import random
import zipfile
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple, Union


# Defaults
DEFAULT_FOV_DEG = 30.0
DEFAULT_MAX_RANGE_M = 10.0
DEFAULT_STRIDE = 5

LIDAR_FALLBACK_DISTANCE_M = 999.0


def split_zip_virtual_path(path_str: str) -> Optional[Tuple[str, str]]:
    """Parse a Windows Explorer-style zip virtual path.

    Example:
      C:\data\rellis.zip\Rellis-3D\00000\os1_cloud_node_kitti_bin

    Returns:
      (zip_path, inner_prefix_posix_with_trailing_slash)

    If path_str does not contain ".zip" returns None.
    """
    if path_str is None:
        return None

    lower = path_str.lower()
    if ".zip" not in lower:
        return None

    idx = lower.index(".zip") + 4
    zip_path = path_str[:idx]
    inner = path_str[idx:]
    inner = inner.lstrip("\\/")
    inner_posix = inner.replace("\\", "/")
    if inner_posix.startswith("./"):
        inner_posix = inner_posix[2:]
    if inner_posix and not inner_posix.endswith("/"):
        inner_posix += "/"

    return zip_path, inner_posix


def list_kitti_bins(source_path: str) -> List[Tuple[str, Union[Path, Tuple[str, str]]]]:
    """Return normalized references to *.bin frames.

    Each element is either:
      ("dir", Path("/abs/.../000123.bin"))
      ("zip", ("/abs/.../dataset.zip", "inner/prefix/000123.bin"))
    """
    z = split_zip_virtual_path(source_path)
    if z is not None:
        zip_path, inner_prefix = z
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        refs: List[Tuple[str, Union[Path, Tuple[str, str]]]] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if inner_prefix and not name.startswith(inner_prefix):
                    continue
                if name.lower().endswith(".bin"):
                    refs.append(("zip", (zip_path, name)))

        if not refs:
            raise FileNotFoundError(f"No .bin files found in zip under prefix '{inner_prefix}'")
        return refs

    d = Path(source_path)
    if not d.exists():
        raise FileNotFoundError(f"RELLIS path not found: {source_path}")

    if d.is_dir():
        bins = sorted(d.glob("*.bin"))
        if bins:
            return [("dir", p) for p in bins]

        # common folder name
        for sub in d.rglob("os1_cloud_node_kitti_bin"):
            if sub.is_dir():
                b = sorted(sub.glob("*.bin"))
                if b:
                    return [("dir", p) for p in b]

        # fallback: any .bin
        bins_all = [p for p in d.rglob("*.bin")]
        if not bins_all:
            raise FileNotFoundError(f"No .bin files found under directory tree: {source_path}")

        bins_pref = [
            p for p in bins_all
            if ("kitti" in str(p).lower()) or ("os1_cloud_node" in str(p).lower())
        ]
        bins_use = sorted(bins_pref) if bins_pref else sorted(bins_all)
        return [("dir", p) for p in bins_use]

    raise FileNotFoundError(f"RELLIS bin directory not found: {source_path}")


def read_kitti_bin_floats(ref: Tuple[str, Union[Path, Tuple[str, str]]]) -> array:
    """Read KITTI .bin into array('f') float stream: x,y,z,i repeating."""
    kind, payload = ref
    data = array("f")

    if kind == "dir":
        path = payload
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
        data.frombytes(raw)
        return data

    raise ValueError(f"Unknown ref kind: {kind}")


def min_forward_distance_from_floats(
    floats: array,
    fov_deg: float = DEFAULT_FOV_DEG,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    stride: int = DEFAULT_STRIDE,
) -> Optional[float]:
    """Compute min planar distance sqrt(x^2 + y^2) within:

    - x > 0 (forward)
    - |atan2(y,x)| <= fov/2
    - distance <= max_range

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
    rng: Optional[random.Random] = None,
) -> Tuple[float, str]:
    """Pick a random KITTI frame from the RELLIS source and return (distance_m, note)."""
    refs = list_kitti_bins(rellis_source_path)
    r = rng if rng is not None else random
    chosen = r.choice(refs)
    floats = read_kitti_bin_floats(chosen)
    dist = min_forward_distance_from_floats(
        floats,
        fov_deg=fov_deg,
        max_range_m=max_range_m,
        stride=stride,
    )

    if dist is None:
        return LIDAR_FALLBACK_DISTANCE_M, "RELLIS LiDAR: no valid points (fallback)"

    return float(dist), "RELLIS LiDAR (KITTI .bin)"


# -----------------------------
# Test helpers
# -----------------------------

def floats_to_bin_bytes(points: List[Tuple[float, float, float, float]]) -> bytes:
    """Build KITTI .bin bytes from list of (x,y,z,i)."""
    a = array("f")
    for x, y, z, i in points:
        a.extend([x, y, z, i])
    return a.tobytes()
