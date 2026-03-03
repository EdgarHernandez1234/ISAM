#!/usr/bin/env python3
"""
space_management.py

1) Optional: cleanup stale camera/GStreamer holders (Jetson-friendly)
2) Prune old "run" artifact folders to keep storage under control.

Default pruning:
- Detect run folders under --base-dir (flat or grouped)
- Keep newest --keep-last runs (default 2)
- Delete older runs and report freed space

Camera cleanup notes:
- This script cannot "release()" objects inside another process.
- It can only terminate processes that are holding camera/GStreamer resources.
# space_management.py — Notes / Ops Guide

This utility does two things:
1) **Prunes old run artifacts** (disk space management)
2) **Optionally cleans up camera/GStreamer holders** (frees NVMM/Argus resources indirectly)

⚠️ Important: A separate script cannot call `release()` on a camera/pipeline object owned by another Python process.
The only external way to free those resources is to **terminate the process(es)** holding them (and optionally restart Argus).

## Typical Usage

### Safe dry-run (recommended first)
```bash
python3 space_management.py --base-dir ~/Desktop/demo_artifacts --keep-last 2 --dry-run

Prune only (keep newest 2 runs)
python3 space_management.py --base-dir ~/Desktop/demo_artifacts --keep-last 2

Prune + camera cleanup (recommended between trial runs)
python3 space_management.py --base-dir ~/Desktop/demo_artifacts --keep-last 2 --cleanup-camera




"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Set


ARTIFACT_EXTS = {".csv", ".mp4", ".json", ".png", ".jpg", ".jpeg", ".log", ".txt"}

# Conservative process signatures that commonly indicate a GStreamer/Argus pipeline
DEFAULT_MEDIA_PATTERNS = [
    r"\bgst-launch-1\.0\b",
    r"\bnvarguscamerasrc\b",
    r"\bgstreamer\b",
]

# Default camera device nodes to check
DEFAULT_CAMERA_DEVICES = ["/dev/video0", "/dev/video1", "/dev/video2", "/dev/video3"]


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} TB"


def is_dangerous_base_dir(p: Path) -> bool:
    p = p.resolve()
    dangerous = {Path("/"), Path.home().resolve()}
    high_level = {Path("/home"), Path("/mnt"), Path("/media"), Path("/var"), Path("/usr"), Path("/opt")}
    return p in dangerous or p in high_level


def dir_size_bytes(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except FileNotFoundError:
                pass
    return total


def dir_last_activity_ts(p: Path) -> float:
    latest = None
    for root, _, files in os.walk(p):
        for f in files:
            fp = Path(root) / f
            try:
                mt = fp.stat().st_mtime
                latest = mt if latest is None else max(latest, mt)
            except FileNotFoundError:
                continue
    if latest is not None:
        return latest
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def looks_like_run_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    try:
        for child in p.iterdir():
            if child.is_file():
                if child.suffix.lower() in ARTIFACT_EXTS:
                    return True
                if "metadata" in child.name.lower():
                    return True
    except FileNotFoundError:
        return False
    return False


@dataclass
class RunFolder:
    path: Path
    last_ts: float
    size_bytes: int


def collect_run_folders(base_dir: Path, match: Optional[str]) -> Tuple[str, List[RunFolder] | List[Tuple[Path, List[RunFolder]]]]:
    base_children = [c for c in base_dir.iterdir() if c.is_dir()]
    if match:
        base_children = [c for c in base_children if fnmatch.fnmatch(c.name, match)]

    flat_candidates = [c for c in base_children if looks_like_run_dir(c)]
    if flat_candidates:
        runs: List[RunFolder] = []
        for d in flat_candidates:
            runs.append(RunFolder(d, dir_last_activity_ts(d), dir_size_bytes(d)))
        return "flat", runs

    grouped: List[Tuple[Path, List[RunFolder]]] = []
    for group in base_children:
        subdirs = [s for s in group.iterdir() if s.is_dir()]
        run_dirs = [s for s in subdirs if looks_like_run_dir(s)]
        if not run_dirs:
            continue
        runs = [RunFolder(d, dir_last_activity_ts(d), dir_size_bytes(d)) for d in run_dirs]
        grouped.append((group, runs))

    if grouped:
        return "grouped", grouped

    return "flat", []


# ---------------------------
# Camera / GStreamer cleanup
# ---------------------------

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_pids_from_text(text: str) -> Set[int]:
    # Extract integers that look like PIDs
    pids = set()
    for m in re.finditer(r"\b(\d{2,})\b", text):
        try:
            pids.add(int(m.group(1)))
        except ValueError:
            pass
    return pids


def pids_using_device(device: str) -> Set[int]:
    """
    Try fuser first (common on Linux), fallback to lsof.
    Returns empty set if tools are not available or device doesn't exist.
    """
    dev_path = Path(device)
    if not dev_path.exists():
        return set()

    # fuser
    if which("fuser"):
        cp = run_cmd(["fuser", device])
        # fuser writes PIDs on stdout (sometimes stderr); parse both
        return parse_pids_from_text((cp.stdout or "") + "\n" + (cp.stderr or ""))

    # lsof
    if which("lsof"):
        cp = run_cmd(["lsof", "-t", device])
        return parse_pids_from_text((cp.stdout or "") + "\n" + (cp.stderr or ""))

    return set()


def list_process_table() -> Dict[int, str]:
    """
    Returns {pid: cmdline}
    """
    cp = run_cmd(["ps", "-eo", "pid=,args="])
    table: Dict[int, str] = {}
    for line in (cp.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmdline = parts[1] if len(parts) > 1 else ""
        table[pid] = cmdline
    return table


def matches_any_pattern(cmdline: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if re.search(pat, cmdline):
            return True
    return False


def kill_pids(pids: Iterable[int], dry_run: bool, exclude: Set[int]) -> None:
    """
    SIGTERM then SIGKILL after short wait.
    """
    pids = [p for p in set(pids) if p not in exclude and p > 1]
    if not pids:
        return

    if dry_run:
        print(f"[DRY-RUN] Would terminate PIDs: {sorted(pids)}")
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            print(f"WARNING: No permission to SIGTERM pid {pid}")

    # brief wait
    time.sleep(0.5)

    # force kill remaining
    for pid in pids:
        try:
            os.kill(pid, 0)  # still alive?
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            print(f"WARNING: No permission to SIGKILL pid {pid}")


def cleanup_camera_gstreamer(
    camera_devices: List[str],
    media_patterns: List[str],
    dry_run: bool,
    force_kill_camera_users: bool,
    restart_nvargus_daemon: bool,
) -> None:
    """
    Conservative cleanup:
    - Find PIDs using /dev/video* via fuser/lsof
    - Intersect with media-pattern processes (gst/argus) unless force_kill_camera_users=True
    - Kill those PIDs (excluding self + parent)
    - Optionally attempt to restart nvargus-daemon (requires sudo)
    """
    exclude = {os.getpid(), os.getppid()}
    proc_table = list_process_table()

    device_pids: Set[int] = set()
    for dev in camera_devices:
        device_pids |= pids_using_device(dev)

    if not device_pids:
        print("Camera cleanup: no processes detected using camera devices (or no fuser/lsof available).")
    else:
        print(f"Camera cleanup: detected device-holder PIDs: {sorted(device_pids)}")

    if force_kill_camera_users:
        target_pids = {p for p in device_pids if p not in exclude}
    else:
        media_pids = {pid for pid, cmd in proc_table.items() if matches_any_pattern(cmd, media_patterns)}
        target_pids = (device_pids & media_pids) - exclude

        if device_pids and not target_pids:
            print("Camera cleanup: device-holder PIDs found, but none matched GStreamer/Argus patterns.")
            print("  If the camera is stuck and you want to kill any holder, re-run with --force-kill-camera-users")

    if target_pids:
        print("Camera cleanup: terminating likely GStreamer/Argus camera holders:")
        for pid in sorted(target_pids):
            print(f"  PID {pid}: {proc_table.get(pid, '(cmdline unavailable)')}")
        kill_pids(target_pids, dry_run=dry_run, exclude=exclude)
    else:
        print("Camera cleanup: nothing to terminate.")

    if restart_nvargus_daemon:
        # This typically needs sudo; we attempt and report errors.
        cmd = ["systemctl", "restart", "nvargus-daemon"]
        print(f"Camera cleanup: attempting {'(dry-run) ' if dry_run else ''}restart of nvargus-daemon")
        if dry_run:
            print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        else:
            if not which("systemctl"):
                print("WARNING: systemctl not found; cannot restart nvargus-daemon.")
            else:
                cp = subprocess.run(cmd, capture_output=True, text=True)
                if cp.returncode != 0:
                    # Most likely permission issue
                    print("WARNING: nvargus-daemon restart failed.")
                    if cp.stderr:
                        print(cp.stderr.strip())


# ---------------------------
# Pruning logic
# ---------------------------

def delete_path(p: Path, dry_run: bool) -> int:
    sz = dir_size_bytes(p)
    if dry_run:
        print(f"[DRY-RUN] Would delete: {p}  ({human_bytes(sz)})")
        return sz
    print(f"Deleting: {p}  ({human_bytes(sz)})")
    shutil.rmtree(p, ignore_errors=True)
    return sz


def prune_run_list(runs: List[RunFolder], keep_last: int, dry_run: bool) -> int:
    if keep_last < 0:
        keep_last = 0
    runs_sorted = sorted(runs, key=lambda r: r.last_ts, reverse=True)
    to_keep = runs_sorted[:keep_last]
    to_delete = runs_sorted[keep_last:]

    if not runs_sorted:
        print("No run folders detected to prune.")
        return 0

    print(f"Detected {len(runs_sorted)} run folder(s). Keeping newest {len(to_keep)}; pruning {len(to_delete)}.")
    for r in to_keep:
        print(f"KEEP  : {r.path}  (last={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r.last_ts))}, size={human_bytes(r.size_bytes)})")

    freed = 0
    for r in to_delete:
        freed += delete_path(r.path, dry_run=dry_run)

    return freed


def main() -> int:
    ap = argparse.ArgumentParser(description="Cleanup camera holders + prune old trial run artifacts.")
    ap.add_argument("--base-dir", type=str, default="~/Desktop/demo_artifacts",
                    help="Directory that contains run folders (default: ~/Desktop/demo_artifacts)")
    ap.add_argument("--keep-last", type=int, default=2,
                    help="How many newest runs to keep (default: 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be deleted / terminated, but do not change anything")
    ap.add_argument("--match", type=str, default=None,
                    help="Optional wildcard match for run dirs (e.g., 'run_*').")
    ap.add_argument("--min-free-gb", type=float, default=None,
                    help="Optional: report whether at least this many GB are free after pruning.")
    ap.add_argument("--force", action="store_true",
                    help="Allow pruning in dangerous base dirs (/, ~, /home, etc).")

    # Camera/GStreamer cleanup switches
    ap.add_argument("--cleanup-camera", action="store_true",
                    help="Attempt to free camera resources by terminating stale GStreamer/Argus holders.")
    ap.add_argument("--camera-devices", type=str, default=",".join(DEFAULT_CAMERA_DEVICES),
                    help="Comma-separated device nodes to check (default: /dev/video0,/dev/video1,...)")
    ap.add_argument("--force-kill-camera-users", action="store_true",
                    help="Aggressive: kill ANY process using the camera devices (not just gst/argus pattern matches).")
    ap.add_argument("--restart-nvargus-daemon", action="store_true",
                    help="Attempt to restart nvargus-daemon (requires permission).")
    ap.add_argument("--media-pattern", action="append", default=[],
                    help="Additional regex pattern(s) to identify media pipeline processes. Can be used multiple times.")

    args = ap.parse_args()
    base_dir = Path(os.path.expanduser(args.base_dir)).resolve()

    if not base_dir.exists() or not base_dir.is_dir():
        print(f"ERROR: base-dir does not exist or is not a directory: {base_dir}")
        return 2

    if is_dangerous_base_dir(base_dir) and not args.force:
        print(f"ERROR: Refusing to run on high-level directory: {base_dir}")
        print("       Pass --force if you are absolutely sure, or point --base-dir to a specific artifacts folder.")
        return 3

    # 1) Optional camera cleanup (run BEFORE pruning / BEFORE next trial opens camera)
    if args.cleanup_camera:
        devices = [d.strip() for d in args.camera_devices.split(",") if d.strip()]
        patterns = DEFAULT_MEDIA_PATTERNS + (args.media_pattern or [])
        print("=== Camera/GStreamer cleanup ===")
        cleanup_camera_gstreamer(
            camera_devices=devices,
            media_patterns=patterns,
            dry_run=args.dry_run,
            force_kill_camera_users=args.force_kill_camera_users,
            restart_nvargus_daemon=args.restart_nvargus_daemon,
        )
        print("=== End camera cleanup ===\n")

    # 2) Prune artifacts
    mode, collected = collect_run_folders(base_dir, match=None)
    total_freed = 0

    if mode == "flat":
        runs: List[RunFolder] = collected  # type: ignore[assignment]
        if args.match:
            runs = [r for r in runs if fnmatch.fnmatch(r.path.name, args.match)]
        total_freed += prune_run_list(runs, keep_last=args.keep_last, dry_run=args.dry_run)

    else:
        grouped: List[Tuple[Path, List[RunFolder]]] = collected  # type: ignore[assignment]
        any_group = False
        for group_path, runs in grouped:
            if args.match:
                runs = [r for r in runs if fnmatch.fnmatch(r.path.name, args.match)]
            if not runs:
                continue
            any_group = True
            print(f"\n=== Group: {group_path} ===")
            total_freed += prune_run_list(runs, keep_last=args.keep_last, dry_run=args.dry_run)

        if not any_group:
            print("No run folders detected to prune (grouped layout scan found nothing).")

    # 3) Optional free-space reporting
    if args.min_free_gb is not None:
        target_free = int(args.min_free_gb * (1024**3))
        usage = shutil.disk_usage(base_dir)
        free_now = usage.free
        print(f"\nFree space now: {human_bytes(free_now)} (target: {human_bytes(target_free)})")
        if free_now < target_free:
            print("WARNING: target free space not met after pruning.")
        else:
            print("Target free space satisfied.")

    print(f"\nTotal space {'that would be freed' if args.dry_run else 'freed'}: {human_bytes(total_freed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())