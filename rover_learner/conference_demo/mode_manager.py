#!/usr/bin/env python3
"""mode_manager.py

ALAM conference_demo — Mode & sensor gating layer.

Why this exists
--------------
If the policy layer treats "gated" == "failed", you can end up in a feedback loop:
  1) policy disables LiDAR -> mode becomes cam-only
  2) cam-only mode reports lidar_ok=False (because it is gated)
  3) policy concludes LiDAR is "lost" and keeps it disabled forever

This file prevents that by tracking *LiDAR heartbeat* independently of whether the
current mode is using LiDAR for distance.

It also supports a temporary stabilization override used by UNBREAKABLE:
briefly drop to a lightweight mode during the "catch" motion, then revert.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Tuple

import cv2
import numpy as np


class OperatingMode(Enum):
    """How the controller should *consume* sensor streams."""
    DUAL_CAM_LIDAR = auto()   # Full perception: A+B cameras, LiDAR
    SINGLE_CAM_LIDAR = auto() # Only camera A used, LiDAR still used
    SINGLE_CAM_ONLY = auto()  # Camera only (LiDAR gated from use)
    HALT = auto()             # Safety stop (visual halt overlay)


@dataclass
class FramePacket:
    """Return type for ModeManager.get_frame()."""
    frame: np.ndarray
    timestamp: float
    cam_a_ok: bool
    cam_b_ok: bool


@dataclass
class LidarPacket:
    """Return type for ModeManager.get_distance()."""
    distance_m: Optional[float]
    ok: bool


class ModeManager:
    """Abstraction layer between providers and the demo controller."""

    def __init__(self, cam_provider: Any, lidar_provider: Any):
        self.cam = cam_provider
        self.lidar = lidar_provider

        self.current_mode: OperatingMode = OperatingMode.DUAL_CAM_LIDAR

        # throttles (read by beta_conf for pacing)
        self.res_scale: float = 1.0
        self.fps_target: int = 30

        # black fallback (dimensions updated from first valid frame)
        self._black_h = 360
        self._black_w = 640
        self._black_frame = np.zeros((self._black_h, self._black_w, 3), dtype=np.uint8)

        # -----------------------------
        # LiDAR heartbeat caching
        # -----------------------------
        self._lidar_last_ok_ts: float = 0.0
        self._lidar_last_distance_m: Optional[float] = None
        self._lidar_last_probe_ts: float = 0.0
        self._lidar_probe_interval_s: float = 0.25
        self._lidar_alive_window_s: float = 1.5

        # -----------------------------
        # Temporary stabilization override (demo-driven)
        # -----------------------------
        self._stabilize_until_ts: float = 0.0
        self._stabilize_mode: OperatingMode = OperatingMode.SINGLE_CAM_LIDAR
        self._stabilize_reason: str = ""

    # ============================================================
    # Mode control
    # ============================================================

    def set_mode(self, new_mode: OperatingMode) -> None:
        """Safe transition between modes without restarting providers."""
        if self.current_mode != new_mode:
            print(f"[ModeManager] Transition: {self.current_mode.name} -> {new_mode.name}")
            self.current_mode = new_mode

    def apply_plan(self, plan: Any) -> None:
        """Apply a ResourcePlan-like object (duck-typed).

        Expected attributes (if present):
          - mode: SystemMode enum or str ("NOMINAL"/"DEGRADED"/"CRITICAL")
          - camera_count: int
          - lidar_on: bool
          - res_scale: float
          - fps_target: int
        """
        now = time.time()

        # expire stabilization automatically
        if self._stabilize_until_ts > 0.0 and now >= self._stabilize_until_ts:
            self.clear_stabilization()

        # throttles
        if hasattr(plan, "res_scale"):
            try:
                self.res_scale = float(getattr(plan, "res_scale"))
            except Exception:
                pass
        if hasattr(plan, "fps_target"):
            try:
                ft = int(getattr(plan, "fps_target"))
                if ft > 0:
                    self.fps_target = ft
            except Exception:
                pass

        # CRITICAL -> HALT
        if _plan_is_critical(plan) or int(getattr(plan, "camera_count", 1)) <= 0:
            self.clear_stabilization()
            self.set_mode(OperatingMode.HALT)
            return

        lidar_on = bool(getattr(plan, "lidar_on", True))
        cam_count = int(getattr(plan, "camera_count", 1))

        if not lidar_on:
            desired = OperatingMode.SINGLE_CAM_ONLY
        else:
            desired = OperatingMode.DUAL_CAM_LIDAR if cam_count >= 2 else OperatingMode.SINGLE_CAM_LIDAR

        # Stabilization override (never forces LiDAR ON if policy disabled it)
        if self.is_stabilizing() and lidar_on:
            self.set_mode(self._stabilize_mode)
        else:
            self.set_mode(desired)

    # ---- stabilization -------------------------------------------------

    def request_stabilization(
        self,
        *,
        duration_s: float = 2.0,
        mode: OperatingMode = OperatingMode.SINGLE_CAM_LIDAR,
        reason: str = "",
    ) -> None:
        """Temporarily force a lighter mode during a motion segment."""
        now = time.time()
        self._stabilize_mode = mode
        self._stabilize_reason = reason
        self._stabilize_until_ts = now + max(0.1, float(duration_s))
        print(f"[ModeManager] Stabilization requested: {mode.name} for {duration_s:.1f}s ({reason})")

    def clear_stabilization(self) -> None:
        if self._stabilize_until_ts > 0.0 and self._stabilize_reason:
            print(f"[ModeManager] Stabilization cleared ({self._stabilize_reason})")
        self._stabilize_until_ts = 0.0
        self._stabilize_reason = ""

    def is_stabilizing(self) -> bool:
        return self._stabilize_until_ts > 0.0 and time.time() < self._stabilize_until_ts

    # ============================================================
    # Sensors
    # ============================================================

    def get_distance(self) -> LidarPacket:
        """Read LiDAR.

        - We probe LiDAR periodically *even if gated*.
        - ok indicates if LiDAR is alive (recent valid probe).
        - distance_m is returned only when mode uses LiDAR.
        """
        if not self.lidar:
            return LidarPacket(distance_m=None, ok=False)

        self._probe_lidar_if_due()
        alive = self._lidar_is_alive()

        if self.current_mode in (OperatingMode.SINGLE_CAM_ONLY, OperatingMode.HALT):
            return LidarPacket(distance_m=None, ok=alive)

        return LidarPacket(distance_m=self._lidar_last_distance_m, ok=alive)

    def get_frame(self) -> FramePacket:
        """Read camera(s), apply gating, return a stable display frame."""
        frame_a, frame_b, ts = self._read_camera_any()
        cam_a_ok = frame_a is not None
        cam_b_ok = frame_b is not None

        ref = frame_a if frame_a is not None else frame_b
        self._update_black_dims(ref)

        if self.current_mode == OperatingMode.HALT:
            out = self._apply_halt_overlay(ref)
            return FramePacket(frame=out, timestamp=ts, cam_a_ok=cam_a_ok, cam_b_ok=cam_b_ok)

        if self.current_mode == OperatingMode.DUAL_CAM_LIDAR:
            out = self._stitch_dual_view(frame_a, frame_b)
        else:
            out = frame_a if frame_a is not None else self._black_frame.copy()

        out = self._apply_res_scale(out)
        return FramePacket(frame=out, timestamp=ts, cam_a_ok=cam_a_ok, cam_b_ok=cam_b_ok)

    # ============================================================
    # Internal helpers
    # ============================================================

    def _read_camera_any(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Supports both single-cam and dual-cam providers."""
        ts = time.time()
        try:
            raw = self.cam.read()
        except Exception:
            return None, None, ts

        # TwoCameraProvider: (fa, fb, ts)
        if isinstance(raw, tuple) and len(raw) == 3:
            fa, fb, t = raw
            return fa, fb, float(t)

        # Single camera: (fa, ts)
        if isinstance(raw, tuple) and len(raw) == 2:
            fa, t = raw
            return fa, None, float(t)

        # Some providers: raw frame only
        if isinstance(raw, np.ndarray):
            return raw, None, ts

        return None, None, ts

    def _stitch_dual_view(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> np.ndarray:
        """Vertical stack A over B (fallbacks if missing)."""
        if a is None and b is None:
            return self._black_frame.copy()
        if a is None:
            return b
        if b is None:
            return a

        if b.shape[:2] != a.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        return np.vstack((a, b))

    def _apply_res_scale(self, frame: np.ndarray) -> np.ndarray:
        """Apply resolution scaling for performance."""
        s = float(self.res_scale)
        if s >= 0.999:
            return frame
        h, w = frame.shape[:2]
        nh = max(8, int(h * s))
        nw = max(8, int(w * s))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def _apply_halt_overlay(self, ref: Optional[np.ndarray]) -> np.ndarray:
        """Visual indicator that the system is halted."""
        if ref is None:
            h, w = self._black_h, self._black_w
        else:
            h, w = ref.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(canvas, "SYSTEM HALTED", (max(10, w // 4), h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        return canvas

    def _update_black_dims(self, ref: Optional[np.ndarray]) -> None:
        if ref is None:
            return
        h, w = ref.shape[:2]
        if (h, w) != (self._black_h, self._black_w):
            self._black_h, self._black_w = h, w
            self._black_frame = np.zeros((h, w, 3), dtype=np.uint8)

    # ---- LiDAR heartbeat ------------------------------------------------

    def _probe_lidar_if_due(self) -> None:
        now = time.time()
        if (now - self._lidar_last_probe_ts) < self._lidar_probe_interval_s:
            return
        self._lidar_last_probe_ts = now

        try:
            d = self.lidar.get_distance_m()
        except Exception:
            d = None

        if d is not None:
            self._lidar_last_distance_m = float(d)
            self._lidar_last_ok_ts = now

    def _lidar_is_alive(self) -> bool:
        return self._lidar_last_ok_ts > 0.0 and (time.time() - self._lidar_last_ok_ts) < self._lidar_alive_window_s


def _plan_is_critical(plan: Any) -> bool:
    """Robust CRITICAL check across enums/strings."""
    if plan is None:
        return False
    mode = getattr(plan, "mode", None)
    if mode is None:
        return False
    if hasattr(mode, "name") and str(getattr(mode, "name")) == "CRITICAL":
        return True
    if isinstance(mode, str) and mode.upper() == "CRITICAL":
        return True
    return False

