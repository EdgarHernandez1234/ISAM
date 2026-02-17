#!/usr/bin/env python3
"""
power_saver_degradation.py
==========================

ALAM Power Saver + Degradation Policy Engine (Single Source of Truth)

Folder intent (per your layout):
  rover_learner/
    conference_demo/
      power_saver_degradation.py   <-- THIS FILE
      mode_manager.py
      master_controller.py

ROLE IN THE STACK
-----------------
This module computes a ResourcePlan from:
  - Jetson "system health" signals: CPU temp, free RAM, CPU load (best-effort)
  - Sensor heartbeat timestamps: camera + LiDAR (from master_controller)
  - Optional sensor availability hints: camera_count_ok, lidar_seen_recently

The ResourcePlan is then enforced by ModeManager (gating/throttling) and used by
master_controller (UI/logging + possible RETURN_HOME override).

DESIGN CHOICE: Prefer gating, not restarting
--------------------------------------------
Conference demos are more stable if we keep drivers "hot" and *ignore* data rather than
repeatedly closing/reopening device drivers.

This file is dependency-light (no OpenCV, no ROS, no psutil).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

# ------------------------------------------------------------
# Policy constants (tune after Jetson profiling)
# ------------------------------------------------------------

# Temperature (°C)
THERMAL_WARNING_C = 70.0
THERMAL_CRITICAL_C = 80.0

# Free RAM (MB) - MemAvailable from /proc/meminfo
RAM_WARNING_MB = 350.0
RAM_CRITICAL_MB = 250.0

# CPU load / core (1-min loadavg / cores)
CPU_LOAD_WARN = 0.85
CPU_LOAD_CRIT = 1.20

# Sensor heartbeat timeouts (seconds)
CAM_TIMEOUT_S = 0.50
LIDAR_TIMEOUT_S = 0.80

# Anti-flap hysteresis: require conditions to be good for this long before upgrading mode
HYSTERESIS_S = 3.0


class SystemMode(Enum):
    """High-level system state used across the demo stack."""
    NOMINAL = auto()     # 2-cam + LiDAR
    DEGRADED = auto()    # 1-cam + (maybe) LiDAR
    CRITICAL = auto()    # Return home / end demo


@dataclass(frozen=True)
class ResourcePlan:
    """
    Policy output: what the system should do *now*.

    - camera_count / lidar_on: enforced by ModeManager gating
    - res_scale / fps_target: throttles to reduce compute + heat
    - action_hint: what controller should do if we enter CRITICAL
    """
    mode: SystemMode
    camera_count: int
    lidar_on: bool

    res_scale: float
    fps_target: int

    # action hint is a string because rl_safety_supervisor uses string actions
    # (RoverAction.RETURN_HOME, etc.)
    action_hint: str

    # explainability
    reasons: List[str] = field(default_factory=list)

    # dashboard stats
    cpu_temp_c: float = 0.0
    ram_free_mb: float = 0.0
    cpu_load_per_core: float = 0.0

    # optional: feed into SafetySupervisor (0..1)
    health_score: float = 1.0


class SystemHealthMonitor:
    """
    Computes ResourcePlan deterministically.
    Holds hysteresis state to prevent flapping between NOMINAL<->DEGRADED.
    """

    def __init__(
        self,
        thermal_warning_c: float = THERMAL_WARNING_C,
        thermal_critical_c: float = THERMAL_CRITICAL_C,
        ram_warning_mb: float = RAM_WARNING_MB,
        ram_critical_mb: float = RAM_CRITICAL_MB,
        cpu_load_warn: float = CPU_LOAD_WARN,
        cpu_load_crit: float = CPU_LOAD_CRIT,
        cam_timeout_s: float = CAM_TIMEOUT_S,
        lidar_timeout_s: float = LIDAR_TIMEOUT_S,
        hysteresis_s: float = HYSTERESIS_S,
    ):
        self.thermal_warning_c = float(thermal_warning_c)
        self.thermal_critical_c = float(thermal_critical_c)
        self.ram_warning_mb = float(ram_warning_mb)
        self.ram_critical_mb = float(ram_critical_mb)
        self.cpu_load_warn = float(cpu_load_warn)
        self.cpu_load_crit = float(cpu_load_crit)
        self.cam_timeout_s = float(cam_timeout_s)
        self.lidar_timeout_s = float(lidar_timeout_s)
        self.hysteresis_s = float(hysteresis_s)

        # mode hysteresis state
        self._last_mode: SystemMode = SystemMode.NOMINAL
        self._last_mode_change_ts: float = time.time()

        print("[HealthPolicy] Policy Engine Online.")

    # -------------------- Jetson/Linux metrics (best-effort) --------------------

    def _read_temp_c(self) -> Optional[float]:
        """
        Best-effort CPU temperature read (°C).
        Returns None if unavailable (desktop/WSL).
        """
        paths = (
            "/sys/devices/virtual/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone0/temp",
        )
        for p in paths:
            try:
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        raw = f.read().strip()
                    if not raw:
                        continue
                    v = float(raw)
                    if v > 1000.0:
                        v /= 1000.0
                    return float(v)
            except Exception:
                continue
        return None

    def _read_mem_available_mb(self) -> Optional[float]:
        """
        Best-effort MemAvailable read (MB).
        Returns None if unavailable.
        """
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = float(line.split()[1])
                        return kb / 1024.0
        except Exception:
            pass
        return None

    def _read_cpu_load_per_core(self) -> Optional[float]:
        """
        1-minute load average per core.
        Returns None if unavailable (Windows).
        """
        try:
            one, _, _ = os.getloadavg()
            cores = os.cpu_count() or 1
            return float(one) / float(cores)
        except Exception:
            return None

    # -------------------- Policy evaluation --------------------

    def get_plan(
        self,
        lidar_last_ts: float,
        cam_last_ts: float,
        *,
        camera_count_ok: int = 2,
        lidar_seen_recently: Optional[bool] = None,
        cam_seen_recently: Optional[bool] = None,
    ) -> ResourcePlan:
        """
        Generate a ResourcePlan.

        Inputs:
          - lidar_last_ts / cam_last_ts: last-good timestamps (0 if never seen)
          - camera_count_ok: how many camera streams look valid this cycle (0..2)
          - lidar_seen_recently / cam_seen_recently: direct provider OK flags (optional)

        Output:
          - ResourcePlan to be enforced by ModeManager + master_controller.
        """
        now = time.time()

        # safe defaults for desktop/sim
        temp = self._read_temp_c()
        ram = self._read_mem_available_mb()
        load = self._read_cpu_load_per_core()
        if temp is None:
            temp = 45.0
        if ram is None:
            ram = 1000.0
        if load is None:
            load = 0.30

        camera_count_ok = max(0, min(2, int(camera_count_ok)))

        # heartbeat loss checks (only if ever seen)
        cam_lost = (cam_last_ts > 0.0) and ((now - cam_last_ts) > self.cam_timeout_s)
        lidar_lost = (lidar_last_ts > 0.0) and ((now - lidar_last_ts) > self.lidar_timeout_s)

        # provider hints override timeouts if supplied
        if cam_seen_recently is not None:
            cam_lost = not bool(cam_seen_recently)
        if lidar_seen_recently is not None:
            lidar_lost = not bool(lidar_seen_recently)

        # system thresholds
        too_hot = temp >= self.thermal_critical_c
        warn_hot = temp >= self.thermal_warning_c
        out_of_ram = ram <= self.ram_critical_mb
        low_ram = ram <= self.ram_warning_mb
        cpu_crit = load >= self.cpu_load_crit
        cpu_warn = load >= self.cpu_load_warn

        reasons: List[str] = []

        # =========================================================
        # CRITICAL (return home)
        # =========================================================
        if too_hot or out_of_ram or cpu_crit or cam_lost or camera_count_ok == 0:
            if too_hot:
                reasons.append(f"HOT({temp:.1f}C)")
            if out_of_ram:
                reasons.append(f"MEM_CRIT({ram:.0f}MB)")
            if cpu_crit:
                reasons.append(f"CPU_CRIT({load:.2f}/core)")
            if cam_lost:
                reasons.append("CAM_LOST")
            if camera_count_ok == 0:
                reasons.append("NO_CAMERA_FRAMES")
            if lidar_lost:
                reasons.append("LIDAR_LOST")

            plan = ResourcePlan(
                mode=SystemMode.CRITICAL,
                camera_count=0,
                lidar_on=False,
                res_scale=0.0,
                fps_target=0,
                action_hint="RETURN_HOME",
                reasons=reasons,
                cpu_temp_c=float(temp),
                ram_free_mb=float(ram),
                cpu_load_per_core=float(load),
                health_score=0.0,
            )
            self._update_mode(plan.mode, now)
            return plan

        # =========================================================
        # DEGRADED (power saver)
        # =========================================================
        if warn_hot or low_ram or cpu_warn or camera_count_ok < 2 or lidar_lost:
            if warn_hot:
                reasons.append(f"WARN_HEAT({temp:.1f}C)")
            if low_ram:
                reasons.append(f"LOW_MEM({ram:.0f}MB)")
            if cpu_warn:
                reasons.append(f"CPU_WARN({load:.2f}/core)")
            if camera_count_ok < 2:
                reasons.append("CAM2_UNAVAILABLE")
            if lidar_lost:
                reasons.append("LIDAR_LOST")

            # If lidar is lost, explicitly turn it off (camera-only fallback)
            lidar_on = not lidar_lost

            plan = ResourcePlan(
                mode=SystemMode.DEGRADED,
                camera_count=1,
                lidar_on=lidar_on,
                res_scale=0.5,
                fps_target=10 if not lidar_lost else 12,
                action_hint="CONTINUE",
                reasons=reasons,
                cpu_temp_c=float(temp),
                ram_free_mb=float(ram),
                cpu_load_per_core=float(load),
                health_score=0.7,
            )

            # hysteresis: don't bounce back to NOMINAL immediately
            plan = self._apply_hysteresis(plan, now)
            return plan

        # =========================================================
        # NOMINAL
        # =========================================================
        plan = ResourcePlan(
            mode=SystemMode.NOMINAL,
            camera_count=2,
            lidar_on=True,
            res_scale=1.0,
            fps_target=30,
            action_hint="CONTINUE",
            reasons=[],
            cpu_temp_c=float(temp),
            ram_free_mb=float(ram),
            cpu_load_per_core=float(load),
            health_score=1.0,
        )

        plan = self._apply_hysteresis(plan, now)
        return plan

    # -------------------- hysteresis helpers --------------------

    def _update_mode(self, new_mode: SystemMode, now: float) -> None:
        if new_mode != self._last_mode:
            self._last_mode = new_mode
            self._last_mode_change_ts = now

    def _apply_hysteresis(self, candidate: ResourcePlan, now: float) -> ResourcePlan:
        """
        If we're trying to upgrade from DEGRADED -> NOMINAL too quickly,
        hold DEGRADED until hysteresis window passes.
        """
        last = self._last_mode
        since = now - self._last_mode_change_ts

        # Entering DEGRADED is immediate
        if candidate.mode == SystemMode.DEGRADED and last != SystemMode.DEGRADED:
            self._update_mode(candidate.mode, now)
            return candidate

        # Upgrading back to NOMINAL: wait hysteresis_s
        if candidate.mode == SystemMode.NOMINAL and last == SystemMode.DEGRADED and since < self.hysteresis_s:
            # Hold degraded until stable
            held = ResourcePlan(
                mode=SystemMode.DEGRADED,
                camera_count=1,
                lidar_on=candidate.lidar_on,
                res_scale=0.5,
                fps_target=10,
                action_hint="CONTINUE",
                reasons=["HYSTERESIS_HOLD"],
                cpu_temp_c=candidate.cpu_temp_c,
                ram_free_mb=candidate.ram_free_mb,
                cpu_load_per_core=candidate.cpu_load_per_core,
                health_score=0.7,
            )
            return held

        self._update_mode(candidate.mode, now)
        return candidate


if __name__ == "__main__":
    mon = SystemHealthMonitor()
    p = mon.get_plan(time.time(), time.time(), camera_count_ok=2)
    print(p)
