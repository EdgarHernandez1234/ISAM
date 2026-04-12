from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

from .adapters import BaseDriveCommand

# Resolve the real rover_learner root directly so the standalone navigation package
# under ~/Desktop/rover_learner/navigation is importable from operator_app.
THIS_FILE = Path(__file__).resolve()
ROVER_ROOT: Optional[Path] = None
for parent in THIS_FILE.parents:
    if (parent / "navigation").is_dir():
        ROVER_ROOT = parent
        break

# Also try the explicit Desktop path the project is using.
if ROVER_ROOT is None:
    explicit_root = Path.home() / "Desktop" / "rover_learner"
    if (explicit_root / "navigation").is_dir():
        ROVER_ROOT = explicit_root

if ROVER_ROOT is not None and str(ROVER_ROOT) not in sys.path:
    sys.path.insert(0, str(ROVER_ROOT))

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    HAS_ROS2 = True
except Exception:
    rclpy = None
    Node = object  # type: ignore
    LaserScan = None  # type: ignore
    String = None  # type: ignore
    HAS_ROS2 = False

try:
    from navigation.navigation import Navigator, NavigatorConfig
    from navigation.config import WaypointFollowerConfig
    from navigation.types import NavMode, Pose2D, Waypoint
except ModuleNotFoundError as exc:
    raise RuntimeError(
        f"Could not import navigation package. "
        f"Checked rover_learner root={ROVER_ROOT!s}. "
        f"Expected navigation folder at ~/Desktop/rover_learner/navigation/"
    ) from exc


class _AutonomyBridgeNode(Node):
    def __init__(
        self,
        controller: "AutonomyController",
        pose_topic: str,
        scan_topic: str,
        arm_status_topic: str,
    ) -> None:
        super().__init__("alam_autonomy_controller")
        self._controller = controller
        self.create_subscription(String, str(pose_topic), self._on_pose, 10)
        self.create_subscription(LaserScan, str(scan_topic), self._on_scan, 10)
        self.create_subscription(String, str(arm_status_topic), self._on_arm_status, 10)

    def _on_pose(self, msg: String) -> None:
        self._controller._on_pose_msg(msg)

    def _on_scan(self, msg: LaserScan) -> None:
        self._controller._on_scan_msg(msg)

    def _on_arm_status(self, msg: String) -> None:
        self._controller._on_arm_status_msg(msg)


class AutonomyController:
    """
    Stage C Step 1 autonomy controller.

    Scope for this step:
    - keep Jetson-led autonomy and manual fallback exactly as-is
    - keep desktop proxies unchanged
    - keep arm motion on the validated /arm_preset_cmd -> desktop preset proxy path
    - add ONE gated autonomous arm trigger: APPROACH_PICKUP

    Flow:
    1) Leave base and follow search route.
    2) On first route completion, request APPROACH_PICKUP.
    3) Hold base motion while the arm preset runs.
    4) When arm status reports that the preset is no longer busy, enter AUTO_PICKUP_HOLD.

    This validates autonomous arm handoff without yet chaining SCOOP/CARRY/DUMP.
    """

    def __init__(
        self,
        pose_topic: str = "/alam/rover_pose_json",
        scan_topic: str = "/scan",
        arm_status_topic: str = "/alam/arm_preset_status_json",
    ) -> None:
        self.enabled = False
        self.phase = "DISABLED"
        self.reason = "Autonomy disabled."
        self._phase_started = 0.0
        self._last_cmd = BaseDriveCommand()

        self._pose_topic = str(pose_topic)
        self._scan_topic = str(scan_topic)
        self._arm_status_topic = str(arm_status_topic)

        self._node: Optional[Node] = None
        self._pose: Optional[Pose2D] = None
        self._pose_stamp = 0.0
        self._scan_stamp = 0.0
        self._min_distance_m: Optional[float] = None

        self._arm_busy = False
        self._arm_stamp = 0.0
        self._arm_preset = ""
        self._arm_step = ""

        self._start_pose: Optional[Pose2D] = None
        self._route_loops = 0
        self._return_home_requested = False

        # Stage C Step 1 arm-demo state
        self._arm_demo_completed = False
        self._pending_arm_preset: Optional[str] = None
        self._issued_arm_preset: Optional[str] = None
        self._arm_wait_started = 0.0

        follower_cfg = WaypointFollowerConfig(
            max_v_mps=0.16,
            max_w_rps=0.45,
            cruise_v_mps=0.10,
            min_v_mps=0.04,
            heading_kp=0.70,
            waypoint_radius_m=0.45,
            slow_down_radius_m=1.20,
            heading_err_stop_rad=1.35,
        )
        nav_cfg = NavigatorConfig(
            follower=follower_cfg,
            max_v_mps=0.16,
            max_w_rps=0.45,
        )
        self.nav = Navigator(cfg=nav_cfg)
        self.nav.set_laser_waypoint(Waypoint(6.0, 0.0, meta={"label": "LASER"}))

        if HAS_ROS2:
            try:
                if not rclpy.ok():
                    rclpy.init()
                self._node = _AutonomyBridgeNode(
                    self,
                    pose_topic=self._pose_topic,
                    scan_topic=self._scan_topic,
                    arm_status_topic=self._arm_status_topic,
                )
            except Exception:
                self._node = None

    @property
    def last_command(self) -> BaseDriveCommand:
        return self._last_cmd

    def pending_arm_preset(self) -> Optional[str]:
        return self._pending_arm_preset

    def mark_arm_command_issued(self, preset_name: str) -> None:
        self._issued_arm_preset = str(preset_name)
        self._pending_arm_preset = None
        self._arm_wait_started = time.monotonic()

    def enable(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self.enabled = True
        self.phase = "LEAVE_BASE"
        self.reason = "Autonomy enabled: leaving base."
        self._phase_started = now
        self._last_cmd = BaseDriveCommand()
        self._start_pose = self._pose
        self._route_loops = 0
        self._return_home_requested = False
        self._arm_demo_completed = False
        self._pending_arm_preset = None
        self._issued_arm_preset = None
        self._arm_wait_started = 0.0
        self.nav.reset(pose=self._pose)
        self._rebuild_search_route()

    def disable(self, reason: str = "Autonomy disabled.", now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self.enabled = False
        self.phase = "DISABLED"
        self.reason = str(reason)
        self._phase_started = now
        self._last_cmd = BaseDriveCommand()
        self._return_home_requested = False
        self._pending_arm_preset = None
        self._issued_arm_preset = None

    def close(self) -> None:
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass

    def request_return_home(self) -> None:
        self._return_home_requested = True

    def _set_phase(self, phase: str, reason: str, now: float) -> None:
        self.phase = str(phase)
        self.reason = str(reason)
        self._phase_started = float(now)

    def _on_pose_msg(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
            yaw = float(data.get("yaw", 0.0))
            self._pose = Pose2D(x_m=x, y_m=y, yaw_rad=yaw)
            self._pose_stamp = time.time()
        except Exception:
            return

    def _on_scan_msg(self, msg: LaserScan) -> None:
        self._scan_stamp = time.time()
        try:
            vals = [float(r) for r in msg.ranges if math.isfinite(float(r)) and float(r) > 0.0]
            self._min_distance_m = min(vals) if vals else None
        except Exception:
            self._min_distance_m = None

    def _on_arm_status_msg(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self._arm_busy = bool(data.get("busy", False))
            self._arm_preset = str(data.get("preset", "") or "")
            self._arm_step = str(data.get("step", "") or "")
            self._arm_stamp = time.time()
        except Exception:
            return

    def _spin_once(self) -> None:
        if self._node is None or not HAS_ROS2:
            return
        try:
            rclpy.spin_once(self._node, timeout_sec=0.0)
        except Exception:
            pass

    def _fresh_pose(self) -> bool:
        return self._pose is not None and (time.time() - self._pose_stamp) < 1.5

    def _fresh_arm_status(self) -> bool:
        return self._arm_stamp > 0.0 and (time.time() - self._arm_stamp) < 2.5

    def _relative_waypoint(self, pose: Pose2D, dx_m: float, dy_m: float, label: str) -> Waypoint:
        c = math.cos(float(pose.yaw_rad))
        s = math.sin(float(pose.yaw_rad))
        wx = float(pose.x_m) + (dx_m * c - dy_m * s)
        wy = float(pose.y_m) + (dx_m * s + dy_m * c)
        return Waypoint(wx, wy, meta={"label": label})

    def _rebuild_search_route(self) -> None:
        anchor = self._pose or self._start_pose or Pose2D()
        route = [
            self._relative_waypoint(anchor, 1.2, 0.0, "SEARCH_1"),
            self._relative_waypoint(anchor, 2.2, 0.4, "SEARCH_2"),
            self._relative_waypoint(anchor, 3.2, 0.0, "SEARCH_3"),
            self._relative_waypoint(anchor, 2.4, -0.5, "SEARCH_4"),
            self._relative_waypoint(anchor, 1.4, -0.2, "SEARCH_5"),
        ]
        self.nav.set_route(route)

    def _hold_cmd(self, speed_scale: float) -> BaseDriveCommand:
        cmd = BaseDriveCommand(linear=0.0, angular=0.0, speed_scale=speed_scale)
        self._last_cmd = cmd
        return cmd

    def tick(self, now: Optional[float] = None, speed_scale: float = 0.25) -> BaseDriveCommand:
        now = time.monotonic() if now is None else float(now)
        self._spin_once()

        capped_scale = max(0.10, min(float(speed_scale), 0.35))

        if not self.enabled:
            return self._hold_cmd(capped_scale)

        if not self._fresh_pose():
            self._set_phase("WAIT_FOR_POSE", "Waiting for fresh rover pose.", now)
            return self._hold_cmd(capped_scale)

        # Stage C Step 1: keep avoidance disabled while validating preset handoff.
        self.nav.update_lidar(min_distance_m=None)

        # Arm-demo hold phases
        if self._pending_arm_preset == "APPROACH_PICKUP":
            self._set_phase("AUTO_ARM_APPROACH_REQUEST", "Autonomy requested APPROACH_PICKUP.", now)
            return self._hold_cmd(capped_scale)

        if self._issued_arm_preset == "APPROACH_PICKUP":
            # Wait for the desktop arm preset proxy to report busy, then wait until it clears.
            if self._fresh_arm_status() and self._arm_busy:
                self._set_phase("AUTO_ARM_APPROACH", "Autonomous APPROACH_PICKUP in progress.", now)
                return self._hold_cmd(capped_scale)

            if self._fresh_arm_status() and (not self._arm_busy):
                # Treat a non-busy status after issue as completion of the first autonomous arm demo.
                self._arm_demo_completed = True
                self._issued_arm_preset = None
                self._set_phase("AUTO_PICKUP_HOLD", "Autonomous APPROACH_PICKUP complete; holding for next stage.", now)
                return self._hold_cmd(capped_scale)

            # Status not fresh yet; continue holding.
            self._set_phase("AUTO_ARM_APPROACH", "Waiting for arm preset status.", now)
            return self._hold_cmd(capped_scale)

        if self._arm_demo_completed:
            self._set_phase("AUTO_PICKUP_HOLD", "Autonomous APPROACH_PICKUP complete; holding for next stage.", now)
            return self._hold_cmd(capped_scale)

        nav_mode = NavMode.SEARCH_ROUTE

        if self._return_home_requested:
            nav_mode = NavMode.GO_HOME
            self._set_phase("RETURN_HOME", "Autonomy returning home via breadcrumbs.", now)
        elif self.phase in ("DISABLED", "LEAVE_BASE", "WAIT_FOR_POSE"):
            self._set_phase("LEAVE_BASE", "Autonomy leaving base.", now)
            if self._start_pose is not None and self._pose is not None:
                dx = float(self._pose.x_m) - float(self._start_pose.x_m)
                dy = float(self._pose.y_m) - float(self._start_pose.y_m)
                if math.hypot(dx, dy) >= 0.8:
                    self._set_phase("SEARCH_ROUTE", "Autonomy following search route.", now)
        else:
            self._set_phase("SEARCH_ROUTE", "Autonomy following search route.", now)

        proposal = self.nav.step(
            nav_mode=nav_mode,
            timestamp_s=time.time(),
            vision={},
            record_breadcrumbs=True,
        )

        cmd_v = float(proposal.twist.v_mps)
        cmd_w = float(proposal.twist.w_rps)

        # Gentle debug-phase nudge: if the controller is still in a pivot branch but the
        # angular rate is moderate, allow a very small forward creep instead of a pure turn.
        dbg = proposal.debug or {}
        heading_err = None
        try:
            heading_err = float(dbg.get("heading_err")) if dbg.get("heading_err") is not None else None
        except Exception:
            heading_err = None
        if abs(cmd_v) < 1e-4 and abs(cmd_w) > 1e-3 and heading_err is not None and abs(heading_err) < 1.75:
            cmd_v = 0.03

        cmd = BaseDriveCommand(
            linear=cmd_v,
            angular=cmd_w,
            speed_scale=capped_scale,
        )

        if proposal.done and nav_mode == NavMode.SEARCH_ROUTE:
            # Stage C Step 1: trigger the first autonomous arm preset and hold base motion.
            self._pending_arm_preset = "APPROACH_PICKUP"
            self._set_phase("AUTO_ARM_APPROACH_REQUEST", "Search route complete; requesting APPROACH_PICKUP.", now)
            return self._hold_cmd(capped_scale)

        if proposal.done and nav_mode == NavMode.GO_HOME:
            self._set_phase("AT_HOME", "Autonomy completed breadcrumb return.", now)
            return self._hold_cmd(capped_scale)

        self._last_cmd = cmd
        return cmd
