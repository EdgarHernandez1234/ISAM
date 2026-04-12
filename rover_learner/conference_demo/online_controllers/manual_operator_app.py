from __future__ import annotations

import time
from datetime import datetime
from typing import Optional
from uuid import uuid4

from .adapters import (
    ArmPresetExecutor,
    BaseDriveCommand,
    ControllerAdapter,
    ControllerSnapshot,
    DashboardSink,
    RoverDriveAdapter,
)
from .autonomy_controller import AutonomyController
from .config import OperatorConfig
from .dashboard_bridge import OperatorStatePublisher
from .enums import ArmPreset, ControlMode, MissionPhase, SafetyState
from .event_logger import MissionEventLogger
from .models import EventRecord, RunState


PRESET_BUSY_HINT_SECS = {
    ArmPreset.STOW: 2.6,
    ArmPreset.APPROACH_PICKUP: 2.2,
    ArmPreset.SCOOP_PICKUP: 2.8,
    ArmPreset.CARRY: 3.0,
    ArmPreset.DUMP: 2.6,
    ArmPreset.RETURN_TO_STOW: 2.4,
}

MANUAL_OVERRIDE_DEADBAND = 0.18


class ManualOperatorApp:
    """Operator state machine with ROS telemetry publishing for the laptop dashboard shell."""

    def __init__(
        self,
        config: OperatorConfig,
        controller: ControllerAdapter,
        rover_drive: RoverDriveAdapter,
        arm_executor: ArmPresetExecutor,
        dashboard: DashboardSink,
    ) -> None:
        self.config = config
        self.controller = controller
        self.rover_drive = rover_drive
        self.arm_executor = arm_executor
        self.dashboard = dashboard

        self.config.ensure_dirs()
        run_id = f"{self.config.run_id_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        self.state = RunState(
            run_id=run_id,
            control_mode=ControlMode.BASE,
            mission_phase=MissionPhase.PRESTART,
            safety_state=SafetyState.SAFE,
            speed_scale=self.config.default_speed_scale,
            operator_message="Confirm controller connected and mission state clear.",
        )
        self._speed_index = self.config.default_speed_index
        self._last_dashboard_push = 0.0
        self._sample_period = 0.5
        self._last_sample_ts = 0.0
        self._running = False
        self._prev_snapshot = ControllerSnapshot()
        self._last_snapshot = ControllerSnapshot()
        self._last_base_command = BaseDriveCommand()
        self._input_source = "idle"
        self._controller_name = ""
        self._controller_status = ""
        self._arm_busy_until = 0.0
        self._arm_status_text = "IDLE"

        self._control_domain = "MANUAL"
        self._autonomy = AutonomyController()
        self._last_autonomy_phase = "DISABLED"

        self.event_logger = MissionEventLogger(self.config.logs_dir / f"{run_id}.csv")
        self.state_publisher = OperatorStatePublisher()

    def start(self) -> None:
        self._running = True
        self._autonomy.disable("Autonomy disabled at startup.")
        self._control_domain = "MANUAL"
        self._last_autonomy_phase = self._autonomy.phase
        self._emit_event("operator_session_started", "Manual operator app entered.")
        self._set_phase(MissionPhase.AT_BASE, "Ready at home base.")
        self.push_dashboard(force=True)

    def stop(self) -> None:
        try:
            self._autonomy.close()
        except Exception:
            pass
        self._autonomy.disable("Operator app stopped.")
        self._control_domain = "MANUAL"
        self.rover_drive.halt()
        self._last_base_command = BaseDriveCommand()
        self._input_source = "stopped"
        self._arm_busy_until = 0.0
        self._arm_status_text = "IDLE"
        self._emit_event("operator_session_stopped", "Manual operator app exited.")
        self._running = False
        self.push_dashboard(force=True)
        self.event_logger.close()
        self.state_publisher.close()

    def run_console_loop(self) -> None:
        self.start()
        period = 1.0 / max(1.0, self.config.manual_loop_hz)
        try:
            while self._running:
                self.tick()
                time.sleep(period)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def tick(self) -> None:
        now = time.monotonic()
        if self._arm_busy_until > 0.0 and now >= self._arm_busy_until:
            self._arm_busy_until = 0.0
            self._arm_status_text = "IDLE"

        snapshot = self.controller.poll()
        self._last_snapshot = snapshot
        self._controller_name = getattr(self.controller, "controller_name", "") or ""
        self._controller_status = getattr(self.controller, "status_text", "") or ""

        self._handle_global_buttons(snapshot)

        if self.state.safety_state == SafetyState.HALTED:
            self._last_base_command = BaseDriveCommand()
            self._input_source = "halt"

        elif self._control_domain == "AUTONOMOUS":
            self._handle_autonomous_mode(snapshot)

        elif self.state.control_mode == ControlMode.BASE and self.state.safety_state == SafetyState.SAFE:
            self._handle_base_mode(snapshot)

        elif self.state.control_mode == ControlMode.ARM and self.state.safety_state == SafetyState.SAFE:
            self._last_base_command = BaseDriveCommand()
            self._input_source = "arm_mode"
            self._handle_arm_mode(snapshot)

        else:
            self._last_base_command = BaseDriveCommand()
            self._input_source = "halt"

        self.push_dashboard()
        if (time.time() - self._last_sample_ts) >= self._sample_period:
            self.event_logger.log_sample(self.state, self._telemetry_extra())
            self._last_sample_ts = time.time()
        self._prev_snapshot = snapshot

    def _telemetry_extra(self) -> dict:
        return {
            "arm_busy": self.arm_busy,
            "arm_status_text": self._arm_status_text,
            "controller_connected": bool(self._last_snapshot.connected),
            "controller_name": self._controller_name,
            "input_source": self._input_source,
            "base_linear": float(self._last_base_command.linear),
            "base_angular": float(self._last_base_command.angular),
            "cmd_linear": float(self._last_base_command.linear),
            "cmd_angular": float(self._last_base_command.angular),
            "control_domain": self._control_domain,
            "autonomy_enabled": self._autonomy.enabled,
            "autonomy_phase": self._autonomy.phase,
            "autonomy_reason": self._autonomy.reason,
            "autonomy_cmd_linear": float(self._autonomy.last_command.linear),
            "autonomy_cmd_angular": float(self._autonomy.last_command.angular),
            "autonomy_target_preset": self._autonomy.pending_arm_preset(),
        }

    @property
    def arm_busy(self) -> bool:
        return self._arm_busy_until > time.monotonic()

    def push_dashboard(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_dashboard_push) < (1.0 / max(0.5, self.config.dashboard_refresh_hz)):
            return

        payload = {
            "summary": (
                f"{self.state.summary_line()} "
                f"domain={self._control_domain} auto_phase={self._autonomy.phase}"
            ),
            "published_at_epoch": now,
            "run_id": self.state.run_id,
            "control_mode": self.state.control_mode.value,
            "control_domain": self._control_domain,
            "autonomy_enabled": self._autonomy.enabled,
            "autonomy_phase": self._autonomy.phase,
            "autonomy_reason": self._autonomy.reason,
            "autonomy_cmd_linear": float(self._autonomy.last_command.linear),
            "autonomy_cmd_angular": float(self._autonomy.last_command.angular),
            "autonomy_target_preset": self._autonomy.pending_arm_preset(),
            "mission_phase": self.state.mission_phase.value,
            "safety_state": self.state.safety_state.value,
            "speed_scale": self.state.speed_scale,
            "last_arm_preset": self.state.last_arm_preset.value if self.state.last_arm_preset else None,
            "carrying_regolith": self.state.flags.carrying_regolith,
            "dumped_at_laser": self.state.flags.dumped_at_laser,
            "returned_home": self.state.flags.returned_home,
            "mission_success": self.state.flags.mission_success,
            "event_count": self.state.event_count,
            "last_event": self.state.last_event.name if self.state.last_event else None,
            "operator_message": self.state.operator_message,
            "controller_connected": bool(self._last_snapshot.connected),
            "controller_name": self._controller_name,
            "controller_status": self._controller_status,
            "input_source": self._input_source,
            "base_linear": float(self._last_base_command.linear),
            "base_angular": float(self._last_base_command.angular),
            "cmd_linear": float(self._last_base_command.linear),
            "cmd_angular": float(self._last_base_command.angular),
            "base_speed_scale": float(self._last_base_command.speed_scale),
            "arm_busy": self.arm_busy,
            "arm_status_text": self._arm_status_text,
        }
        self.dashboard.update(payload)
        self.state_publisher.publish(payload)
        self._last_dashboard_push = now

    def _handle_global_buttons(self, snapshot: ControllerSnapshot) -> None:
        if self._pressed(snapshot.share_pressed, self._prev_snapshot.share_pressed):
            self._toggle_control_domain()

        if self._pressed(snapshot.cross_pressed, self._prev_snapshot.cross_pressed):
            self._toggle_mode()

        if self._pressed(snapshot.circle_pressed, self._prev_snapshot.circle_pressed):
            self._enter_halt("Operator halt requested.")

        if self._pressed(snapshot.options_pressed, self._prev_snapshot.options_pressed):
            self._clear_halt()

        if self._pressed(snapshot.triangle_pressed, self._prev_snapshot.triangle_pressed):
            self.state.flags.returned_home = True
            self._set_phase(MissionPhase.RETURN_HOME, "Operator marked returned home.")
            self._emit_event("returned_home", "Triangle pressed.")
            if self._control_domain == "AUTONOMOUS":
                try:
                    self._autonomy.request_return_home()
                except Exception:
                    pass

        if self._pressed(snapshot.square_pressed, self._prev_snapshot.square_pressed):
            self.state.flags.mission_success = True
            self._set_phase(MissionPhase.COMPLETE, "Operator marked mission success.")
            self._emit_event("mission_success", "Square pressed.")

        if self._pressed(snapshot.r2_pressed, self._prev_snapshot.r2_pressed):
            self._speed_step(+1)

        if self._pressed(snapshot.l2_pressed, self._prev_snapshot.l2_pressed):
            self._speed_step(-1)

    def _handle_base_mode(self, snapshot: ControllerSnapshot) -> None:
        linear = -float(snapshot.left_stick_y) * self.state.speed_scale
        angular = float(snapshot.left_stick_x) * self.state.speed_scale
        self._input_source = "ps4" if snapshot.connected else "idle"

        if abs(linear) > 1e-3 or abs(angular) > 1e-3:
            cmd = BaseDriveCommand(linear=linear, angular=angular, speed_scale=self.state.speed_scale)
            self._last_base_command = cmd
            self.rover_drive.send_base_command(cmd)
            if self.state.mission_phase in (MissionPhase.PRESTART, MissionPhase.AT_BASE):
                self._emit_event("departed_base", "Rover began moving away from home base.")
                self._set_phase(MissionPhase.BASE_DEPARTURE, "Rover departing home base.")
            elif self.state.mission_phase == MissionPhase.BASE_DEPARTURE:
                self._set_phase(MissionPhase.SEARCHING, "Searching for regolith target zone.")
        else:
            cmd = BaseDriveCommand(linear=0.0, angular=0.0, speed_scale=self.state.speed_scale)
            self._last_base_command = cmd
            self.rover_drive.send_base_command(cmd)
            self._input_source = "idle"

    def _handle_autonomous_mode(self, snapshot: ControllerSnapshot) -> None:
        if (
            abs(float(snapshot.left_stick_x)) >= MANUAL_OVERRIDE_DEADBAND
            or abs(float(snapshot.left_stick_y)) >= MANUAL_OVERRIDE_DEADBAND
        ):
            self._disable_autonomy(
                event_name="manual_override_from_autonomy",
                message="Manual override detected on left stick. Back in BASE mode.",
            )
            self._handle_base_mode(snapshot)
            return

        cmd = self._autonomy.tick(now=time.monotonic(), speed_scale=self.state.speed_scale)
        self._last_base_command = cmd
        self._input_source = "autonomy"
        self.rover_drive.send_base_command(cmd)

        pending = self._autonomy.pending_arm_preset()
        if pending == ArmPreset.APPROACH_PICKUP.value and not self.arm_busy:
            self._issue_arm_preset(ArmPreset.APPROACH_PICKUP, source="autonomy")
            try:
                self._autonomy.mark_arm_command_issued(ArmPreset.APPROACH_PICKUP.value)
            except Exception:
                pass

        if self._autonomy.phase != self._last_autonomy_phase:
            self._emit_event("autonomy_phase_changed", self._autonomy.phase)
            self._last_autonomy_phase = self._autonomy.phase

        if self._autonomy.phase == "LEAVE_BASE":
            self._set_phase(MissionPhase.BASE_DEPARTURE, self._autonomy.reason)
        elif self._autonomy.phase == "SEARCH_ROUTE":
            self._set_phase(MissionPhase.SEARCHING, self._autonomy.reason)
        elif self._autonomy.phase in ("AUTO_ARM_APPROACH_REQUEST", "AUTO_ARM_APPROACH", "AUTO_PICKUP_HOLD"):
            self._set_phase(MissionPhase.PICKUP, self._autonomy.reason)
        elif self._autonomy.phase == "RETURN_HOME":
            self._set_phase(MissionPhase.RETURN_HOME, self._autonomy.reason)
        else:
            self.state.operator_message = self._autonomy.reason

    def _handle_arm_mode(self, snapshot: ControllerSnapshot) -> None:
        preset: Optional[ArmPreset] = None

        if self._pressed(snapshot.dpad_up_pressed, self._prev_snapshot.dpad_up_pressed):
            preset = ArmPreset.STOW
        elif self._pressed(snapshot.dpad_right_pressed, self._prev_snapshot.dpad_right_pressed):
            preset = ArmPreset.APPROACH_PICKUP
        elif self._pressed(snapshot.dpad_down_pressed, self._prev_snapshot.dpad_down_pressed):
            preset = ArmPreset.SCOOP_PICKUP
        elif self._pressed(snapshot.dpad_left_pressed, self._prev_snapshot.dpad_left_pressed):
            preset = ArmPreset.CARRY
        elif self._pressed(snapshot.l1_pressed, self._prev_snapshot.l1_pressed):
            preset = ArmPreset.DUMP
        elif self._pressed(snapshot.r1_pressed, self._prev_snapshot.r1_pressed):
            preset = ArmPreset.RETURN_TO_STOW

        if preset is None:
            return

        self._issue_arm_preset(preset, source="manual")

    def _issue_arm_preset(self, preset: ArmPreset, source: str) -> None:
        self.rover_drive.halt()
        self._last_base_command = BaseDriveCommand()
        self.state.last_arm_preset = preset
        self.arm_executor.execute(preset)
        self._arm_busy_until = time.monotonic() + PRESET_BUSY_HINT_SECS.get(preset, 2.5)
        self._arm_status_text = f"ISSUED:{preset.value}"

        if source == "autonomy":
            self._emit_event("autonomy_arm_preset_issued", preset.value)
        else:
            self._emit_event("arm_preset_issued", preset.value)

        if preset == ArmPreset.STOW:
            self.state.operator_message = "Arm returned to STOW."
        elif preset == ArmPreset.APPROACH_PICKUP:
            self._set_phase(MissionPhase.PICKUP, "Approaching pickup posture.")
        elif preset == ArmPreset.SCOOP_PICKUP:
            self.state.flags.carrying_regolith = True
            self._emit_event("pickup_regolith", "SCOOP_PICKUP preset triggered.")
            self._set_phase(MissionPhase.PICKUP, "Payload marked as acquired.")
        elif preset == ArmPreset.CARRY:
            self.state.flags.carrying_regolith = True
            self._set_phase(MissionPhase.TRANSIT_TO_LASER, "Carry posture reached.")
        elif preset == ArmPreset.DUMP:
            self.state.flags.carrying_regolith = False
            self.state.flags.dumped_at_laser = True
            self._emit_event("dumped_at_laser", "DUMP preset triggered.")
            self._set_phase(MissionPhase.DUMPING, "Dump action triggered.")
        elif preset == ArmPreset.RETURN_TO_STOW:
            self._set_phase(MissionPhase.RETURN_HOME, "Arm returned toward travel posture.")

    def _toggle_control_domain(self) -> None:
        if self.state.safety_state == SafetyState.HALTED:
            self.state.operator_message = "Cannot toggle autonomy while halted. Clear halt first."
            return

        if self._control_domain == "AUTONOMOUS":
            self._disable_autonomy(
                event_name="autonomy_disabled",
                message="Autonomy disabled. Back in MANUAL BASE mode.",
            )
            return

        self.rover_drive.halt()
        self._last_base_command = BaseDriveCommand()
        self.state.control_mode = ControlMode.BASE
        self._control_domain = "AUTONOMOUS"
        self._autonomy.enable()
        self._last_autonomy_phase = self._autonomy.phase
        self.state.operator_message = "Autonomous BASE mode enabled."
        self._emit_event("autonomy_enabled", self.state.operator_message)

    def _disable_autonomy(self, event_name: str, message: str) -> None:
        self._autonomy.disable(reason=message)
        self._control_domain = "MANUAL"
        self.state.control_mode = ControlMode.BASE
        self.state.operator_message = message
        self._emit_event(event_name, message)

    def _toggle_mode(self) -> None:
        if self.state.safety_state == SafetyState.HALTED:
            self.state.operator_message = "Cannot toggle modes while halted. Clear halt first."
            return

        if self._control_domain != "MANUAL":
            self.state.operator_message = "Disable autonomy before switching BASE/ARM."
            return

        self.state.control_mode = (
            ControlMode.ARM if self.state.control_mode == ControlMode.BASE else ControlMode.BASE
        )
        self.state.operator_message = f"Operator switched to {self.state.control_mode.value} mode."
        self._emit_event("mode_changed", self.state.operator_message)

    def _enter_halt(self, reason: str) -> None:
        self._autonomy.disable("Autonomy disabled by HALT.")
        self._control_domain = "MANUAL"
        self.state.control_mode = ControlMode.HALT
        self.state.safety_state = SafetyState.HALTED
        self.state.operator_message = reason
        self._last_base_command = BaseDriveCommand()
        self._input_source = "halt"
        self.rover_drive.halt()
        self._emit_event("halted", reason)

    def _clear_halt(self) -> None:
        self._autonomy.disable("Autonomy disabled after halt clear.")
        self._control_domain = "MANUAL"
        self.state.control_mode = ControlMode.BASE
        self.state.safety_state = SafetyState.SAFE
        self.state.operator_message = "Halt cleared. Back in MANUAL BASE mode."
        self._emit_event("halt_cleared", self.state.operator_message)

    def _speed_step(self, delta: int) -> None:
        new_index = min(max(self._speed_index + delta, 0), len(self.config.speed_ladder) - 1)
        if new_index == self._speed_index:
            return
        self._speed_index = new_index
        self.state.speed_scale = self.config.speed_ladder[self._speed_index]
        self._emit_event("speed_scale_changed", f"Speed scale now {self.state.speed_scale:.2f}")

    def _set_phase(self, phase: MissionPhase, message: str) -> None:
        self.state.mission_phase = phase
        self.state.operator_message = message

    def _emit_event(self, name: str, details: str = "") -> None:
        event = EventRecord(name=name, timestamp=datetime.now(), details=details)
        self.state.last_event = event
        self.state.event_count += 1
        self.event_logger.log_event(self.state, event, self._telemetry_extra())
        print(f"[event] {event.timestamp.isoformat(timespec='seconds')} {name} :: {details}")

    @staticmethod
    def _pressed(current: bool, previous: bool) -> bool:
        return bool(current and not previous)
