"""
rl_safety_supervisor_arduino.py

Parallel safety supervisor that adds an Arduino interlock evaluator
(E-stop + watchdog) without modifying rover_learner/rl_safety_supervisor.py.

Use when you want a hard interlock layer in the same arbitration framework.

Key behavior:
- ESTOP or WATCHDOG => CRITICAL + STOP override
- Stale/missing Arduino status => CRITICAL (if require_arduino=True), else CAUTION
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Optional, Tuple, List

from rover_learner.arduino_interlock import ArduinoInterlock, ArduinoStatus
from rover_learner.rl_safety_supervisor import (
    Observation,
    ActionProposal,
    Decision,
    SafetySignal,
    SafetyLevel,
    RoverAction,
    SafetySupervisor,
    SupervisorConfig,
)


@dataclass
class ArduinoInterlockConfig:
    require_arduino: bool = True
    stale_level: SafetyLevel = SafetyLevel.CRITICAL
    stale_override: str = RoverAction.STOP
    estop_override: str = RoverAction.STOP
    watchdog_override: str = RoverAction.STOP
    ttl_s: float = 0.25


class ArduinoInterlockEvaluator:
    def __init__(self, interlock: ArduinoInterlock, cfg: ArduinoInterlockConfig = ArduinoInterlockConfig()):
        self.interlock = interlock
        self.cfg = cfg

    def evaluate(self, obs: Observation) -> Optional[SafetySignal]:
        # obs unused currently; kept for signature compatibility.
        now = time.time()

        alive = self.interlock.is_alive(now=now)
        st = self.interlock.get_status()

        if not alive or st is None:
            if not self.cfg.require_arduino:
                return SafetySignal(
                    source="arduino_interlock",
                    level=SafetyLevel.CAUTION,
                    score=0.4,
                    reason="Arduino interlock missing/stale (not required)",
                    override_action=None,
                    ttl_s=self.cfg.ttl_s,
                )
            return SafetySignal(
                source="arduino_interlock",
                level=self.cfg.stale_level,
                score=1.0,
                reason="Arduino interlock missing/stale",
                override_action=self.cfg.stale_override,
                ttl_s=self.cfg.ttl_s,
            )

        # Hard overrides
        if st.estop:
            return SafetySignal(
                source="arduino_estop",
                level=SafetyLevel.CRITICAL,
                score=1.0,
                reason="E-stop active (pressed or circuit open)",
                override_action=self.cfg.estop_override,
                ttl_s=self.cfg.ttl_s,
            )

        if st.wd:
            return SafetySignal(
                source="arduino_watchdog",
                level=SafetyLevel.CRITICAL,
                score=0.9,
                reason="Watchdog active (PING timeout)",
                override_action=self.cfg.watchdog_override,
                ttl_s=self.cfg.ttl_s,
            )

        if not st.is_interlock_safe:
            # Not armed / unsafe but no explicit flags
            return SafetySignal(
                source="arduino_interlock",
                level=SafetyLevel.DANGER,
                score=0.7,
                reason=f"Interlock not safe ({st.reason})",
                override_action=RoverAction.HOLD,
                ttl_s=self.cfg.ttl_s,
            )

        return None


class SafetySupervisorArduino:
    """
    Wraps a base SafetySupervisor, injecting Arduino safety signals.
    """
    def __init__(
        self,
        base: SafetySupervisor,
        arduino_eval: ArduinoInterlockEvaluator,
        cfg: SupervisorConfig = SupervisorConfig(),
    ):
        self.base = base
        self.arduino_eval = arduino_eval
        self.cfg = cfg

    @staticmethod
    def default(interlock: ArduinoInterlock, *, require_arduino: bool = True) -> "SafetySupervisorArduino":
        base = SafetySupervisor.default()
        ar_cfg = ArduinoInterlockConfig(require_arduino=require_arduino)
        return SafetySupervisorArduino(base=base, arduino_eval=ArduinoInterlockEvaluator(interlock, ar_cfg), cfg=base.cfg)

    def evaluate_all(self, obs: Observation) -> Tuple[SafetySignal, ...]:
        signals: List[SafetySignal] = list(self.base.evaluate_all(obs))

        s = self.arduino_eval.evaluate(obs)
        if s is not None and s.is_active():
            signals.append(s)

        signals.sort(key=lambda x: (int(x.level), float(x.score)), reverse=True)
        return tuple(signals)

    def apply(self, proposal: ActionProposal, obs: Observation) -> Decision:
        signals = self.evaluate_all(obs)
        if not signals:
            return Decision(proposed_action=proposal.action, final_action=proposal.action, reason="No safety overrides", signals=())

        top = signals[0]
        if top.override_action and top.level >= self.cfg.override_at_or_above:
            return Decision(
                proposed_action=proposal.action,
                final_action=str(top.override_action),
                reason=f"SAFETY OVERRIDE: {top.source}: {top.reason}",
                signals=signals,
            )

        return Decision(
            proposed_action=proposal.action,
            final_action=proposal.action,
            reason=f"Advisory safety signals present (top={top.source}:{top.level.name})",
            signals=signals,
        )


@dataclass
class ShieldedControllerArduino:
    policy: Any
    supervisor: SafetySupervisorArduino

    def step(self, obs: Observation) -> Decision:
        proposal = self.policy.propose(obs)
        return self.supervisor.apply(proposal, obs)
