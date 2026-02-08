#!/usr/bin/env python3
"""core.py (rover_learner)

Shared business logic for the live stack.
This module should be importable without cv2/ultralytics/ROS2 so unit tests can run anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from .rl_safety_supervisor import Observation, ShieldedController, Decision


@dataclass(frozen=True)
class Perception:
    pred_class: str
    pred_conf: float


@dataclass(frozen=True)
class Telemetry:
    """Optional runtime telemetry (can be None during early integration)."""
    joint_error_norm: Optional[float] = None
    motor_current_a: Optional[float] = None
    stall_flag: Optional[bool] = None
    health_score: Optional[float] = None


@dataclass(frozen=True)
class StepInputs:
    perception: Perception
    distance_m: Optional[float]
    telemetry: Telemetry = Telemetry()


@dataclass(frozen=True)
class StepOutput:
    proposed_action: str
    final_action: str
    reason: str
    signals: Dict[str, Any]


def build_observation(inp: StepInputs) -> Observation:
    """Convert loosely-typed inputs into the structured Observation used by policy+shield."""
    obs = Observation.from_perception(inp.perception.pred_class, inp.perception.pred_conf, inp.distance_m)
    # attach optional telemetry
    return Observation(
        pred_class=obs.pred_class,
        pred_conf=obs.pred_conf,
        distance_m=obs.distance_m,
        joint_error_norm=inp.telemetry.joint_error_norm,
        motor_current_a=inp.telemetry.motor_current_a,
        stall_flag=inp.telemetry.stall_flag,
        health_score=inp.telemetry.health_score,
        timestamp=obs.timestamp,
    )


def step_with_safety(controller: ShieldedController, inp: StepInputs) -> StepOutput:
    """Run one decision step (policy proposal + safety supervisor enforcement)."""
    obs = build_observation(inp)
    decision: Decision = controller.step(obs)
    # Make a log-friendly dict of signals
    sigs = {
        "signals": [s.__dict__ for s in decision.signals],
        "proposed_action": decision.proposed_action,
        "final_action": decision.final_action,
    }
    return StepOutput(
        proposed_action=decision.proposed_action,
        final_action=decision.final_action,
        reason=decision.reason,
        signals=sigs,
    )
