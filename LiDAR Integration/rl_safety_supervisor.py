#!/usr/bin/env python3
"""
rl_safety_supervisor.py

Shielded RL pattern (Possibility #1):
  - RL/Policy proposes an action (what we'd like to do)
  - Safety Supervisor evaluates three independent failsafe measures and can:
      * veto (STOP / RETREAT / RETURN_HOME)
      * clamp (DEGRADED mode)
  - Final decision is deterministic and loggable.

This module is designed to be:
  - Modular (swap policies and safety evaluators)
  - Testable (pure-Python unit tests, no cv2/ultralytics/numpy required)
  - Reusable (works in demos and ROS2 nodes)

It pairs naturally with your current demo pipeline:
  - demo_decider_lidar.py already produces: (pred_class, pred_conf, distance_m)
    and uses LiDAR distance logic plus gating heuristics. This module formalizes
    that into a policy + supervisor with structured safety signals.
  - lidar_provider.py already gives a clean get_distance_m() interface; you can
    use it to drive the hazard risk scorer's distance input.

Usage sketch (inside your demo after you compute pred_class/pred_conf/distance):
    from rl_safety_supervisor import (
        Observation, HeuristicPolicy, SafetySupervisor, ShieldedController,
        RoverAction
    )

    obs = Observation.from_perception(pred_class, pred_conf, distance_m)
    ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())
    decision = ctrl.step(obs)
    print(decision.final_action, decision.reason, decision.signals)

CLI demo (no CV/YOLO required):
    python3 rl_safety_supervisor.py --demo --distance-m 1.2 --pred-class clean --pred-conf 0.92

Optional: integrate real LiDAR provider (offline RELLIS or ROS2):
    python3 rl_safety_supervisor.py --demo --lidar-mode rellis_bin --lidar-source "/path/to/os1_cloud_node_kitti_bin"
"""

from __future__ import annotations

import argparse
import time
import unittest
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Protocol, Tuple


# -----------------------------
# Core Types
# -----------------------------

class RoverAction(str):
    """Actions that the rover/arm layer could execute (extend as needed)."""
    SCOOP = "SCOOP"
    BYPASS = "BYPASS"
    APPROACH = "APPROACH"
    HOLD = "HOLD"
    STOP = "STOP"
    RETREAT = "RETREAT"
    RETURN_HOME = "RETURN_HOME"
    DEGRADED = "DEGRADED"  # clamp behavior, reduced speed/torque, conservative planning


class SafetyLevel(IntEnum):
    """Increasing severity; used for deterministic arbitration."""
    INFO = 0
    CAUTION = 1
    DANGER = 2
    CRITICAL = 3


@dataclass(frozen=True)
class SafetySignal:
    """
    A single evaluator's output. Keep these small and loggable.
    - score is evaluator-specific but should be monotonic (higher = worse).
    - override_action can be None for advisory signals.
    """
    source: str
    level: SafetyLevel
    score: float
    reason: str
    override_action: Optional[str] = None
    ttl_s: float = 0.0  # time-to-live; 0 means "use immediately only"
    ts: float = field(default_factory=lambda: time.time())

    def is_active(self, now: Optional[float] = None) -> bool:
        if self.ttl_s <= 0:
            return True
        now = time.time() if now is None else float(now)
        return (now - self.ts) <= self.ttl_s


@dataclass
class Observation:
    """
    Minimal observation used by policy + safety.
    """
    pred_class: str
    pred_conf: float
    distance_m: Optional[float]

    joint_error_norm: Optional[float] = None
    motor_current_a: Optional[float] = None
    stall_flag: Optional[bool] = None
    health_score: Optional[float] = None

    timestamp: float = field(default_factory=lambda: time.time())

    @staticmethod
    def from_perception(pred_class: str, pred_conf: float, distance_m: Optional[float]) -> "Observation":
        return Observation(
            pred_class=str(pred_class),
            pred_conf=float(pred_conf),
            distance_m=None if distance_m is None else float(distance_m),
        )


@dataclass(frozen=True)
class ActionProposal:
    action: str
    confidence: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    proposed_action: str
    final_action: str
    reason: str
    signals: Tuple[SafetySignal, ...]


# -----------------------------
# Policy Interface (RL slot)
# -----------------------------

class Policy(Protocol):
    """RL policy interface: propose an action from an observation."""
    def propose(self, obs: Observation) -> ActionProposal:
        ...


class HeuristicPolicy:
    """
    Placeholder policy that mirrors your current demo gating idea.
    Replace with a real RL policy later without changing supervisor code.
    """
    def __init__(self, conf_thresh: float = 0.60, max_scoop_dist: float = 2.50):
        self.conf_thresh = float(conf_thresh)
        self.max_scoop_dist = float(max_scoop_dist)

    def propose(self, obs: Observation) -> ActionProposal:
        cls = (obs.pred_class or "").lower()
        looks_dirty = any(w in cls for w in ("dirty", "dirt", "trash", "plastic"))

        if looks_dirty:
            return ActionProposal(action=RoverAction.BYPASS, confidence=1.0, meta={"policy_reason": "looks_dirty"})
        if obs.pred_conf < self.conf_thresh:
            return ActionProposal(action=RoverAction.BYPASS, confidence=1.0, meta={"policy_reason": "low_conf"})
        if obs.distance_m is None or obs.distance_m > self.max_scoop_dist:
            return ActionProposal(action=RoverAction.BYPASS, confidence=1.0, meta={"policy_reason": "too_far_or_unknown"})
        return ActionProposal(action=RoverAction.SCOOP, confidence=1.0, meta={"policy_reason": "clean_confident_close"})


# -----------------------------
# Safety evaluators (3 failsafes)
# -----------------------------

@dataclass
class HazardRiskConfig:
    stop_distance_m: float = 0.60
    retreat_distance_m: float = 0.35
    max_valid_distance_m: float = 30.0
    unknown_distance_level: SafetyLevel = SafetyLevel.CAUTION


class HazardRiskScorer:
    def __init__(self, cfg: HazardRiskConfig):
        self.cfg = cfg

    def evaluate(self, obs: Observation) -> Optional[SafetySignal]:
        d = obs.distance_m

        if d is None:
            return SafetySignal(
                source="hazard_risk",
                level=self.cfg.unknown_distance_level,
                score=0.5,
                reason="LiDAR distance unavailable",
                override_action=RoverAction.DEGRADED if self.cfg.unknown_distance_level >= SafetyLevel.CAUTION else None,
                ttl_s=0.5,
            )

        d = float(d)
        if d <= 0.0 or d > self.cfg.max_valid_distance_m:
            return SafetySignal(
                source="hazard_risk",
                level=SafetyLevel.CAUTION,
                score=0.6,
                reason=f"LiDAR distance out-of-range ({d:.2f}m)",
                override_action=RoverAction.DEGRADED,
                ttl_s=0.5,
            )

        if d <= self.cfg.retreat_distance_m:
            return SafetySignal(
                source="hazard_risk",
                level=SafetyLevel.CRITICAL,
                score=1.0,
                reason=f"Intrusion/collision imminent (d={d:.2f}m <= {self.cfg.retreat_distance_m:.2f}m)",
                override_action=RoverAction.RETREAT,
                ttl_s=0.5,
            )
        if d <= self.cfg.stop_distance_m:
            return SafetySignal(
                source="hazard_risk",
                level=SafetyLevel.DANGER,
                score=0.9,
                reason=f"Unsafe proximity (d={d:.2f}m <= {self.cfg.stop_distance_m:.2f}m)",
                override_action=RoverAction.STOP,
                ttl_s=0.5,
            )

        return None


@dataclass
class OffNominalConfig:
    joint_error_stop: float = 0.75
    current_stop_a: float = 3.0
    stall_is_critical: bool = True


class OffNominalDetector:
    def __init__(self, cfg: OffNominalConfig):
        self.cfg = cfg

    def evaluate(self, obs: Observation) -> Optional[SafetySignal]:
        if obs.stall_flag is True:
            lvl = SafetyLevel.CRITICAL if self.cfg.stall_is_critical else SafetyLevel.DANGER
            return SafetySignal(
                source="off_nominal",
                level=lvl,
                score=1.0,
                reason="Arm stall detected",
                override_action=RoverAction.STOP if lvl < SafetyLevel.CRITICAL else RoverAction.RETREAT,
                ttl_s=1.0,
            )

        if obs.joint_error_norm is not None and obs.joint_error_norm >= self.cfg.joint_error_stop:
            return SafetySignal(
                source="off_nominal",
                level=SafetyLevel.DANGER,
                score=float(obs.joint_error_norm),
                reason=f"High joint tracking error ({obs.joint_error_norm:.2f} >= {self.cfg.joint_error_stop:.2f})",
                override_action=RoverAction.STOP,
                ttl_s=1.0,
            )

        if obs.motor_current_a is not None and obs.motor_current_a >= self.cfg.current_stop_a:
            return SafetySignal(
                source="off_nominal",
                level=SafetyLevel.CAUTION,
                score=float(obs.motor_current_a),
                reason=f"Elevated motor current ({obs.motor_current_a:.2f}A >= {self.cfg.current_stop_a:.2f}A)",
                override_action=RoverAction.DEGRADED,
                ttl_s=1.0,
            )

        return None


@dataclass
class HealthModeConfig:
    degraded_below: float = 0.60
    return_home_below: float = 0.35
    unknown_health_level: SafetyLevel = SafetyLevel.INFO


class HealthAwareModes:
    def __init__(self, cfg: HealthModeConfig):
        self.cfg = cfg

    def evaluate(self, obs: Observation) -> Optional[SafetySignal]:
        hs = obs.health_score
        if hs is None:
            if self.cfg.unknown_health_level == SafetyLevel.INFO:
                return None
            return SafetySignal(
                source="health_mode",
                level=self.cfg.unknown_health_level,
                score=0.5,
                reason="Health score unavailable",
                override_action=RoverAction.DEGRADED if self.cfg.unknown_health_level >= SafetyLevel.CAUTION else None,
                ttl_s=2.0,
            )

        hs = float(hs)
        if hs <= self.cfg.return_home_below:
            return SafetySignal(
                source="health_mode",
                level=SafetyLevel.DANGER,
                score=1.0 - hs,
                reason=f"Low health score ({hs:.2f} <= {self.cfg.return_home_below:.2f})",
                override_action=RoverAction.RETURN_HOME,
                ttl_s=5.0,
            )
        if hs <= self.cfg.degraded_below:
            return SafetySignal(
                source="health_mode",
                level=SafetyLevel.CAUTION,
                score=1.0 - hs,
                reason=f"Degraded health ({hs:.2f} <= {self.cfg.degraded_below:.2f})",
                override_action=RoverAction.DEGRADED,
                ttl_s=5.0,
            )
        return None


# -----------------------------
# Safety Supervisor (arbiter)
# -----------------------------

@dataclass
class SupervisorConfig:
    override_at_or_above: SafetyLevel = SafetyLevel.DANGER


class SafetySupervisor:
    def __init__(
        self,
        hazard: HazardRiskScorer,
        off_nominal: OffNominalDetector,
        health: HealthAwareModes,
        cfg: SupervisorConfig = SupervisorConfig(),
    ):
        self.hazard = hazard
        self.off_nominal = off_nominal
        self.health = health
        self.cfg = cfg

    @staticmethod
    def default() -> "SafetySupervisor":
        return SafetySupervisor(
            hazard=HazardRiskScorer(HazardRiskConfig()),
            off_nominal=OffNominalDetector(OffNominalConfig()),
            health=HealthAwareModes(HealthModeConfig()),
            cfg=SupervisorConfig(),
        )

    def evaluate_all(self, obs: Observation) -> Tuple[SafetySignal, ...]:
        signals: List[SafetySignal] = []
        for evaluator in (self.hazard, self.off_nominal, self.health):
            s = evaluator.evaluate(obs)
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


# -----------------------------
# Shielded Controller (policy + supervisor)
# -----------------------------

@dataclass
class ShieldedController:
    policy: Policy
    supervisor: SafetySupervisor

    def step(self, obs: Observation) -> Decision:
        proposal = self.policy.propose(obs)
        return self.supervisor.apply(proposal, obs)


# -----------------------------
# CLI Demo (optionally uses lidar_provider.py)
# -----------------------------

def _maybe_get_lidar_distance(args) -> Optional[float]:
    if not args.lidar_mode:
        return None

    # Import lazily so this file works without ROS2 packages installed.
    from lidar_provider import (  # type: ignore
        RellisBinProvider,
        ROS2LaserScanProvider,
        ROS2PointCloud2Provider,
        SmoothingConfig,
    )

    smoothing = SmoothingConfig(window=max(1, int(args.lidar_window)), timeout_s=float(args.lidar_timeout_s))

    if args.lidar_mode == "rellis_bin":
        if not args.lidar_source:
            raise SystemExit("[ERROR] --lidar-source is required for --lidar-mode rellis_bin")
        p = RellisBinProvider(
            args.lidar_source,
            fov_deg=float(args.fov_deg),
            max_range_m=float(args.max_range_m),
            stride=int(args.stride),
            smoothing=smoothing,
        )
        return p.get_distance_m()

    if args.lidar_mode == "ros2_scan":
        p = ROS2LaserScanProvider(
            topic=str(args.lidar_topic),
            fov_deg=float(args.fov_deg),
            max_range_m=float(args.max_range_m),
            smoothing=smoothing,
        )
        p.start()
        try:
            for _ in range(10):
                p.spin_once(timeout_s=0.1)
                d = p.get_distance_m()
                if d is not None:
                    return d
            return p.get_distance_m()
        finally:
            p.shutdown()

    if args.lidar_mode == "ros2_points":
        p = ROS2PointCloud2Provider(
            topic=str(args.lidar_topic),
            fov_deg=float(args.fov_deg),
            max_range_m=float(args.max_range_m),
            stride=int(args.stride),
            smoothing=smoothing,
        )
        p.start()
        try:
            for _ in range(10):
                p.spin_once(timeout_s=0.1)
                d = p.get_distance_m()
                if d is not None:
                    return d
            return p.get_distance_m()
        finally:
            p.shutdown()

    raise SystemExit(f"[ERROR] Unknown lidar mode: {args.lidar_mode}")


def run_demo(args) -> int:
    distance_m = args.distance_m
    if distance_m is None:
        distance_m = _maybe_get_lidar_distance(args)

    obs = Observation(
        pred_class=str(args.pred_class),
        pred_conf=float(args.pred_conf),
        distance_m=None if distance_m is None else float(distance_m),
        joint_error_norm=args.joint_error_norm,
        motor_current_a=args.motor_current_a,
        stall_flag=bool(args.stall_flag) if args.stall_flag else None,
        health_score=args.health_score,
    )

    ctrl = ShieldedController(
        policy=HeuristicPolicy(conf_thresh=float(args.conf_thresh), max_scoop_dist=float(args.max_scoop_dist)),
        supervisor=SafetySupervisor.default(),
    )

    decision = ctrl.step(obs)

    print("\n=== SHIELDED DECISION (Policy + Safety Supervisor) ===")
    print(f"Policy proposed : {decision.proposed_action}")
    print(f"Final action    : {decision.final_action}")
    print(f"Reason          : {decision.reason}")
    if decision.signals:
        print("\nSafety signals (sorted):")
        for s in decision.signals:
            print(f" - [{s.level.name}] {s.source} | score={s.score:.3f} | override={s.override_action} | {s.reason}")
    else:
        print("\nSafety signals  : none")

    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Shielded RL demo: policy proposes, supervisor enforces safety")
    ap.add_argument("--test", action="store_true", help="Run unit tests and exit")
    ap.add_argument("--demo", action="store_true", help="Run a single decision demo")

    ap.add_argument("--pred-class", default="clean", help="Predicted class label (e.g., clean/dirty)")
    ap.add_argument("--pred-conf", type=float, default=0.90, help="Classifier confidence (0..1)")
    ap.add_argument("--distance-m", type=float, default=None, help="LiDAR distance in meters (optional if using lidar_provider)")

    ap.add_argument("--conf-thresh", type=float, default=0.60)
    ap.add_argument("--max-scoop-dist", type=float, default=2.50)

    ap.add_argument("--joint-error-norm", type=float, default=None)
    ap.add_argument("--motor-current-a", type=float, default=None)
    ap.add_argument("--stall-flag", action="store_true", help="If set, forces stall condition")

    ap.add_argument("--health-score", type=float, default=None, help="0..1 health score (lower triggers degraded/return)")

    ap.add_argument("--lidar-mode", choices=["rellis_bin", "ros2_scan", "ros2_points"], default=None)
    ap.add_argument("--lidar-source", default="", help="(rellis_bin) path to *.bin directory or virtual zip path")
    ap.add_argument("--lidar-topic", default="/scan", help="(ros2_*) topic name")
    ap.add_argument("--lidar-window", type=int, default=5)
    ap.add_argument("--lidar-timeout-s", type=float, default=1.0)

    ap.add_argument("--fov-deg", type=float, default=30.0)
    ap.add_argument("--max-range-m", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=5)

    return ap


# -----------------------------
# Unit tests (pure Python)
# -----------------------------

class TestSupervisorArbitration(unittest.TestCase):
    def test_no_signals_keeps_policy(self):
        ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())
        obs = Observation.from_perception("clean", 0.99, 2.0)
        d = ctrl.step(obs)
        self.assertEqual(d.final_action, RoverAction.SCOOP)

    def test_hazard_stop_overrides(self):
        ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())
        obs = Observation.from_perception("clean", 0.99, 0.50)
        d = ctrl.step(obs)
        self.assertEqual(d.final_action, RoverAction.STOP)
        self.assertTrue(any(s.source == "hazard_risk" for s in d.signals))

    def test_health_return_home_overrides(self):
        ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())
        obs = Observation.from_perception("clean", 0.99, 2.0)
        obs.health_score = 0.20  # type: ignore
        d = ctrl.step(obs)
        self.assertEqual(d.final_action, RoverAction.RETURN_HOME)


def run_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()

    if args.test:
        return run_tests()

    if args.demo:
        return run_demo(args)

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
