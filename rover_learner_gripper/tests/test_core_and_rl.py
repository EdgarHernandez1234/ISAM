from rover_learner.core import StepInputs, Perception, Telemetry, step_with_safety
from rover_learner.rl_safety_supervisor import HeuristicPolicy, SafetySupervisor, ShieldedController

def test_step_with_safety_overrides_close_hazard():
    ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())
    inp = StepInputs(Perception("dirty", 0.99), distance_m=0.25, telemetry=Telemetry())
    out = step_with_safety(ctrl, inp)
    assert out.final_action in ("STOP", "RETREAT")
    assert out.proposed_action in ("SCOOP", "BYPASS", "APPROACH", "HOLD", "STOP", "RETREAT", "RETURN_HOME", "DEGRADED")

def test_step_with_safety_return_home_on_low_health():
    ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())
    inp = StepInputs(Perception("clean", 0.8), distance_m=2.0, telemetry=Telemetry(health_score=0.1))
    out = step_with_safety(ctrl, inp)
    assert out.final_action in ("RETURN_HOME", "STOP", "DEGRADED")
