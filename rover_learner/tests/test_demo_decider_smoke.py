import subprocess, sys, pathlib

def test_demo_decider_check_rl_runs():
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "-m", "rover_learner.demo_decider", "--check-rl"]
    p = subprocess.run(cmd, cwd=str(root_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.returncode == 0, p.stderr
    assert "[OK] rl safety override" in p.stdout
