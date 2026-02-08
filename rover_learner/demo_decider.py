#!/usr/bin/env python3
"""demo_decider.py (rover_learner)

LIVE Jetson demo: Camera + LiDAR + ML model + RL safety shield + logging.

This script is *online mode* and should live in rover_learner/.
Your offline/replay scripts stay in rover_decider/.

----------------------------
HOW TO RUN (incremental)
----------------------------

0) Environment prep (Jetson):
   - Camera verified working (nvarguscamerasrc works, /dev/video0 exists)
   - LiDAR driver running and publishing /scan (ROS2 LaserScan), e.g. Slamtec rplidar_ros
   - Optional: install deps
       sudo apt update
       sudo apt install -y python3-opencv python3-pip
       pip3 install ultralytics pytest

1) Quick component checks (no motion control, safe):
   A) Camera online check (CSI):
       python3 -m rover_learner.demo_decider --check-camera --camera csi --sensor-id 0

   B) LiDAR online check (ROS2 /scan must be publishing):
       python3 -m rover_learner.demo_decider --check-lidar --lidar ros2 --lidar-topic /scan

   C) RL layer import + deterministic override check:
       python3 -m rover_learner.demo_decider --check-rl

   D) ML model check (use your saved model path):
       python3 -m rover_learner.demo_decider --check-ml --model /path/to/sand_cls_clean_vs_dirt.pt --camera csi --sensor-id 0

   E) Full stack check (camera + lidar + rl + ml), exits after one step:
       python3 -m rover_learner.demo_decider --check-all --model /path/to/sand_cls_clean_vs_dirt.pt --camera csi --sensor-id 0 --lidar ros2 --lidar-topic /scan

2) Full live demo loop (safe; no arm required):
   python3 -m rover_learner.demo_decider \
     --model /path/to/sand_cls_clean_vs_dirt.pt \
     --camera csi --sensor-id 0 \
     --lidar ros2 --lidar-topic /scan \
     --hz 2 --num-steps 200

3) Forced safety mode demos (prove supervisor overrides):
   - Force "too close" hazard (should STOP/RETREAT):
       python3 -m rover_learner.demo_decider --demo-safety --force-distance 0.25

   - Force low health (should RETURN_HOME):
       python3 -m rover_learner.demo_decider --demo-safety --force-health 0.15

   - Force stall flag (should STOP):
       python3 -m rover_learner.demo_decider --demo-safety --force-stall

Notes:
  - For unit tests (no hardware): `pytest -q rover_learner/tests` or `python3 -m unittest`.
  - You can run a mock full-stack step without ultralytics by using `--model mock --camera mock --lidar mock`.

----------------------------
What this script proves
----------------------------
  ✓ LiDAR is online (distance updates)
  ✓ Camera is online (frames read)
  ✓ ML model is active (pred_class/conf updates)
  ✓ RL safety layer is referenced (proposed vs final action can differ)
  ✓ Forced safety demos trigger deterministic overrides

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

from .camera_provider import CSICameraProvider, USBCameraProvider, MockCameraProvider
from .lidar_provider import ROS2LaserScanProvider, MockLidarProvider
from .logger import CSVDecisionLogger, DecisionFrame
from .rl_safety_supervisor import HeuristicPolicy, SafetySupervisor, ShieldedController
from .core import StepInputs, Perception, Telemetry, step_with_safety


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="rover_learner live demo decider")
    p.add_argument("--model", type=str, default="mock", help="Path to Ultralytics model .pt, or 'mock'")
    p.add_argument("--camera", type=str, default="csi", choices=["csi", "usb", "mock"])
    p.add_argument("--sensor-id", type=int, default=0)
    p.add_argument("--usb-index", type=int, default=0)
    p.add_argument("--lidar", type=str, default="ros2", choices=["ros2", "mock"])
    p.add_argument("--lidar-topic", type=str, default="/scan")
    p.add_argument("--hz", type=float, default=2.0)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--no-display", action="store_true")

    # checks
    p.add_argument("--check-camera", action="store_true")
    p.add_argument("--check-lidar", action="store_true")
    p.add_argument("--check-ml", action="store_true")
    p.add_argument("--check-rl", action="store_true")
    p.add_argument("--check-all", action="store_true")

    # forced safety demo
    p.add_argument("--demo-safety", action="store_true", help="Run 3-5 steps using forced safety fields only.")
    p.add_argument("--force-distance", type=float, default=None, help="Override LiDAR distance (m) for demo.")
    p.add_argument("--force-health", type=float, default=None, help="Override health_score (0..1) for demo.")
    p.add_argument("--force-stall", action="store_true", help="Set stall_flag=True for demo.")
    p.add_argument("--force-joint-error", type=float, default=None)
    p.add_argument("--force-motor-current", type=float, default=None)

    return p.parse_args()


class _MockModel:
    def __init__(self):
        self._toggle = False
    def predict(self, frame) -> Tuple[str, float]:
        self._toggle = not self._toggle
        return ("clean" if self._toggle else "dirty", 0.88 if self._toggle else 0.91)


class _UltralyticsModel:
    def __init__(self, path: str):
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. pip3 install ultralytics")
        self._model = YOLO(path)

    def predict(self, frame) -> Tuple[str, float]:
        # Expect single-image inference; return best class and conf
        res = self._model(frame, verbose=False)[0]
        # classification models: res.probs has top1 + top1conf
        if hasattr(res, "probs") and res.probs is not None:
            idx = int(res.probs.top1)
            conf = float(res.probs.top1conf)
            # names mapping
            names = getattr(res, "names", None) or getattr(self._model, "names", None) or {}
            label = names.get(idx, str(idx)) if isinstance(names, dict) else str(idx)
            return str(label).lower(), conf
        # fallback: treat as unknown
        return "unknown", 0.0


def make_camera(args):
    if args.camera == "mock":
        return MockCameraProvider()
    if args.camera == "usb":
        return USBCameraProvider(index=args.usb_index)
    return CSICameraProvider(sensor_id=args.sensor_id)


def make_lidar(args):
    if args.lidar == "mock":
        return MockLidarProvider(distance_m=2.0)
    p = ROS2LaserScanProvider(topic=args.lidar_topic)
    p.start()
    return p


def make_model(args):
    if str(args.model).lower() == "mock":
        return _MockModel()
    return _UltralyticsModel(args.model)


def check_camera(cam) -> None:
    frame, ts = cam.read()
    print(f"[OK] camera read at {ts:.3f}; frame={type(frame)}")

def check_lidar(lidar) -> None:
    t0 = time.time()
    while time.time() - t0 < 3.0:
        d = lidar.get_distance_m()
        if d is not None:
            print(f"[OK] lidar distance_m={d:.3f}")
            return
        time.sleep(0.1)
    raise RuntimeError("LiDAR did not produce a distance within 3s. Is /scan publishing?")

def check_rl() -> None:
    ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())
    out = step_with_safety(ctrl, StepInputs(Perception("dirty", 0.99), distance_m=0.25))
    assert out.final_action in ("STOP", "RETREAT"), f"unexpected final_action: {out.final_action}"
    print(f"[OK] rl safety override: proposed={out.proposed_action} final={out.final_action}")

def check_ml(model, cam) -> None:
    frame, _ = cam.read()
    pred_class, pred_conf = model.predict(frame)
    print(f"[OK] ml prediction: class={pred_class} conf={pred_conf:.3f}")


def overlay_text(img, lines):
    if cv2 is None:
        return img
    y = 30
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28
    return img


def main() -> None:
    args = parse_args()

    # ----------------------------
    # Checks are incremental and should not require unrelated hardware.
    # ----------------------------
    if args.check_rl and not (args.check_camera or args.check_lidar or args.check_ml or args.check_all):
        check_rl()
        return

    cam = None
    lidar = None
    model = None

    try:
        if args.check_camera or args.check_ml or args.check_all:
            cam = make_camera(args)
        if args.check_lidar or args.check_all:
            lidar = make_lidar(args)
        if args.check_ml or args.check_all:
            model = make_model(args)

        # checks
        if args.check_camera or args.check_all:
            assert cam is not None
            check_camera(cam)
        if args.check_lidar or args.check_all:
            assert lidar is not None
            check_lidar(lidar)
        if args.check_rl or args.check_all:
            check_rl()
        if args.check_ml or args.check_all:
            assert model is not None and cam is not None
            check_ml(model, cam)

        if args.check_camera or args.check_lidar or args.check_rl or args.check_ml or args.check_all:
            return

    finally:
        # Close resources opened during checks (if any)
        try:
            if hasattr(lidar, "close"):
                lidar.close()
        except Exception:
            pass
        try:
            if hasattr(cam, "close"):
                cam.close()
        except Exception:
            pass

    # ----------------------------
    # Full live loop starts here
    # ----------------------------
    cam = make_camera(args)
    lidar = make_lidar(args)
    model = make_model(args)

    # controller + logger

    ctrl = ShieldedController(policy=HeuristicPolicy(), supervisor=SafetySupervisor.default())
    out_csv = Path(__file__).resolve().parent / "logs" / f"live_decision_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    logger = CSVDecisionLogger(out_csv)

    display = (not args.no_display) and (cv2 is not None)

    def forced_telemetry() -> Telemetry:
        return Telemetry(
            joint_error_norm=args.force_joint_error,
            motor_current_a=args.force_motor_current,
            stall_flag=True if args.force_stall else None,
            health_score=args.force_health,
        )

    try:
        period = 1.0 / max(args.hz, 0.1)

        if args.demo_safety:
            # Run a few steps using forced fields; no need for camera/lidar
            for i in range(5):
                dist = args.force_distance if args.force_distance is not None else 2.0
                inp = StepInputs(Perception("dirty", 0.95), distance_m=dist, telemetry=forced_telemetry())
                out = step_with_safety(ctrl, inp)
                print(f"[DEMO] step={i} dist={dist} proposed={out.proposed_action} final={out.final_action} reason={out.reason}")
                time.sleep(0.2)
            return

        for step in range(int(args.num_steps)):
            t0 = time.time()
            frame, ts = cam.read()
            pred_class, pred_conf = model.predict(frame)

            dist = args.force_distance if args.force_distance is not None else lidar.get_distance_m()

            inp = StepInputs(
                Perception(pred_class, pred_conf),
                distance_m=dist,
                telemetry=forced_telemetry(),
            )
            out = step_with_safety(ctrl, inp)

            safety_json = CSVDecisionLogger.safety_to_json(out.signals)

            logger.log(DecisionFrame(
                ts=ts,
                frame_id=step,
                pred_class=pred_class,
                pred_conf=float(pred_conf),
                distance_m=None if dist is None else float(dist),
                proposed_action=out.proposed_action,
                final_action=out.final_action,
                reason=out.reason,
                safety_json=safety_json,
            ))

            print(f"step={step} class={pred_class} conf={pred_conf:.2f} dist={dist} proposed={out.proposed_action} final={out.final_action}")

            if display and cv2 is not None and not isinstance(frame, dict):
                lines = [
                    f"class={pred_class} conf={pred_conf:.2f}",
                    f"dist_m={dist if dist is not None else 'None'}",
                    f"policy={out.proposed_action} final={out.final_action}",
                    f"reason={out.reason}",
                ]
                overlay_text(frame, lines)
                cv2.imshow("rover_learner live decider", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        try:
            logger.close()
        except Exception:
            pass
        # Close lidar node if ROS2 provider
        try:
            if hasattr(lidar, "close"):
                lidar.close()
        except Exception:
            pass
        try:
            if hasattr(cam, "close"):
                cam.close()
        except Exception:
            pass
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()
