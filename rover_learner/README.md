# rover_learner (online / live mode)

Purpose:
- Take the offline-trained ML model (Ultralytics classifier) and run it on **live camera frames**
- Take **live LiDAR** (ROS2 `/scan`) and compute `min_forward_distance_m`
- Route every proposed action through `rl_safety_supervisor` to enforce failsafes
- Log every decision to CSV for later analysis

## Quick start
- Run unit tests on any machine:
  - `pytest -q rover_learner/tests`
  - or `python3 -m unittest discover -s rover_learner/tests -v`

- Live component checks on Jetson:
  - `python3 -m rover_learner.demo_decider --check-all --model /path/to/model.pt --camera csi --sensor-id 0 --lidar ros2 --lidar-topic /scan`

- Forced safety demos:
  - `python3 -m rover_learner.demo_decider --demo-safety --force-distance 0.25`
  - `python3 -m rover_learner.demo_decider --demo-safety --force-health 0.15`
  - `python3 -m rover_learner.demo_decider --demo-safety --force-stall`
