"""rover_learner

Online (live) runtime package:
  - Uses the offline-trained ML model (Ultralytics YOLO classifier) for live camera frames
  - Uses live LiDAR for hazard distance
  - Routes every action through rl_safety_supervisor (shielded controller)

Keep rover_decider/ as your offline/replay/bench package.
"""

__all__ = [
    "core",
    "logger",
    "camera_provider",
    "lidar_provider",
    "rl_safety_supervisor",
]
