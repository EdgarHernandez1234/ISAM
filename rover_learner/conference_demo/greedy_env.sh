#!/usr/bin/env bash
# Source this file before launching the operator app in greedy-autonomy mode.

# ROS networking for the Jetson <-> desktop dual-system setup
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER=192.168.50.2:11811

# Controller-app autonomy backend selection
export ALAM_AUTONOMY_BACKEND=greedy

# Greedy backend model + persistence
export ALAM_GREEDY_MODEL_PATH="${ALAM_GREEDY_MODEL_PATH:-$HOME/Desktop/models/regolith/best_yolov26.onnx}"
export ALAM_GREEDY_QTABLE_PATH="${ALAM_GREEDY_QTABLE_PATH:-$HOME/Desktop/rover_learner/conference_demo/q_table_greedy_operator_backend.json}"

# Gazebo proxy topics coming from the desktop proxy bridge
export ALAM_GREEDY_FRONT_TOPIC="${ALAM_GREEDY_FRONT_TOPIC:-/sim/rs_front/image_raw}"
export ALAM_GREEDY_BACK_TOPIC="${ALAM_GREEDY_BACK_TOPIC:-/sim/rs_back/image_raw}"
export ALAM_GREEDY_SCAN_TOPIC="${ALAM_GREEDY_SCAN_TOPIC:-/scan}"
export ALAM_GREEDY_POSE_TOPIC="${ALAM_GREEDY_POSE_TOPIC:-/alam/rover_pose_json}"
export ALAM_GREEDY_IMAGE_TRANSPORT="${ALAM_GREEDY_IMAGE_TRANSPORT:-raw}"

# Greedy RL tuning defaults
export ALAM_GREEDY_EPSILON="${ALAM_GREEDY_EPSILON:-0.45}"
export ALAM_GREEDY_MIN_EPSILON="${ALAM_GREEDY_MIN_EPSILON:-0.15}"
export ALAM_GREEDY_EPSILON_DECAY="${ALAM_GREEDY_EPSILON_DECAY:-0.999}"
export ALAM_GREEDY_ALPHA="${ALAM_GREEDY_ALPHA:-0.20}"
export ALAM_GREEDY_GAMMA="${ALAM_GREEDY_GAMMA:-0.95}"

# Action smoothing defaults
export ALAM_GREEDY_ACTION_HOLD_FRAMES="${ALAM_GREEDY_ACTION_HOLD_FRAMES:-5}"
export ALAM_GREEDY_ACTION_SWITCH_MARGIN="${ALAM_GREEDY_ACTION_SWITCH_MARGIN:-0.18}"
export ALAM_GREEDY_ACTION_SWITCH_COOLDOWN_S="${ALAM_GREEDY_ACTION_SWITCH_COOLDOWN_S:-0.40}"

# Optional one-shot reset for the learned Q-table on next launch.
# Set to 1 only when you intentionally want a fresh table.
export ALAM_GREEDY_RESET_QTABLE="${ALAM_GREEDY_RESET_QTABLE:-0}"
