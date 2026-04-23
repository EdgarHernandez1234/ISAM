#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/humble/setup.bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/greedy_env.sh"

echo "=== Greedy operator environment ==="
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "ROS_DISCOVERY_SERVER=${ROS_DISCOVERY_SERVER}"
echo "ALAM_AUTONOMY_BACKEND=${ALAM_AUTONOMY_BACKEND}"
echo "ALAM_GREEDY_MODEL_PATH=${ALAM_GREEDY_MODEL_PATH}"
echo "ALAM_GREEDY_QTABLE_PATH=${ALAM_GREEDY_QTABLE_PATH}"
echo "ALAM_GREEDY_FRONT_TOPIC=${ALAM_GREEDY_FRONT_TOPIC}"
echo "ALAM_GREEDY_BACK_TOPIC=${ALAM_GREEDY_BACK_TOPIC}"
echo "ALAM_GREEDY_SCAN_TOPIC=${ALAM_GREEDY_SCAN_TOPIC}"
echo "ALAM_GREEDY_POSE_TOPIC=${ALAM_GREEDY_POSE_TOPIC}"
echo "ALAM_GREEDY_IMAGE_TRANSPORT=${ALAM_GREEDY_IMAGE_TRANSPORT}"

echo
echo "=== Topic discovery ==="
ros2 topic list | sort | grep -E "rs_front|rs_back|image_raw|scan|alam|pose" || true

echo
echo "=== Python compile check ==="
cd "$HOME/Desktop/rover_learner/conference_demo/operator_app/operator_app"
python3 -m py_compile manual_operator_app.py rover_operator/greedy_autonomy_backend.py
echo "[OK] Compile check passed"
