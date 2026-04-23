#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/humble/setup.bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/greedy_env.sh"

APP_DIR="$HOME/Desktop/rover_learner/conference_demo/operator_app/operator_app"
MAIN_FILE="${APP_DIR}/run_operator_menu.py"

if [[ ! -f "${MAIN_FILE}" ]]; then
  echo "[ERROR] Could not find ${MAIN_FILE}"
  exit 1
fi

echo "[INFO] Launching operator app in greedy-autonomy mode"
echo "[INFO] ALAM_AUTONOMY_BACKEND=${ALAM_AUTONOMY_BACKEND}"
echo "[INFO] MODEL=${ALAM_GREEDY_MODEL_PATH}"
echo "[INFO] FRONT_TOPIC=${ALAM_GREEDY_FRONT_TOPIC}"
echo "[INFO] BACK_TOPIC=${ALAM_GREEDY_BACK_TOPIC}"
echo "[INFO] SCAN_TOPIC=${ALAM_GREEDY_SCAN_TOPIC}"
echo "[INFO] POSE_TOPIC=${ALAM_GREEDY_POSE_TOPIC}"
echo "[INFO] QTABLE=${ALAM_GREEDY_QTABLE_PATH}"

cd "${APP_DIR}"
exec python3 run_operator_menu.py
