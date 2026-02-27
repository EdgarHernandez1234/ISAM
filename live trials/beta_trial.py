#!/usr/bin/env bash
set -euo pipefail
# make them executable: chmod +x launchers/alpha_trial.sh launchers/beta_trial.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

echo "=== ALAM Beta Trial Launcher ==="
echo "Repo: $REPO_ROOT"
echo

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  echo "[env] Activating .venv"
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
elif [[ -f "$REPO_ROOT/venv/bin/activate" ]]; then
  echo "[env] Activating venv"
  # shellcheck disable=SC1091
  source "$REPO_ROOT/venv/bin/activate"
else
  echo "[env] No venv found (.venv/ or venv/). Using system python."
fi

for ROS_DISTRO in humble iron jazzy foxy galactic; do
  if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    echo "[ros] Sourcing /opt/ros/${ROS_DISTRO}/setup.bash"
    # shellcheck disable=SC1091
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    break
  fi
done

if command -v rviz2 >/dev/null 2>&1; then
  echo "[rviz] Launching RViz2..."
  rviz2 >/dev/null 2>&1 &
  RVIZ_PID=$!
  echo "[rviz] RViz2 PID: ${RVIZ_PID}"
  sleep 1
else
  echo "[rviz] rviz2 not found in PATH. Open RViz manually if needed."
fi

echo
echo ">>> PAUSE: Set up your RViz/simulator now."
read -r -p "Press ENTER to start beta_trial.py... " _

echo
echo "[run] python3 -u beta_trial.py $*"
set +e
python3 -u "$REPO_ROOT/beta_trial.py" "$@"
EXIT_CODE=$?
set -e

echo
echo "[done] beta_trial.py exited with code: ${EXIT_CODE}"
read -r -p "Press ENTER to close this terminal... " _
exit "${EXIT_CODE}"