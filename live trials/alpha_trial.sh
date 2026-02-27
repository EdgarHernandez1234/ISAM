#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root:
# - If this script is in <repo>/launchers/, repo root is parent directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

echo "=== ALAM Alpha Trial Launcher ==="
echo "Repo: $REPO_ROOT"
echo

# Activate a local venv if present
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

# Source ROS2 if installed (best-effort)
for ROS_DISTRO in humble iron jazzy foxy galactic; do
  if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    echo "[ros] Sourcing /opt/ros/${ROS_DISTRO}/setup.bash"
    # shellcheck disable=SC1091
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    break
  fi
done

# Launch RViz2 (best-effort) and pause so the user can set up the sim
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
read -r -p "Press ENTER to start alpha_trial.py... " _

echo
echo "[run] python3 -u alpha_trial.py $*"
set +e
python3 -u "$REPO_ROOT/alpha_trial.py" "$@"
EXIT_CODE=$?
set -e

echo
echo "[done] alpha_trial.py exited with code: ${EXIT_CODE}"
read -r -p "Press ENTER to close this terminal... " _
exit "${EXIT_CODE}"