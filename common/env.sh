#!/usr/bin/env bash
# Source-only; no CUDA/ROS/CAN initialization.
PIPER_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIPER_PROJECT_ROOT
if [[ -f "$PIPER_PROJECT_ROOT/configs/local.env" ]]; then
  source "$PIPER_PROJECT_ROOT/configs/local.env"
fi
export PIPER_OPENPI_DIR="${PIPER_OPENPI_DIR:-$PIPER_PROJECT_ROOT/deps/openpi}"
export PIPER_RLINF_DIR="${PIPER_RLINF_DIR:-$PIPER_PROJECT_ROOT/deps/rlinf}"
export PIPER_PYTHON="${PIPER_PYTHON:-$PIPER_PROJECT_ROOT/.venv/bin/python}"
export PIPER_ROS_PYTHON="${PIPER_ROS_PYTHON:-/usr/bin/python3}"
export PIPER_ROS_WS="${PIPER_ROS_WS:-$PIPER_PROJECT_ROOT/robot/cobot_magic/Piper_ros_private-ros-noetic}"
export REPO_PATH="$PIPER_RLINF_DIR"
export RLINF_DIR="$PIPER_RLINF_DIR"
export RLINF_OPENPI_DIR="$PIPER_OPENPI_DIR"
export PYTHONPATH="$PIPER_RLINF_DIR:$PIPER_OPENPI_DIR/src:$PIPER_OPENPI_DIR/packages/openpi-client/src:$PIPER_OPENPI_DIR:$PIPER_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
