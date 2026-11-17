#!/usr/bin/env bash
RECAP_RELEASE_ROOT="${PIPER_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
source "$RECAP_RELEASE_ROOT/common/env.sh"
export RECAP_REPO_DIR="$PIPER_RLINF_DIR"
export RECAP_PYTHON="$PIPER_PYTHON"
export RECAP_CLIENT_PYTHON="${RECAP_CLIENT_PYTHON:-$PIPER_PYTHON}"
export RECAP_NATIVE_OPENPI="$PIPER_OPENPI_DIR"
export RECAP_SOURCE_CONFIG_FILE="${RECAP_SOURCE_CONFIG_FILE:-$PIPER_OPENPI_DIR/src/openpi/training/config.py}"
export RECAP_SOURCE_CONFIG="${RECAP_SOURCE_CONFIG:-${PIPER_CONFIG_NAME:-pi05_connect_strings}}"
export RECAP_PI05_JAX_DIR="${PIPER_JAX_CHECKPOINT:-$PIPER_PROJECT_ROOT/artifacts/pi05}"
export RECAP_NORM_STATS="${PIPER_NORM_STATS:-$RECAP_PI05_JAX_DIR/assets/all_wire_data/norm_stats.json}"
export RECAP_RAW_DIR="${RECAP_RAW_DIR:-$PIPER_PROJECT_ROOT/data/recap/raw}"
export RECAP_DATASET="${RECAP_DATASET:-$PIPER_PROJECT_ROOT/data/recap/round_001/lerobot}"
export RECAP_SFT_DATASET="${RECAP_SFT_DATASET:-$PIPER_PROJECT_ROOT/data/recap/demos_copy}"
export RECAP_INT_DATASET="${RECAP_INT_DATASET:-$PIPER_PROJECT_ROOT/data/recap/interventions}"
export RECAP_PI05_TORCH_DIR="${RECAP_PI05_TORCH_DIR:-$PIPER_PROJECT_ROOT/artifacts/recap/pi05_torch}"
export RECAP_SIGLIP_DIR="${RECAP_SIGLIP_DIR:-$PIPER_PROJECT_ROOT/artifacts/recap/siglip2-so400m-patch14-224}"
export RECAP_GEMMA_DIR="${RECAP_GEMMA_DIR:-$PIPER_PROJECT_ROOT/artifacts/recap/gemma-3-270m}"
export RECAP_RUN_DIR="${RECAP_RUN_DIR:-$PIPER_PROJECT_ROOT/runs/recap/round_001}"
export RECAP_FOLLOWER_MUX_ROS="$PIPER_PROJECT_ROOT/robot/follower_hold"
export RECAP_RAY_ADDRESS="${RECAP_RAY_ADDRESS:-127.0.0.1:6385}"
export RAY_ADDRESS="$RECAP_RAY_ADDRESS"
export HF_DATASETS_CACHE="${RECAP_DATASETS_CACHE:-$PIPER_PROJECT_ROOT/artifacts/recap/datasets_cache}"
if [[ -z "${RECAP_TASK:-}" ]]; then
 RECAP_TASK="$("$RECAP_PYTHON" -m rlinf.recap_piper.cli task --config-file "$RECAP_SOURCE_CONFIG_FILE" --config-name "$RECAP_SOURCE_CONFIG")" || return 1
 export RECAP_TASK
fi
