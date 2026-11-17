#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
STAGE="${1:?Expected collect|convert|norm-stats|train|serve|client}"
shift
case "$STAGE" in
 collect) exec "$PIPER_ROS_PYTHON" "$PIPER_PROJECT_ROOT/robot/cobot_magic/collect_data/new_collect_data.py" "$@" ;;
 convert) exec "$PIPER_PYTHON" "$PIPER_PROJECT_ROOT/robot/cobot_magic/collect_data/hdf5_lerobot.py" "$@" ;;
 norm-stats) cd "$PIPER_OPENPI_DIR"; exec "$PIPER_PYTHON" scripts/compute_norm_stats.py "${PIPER_CONFIG_NAME:-pi05_connect_strings}" "$@" ;;
 train) cd "$PIPER_OPENPI_DIR"; exec "$PIPER_PYTHON" scripts/train.py "${PIPER_CONFIG_NAME:-pi05_connect_strings}" "$@" ;;
 serve) : "${PIPER_JAX_CHECKPOINT:?Set PIPER_JAX_CHECKPOINT}"; cd "$PIPER_OPENPI_DIR"; exec "$PIPER_PYTHON" scripts/serve_policy.py --env ALOHA --port "${PIPER_BASE_PORT:-6016}" policy:checkpoint --policy.config "${PIPER_CONFIG_NAME:-pi05_connect_strings}" --policy.dir "$PIPER_JAX_CHECKPOINT" "$@" ;;
 client) exec bash "$PIPER_PROJECT_ROOT/recap/client.sh" "$@" ;;
 *) echo "Unknown pi05 stage: $STAGE" >&2; exit 2 ;;
esac
