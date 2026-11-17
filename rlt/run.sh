#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"
STAGE="${1:?Expected stage1|serve|preflight|stage2|stage2-base|replay}"
shift
cd "$PIPER_RLINF_DIR"
export RLINF_NODE_RANK="${RLINF_NODE_RANK:-0}"
export RLINF_JAX_REF_HOST="${RLINF_JAX_REF_HOST:-${PIPER_ROBOT_IP:-127.0.0.1}}"
export RLINF_JAX_PREFIX_HOST="${RLINF_JAX_PREFIX_HOST:-$RLINF_JAX_REF_HOST}"
export RLINF_JAX_REF_PORT="${RLINF_JAX_REF_PORT:-6006}"
export RLINF_PRESERVE_ROS_MASTER=1
export PIPER_ROBOT_IP="${PIPER_ROBOT_IP:-127.0.0.1}"
case "$STAGE" in
 stage1)
  : "${PIPER_JAX_CHECKPOINT:?Set PIPER_JAX_CHECKPOINT}" "${PIPER_DEMO_DATASET:?Set PIPER_DEMO_DATASET}"
  exec "$PIPER_PYTHON" -m torch.distributed.run --standalone --nproc_per_node "${PIPER_GPUS:-1}" toolkits/realworld_check/train_piper_rlt_token_native_jax.py --config-name "${PIPER_CONFIG_NAME:-pi05_connect_strings}" --jax-checkpoint "$PIPER_JAX_CHECKPOINT" --dataset-dir "$PIPER_DEMO_DATASET" --output-dir "${PIPER_RLT_STAGE1_OUTPUT:-$PIPER_PROJECT_ROOT/runs/rlt/stage1}" "$@"
  ;;
 serve)
  : "${PIPER_JAX_CHECKPOINT:?Set PIPER_JAX_CHECKPOINT}" "${RLINF_STAGE1_RLT_TOKEN:?Set RLINF_STAGE1_RLT_TOKEN}"
  exec "$PIPER_PYTHON" toolkits/realworld_check/serve_piper_rlt_compact.py --config "${PIPER_CONFIG_NAME:-pi05_connect_strings}" --checkpoint-dir "$PIPER_JAX_CHECKPOINT" --rlt-token-checkpoint "$RLINF_STAGE1_RLT_TOKEN" --port "$RLINF_JAX_REF_PORT" "$@"
  ;;
 preflight) exec "$PIPER_PYTHON" toolkits/realworld_check/preflight_piper_rlt_stage2.py --ray-address "${PIPER_RLT_RAY_ADDRESS:-127.0.0.1:6379}" "$@" ;;
 stage2|stage2-base)
  : "${RLINF_STAGE1_RLT_TOKEN:?Set RLINF_STAGE1_RLT_TOKEN}"
  CONFIG=realworld_piper_rlt_stage2_ac_mlp
  if [[ "$STAGE" == stage2-base ]]; then CONFIG=piper_rlt_base; fi
  CONFIG_ONLY=false
  for argument in "$@"; do
   if [[ "$argument" == --cfg || "$argument" == --help ]]; then CONFIG_ONLY=true; fi
  done
  if [[ "$CONFIG_ONLY" == false ]]; then
   source /opt/ros/noetic/setup.bash
   source "$PIPER_ROS_WS/devel/setup.bash"
   export PYTHONPATH="$PIPER_RLINF_DIR:$PIPER_OPENPI_DIR/src:$PIPER_OPENPI_DIR/packages/openpi-client/src:$PYTHONPATH"
   "$PIPER_PYTHON" toolkits/realworld_check/preflight_piper_rlt_stage2.py --ray-address "${PIPER_RLT_RAY_ADDRESS:-127.0.0.1:6379}"
  fi
  # Ray is explicitly started by the operator after exporting this environment.
  export RAY_ADDRESS="${PIPER_RLT_RAY_ADDRESS:-127.0.0.1:6379}"
  exec "$PIPER_PYTHON" examples/embodiment/train_async.py --config-path "$PIPER_RLINF_DIR/examples/embodiment/config" --config-name "$CONFIG" "runner.logger.log_path=$PIPER_PROJECT_ROOT/runs/rlt/stage2" "algorithm.replay_buffer.load_path=${RLINF_REPLAY_LOAD_PATH:-null}" "$@"
  ;;
 replay)
  : "${RLINF_STAGE1_RLT_TOKEN:?Set RLINF_STAGE1_RLT_TOKEN}" "${RLINF_REPLAY_LOAD_PATH:?Set RLINF_REPLAY_LOAD_PATH}"
  exec "$PIPER_PYTHON" examples/embodiment/train_piper_rlt_stage2_replay_only.py --config-path "$PIPER_RLINF_DIR/examples/embodiment/config" --config-name "${PIPER_RLT_CONFIG:-realworld_piper_rlt_stage2_ac_mlp}" "runner.logger.log_path=$PIPER_PROJECT_ROOT/runs/rlt/replay" "runner.max_steps=${PIPER_REPLAY_STEPS:-2000}" runner.resume_dir=null algorithm.replay_buffer.auto_save=false "$@"
  ;;
 *) echo "Unknown RLT stage: $STAGE" >&2; exit 2 ;;
esac
