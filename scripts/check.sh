#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"
cd "$PIPER_PROJECT_ROOT"
export RECAP_NATIVE_OPENPI="$PIPER_OPENPI_DIR"
export PYTHONDONTWRITEBYTECODE=1
"$PIPER_PYTHON" scripts/bootstrap.py --verify-only
"$PIPER_PYTHON" -m ruff check scripts tests conftest.py
"$PIPER_PYTHON" -m ruff format --check scripts tests conftest.py
"$PIPER_PYTHON" -m pytest -c pytest.ini -q -rs -p no:cacheprovider \
  tests robot/cobot_magic/Piper_ros_private-ros-noetic/src/piper/test \
  deps/rlinf/tests/unit_tests/test_piper_recap.py \
  deps/rlinf/tests/unit_tests/test_recap_mux_guard.py \
  deps/rlinf/tests/unit_tests/test_recap_client_blend.py \
  deps/rlinf/tests/unit_tests/test_recap_alignment_schedule.py \
  deps/rlinf/tests/unit_tests/test_rlt_takeover_as_failure.py \
  deps/rlinf/tests/unit_tests/test_rlt_rtc_alignment.py \
  deps/rlinf/tests/unit_tests/test_piper_four_can_rlt.py \
  deps/rlinf/tests/unit_tests/test_rlt_residual_and_teaching.py \
  deps/rlinf/tests/unit_tests/test_rlt_action_space.py \
  deps/rlinf/tests/unit_tests/test_rlt_correction_continuity.py \
  deps/rlinf/tests/unit_tests/test_piper_mux_executor.py \
  deps/rlinf/tests/unit_tests/test_piper_leader_takeover.py
