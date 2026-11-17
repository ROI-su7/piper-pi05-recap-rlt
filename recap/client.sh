#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"
exec bash "$PIPER_RLINF_DIR/examples/recap_piper/run_client.sh" "$@"
