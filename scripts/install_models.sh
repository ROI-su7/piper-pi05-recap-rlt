#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$ROOT/.venv}"
if [[ -e "$VENV" ]]; then echo "Refusing to overwrite existing environment: $VENV" >&2; exit 2; fi
python3.11 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$ROOT/requirements/models.txt"
"$PY" -m pip install 'git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
"$PY" -m pip install --no-deps -e "$ROOT/deps/openpi/packages/openpi-client" -e "$ROOT/deps/openpi" -e "$ROOT/deps/rlinf"
# OpenPI pi0.5 requires its exact Transformers implementation patches in this new venv.
"$PY" - "$ROOT" <<'PY'
import importlib.util,shutil
from pathlib import Path
import sys
root=Path(sys.argv[1]);target=Path(importlib.util.find_spec('transformers').origin).parent
source=root/'deps/openpi/src/openpi/models_pytorch/transformers_replace'
shutil.copytree(source,target,dirs_exist_ok=True)
PY
"$PY" -m pip freeze > "$VENV/installed-packages.txt"
echo "Activate: source $VENV/bin/activate"
