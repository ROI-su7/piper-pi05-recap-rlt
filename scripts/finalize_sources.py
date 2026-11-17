"""Record the hash of every published patch/overlay after intentional edits."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "sources.lock.json"
lock = json.loads(p.read_text())
for operation in lock["operations"]:
    operation["sha256"] = hashlib.sha256((ROOT / operation["source"]).read_bytes()).hexdigest()
p.write_text(json.dumps(lock, indent=2) + "\n")
