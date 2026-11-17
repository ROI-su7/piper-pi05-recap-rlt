"""Portable source, publication and route-isolation contracts."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_published_sources_match_lock():
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    assert set(lock["dependencies"]) == {"openpi", "rlinf"}
    for op in lock["operations"]:
        assert hashlib.sha256((ROOT / op["source"]).read_bytes()).hexdigest() == op["sha256"]


def test_scoped_openpi_presets_only_use_three_cameras():
    tree = ast.parse((ROOT / "common/source/openpi/src/openpi/training/config.py").read_text())
    values = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_CONFIGS" for t in n.targets)
    )
    names = [next(k.value.value for k in n.keywords if k.arg == "name") for n in values.elts]
    assert set(names) == {"pi05_wire", "pi05_wire_all_data_full", "pi05_connect_strings"}
    assert len(values.elts) == 3


def test_algorithm_implementations_are_owned_by_distinct_directories():
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    sources = [op["source"] for op in lock["operations"]]
    assert any(s.startswith("recap/source/rlinf/rlinf/recap_piper/") for s in sources)
    assert any(s.startswith("rlt/source/rlinf/rlinf/algorithms/rlt/") for s in sources)
    assert (ROOT / "recap/configs/piper_recap_cfg.yaml").exists()
    assert (ROOT / "rlt/configs/realworld_piper_rlt_stage2_ac_mlp.yaml").exists()


def test_python_shell_and_ros_sources_parse():
    import subprocess
    import xml.etree.ElementTree as ET

    for folder in ["common", "recap", "rlt", "robot", "scripts"]:
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file():
                continue
            if path.suffix == ".py":
                ast.parse(path.read_text(), filename=str(path))
            elif path.suffix == ".sh":
                subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
            elif path.suffix in {".xml", ".launch"}:
                ET.parse(path)
