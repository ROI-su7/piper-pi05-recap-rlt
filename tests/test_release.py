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


def test_refresh_keeps_old_generated_tree(tmp_path, monkeypatch):
    import importlib.util
    import io
    import tarfile

    spec = importlib.util.spec_from_file_location(
        "release_bootstrap", ROOT / "scripts/bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    target = tmp_path / "deps/openpi"
    target.mkdir(parents=True)
    (target / ".piper-source.json").write_text('{"signature":"previous"}')
    (target / "old.txt").write_text("preserve me")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        item = tarfile.TarInfo("new.txt")
        item.size = 3
        archive.addfile(item, io.BytesIO(b"new"))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: payload.getvalue())
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: None)
    lock = {"dependencies": {"openpi": {"revision": "new", "url": "unused"}}, "operations": []}
    module.materialize("openpi", lock, local=tmp_path, refresh=True)
    assert (target / "new.txt").read_text() == "new"
    backup = next((tmp_path / "deps").glob("openpi.previous_*"))
    assert (backup / "old.txt").read_text() == "preserve me"


def test_refresh_refuses_unrecognized_tree(tmp_path, monkeypatch):
    import importlib.util

    import pytest

    spec = importlib.util.spec_from_file_location(
        "release_bootstrap_unknown", ROOT / "scripts/bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    target = tmp_path / "deps/openpi"
    target.mkdir(parents=True)
    (target / "user.txt").write_text("untouched")
    with pytest.raises(FileExistsError):
        module.materialize("openpi", {"dependencies": {}, "operations": []}, refresh=True)
    assert (target / "user.txt").read_text() == "untouched"
