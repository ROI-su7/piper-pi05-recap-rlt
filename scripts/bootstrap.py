"""Reconstruct pinned dependencies and apply only the published Piper changes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(name, lock, local=None):
    target = ROOT / "deps" / name
    signature = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    marker = target / ".piper-source.json"
    if target.exists():
        if marker.is_file() and json.loads(marker.read_text())["signature"] == signature:
            return target
        raise FileExistsError(
            f"{target} already exists with different/uncommitted provenance; use a fresh destination"
        )
    target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=target.parent) as folder:
        folder = Path(folder)
        spec = lock["dependencies"][name]
        if local:
            archive = subprocess.check_output(
                ["git", "-C", str(Path(local).resolve()), "archive", spec["revision"]]
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(folder, filter="data")
            subprocess.run(["git", "init", "-q", str(folder)], check=True)
        else:
            subprocess.run(["git", "clone", "--no-checkout", spec["url"], str(folder)], check=True)
            subprocess.run(
                ["git", "-C", str(folder), "checkout", "--detach", spec["revision"]], check=True
            )
        for operation in lock["operations"]:
            if operation["dependency"] != name:
                continue
            source = ROOT / operation["source"]
            if digest(source) != operation["sha256"]:
                raise ValueError(f"Published source changed: {source}")
            if operation["kind"] == "patch":
                subprocess.run(
                    ["git", "-C", str(folder), "apply", "--check", str(source)], check=True
                )
                subprocess.run(["git", "-C", str(folder), "apply", str(source)], check=True)
            else:
                dest = folder / operation["file"]
                if not dest.resolve().is_relative_to(folder.resolve()):
                    raise ValueError("Unsafe source path")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        (folder / ".piper-source.json").write_text(
            json.dumps({"signature": signature, "revision": spec["revision"]}, indent=2) + "\n"
        )
        # Rename the fully prepared tree into place, including reproducibility metadata.
        folder.rename(target)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--local-openpi", help="Read-only local git object source for offline reconstruction"
    )
    p.add_argument(
        "--local-rlinf", help="Read-only local git object source for offline reconstruction"
    )
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    for operation in lock["operations"]:
        if digest(ROOT / operation["source"]) != operation["sha256"]:
            raise ValueError(f"Source checksum mismatch: {operation['source']}")
    if args.verify_only:
        print(f"Verified {len(lock['operations'])} pinned Piper changes")
        return
    for name, local in [("openpi", args.local_openpi), ("rlinf", args.local_rlinf)]:
        print(materialize(name, lock, local))


if __name__ == "__main__":
    main()
