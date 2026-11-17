"""Replay real recorded frames through the project's actual episode/HDF5 converters."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from video import VideoReader

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def sample(path, frames, kind):
    root = Path(path)
    info = json.loads((root / "meta/info.json").read_text())
    rows = []
    groups = {"success": 0, "failure": 0, "human": 0}
    for parquet in sorted((root / "data").rglob("episode_*.parquet")):
        if kind == "demo" and len(rows) >= 2:
            break
        table = pq.read_table(parquet).to_pydict()
        n = len(table["action"])
        if n < frames:
            continue
        ok = True if kind == "demo" else bool(np.asarray(table["is_success"][-1]).reshape(-1)[0])
        human = np.asarray(table.get("intervention", np.zeros(n)), bool).reshape(n, -1).any(-1)
        category = (
            "human"
            if kind != "demo" and ok and human[-frames:].any()
            else "success"
            if ok
            else "failure"
        )
        if kind == "demo":
            category = "success"
        limit = 2 if category != "human" else 1
        if groups[category] >= limit:
            continue
        if kind == "demo" and len(rows) >= 2:
            break
        episode = int(table["episode_index"][0])
        start = n - frames
        images = {}
        for camera in CAMERAS:
            key = "observation.images." + camera
            if info["features"][key]["dtype"] == "video":
                template = info.get(
                    "video_path",
                    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                )
                video = root / template.format(
                    episode_chunk=episode // int(info.get("chunks_size", 1000)),
                    video_key=key,
                    episode_index=episode,
                )
                reader = VideoReader(str(video))
                reader.skip(start)
                images[camera] = np.stack([reader.read() for _ in range(frames)])
                reader.close()
            else:
                decoded = []
                for item in table[key][start:]:
                    import io

                    source = io.BytesIO(item["bytes"]) if item.get("bytes") else root / item["path"]
                    decoded.append(np.asarray(Image.open(source).convert("RGB")))
                images[camera] = np.stack(decoded)

        def column(key, fallback):
            return np.asarray(table[key][start:] if key in table else fallback, np.float32).reshape(
                frames, 14
            )

        state = column("observation.state", None)
        action = column("action", None)
        rows.append(
            {
                "kind": kind,
                "source": str(parquet),
                "original_frames": n,
                "original_start": start,
                "source_episode": episode,
                "fps": int(info["fps"]),
                "success": ok,
                "images": images,
                "state": state,
                "action": action,
                "policy_action": column("policy_action", action),
                "leader_action": column("leader_action", action),
                "human": human[start:],
                "timestamp": np.asarray(table["timestamp"][start:], np.float64),
            }
        )
        groups[category] += 1
        if kind != "demo" and all(
            groups[c] >= x for c, x in [("success", 2), ("failure", 2), ("human", 1)]
        ):
            break
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", required=True)
    p.add_argument("--rollout", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=64)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from rlinf.recap_piper.data import EpisodeWriter, export_lerobot
    from rlinf.recap_piper.source import native_config

    task, _ = native_config(
        ROOT / "deps/openpi/src/openpi/training/config.py", "pi05_connect_strings"
    )
    samples = sample(args.demo, args.frames, "demo") + sample(args.rollout, args.frames, "rollout")
    raw = out / "hdf5"
    raw.mkdir()
    provenance = []
    for i, row in enumerate(samples):
        if row["kind"] == "demo":
            with h5py.File(raw / f"episode_{i:06d}.hdf5", "w") as h:
                h.attrs["sim"] = False
                h.attrs["compress"] = False
                h.attrs["color_order"] = "rgb"
                h.create_dataset("observations/qpos", data=row["state"])
                h.create_dataset("action", data=row["action"])
                for c in CAMERAS:
                    h.create_dataset("observations/images/" + c, data=row["images"][c])
        else:
            writer = EpisodeWriter(out / "raw_rollout", task, row["fps"])
            for t in range(args.frames):
                writer.add_frame(
                    {
                        "state": row["state"][t],
                        "action": row["action"][t],
                        "policy_action": row["policy_action"][t],
                        "leader_action": row["leader_action"][t],
                        "timestamp": float(row["timestamp"][t]),
                        "intervention": bool(row["human"][t]),
                        "policy_valid": True,
                        "leader_valid": True,
                        "images": {c: row["images"][c][t] for c in CAMERAS},
                    }
                )
            writer.finish(row["success"])
        provenance.append(
            {
                k: row[k]
                for k in [
                    "kind",
                    "source",
                    "original_frames",
                    "original_start",
                    "source_episode",
                    "fps",
                    "success",
                ]
            }
            | {"frames": args.frames, "intervention_frames": int(row["human"].sum())}
        )
    # Run the production converter's API with lossless PNG for exact RGB checks.
    file = ROOT / "robot/cobot_magic/collect_data/hdf5_lerobot.py"
    spec = importlib.util.spec_from_file_location("cobot_converter", file)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    config = dataclasses.replace(
        m.DEFAULT_DATASET_CONFIG, use_videos=False, image_writer_processes=0, image_writer_threads=4
    )
    ds = m.create_empty_dataset(
        repo_id="local/offline_demos",
        root=out / "demos",
        robot_type="piper",
        fps=30,
        cameras=list(CAMERAS),
        include_velocity=False,
        include_effort=False,
        dataset_config=config,
    )
    ds = m.populate_dataset(
        ds,
        sorted(raw.glob("*.hdf5")),
        task,
        list(CAMERAS),
        include_velocity=False,
        include_effort=False,
        include_intervention=False,
        camera_renames={},
    )
    export_lerobot(out / "raw_rollout", out / "rollout", "local/offline_rollout", use_videos=False)
    export_lerobot(
        out / "raw_rollout",
        out / "interventions",
        "local/offline_interventions",
        interventions_only=True,
        min_segment_frames=1,
        use_videos=False,
    )
    # Validate actual converted pixels/commands, frame by frame, against source recordings.
    for kind in ["demo", "rollout"]:
        selected = [r for r in samples if r["kind"] == kind]
        folder = out / ("demos" if kind == "demo" else "rollout")
        paths = sorted((folder / "data").rglob("episode_*.parquet"))
        assert len(paths) == len(selected)
        # Rollout source UUID sorting differs; match physical action arrays rather than filenames.
        remaining = list(selected)
        for path in paths:
            table = pq.read_table(path).to_pydict()
            actions = np.asarray(table["action"])
            match = next(r for r in remaining if np.allclose(actions, r["action"], atol=1e-6))
            np.testing.assert_allclose(
                np.asarray(table["observation.state"]), match["state"], atol=1e-6
            )
            for c in CAMERAS:
                for t, item in enumerate(table["observation.images." + c]):
                    import io

                    image = Image.open(
                        io.BytesIO(item["bytes"]) if item.get("bytes") else folder / item["path"]
                    ).convert("RGB")
                    np.testing.assert_array_equal(np.asarray(image), match["images"][c][t])
            remaining = [r for r in remaining if r is not match]
    report = {
        "hardware_collection": False,
        "real_recording_replay": True,
        "frames_per_clip": args.frames,
        "episodes": provenance,
        "roundtrip": "state/action/RGB exact or float tolerance1e-6",
        "task": task,
        "output": str(out),
    }
    (out / "data_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
