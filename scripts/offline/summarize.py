"""Collect measured stage evidence, without copying frames or weights into the repository."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    report = json.loads((out / "validation.json").read_text())
    human = 0
    human_positive = 0
    labels = 0
    positive = 0
    for name in ["demos", "rollout"]:
        root = out / "data" / name
        side = pq.read_table(root / "meta/advantages_piper_recap_adv_v1.parquet").to_pydict()
        lookup = {
            (int(e), int(f)): bool(a)
            for e, f, a in zip(
                side["episode_index"], side["frame_index"], side["advantage"], strict=True
            )
        }
        count = 0
        for path in (root / "data").rglob("*.parquet"):
            rows = pq.read_table(path).to_pydict()
            for i, (episode, frame) in enumerate(
                zip(rows["episode_index"], rows["frame_index"], strict=True)
            ):
                flag = lookup[(int(episode), int(frame))]
                count += 1
                if name == "demos":
                    assert flag
                operator = bool(
                    np.asarray(rows.get("intervention", np.zeros(len(rows["action"])))[i])
                    .reshape(-1)
                    .any()
                )
                human += int(operator)
                human_positive += int(operator and flag)
        assert count == len(lookup)
        labels += count
        positive += sum(lookup.values())
    assert human and human_positive == human
    token = torch.load(
        out / "rlt_token/step_2/rlt_token.pt", map_location="cpu", weights_only=True, mmap=True
    )
    assert (
        token["step"] == 2
        and token["format_version"] == 2
        and token["image_only"]
        and token["prefix_seq_len"] == 768
    )
    losses = [
        json.loads(line)["rlt_loss"]
        for line in (out / "token.log").read_text().splitlines()
        if line.startswith('{"step":')
    ]
    assert len(losses) == 2 and np.isfinite(losses).all()
    replay = json.loads((out / "rlt_replay/offline_replay.json").read_text())
    assert replay["encoder_fingerprint"] and replay["transitions"] == len(replay["rows"])
    report.update(
        labels={
            "total": labels,
            "positive": positive,
            "negative": labels - positive,
            "human_frames": human,
            "human_positive": human_positive,
        },
        rlt={
            "token_parameters": sum(t.numel() for t in token["rlt_module"].values()),
            "token_losses": losses,
            "token_steps": token["step"],
            "transitions": replay["transitions"],
            "valid_recorded_actions": sum(row["valid_steps"] for row in replay["rows"]),
            "human_recorded_actions": sum(row["human_steps"] for row in replay["rows"]),
            "encoder_fingerprint": replay["encoder_fingerprint"],
            "actor_updates": report["reports"]["rlt_infer"]["actor_optimizer_step"],
            "critic_updates": report["reports"]["rlt_infer"]["critic_optimizer_step"],
        },
    )
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["labels", "rlt"]}, indent=2))


if __name__ == "__main__":
    main()
