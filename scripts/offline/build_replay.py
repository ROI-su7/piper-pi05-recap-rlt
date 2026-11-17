"""Recompute real native pi0.5/Token features and persist actual recorded-action replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from native_infer import observation
from runtime import install_recipe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--token", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    recipe, cfg, trained = install_recipe(args.recipe)
    from openpi.policies import policy_config
    from rlinf.data.embodied_io_struct import Trajectory
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer
    from rlinf.models.embodiment.native_jax_rlt.compact_server import CompactRLTPolicy

    original = policy_config.create_trained_policy(
        cfg,
        str(trained),
        default_prompt=recipe["task"],
        return_rlt_prefix=True,
        rlt_prefix_seq_len=768,
        rlt_token_checkpoint=str(args.token),
    )
    policy = CompactRLTPolicy(original)
    buffer = TrajectoryReplayBuffer(
        seed=42,
        enable_cache=True,
        cache_size=1000,
        sample_window_size=1000,
        auto_save=True,
        auto_save_path=str(args.output),
        sample_strategy="rlt_stratified",
        takeover_as_failure=True,
    )
    rows = []
    first_obs = None
    traces = []
    for episode, path in enumerate(sorted((args.dataset / "data").rglob("*.parquet"))):
        table = pq.read_table(path).to_pydict()
        n = len(table["action"])
        cache = {}

        def predict(index):
            if index not in cache:
                obs = observation(args.dataset, table, index, recipe["task"])
                result = policy.infer(obs)
                action = np.asarray(result["actions"], np.float32)
                z = np.asarray(result["rlt_z_rl"], np.float32)
                assert action.shape == (50, 14) and z.shape == (2048,) and np.isfinite(z).all()
                assert (
                    result["rlt_encoder_fingerprint"] == policy.metadata["rlt_encoder_fingerprint"]
                )
                cache[index] = {
                    "z_rl": torch.from_numpy(z.copy())[None, None],
                    "proprio": torch.tensor(table["observation.state"][index], dtype=torch.float32)[
                        None, None
                    ],
                    "ref_chunk": torch.from_numpy(action[:10].copy())[None, None],
                    # Offline replay has no live executor queue/bridge; explicitly zero it.
                    "bridge_delta": torch.zeros(1, 1, 14),
                }
            return cache[index]

        for start in range(0, n - 1, 10):
            end = min(start + 10, n - 1)
            count = end - start
            curr, nxt = predict(start), predict(end)
            if first_obs is None:
                first_obs = curr
            action = np.asarray(table["action"][start:end], np.float32)
            padded = np.repeat(action[-1:], 10, axis=0)
            padded[:count] = action
            human = (
                np.asarray(table.get("intervention", np.zeros(n))[start:end], bool)
                .reshape(count, -1)
                .any(-1)
            )
            flags = np.zeros(10, bool)
            flags[:count] = human
            valid = torch.arange(10)[None, None] < count
            rewards = torch.zeros(1, 1, 10)
            done = torch.zeros(1, 1, 10, dtype=torch.bool)
            terminal = end == n - 1
            if terminal:
                rewards[0, 0, count - 1] = float(
                    bool(np.asarray(table["is_success"][-1]).reshape(-1)[0])
                )
                done[0, 0, count - 1] = True
            row = Trajectory(
                max_episode_length=1,
                model_weights_id="offline-native-token",
                actions=torch.from_numpy(padded.reshape(1, 1, 140)),
                intervene_flags=torch.from_numpy(
                    np.repeat(flags[:, None], 14, axis=1).reshape(1, 1, 140)
                ),
                rewards=rewards,
                dones=done,
                terminations=done.clone(),
                truncations=torch.zeros_like(done),
                curr_obs=curr,
                next_obs=nxt,
                forward_inputs={
                    "rlt_episode_id": torch.tensor([[[episode]]]),
                    "action_valid_mask": valid,
                    "reference_warmup": torch.tensor([[[not flags.any()]]]),
                    "actor_switch": torch.zeros(1, 1, 1, dtype=torch.bool),
                },
            )
            rows.append(row)
            traces.append(
                {
                    "episode": str(path),
                    "start": start,
                    "valid_steps": count,
                    "human_steps": int(flags.sum()),
                    "terminal": terminal,
                }
            )
    buffer.add_trajectories(rows)
    buffer.save_checkpoint(str(args.output))
    torch.save(first_obs, args.output / "inference_observation.pt")
    report = {
        "transitions": len(rows),
        "source": "real recorded actions with recomputed native features",
        "encoder_fingerprint": policy.metadata["rlt_encoder_fingerprint"],
        "token": str(args.token),
        "native_checkpoint": str(trained),
        "z_dim": 2048,
        "bridge_delta": "zero; no live RTC executor in offline replay",
        "rows": traces,
        "metadata": policy.metadata,
    }
    (args.output / "offline_replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ["rows", "metadata"]}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
