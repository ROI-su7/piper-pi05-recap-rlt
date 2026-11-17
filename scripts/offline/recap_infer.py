"""Strictly reload the production RECAP checkpoint and infer on real RGB/state."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from native_infer import observation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import hydra
    from omegaconf import OmegaConf
    from rlinf.recap_piper.policy import (
        PiperCFGPolicy,
        load_piper_cfg_model,
        validate_policy_provenance,
    )

    root = Path(__file__).resolve().parents[2]
    with hydra.initialize_config_dir(
        config_dir=str(root / "deps/rlinf/examples/offline_rl/config"), version_base="1.1"
    ):
        cfg = hydra.compose(config_name="piper_recap_cfg")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    validate_policy_provenance(args.checkpoint, cfg.actor.model)
    model = load_piper_cfg_model(cfg.actor.model)
    weights = torch.load(
        args.checkpoint / "model_state_dict/full_weights.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(weights, strict=True)
    policy = PiperCFGPolicy(model, task=os.environ["RECAP_TASK"])
    table = pq.read_table(next((args.dataset / "data").rglob("*.parquet"))).to_pydict()
    obs = observation(args.dataset, table, 0, policy.task)
    object.__setattr__(model.config, "cfgrl_guidance_scale", 1.0)
    torch.manual_seed(123)
    started = time.monotonic()
    plain = np.asarray(policy.infer(obs)["actions"])
    plain_seconds = time.monotonic() - started
    object.__setattr__(model.config, "cfgrl_guidance_scale", 1.5)
    torch.manual_seed(123)
    started = time.monotonic()
    guided = np.asarray(policy.infer(obs)["actions"])
    guided_seconds = time.monotonic() - started
    assert (
        plain.shape == guided.shape == (50, 14)
        and np.isfinite(plain).all()
        and np.isfinite(guided).all()
    )
    np.save(args.output.with_suffix(".npy"), guided)
    report = {
        "checkpoint": str(args.checkpoint),
        "strict_reload": True,
        "shape": [50, 14],
        "finite": True,
        "plain_seconds": plain_seconds,
        "cfg_seconds": guided_seconds,
        "cfg_scale": 1.5,
        "paired_noise_seed": 123,
        "mean_difference": float(np.abs(guided - plain).mean()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
