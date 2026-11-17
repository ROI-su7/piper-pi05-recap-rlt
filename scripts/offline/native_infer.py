"""Reload the actual newly trained JAX checkpoint on real recorded observations."""

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from runtime import install_recipe


def observation(dataset, table, index, task):
    images = {}
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        item = table["observation.images." + camera][index]
        pixel = np.asarray(
            Image.open(
                io.BytesIO(item["bytes"]) if item.get("bytes") else Path(dataset) / item["path"]
            ).convert("RGB")
        )
        images[camera] = pixel.transpose(2, 0, 1)
    return {
        "images": images,
        "state": np.asarray(table["observation.state"][index], np.float32),
        "prompt": task,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix-cache", type=Path)
    args = p.parse_args()
    recipe, cfg, trained = install_recipe(args.recipe)
    from openpi.policies import policy_config

    policy = policy_config.create_trained_policy(
        cfg, str(trained), default_prompt=recipe["task"], return_rlt_prefix=bool(args.prefix_cache)
    )
    dataset = Path(recipe["dataset"])
    table = pq.read_table(next((dataset / "data").rglob("*.parquet"))).to_pydict()
    obs = observation(dataset, table, 0, recipe["task"])
    started = time.monotonic()
    result = policy.infer(obs)
    seconds = time.monotonic() - started
    action = np.asarray(result["actions"])
    assert action.shape == (50, 14) and np.isfinite(action).all()
    np.save(args.output.with_suffix(".npy"), action)
    report = {
        "stage": "native_sft_reload",
        "checkpoint": str(trained),
        "shape": list(action.shape),
        "finite": True,
        "seconds": seconds,
        "real_model": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report)
    if args.prefix_cache:
        import torch

        features = []
        masks = []
        for i in range(2):
            item = policy.infer(observation(dataset, table, i, recipe["task"]))
            features.append(torch.from_numpy(np.asarray(item["rlt_prefix"]).copy()))
            masks.append(torch.from_numpy(np.asarray(item["rlt_prefix_mask"]).copy()))
        torch.save(
            {
                "prefix": torch.stack(features),
                "mask": torch.stack(masks),
                "source_jax_checkpoint": str(trained.resolve()),
                "config_name": cfg.name,
                "real_native_features": True,
            },
            args.prefix_cache,
        )
        print("PREFIX_CACHE", args.prefix_cache, flush=True)


if __name__ == "__main__":
    main()
