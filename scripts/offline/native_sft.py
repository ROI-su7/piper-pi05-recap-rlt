"""Run real native-JAX pi0.5 expert-LoRA optimization on the replayed dataset."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    from flax import nnx
    from openpi.models import pi0_config
    from openpi.shared import nnx_utils
    from openpi.training import config, optimizer, weight_loaders

    original = config.get_config("pi05_connect_strings")
    model = pi0_config.Pi0Config(
        pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    )
    dataset = dataclasses.replace(
        original.data,
        repo_id=str(args.dataset.resolve()),
        assets=dataclasses.replace(
            original.data.assets, assets_dir=str(args.checkpoint.resolve() / "assets")
        ),
    )
    cfg = dataclasses.replace(
        original,
        model=model,
        data=dataset,
        name="pi05_offline_expert_lora",
        exp_name="smoke",
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(args.checkpoint.resolve() / "params")
        ),
        freeze_filter=nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*lora.*")),
        checkpoint_base_dir=str(args.output.resolve()),
        batch_size=1,
        num_workers=0,
        num_train_steps=args.steps,
        log_interval=1,
        save_interval=args.steps,
        keep_period=None,
        ema_decay=None,
        wandb_enabled=False,
        fsdp_devices=1,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=0, peak_lr=1e-5, decay_steps=max(2, args.steps), decay_lr=1e-6
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = {
        "name": cfg.name,
        "base_config": "pi05_connect_strings",
        "model": dataclasses.asdict(model),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "task": dataset.default_prompt,
        "steps": args.steps,
        "trainable": "action expert LoRA only",
    }
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    source = (
        "TrainConfig(name="
        + repr(cfg.name)
        + ', model=Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"), data=DataConfig(default_prompt='
        + repr(dataset.default_prompt)
        + "))\n"
    )
    (args.output / "source_config.py").write_text(source)
    spec = importlib.util.spec_from_file_location(
        "native_openpi_train", root / "deps/openpi/scripts/train.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    trainer.main(cfg)
    print("SFT_CHECKPOINT", cfg.checkpoint_dir / str(args.steps - 1), flush=True)


if __name__ == "__main__":
    main()
