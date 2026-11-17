"""Prove real optimizer updates, strict actor reload and production RLT prediction."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--observation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    dcp = args.output.with_name("rlt_optimizer_evidence.pt")
    dcp_to_torch_save(str(args.checkpoint / "dcp_checkpoint"), str(dcp))
    saved = torch.load(dcp, map_location="cpu", weights_only=False)["fsdp_checkpoint"]
    state = saved["optimizers"]["state"]
    actor_steps = [
        float(v["step"]) for k, v in state.items() if not k.startswith("q_head.") and "step" in v
    ]
    critic_steps = [
        float(v["step"]) for k, v in state.items() if k.startswith("q_head.") and "step" in v
    ]
    assert actor_steps and critic_steps and min(actor_steps) >= 1 and min(critic_steps) >= 1
    for v in state.values():
        for x in v.values():
            if torch.is_tensor(x):
                assert torch.isfinite(x).all()
    import hydra
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.mlp_policy import get_model

    with hydra.initialize_config_dir(
        config_dir=str(root / "deps/rlinf/examples/embodiment/config"), version_base="1.1"
    ):
        cfg = hydra.compose(config_name="piper_rlt_base")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    model = get_model(cfg.actor.model, torch_dtype=torch.float32)
    weights = torch.load(
        args.checkpoint / "model_state_dict/full_weights.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(weights, strict=True)
    model.to("cuda").eval()
    obs = {
        k: v[0].to(device="cuda", dtype=torch.float32)
        for k, v in torch.load(args.observation, map_location="cpu", weights_only=True).items()
    }
    with torch.inference_mode():
        action, result = model.predict_action_batch(obs, mode="eval")
    action = action.detach().float().cpu().numpy()
    assert action.shape == (1, 10, 14) and np.isfinite(action).all()
    np.save(args.output.with_suffix(".npy"), action)
    report = {
        "strict_reload": True,
        "checkpoint": str(args.checkpoint),
        "shape": list(action.shape),
        "finite": True,
        "actor_optimizer_step": max(actor_steps),
        "critic_optimizer_step": max(critic_steps),
        "optimizer_state_finite": True,
        "z_dim": 2048,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
