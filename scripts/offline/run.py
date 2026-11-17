"""Run every real-model offline stage on recorded data, using an owned local Ray head."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def ray_head(env, port, log_path):
    probe = socket.socket()
    probe.bind(("127.0.0.1", port))
    probe.close()
    binary = str(Path(sys.executable).parent / "ray")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                binary,
                "start",
                "--head",
                f"--port={port}",
                "--node-ip-address=127.0.0.1",
                "--num-cpus=6",
                "--num-gpus=1",
                "--include-dashboard=false",
                "--object-store-memory=268435456",
                f"--temp-dir=/tmp/pvo-{port}",
                "--disable-usage-stats",
                "--block",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(60):
                ready = subprocess.run(
                    [binary, "status", f"--address=127.0.0.1:{port}"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                if ready.returncode == 0:
                    break
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text()[-4000:])
                time.sleep(1)
            else:
                raise RuntimeError("Validation Ray head did not become ready")
            yield
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["demo", "rollout", "checkpoint", "siglip", "gemma", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--ray-port", type=int, default=6386)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    if args.frames < 51:
        raise ValueError("Need at least51 observations for a complete50-action chunk")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Real-model offline validation requires CUDA; no fake fallback")
    paths = {
        k: str(getattr(args, k).resolve())
        for k in ["demo", "rollout", "checkpoint", "siglip", "gemma"]
    }
    plan = out / "validation_plan.json"
    if plan.exists() and json.loads(plan.read_text()) != paths:
        raise ValueError("Cannot resume with different real datasets or model assets")
    plan.write_text(json.dumps(paths, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        PIPER_PYTHON=sys.executable,
        PIPER_OFFLINE_VALIDATION="1",
        PYTHONPATH=f"{ROOT}/common/offline_guard:{ROOT}/deps/rlinf:{ROOT}/deps/openpi/src:{ROOT}/deps/openpi/packages/openpi-client/src:{ROOT}/deps/openpi",
        RLINF_DIR=str(ROOT / "deps/rlinf"),
        HF_HUB_OFFLINE="1",
        WANDB_MODE="disabled",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        RECAP_SFT_DATASET=str(out / "data/demos"),
        RECAP_DATASET=str(out / "data/rollout"),
        RECAP_INT_DATASET=str(out / "data/interventions"),
        RECAP_RUN_DIR=str(out / "recap"),
        RECAP_SIGLIP_DIR=paths["siglip"],
        RECAP_GEMMA_DIR=paths["gemma"],
        RECAP_PI05_TORCH_DIR=str(out / "pi05_torch"),
        RAY_DEDUP_LOGS="0",
    )
    events = []

    def run(name, command, artifact):
        artifact = Path(artifact)
        if artifact.is_file():
            if not args.resume:
                raise FileExistsError(artifact)
            events.append({"stage": name, "resumed": True, "artifact": str(artifact)})
            return
        print("Running", name, flush=True)
        started = time.monotonic()
        with (out / (name + ".log")).open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode or not artifact.is_file():
            raise RuntimeError(f"{name} failed; inspect {out / (name + '.log')}")
        events.append(
            {"stage": name, "seconds": time.monotonic() - started, "artifact": str(artifact)}
        )
        (out / "stage_events.json").write_text(json.dumps(events, indent=2) + "\n")

    py = sys.executable
    run(
        "prepare",
        [
            py,
            "scripts/offline/prepare.py",
            "--demo",
            paths["demo"],
            "--rollout",
            paths["rollout"],
            "--output",
            str(out / "data"),
            "--frames",
            str(args.frames),
        ],
        out / "data/data_report.json",
    )
    run(
        "native_sft",
        [
            py,
            "scripts/offline/native_sft.py",
            "--checkpoint",
            paths["checkpoint"],
            "--dataset",
            str(out / "data/demos"),
            "--output",
            str(out / "native_sft"),
        ],
        out / "native_sft/pi05_offline_expert_lora/smoke/1/params/_METADATA",
    )
    recipe = out / "native_sft/recipe.json"
    trained = out / "native_sft/pi05_offline_expert_lora/smoke/1"
    env["RECAP_TASK"] = json.loads(recipe.read_text())["task"]
    env.update(
        PIPER_JAX_CHECKPOINT=str(trained),
        PIPER_NORM_STATS=str(trained / "assets/all_wire_data/norm_stats.json"),
        RECAP_SOURCE_CONFIG_FILE=str(out / "native_sft/source_config.py"),
        RECAP_SOURCE_CONFIG="pi05_offline_expert_lora",
    )
    run(
        "native_infer",
        [
            py,
            "scripts/offline/native_infer.py",
            "--recipe",
            str(recipe),
            "--output",
            str(out / "native_infer.json"),
            "--prefix-cache",
            str(out / "prefix_cache.pt"),
        ],
        out / "prefix_cache.pt",
    )
    token = out / "rlt_token/step_2/rlt_token.pt"
    run(
        "token",
        [
            py,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            "1",
            "scripts/offline/train_token.py",
            "--recipe",
            str(recipe),
            "--output",
            str(out / "rlt_token"),
        ],
        token,
    )
    run(
        "returns",
        ["bash", "recap/run.sh", "returns"],
        out / "data/rollout/meta/returns_piper_recap_v1.parquet",
    )
    run("convert", ["bash", "recap/run.sh", "convert"], out / "pi05_torch/config.json")
    value = out / "recap/value/piper_recap_value/checkpoints/global_step_2/actor"
    cfg = out / "recap/cfg/piper_recap_cfg/checkpoints/global_step_2/actor"
    env.update(
        RECAP_VALUE_CHECKPOINT=str(value),
        RECAP_RAY_ADDRESS=f"127.0.0.1:{args.ray_port}",
        RAY_ADDRESS=f"127.0.0.1:{args.ray_port}",
    )
    common = [
        "runner.max_steps=2",
        "runner.max_epochs=2",
        "runner.save_interval=2",
        "actor.global_batch_size=1",
        "actor.micro_batch_size=1",
        "actor.optim.lr_warmup_steps=0",
        "actor.optim.total_training_steps=2",
    ]
    if not all(
        path.is_file()
        for path in [
            value / "model_state_dict/full_weights.pt",
            cfg / "model_state_dict/full_weights.pt",
        ]
    ):
        with ray_head(env, args.ray_port, out / "ray_head.log"):
            run(
                "value",
                ["bash", "recap/run.sh", "value", *common, "data.train_num_workers=0"],
                value / "model_state_dict/full_weights.pt",
            )
            run(
                "advantages",
                ["bash", "recap/run.sh", "advantages", "advantage.batch_size=1"],
                out / "data/demos/meta/advantages_piper_recap_adv_v1.parquet",
            )
            run(
                "recap_cfg",
                ["bash", "recap/run.sh", "cfg", *common],
                cfg / "model_state_dict/full_weights.pt",
            )
    source_env = 'source deps/rlinf/examples/recap_piper/env.sh\nexec "$PIPER_PYTHON" scripts/offline/recap_infer.py --checkpoint "$1" --dataset "$2" --output "$3"'
    run(
        "recap_infer",
        [
            "bash",
            "-c",
            source_env,
            "offline",
            str(cfg),
            str(out / "data/rollout"),
            str(out / "recap_infer.json"),
        ],
        out / "recap_infer.json",
    )
    replay = out / "rlt_replay"
    run(
        "build_replay",
        [
            py,
            "scripts/offline/build_replay.py",
            "--recipe",
            str(recipe),
            "--dataset",
            str(out / "data/rollout"),
            "--token",
            str(token),
            "--output",
            str(replay),
        ],
        replay / "offline_replay.json",
    )
    actor = out / "rlt_stage2/piper_wire_rlt_stage2/checkpoints/global_step_2/actor"
    env.update(
        PIPER_RLT_CONFIG="piper_rlt_base",
        RLINF_STAGE1_RLT_TOKEN=str(token),
        RLINF_REPLAY_LOAD_PATH=str(replay),
        PIPER_REPLAY_STEPS="2",
        PIPER_RLT_RAY_ADDRESS=f"127.0.0.1:{args.ray_port + 1}",
        RAY_ADDRESS=f"127.0.0.1:{args.ray_port + 1}",
        RLINF_JAX_REF_HOST="127.0.0.1",
        RLINF_JAX_REF_PORT="6006",
    )
    if not (actor / "model_state_dict/full_weights.pt").is_file():
        with ray_head(env, args.ray_port + 1, out / "rlt_ray.log"):
            run(
                "rlt_stage2",
                [
                    "bash",
                    "rlt/run.sh",
                    "replay",
                    f"runner.logger.log_path={out}/rlt_stage2",
                    "runner.save_interval=2",
                    "actor.global_batch_size=4",
                    "actor.micro_batch_size=2",
                    "algorithm.update_epoch=1",
                    "algorithm.rlt_schedule.enable=false",
                    "algorithm.replay_buffer.min_buffer_size=1",
                    "algorithm.train_actor_steps=1",
                ],
                actor / "model_state_dict/full_weights.pt",
            )
    run(
        "rlt_infer",
        [
            py,
            "scripts/offline/rlt_infer.py",
            "--checkpoint",
            str(actor),
            "--observation",
            str(replay / "inference_observation.pt"),
            "--output",
            str(out / "rlt_infer.json"),
        ],
        out / "rlt_infer.json",
    )
    reports = {
        name: json.loads((out / (name + ".json")).read_text())
        for name in ["native_infer", "recap_infer", "rlt_infer"]
    }
    assert all(report["finite"] for report in reports.values())
    assert reports["recap_infer"]["strict_reload"] and reports["rlt_infer"]["strict_reload"]
    assert (
        reports["rlt_infer"]["actor_optimizer_step"] >= 1
        and reports["rlt_infer"]["critic_optimizer_step"] >= 1
    )
    summary = {
        "complete": True,
        "profile": "real-recording small-step end-to-end",
        "robot_started": False,
        "steps": {"native_sft": 2, "value": 2, "recap_cfg": 2, "rlt_token": 2},
        "reports": reports,
        "assets": paths,
        "data": json.loads((out / "data/data_report.json").read_text()),
        "events": events,
    }
    (out / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    subprocess.run(
        [py, "scripts/offline/summarize.py", "--output", str(out)], cwd=ROOT, env=env, check=True
    )
    print(out / "validation.json", flush=True)


if __name__ == "__main__":
    main()
