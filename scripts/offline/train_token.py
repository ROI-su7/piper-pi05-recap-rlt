"""Train the production RLT token on the freshly optimized native-JAX policy."""

import argparse
import importlib.util
import sys
from pathlib import Path

from runtime import install_recipe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    recipe, config, trained = install_recipe(args.recipe)
    path = (
        Path(__file__).resolve().parents[2]
        / "deps/rlinf/toolkits/realworld_check/train_piper_rlt_token_native_jax.py"
    )
    spec = importlib.util.spec_from_file_location("production_token_trainer", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    sys.argv = [
        str(path),
        "--jax-checkpoint",
        str(trained),
        "--dataset-dir",
        recipe["dataset"],
        "--output-dir",
        args.output,
        "--config-name",
        config.name,
        "--steps",
        "2",
        "--accumulate",
        "1",
        "--save-every",
        "2",
        "--cpu-offload",
        "--warmup",
        "0",
        "--prefix-cache",
        str(Path(args.recipe).parent.parent / "prefix_cache.pt"),
    ]
    raise SystemExit(m.main())


if __name__ == "__main__":
    main()
