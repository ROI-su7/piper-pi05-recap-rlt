"""Reconstruct a validation model recipe consistently across native pipeline stages."""

import dataclasses
import json
from pathlib import Path


def install_recipe(path):
    from openpi.models import pi0_config
    from openpi.training import config

    recipe = json.loads(Path(path).read_text())
    base = config.get_config(recipe["base_config"])
    fields = {f.name for f in dataclasses.fields(pi0_config.Pi0Config)}
    kwargs = {k: v for k, v in recipe["model"].items() if k in fields}
    if "image_keys" in kwargs:
        kwargs["image_keys"] = tuple(kwargs["image_keys"])
    model = pi0_config.Pi0Config(**kwargs)
    trained = Path(path).parent / recipe["name"] / "smoke" / str(recipe["steps"] - 1)
    data = dataclasses.replace(
        base.data,
        repo_id=recipe["dataset"],
        assets=dataclasses.replace(base.data.assets, assets_dir=str(trained / "assets")),
    )
    custom = dataclasses.replace(base, name=recipe["name"], model=model, data=data)
    config._CONFIGS_DICT[recipe["name"]] = custom
    return recipe, custom, trained
