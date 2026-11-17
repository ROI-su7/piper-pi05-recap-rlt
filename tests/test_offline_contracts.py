"""Regression checks for the concrete failures found in the real-model offline run."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_converter_lossless_images_use_image_metadata(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("lerobot")
    import dataclasses

    converter = load_file(
        "published_cobot_converter", ROOT / "robot/cobot_magic/collect_data/hdf5_lerobot.py"
    )
    cfg = dataclasses.replace(
        converter.DEFAULT_DATASET_CONFIG,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    dataset = converter.create_empty_dataset(
        repo_id="local/regression",
        root=tmp_path / "dataset",
        robot_type="piper",
        fps=30,
        cameras=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        include_velocity=False,
        include_effort=False,
        dataset_config=cfg,
    )
    assert not dataset.meta.video_keys
    assert all(
        dataset.meta.features["observation.images." + c]["dtype"] == "image"
        for c in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    )


def test_mixed_sft_rollout_share_piper_camera_schema():
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    source = (
        ROOT
        / "deps/rlinf/examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py"
    )
    module = load_file("published_advantages", source)
    sample = {
        "observation.state": np.zeros(14, np.float32),
        "task": "connect the lights",
        "observation.images.cam_high": np.zeros((3, 16, 16), np.float32),
        "observation.images.cam_left_wrist": np.full((3, 16, 16), 0.25, np.float32),
        "observation.images.cam_right_wrist": np.full((3, 16, 16), 0.75, np.float32),
    }
    for kind in ["sft", "rollout"]:
        robot = module.resolve_dataset_robot_type({"robot_type": "piper"}, {"type": kind})
        obs = module.build_obs(sample, robot, {})
        np.testing.assert_array_equal(
            obs["observation/extra_view_image-0"], sample["observation.images.cam_left_wrist"]
        )
        np.testing.assert_array_equal(
            obs["observation/extra_view_image-1"], sample["observation.images.cam_right_wrist"]
        )
    assert (
        module.resolve_dataset_robot_type({"robot_type": "piper"}, {"robot_type": "droid"})
        == "droid"
    )


def test_offline_process_forbids_can(monkeypatch):
    import os
    import subprocess

    env = os.environ.copy()
    env["PIPER_OFFLINE_VALIDATION"] = "1"
    env["PYTHONPATH"] = str(ROOT / "common/offline_guard")
    code = """import socket, ssl
assert socket.socket.__module__ == 'sitecustomize'
try:
    socket.socket(getattr(socket, 'AF_CAN', 29), socket.SOCK_RAW)
except RuntimeError as error:
    assert 'SocketCAN' in str(error)
else:
    raise AssertionError('CAN guard failed')
"""
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)
