from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_nvs_sos.py"
SPEC = importlib.util.spec_from_file_location("run_nvs_sos", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sliding_near_has_four_windows_and_three_targets_each() -> None:
    samples = MODULE._protocol_samples(range(500), "sliding-near")

    assert len(samples) == 12
    assert samples[0] == {
        "protocol": "sliding-near",
        "window_start": 0,
        "context_frames": [0, 49],
        "target_frame": 55,
    }
    assert samples[-1] == {
        "protocol": "sliding-near",
        "window_start": 300,
        "context_frames": [300, 349],
        "target_frame": 370,
    }


def test_diagnostic_protocols_use_same_local_windows() -> None:
    identity = MODULE._protocol_samples(range(500), "identity")
    interpolation = MODULE._protocol_samples(range(500), "interpolation")

    assert [sample["target_frame"] for sample in identity] == [49, 149, 249, 349]
    assert [sample["target_frame"] for sample in interpolation] == [24, 124, 224, 324]
    assert all(sample["context_frames"][0] <= sample["target_frame"] for sample in interpolation)
    assert all(sample["target_frame"] <= sample["context_frames"][1] for sample in interpolation)


def test_far_protocol_preserves_original_fractional_design() -> None:
    far = MODULE._protocol_samples(range(500), "far", max_targets=5)

    assert len(far) == 5
    assert all(sample["context_frames"] == [0, 199] for sample in far)
    assert [sample["target_frame"] for sample in far] == [300, 349, 399, 449, 499]


def test_missing_frames_drop_only_affected_windows_and_targets() -> None:
    frames = [frame for frame in range(500) if frame not in {149, 260}]
    samples = MODULE._protocol_samples(frames, "sliding-near")

    assert len(samples) == 8
    assert not any(sample["window_start"] == 100 for sample in samples)
    assert not any(sample["target_frame"] == 260 for sample in samples)


def test_target_baseline_is_relative_to_nearest_context() -> None:
    first = np.eye(4)
    second = np.eye(4)
    target = np.eye(4)
    second[0, 3] = 0.10
    target[0, 3] = 0.13

    baseline = MODULE._target_baseline([0, 49], [first, second], 55, target)

    assert baseline["nearest_context_frame"] == 49
    assert baseline["frame_gap"] == 6
    assert np.isclose(baseline["translation_m"], 0.03)
    assert np.isclose(baseline["rotation_deg"], 0.0)
