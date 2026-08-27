"""Contracts for the ten-model RPX NVS paper slate."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from nvs_models import (  # noqa: E402
    MODEL_DISPLAY_NAMES,
    MODEL_REGISTRY,
    MODEL_SPECS,
    PAPER_MODEL_KEYS,
    models_by_track,
)
from nvs_models.external_bridge import ExternalBridgeNVS  # noqa: E402


EXPECTED = {
    "dn_splatter", "splatam", "rtg_slam", "gaus_slam",
    "depthsplat", "mvsplat", "pixelsplat", "nopo_splat",
    "lightgaussian", "compgs",
}


def test_registry_contains_exact_ten_model_paper_slate() -> None:
    assert set(PAPER_MODEL_KEYS) == EXPECTED
    assert EXPECTED <= set(MODEL_REGISTRY)
    assert EXPECTED <= set(MODEL_DISPLAY_NAMES)


def test_slate_is_four_rgbd_four_rgb_two_compact() -> None:
    assert len(models_by_track("rgbd")) == 4
    assert len(models_by_track("rgb")) == 4
    assert len(models_by_track("compact")) == 2
    assert all(MODEL_SPECS[key].uses_sensor_depth for key in models_by_track("rgbd"))
    assert not any(MODEL_SPECS[key].uses_sensor_depth for key in models_by_track("rgb"))


def test_external_rgbd_bridge_requires_depth() -> None:
    adapter = ExternalBridgeNVS(MODEL_SPECS["splatam"], device="cpu")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    pose = np.eye(4)
    with pytest.raises(ValueError, match="depth map"):
        adapter([rgb], [], [pose], pose)


def test_external_bridge_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = tmp_path / "fake_bridge.py"
    bridge.write_text(
        """#!/usr/bin/env python3
import argparse
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
with np.load(a.input) as data:
    rgb = data['context_rgbs'][0]
    depth = data['context_depths'][0]
np.savez_compressed(a.output, rgb=rgb, depth=depth)
""",
        encoding="utf-8",
    )
    bridge.chmod(bridge.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("RPX_NVS_SPLATAM_BRIDGE", str(bridge))
    adapter = ExternalBridgeNVS(MODEL_SPECS["splatam"], device="cpu")
    rgb = np.full((8, 8, 3), 17, dtype=np.uint8)
    depth = np.full((8, 8), 1.5, dtype=np.float32)
    pose = np.eye(4)

    output = adapter([rgb], [depth], [pose], pose)

    np.testing.assert_array_equal(output["rgb"], rgb)
    np.testing.assert_allclose(output["depth"], depth)
