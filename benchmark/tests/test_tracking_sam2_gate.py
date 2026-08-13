from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "benchmark" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_tracking  # noqa: E402
import run_tracking_gate  # noqa: E402


def test_gate_budgets_are_bounded_and_sequential() -> None:
    assert run_tracking_gate.GATE_FRAMES == {
        "smoke": 2,
        "micro": 8,
        "acceptance": 25,
    }


def test_prediction_resume_is_invalidated_by_adapter_revision(tmp_path: Path) -> None:
    sample = {"rgb": "00000.png"}
    clip = run_tracking.Clip(
        scene="scene_000",
        phase=0,
        split="easy",
        root=tmp_path,
        samples=(sample,),
    )
    prediction_path = run_tracking._prediction_path(tmp_path, clip, sample)
    run_tracking._atomic_mask(
        prediction_path, np.zeros(run_tracking.EXPECTED_SHAPE, dtype=np.int32)
    )
    marker_path = prediction_path.parent / "_complete.json"
    marker_path.write_text(json.dumps({"model": "sam2", "frames": 1, "rpx_git_sha": "a" * 40}))

    assert (
        run_tracking._clip_predictions(clip, clip.samples, tmp_path, "sam2", "a" * 40) is not None
    )
    assert run_tracking._clip_predictions(clip, clip.samples, tmp_path, "sam2", "b" * 40) is None


def test_sam2_overlay_is_pinned_and_uses_sam2_python() -> None:
    dockerfile = (ROOT / "docker/tracking-smoke/Dockerfile.sam2-rpx").read_text()
    builder = (ROOT / "docker/tracking-smoke/build_sam2_rpx.sh").read_text()

    assert "@sha256:b3e0d935b689" in dockerfile
    assert "/opt/rpx-envs/sam2/bin/python -m pip install" in dockerfile
    assert "rpx-stamp-adapter-env" in dockerfile
    assert "sam2-rpx-sha-${short_revision}" in builder
    assert 'if [[ "${base_image}" != *@sha256:* ]]' in builder
