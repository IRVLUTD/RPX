from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "benchmark" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import render_tracking_predictions  # noqa: E402
import run_tracking  # noqa: E402
import run_tracking_gate  # noqa: E402


def test_gate_budgets_are_bounded_and_sequential() -> None:
    assert run_tracking_gate.GATE_FRAMES == {
        "smoke": 2,
        "micro": 8,
        "acceptance": 25,
    }
    assert set(run_tracking_gate.MODEL_PROVENANCE) == {"sam2", "edgetam"}


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


def test_renderer_uses_manifest_rgb_path(tmp_path: Path) -> None:
    snapshot = tmp_path / "datasets--IRVLUTD--RPX" / "snapshots" / "revision"
    manifest_path = snapshot / "manifests" / "object_tracking" / "easy.json"
    rgb_path = snapshot / "unusual" / "scene011" / "frame-00000.jpeg"
    rgb_path.parent.mkdir(parents=True)
    rgb_path.write_bytes(b"rgb")
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "scene_id": "scene011",
                        "phase": 0,
                        "rgb": "unusual/scene011/frame-00000.jpeg",
                    }
                ]
            }
        )
    )

    assert render_tracking_predictions._rgb_index(tmp_path) == {
        ("scene011", "0", "frame-00000"): rgb_path
    }


def test_edgetam_cumulative_overlay_inherits_sam2_digest() -> None:
    dockerfile = (ROOT / "docker/tracking-smoke/Dockerfile.edgetam-cumulative").read_text()
    builder = (ROOT / "docker/tracking-smoke/build_edgetam_rpx.sh").read_text()

    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "7711e012a30a2402c4eaab637bdb00a521302c91" in dockerfile
    assert "/opt/rpx-envs/edgetam/bin/python" in dockerfile
    assert "edgetam-pytorch-noncontiguous.patch" in dockerfile
    assert "sam2-rpx-sha-060b74b287a4" in builder
    assert "RepoDigests" in builder
    assert "edgetam-rpx-sha-${short_revision}" in builder
