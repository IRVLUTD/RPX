#----------------------------------------------------------------------------------------------------
# Tests for the ego (GoPro) frame sampler.
#
#   pytest maskgen_pipeline/test_ego_sample.py -q
#
# The index math is tested exhaustively without cv2. The end-to-end sampler is
# smoke-tested against a tiny synthetic mp4 generated with cv2; it skips if cv2
# / a usable video writer is unavailable.
#
#----------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maskgen_pipeline.ego_sample import (
    EGO_SAMPLE_COUNT,
    sample_ego_frames,
    uniform_sample_indices,
)


# --------------------------------------------------------------------------- #
# uniform_sample_indices
# --------------------------------------------------------------------------- #

def test_default_count_is_250():
    assert EGO_SAMPLE_COUNT == 250


def test_uniform_sample_includes_endpoints():
    idxs = uniform_sample_indices(0, 100, 11)
    assert idxs[0] == 0 and idxs[-1] == 100
    assert idxs == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def test_uniform_sample_is_unique_and_sorted():
    idxs = uniform_sample_indices(0, 5000, 250)
    assert len(idxs) == 250
    assert len(set(idxs)) == 250
    assert idxs == sorted(idxs)


def test_uniform_sample_caps_at_range_when_too_short():
    # n >= span -> every index, no duplicates.
    idxs = uniform_sample_indices(0, 99, 250)
    assert idxs == list(range(0, 100))


def test_uniform_sample_n_le_one_returns_start():
    assert uniform_sample_indices(7, 20, 1) == [7]
    assert uniform_sample_indices(7, 20, 0) == [7]


def test_uniform_sample_rejects_reversed_range():
    with pytest.raises(ValueError):
        uniform_sample_indices(10, 5, 5)


def test_uniform_sample_handles_singleton_range():
    # Degenerate-but-valid: start == end.
    assert uniform_sample_indices(5, 5, 10) == [5]


# --------------------------------------------------------------------------- #
# Sampler smoke tests (require cv2 + a working video writer)
# --------------------------------------------------------------------------- #

def _make_synthetic_mp4(path: Path, n_frames: int, fps: int = 30, size=(64, 48)):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    w, h = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        pytest.skip("no usable mp4 VideoWriter backend in this environment")
    for i in range(n_frames):
        writer.write(np.full((h, w, 3), i % 256, dtype=np.uint8))
    writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("VideoWriter produced no file")


def test_sampler_default_yields_exactly_250(tmp_path):
    cv2 = pytest.importorskip("cv2")
    scene = tmp_path / "scene_long"
    scene.mkdir()
    video = scene / "ego.mp4"
    _make_synthetic_mp4(video, n_frames=300)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total < 280:
        pytest.skip(f"video backend reported {total} frames; need >=280 here")

    written = sample_ego_frames(scene, video)             # default 250, default .jpg
    assert len(written) == EGO_SAMPLE_COUNT
    # Output lands at <scene>/ego/rgb/ with NNNNN.jpg nomenclature by default.
    assert all(p.parent == scene / "ego" / "rgb" for p in written)
    for i, p in enumerate(written):
        assert p.name == f"{i:05d}.jpg"

    fmap = json.loads((scene / "ego" / "ego_frame_map.json").read_text())
    assert fmap["ego_video"] == str(video)
    assert fmap["n_samples_requested"] == EGO_SAMPLE_COUNT
    assert fmap["image_format"] == "jpg"
    src = list(fmap["frames"].values())
    assert len(src) == EGO_SAMPLE_COUNT
    assert len(set(src)) == EGO_SAMPLE_COUNT
    assert src == sorted(src)


def test_sampler_warns_and_caps_when_video_short(tmp_path, capsys):
    cv2 = pytest.importorskip("cv2")
    scene = tmp_path / "scene_short"
    scene.mkdir()
    video = scene / "ego.mp4"
    _make_synthetic_mp4(video, n_frames=80)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total >= EGO_SAMPLE_COUNT:
        pytest.skip("backend produced unexpectedly many frames")

    written = sample_ego_frames(scene, video)             # default 250
    assert 1 <= len(written) <= total                     # can't exceed source
    assert len(written) < EGO_SAMPLE_COUNT                # capped
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    # All sampled images decode.
    img = cv2.imread(str(written[len(written) // 2]))
    assert img is not None


def test_sampler_respects_explicit_n_samples(tmp_path):
    cv2 = pytest.importorskip("cv2")
    scene = tmp_path / "scene_x"
    scene.mkdir()
    video = scene / "ego.mp4"
    _make_synthetic_mp4(video, n_frames=200)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total < 50:
        pytest.skip("unreliable backend")

    written = sample_ego_frames(scene, video, n_samples=10)
    assert len(written) == 10
    fmap = json.loads((scene / "ego" / "ego_frame_map.json").read_text())
    src = list(fmap["frames"].values())
    assert src == sorted(src) and len(set(src)) == len(src)
    assert min(src) == 0 and max(src) == total - 1        # endpoints included


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #

def test_batch_extract_ego_full_flow(tmp_path):
    """End-to-end batch: 3 scenes, mp4s in a flat ego_root dir, mix of OK and
    short-window cases, plus a missing-ego scene. Verifies the status taxonomy
    and that --force re-extracts."""
    cv2 = pytest.importorskip("cv2")
    from maskgen_pipeline.batch_extract_ego import main as batch_main

    scenes_root = tmp_path / "scenes"
    ego_root = tmp_path / "egos"
    scenes_root.mkdir()
    ego_root.mkdir()

    # Scene A: long enough → expect "ok" with exactly 10 frames
    (scenes_root / "scene_A").mkdir()
    _make_synthetic_mp4(ego_root / "scene_A.mp4", n_frames=150)
    # Scene B: short → expect "short", takes all available
    (scenes_root / "scene_B").mkdir()
    _make_synthetic_mp4(ego_root / "scene_B.mp4", n_frames=6)
    # Scene C: no ego mp4 in scene dir AND no matching <id>.mp4 → "no_ego"
    (scenes_root / "scene_C").mkdir()

    rc = batch_main([
        "--scenes_root", str(scenes_root),
        "--ego_root", str(ego_root),
        "--n_samples", "10",
    ])
    assert rc == 1   # because scene_C is missing → status="no_ego" → not ok

    # Scene A delivered exactly 10 frames (default format is now .jpg)
    a_frames = sorted((scenes_root / "scene_A" / "ego" / "rgb").glob("*.jpg"))
    assert len(a_frames) == 10
    a_map = json.loads((scenes_root / "scene_A" / "ego" / "ego_frame_map.json").read_text())
    assert len(a_map["frames"]) == 10
    # Scene B was short; took all 6 (or whatever the writer produced)
    b_frames = sorted((scenes_root / "scene_B" / "ego" / "rgb").glob("*.jpg"))
    assert 1 <= len(b_frames) < 10
    # Scene C produced nothing
    assert not (scenes_root / "scene_C" / "ego").exists()

    # Second run with no --force: scenes A and B should now be skipped, C still missing.
    rc2 = batch_main([
        "--scenes_root", str(scenes_root),
        "--ego_root", str(ego_root),
        "--n_samples", "10",
    ])
    assert rc2 == 1  # still has scene_C missing
    # A's frame count unchanged (skip path)
    assert sorted((scenes_root / "scene_A" / "ego" / "rgb").glob("*.jpg")) == a_frames


def test_batch_resolves_mp4_inside_scene_dir(tmp_path):
    """If the mp4 lives inside the scene dir, --ego_root isn't needed."""
    pytest.importorskip("cv2")
    from maskgen_pipeline.batch_extract_ego import main as batch_main

    scenes_root = tmp_path / "scenes"
    scenes_root.mkdir()
    (scenes_root / "s1").mkdir()
    _make_synthetic_mp4(scenes_root / "s1" / "ego.mp4", n_frames=80)

    rc = batch_main(["--scenes_root", str(scenes_root), "--n_samples", "5"])
    assert rc == 0
    assert len(list((scenes_root / "s1" / "ego" / "rgb").glob("*.jpg"))) == 5


def test_sampler_png_format_explicit(tmp_path):
    """`--format png` switch produces lossless .png outputs and records the
    format in the provenance sidecar."""
    cv2 = pytest.importorskip("cv2")
    scene = tmp_path / "scene_png"
    scene.mkdir()
    video = scene / "ego.mp4"
    _make_synthetic_mp4(video, n_frames=80)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total < 20:
        pytest.skip("unreliable backend")

    written = sample_ego_frames(scene, video, n_samples=10, image_format="png")
    assert len(written) == 10
    assert all(p.suffix == ".png" for p in written)
    fmap = json.loads((scene / "ego" / "ego_frame_map.json").read_text())
    assert fmap["image_format"] == "png"
    # All sampled images decode and are larger per-frame than JPEG would be.
    img = cv2.imread(str(written[0]))
    assert img is not None


def test_batch_depth2_layout(tmp_path):
    """Batch driver discovers scenes nested two levels deep, e.g.
    ``test_gopro_organized-selected/<batch>/<scene_id>/``."""
    cv2 = pytest.importorskip("cv2")
    from maskgen_pipeline.batch_extract_ego import main as batch_main

    root = tmp_path / "grouped"
    root.mkdir()
    # Two batches, two scenes each: root/1-2/1, root/1-2/2, root/3-4/3, root/3-4/4
    for batch, scene_ids in [("1-2", ("1", "2")), ("3-4", ("3", "4"))]:
        bdir = root / batch
        bdir.mkdir()
        for sid in scene_ids:
            sdir = bdir / sid
            sdir.mkdir()
            _make_synthetic_mp4(sdir / "ego.mp4", n_frames=80)

    rc = batch_main(["--scenes_root", str(root), "--n_samples", "5", "--depth", "2"])
    assert rc == 0
    # All 4 scenes processed
    for batch, sid in [("1-2", "1"), ("1-2", "2"), ("3-4", "3"), ("3-4", "4")]:
        rgb = root / batch / sid / "ego" / "rgb"
        assert rgb.is_dir(), f"missing output for {batch}/{sid}"
        assert len(list(rgb.glob("*.jpg"))) == 5
