"""Tests for ``rpx_benchmark.dataset_hub.lossless_convert``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rpx_benchmark.dataset_hub.lossless_convert import (
    CONVERTED_SUFFIX,
    CONVERTIBLE_MODALITIES,
    ConvertSpec,
    _classify,
    _plan_tree,
    convert_capture_tree,
)
from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.exceptions import DatasetError


# --------------------------------------------------------------------------- #
# Pillow / numpy are required by the converter itself; skip the suite cleanly
# if they're absent (e.g. on a minimal CI image).
# --------------------------------------------------------------------------- #

_pillow = pytest.importorskip("PIL.Image", reason="Pillow not installed")
_np = pytest.importorskip("numpy", reason="numpy not installed")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_tree(tmp_path: Path) -> Path:
    """A tiny synthetic capture tree (mos + sos with all modalities)."""
    return generate_mock(
        tmp_path / "src",
        MockSpec(
            multi_object_scenes=2,
            single_object_scenes=2,
            phases_per_multi=2,
            frames_per_phase=3,
            image_size=16,
        ),
    )


# --------------------------------------------------------------------------- #
# _classify — pure unit test (no I/O)
# --------------------------------------------------------------------------- #


def test_classify_rgb_png_is_converted(tmp_path: Path):
    root = tmp_path
    p = root / "mos" / "scene1" / "0" / "rgb" / "00000.png"
    assert _classify(p, root) == "convert"


def test_classify_fisheye_png_is_converted(tmp_path: Path):
    root = tmp_path
    p = root / "mos" / "scene1" / "0" / "fisheye" / "00000.png"
    assert _classify(p, root) == "convert"


def test_classify_depth_png_is_not_converted(tmp_path: Path):
    root = tmp_path
    p = root / "mos" / "scene1" / "0" / "depth" / "00000.png"
    assert _classify(p, root) == "link"


def test_classify_masks_png_is_not_converted(tmp_path: Path):
    root = tmp_path
    p = root / "mos" / "scene1" / "0" / "sam2" / "masks" / "00000.png"
    assert _classify(p, root) == "link"


def test_classify_non_png_files_are_linked(tmp_path: Path):
    root = tmp_path
    for fname in ("cam_pose/00000.json", "sam2/mask_to_object.json", "rgb/notes.txt"):
        p = root / "mos" / "scene1" / "0" / fname
        assert _classify(p, root) == "link", fname


def test_convertible_modalities_set_is_rgb_and_fisheye():
    assert CONVERTIBLE_MODALITIES == frozenset({"rgb", "fisheye"})


# --------------------------------------------------------------------------- #
# End-to-end conversion
# --------------------------------------------------------------------------- #


def _all(root: Path, suffix: str) -> list[Path]:
    return sorted(p for p in root.rglob(f"*{suffix}") if p.is_file())


def test_end_to_end_converts_rgb_and_fisheye_only(mock_tree: Path, tmp_path: Path):
    out = tmp_path / "out"
    res = convert_capture_tree(
        ConvertSpec(src_root=mock_tree, out_root=out, workers=1)
    )

    # Every modality the mock generated is present under out/, with the
    # expected extension change applied selectively.
    assert _all(out / "mos", "/rgb/*.webp" if False else ".webp")  # silence linter

    # rgb/ now contains only .webp, no .png
    for phase_dir in out.rglob("rgb"):
        if not phase_dir.is_dir():
            continue
        assert _all(phase_dir, ".png") == [], f"stray PNG in {phase_dir}"
        assert _all(phase_dir, ".webp"), f"no WebP in {phase_dir}"

    # fisheye/ now contains only .webp, no .png
    for phase_dir in out.rglob("fisheye"):
        if not phase_dir.is_dir():
            continue
        assert _all(phase_dir, ".png") == [], f"stray PNG in {phase_dir}"
        assert _all(phase_dir, ".webp"), f"no WebP in {phase_dir}"

    # depth/ stays PNG
    for phase_dir in out.rglob("depth"):
        if not phase_dir.is_dir():
            continue
        assert _all(phase_dir, ".png"), f"depth lost: {phase_dir}"
        assert _all(phase_dir, ".webp") == []

    # sam2/masks/ stays PNG
    for masks_dir in out.rglob("sam2/masks"):
        if not masks_dir.is_dir():
            continue
        assert _all(masks_dir, ".png"), f"masks lost: {masks_dir}"
        assert _all(masks_dir, ".webp") == []

    # Counts roughly: converted >= linked because every phase has multiple
    # frames in rgb+fisheye and a smaller number of label files. The exact
    # split depends on the mock, but both must be positive.
    assert res.files_converted > 0
    assert res.files_linked > 0


def test_round_trip_pixel_equality(mock_tree: Path, tmp_path: Path):
    """For every converted file, decoded WebP pixels == decoded source PNG pixels."""
    import numpy as np
    from PIL import Image

    out = tmp_path / "out"
    convert_capture_tree(ConvertSpec(src_root=mock_tree, out_root=out, workers=1))

    pairs = 0
    for webp_path in out.rglob("*.webp"):
        rel = webp_path.relative_to(out)
        src_path = mock_tree / rel.with_suffix(".png")
        assert src_path.is_file(), f"missing source for {rel}"
        with Image.open(src_path) as a:
            a.load()
            src_arr = np.asarray(a)
            mode = a.mode
        with Image.open(webp_path) as b:
            b.load()
            if b.mode != mode:
                b = b.convert(mode)
            webp_arr = np.asarray(b)
        assert np.array_equal(src_arr, webp_arr), (
            f"round-trip mismatch at {rel}: shapes {src_arr.shape} vs {webp_arr.shape}"
        )
        pairs += 1
    assert pairs > 0, "no .webp files emitted — the mock had no rgb/fisheye?"


def test_non_converted_files_are_hardlinked(mock_tree: Path, tmp_path: Path):
    """Linked files share an inode with the source — zero extra disk."""
    out = tmp_path / "out"
    convert_capture_tree(ConvertSpec(src_root=mock_tree, out_root=out, workers=1))

    # Pick any depth PNG in the output tree and verify inode parity.
    checked = 0
    for dst in out.rglob("depth/*.png"):
        rel = dst.relative_to(out)
        src = mock_tree / rel
        if not src.is_file():
            continue
        assert dst.stat().st_ino == src.stat().st_ino, f"not hardlinked: {rel}"
        checked += 1
    assert checked > 0, "no depth PNGs in mock — check fixture"


# --------------------------------------------------------------------------- #
# Safety rails
# --------------------------------------------------------------------------- #


def test_refuses_when_src_root_missing(tmp_path: Path):
    with pytest.raises(DatasetError, match="src_root does not exist"):
        convert_capture_tree(
            ConvertSpec(src_root=tmp_path / "nope", out_root=tmp_path / "out")
        )


def test_refuses_when_out_root_equals_src_root(mock_tree: Path):
    with pytest.raises(DatasetError, match="must differ"):
        convert_capture_tree(ConvertSpec(src_root=mock_tree, out_root=mock_tree))


def test_refuses_when_out_root_is_inside_src(mock_tree: Path):
    with pytest.raises(DatasetError, match="overlap"):
        convert_capture_tree(
            ConvertSpec(src_root=mock_tree, out_root=mock_tree / "child")
        )


def test_refuses_non_empty_out_without_overwrite(mock_tree: Path, tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "preexisting.txt").write_text("hi")
    with pytest.raises(DatasetError, match="non-empty"):
        convert_capture_tree(ConvertSpec(src_root=mock_tree, out_root=out))


def test_overwrite_out_allows_non_empty(mock_tree: Path, tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "preexisting.txt").write_text("hi")
    convert_capture_tree(
        ConvertSpec(
            src_root=mock_tree,
            out_root=out,
            overwrite_out=True,
            workers=1,
        )
    )
    assert (out / "preexisting.txt").is_file()  # kept untouched
    assert list(out.rglob("*.webp"))  # something got converted


# --------------------------------------------------------------------------- #
# Dry-run
# --------------------------------------------------------------------------- #


def test_dry_run_does_not_write_anything(mock_tree: Path, tmp_path: Path):
    out = tmp_path / "out"
    res = convert_capture_tree(
        ConvertSpec(src_root=mock_tree, out_root=out, dry_run=True)
    )
    assert res.files_converted > 0
    assert res.files_linked > 0
    # out/ may exist as an empty dir but should contain no files
    assert list(out.rglob("*.webp")) == []
    assert list(out.rglob("*.png")) == []
    assert list(out.rglob("*.json")) == []


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


def test_plan_tree_classifies_every_file(mock_tree: Path):
    plan = _plan_tree(mock_tree)
    assert plan, "mock tree was empty"
    actions = {a for _, a in plan}
    assert actions <= {"convert", "link"}
    # Mock has rgb + fisheye + depth + masks → both actions appear.
    assert "convert" in actions
    assert "link" in actions
