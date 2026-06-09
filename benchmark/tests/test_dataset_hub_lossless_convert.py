"""Tests for ``rpx_benchmark.dataset_hub.lossless_convert``.

Coverage map
------------

* ``_classify`` truth table for every action × modality combination.
* End-to-end conversion against a synthetic mock tree:
    - rgb / fisheye / ego → .webp (8-bit lossless)
    - depth / sam2/masks  → .png at compress_level=9 (still PNG)
    - cam_pose            → per-frame .npy (Option B)
    - everything else     → hard-linked into the output tree
* Per-frame round-trip equality for every re-encoded artefact:
    - PIL.Image.open(webp)  array equals PIL.Image.open(png) array
    - PIL.Image.open(new_png) array equals PIL.Image.open(old_png) array
      AND mode is preserved (palette / I;16 / L stay themselves)
    - np.load(npy) returns (7,) float64 packing [position; orientation]
      whose first 3 / last 4 elements equal the source .npz values.
* Hard-link inode parity for non-converted files.
* Safety rails: refuses overlapping src/out, refuses non-empty out
  without --overwrite-out, dry-run writes nothing.
* skip_* opt-outs route the relevant files through ACTION_LINK instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rpx_benchmark.dataset_hub.lossless_convert import (
    ACTION_CAM_POSE,
    ACTION_LINK,
    ACTION_PNG_RECOMPRESS,
    ACTION_WEBP,
    CAM_POSE_PARENT_DIR,
    PNG_RECOMPRESS_PARENT_DIRS,
    WEBP_PARENT_DIRS,
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


@pytest.fixture
def cam_pose_tree(tmp_path: Path) -> Path:
    """A capture tree with realistic per-frame .npz files under cam_pose/.

    The mock generator's cam_pose layer is JSON, not .npz, so we plant a
    minimal scene by hand to exercise the cam_pose code path.
    """
    import numpy as np

    root = tmp_path / "src"
    cp_dir = root / "mos" / "scene1" / "0" / "cam_pose"
    rgb_dir = root / "mos" / "scene1" / "0" / "rgb"
    cp_dir.mkdir(parents=True)
    rgb_dir.mkdir(parents=True)
    # 3 frames is enough; we don't need many to prove the round-trip.
    for i in range(3):
        pos = np.array([0.1 * i, 0.2 * i, 0.3 * i], dtype=np.float64)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64) + 1e-9 * i
        np.savez(cp_dir / f"{i:05d}.npz", position=pos, orientation=quat)
        # A token RGB so the classifier still finds at least one webp file.
        from PIL import Image

        Image.new("RGB", (8, 8), color=(i, i, i)).save(rgb_dir / f"{i:05d}.png")
    return root


# --------------------------------------------------------------------------- #
# _classify — pure unit tests (no I/O)
# --------------------------------------------------------------------------- #


def test_classify_rgb_png_is_webp(tmp_path: Path):
    p = tmp_path / "mos" / "scene1" / "0" / "rgb" / "00000.png"
    assert _classify(p, tmp_path) == ACTION_WEBP


def test_classify_fisheye_png_is_webp(tmp_path: Path):
    p = tmp_path / "mos" / "scene1" / "0" / "fisheye" / "left" / "00000.png"
    assert _classify(p, tmp_path) == ACTION_WEBP


def test_classify_ego_rgb_png_is_webp(tmp_path: Path):
    """ego/rgb/* lives outside mos/ but still has rgb as a parent dir."""
    p = tmp_path / "scene1" / "ego" / "rgb" / "00000.png"
    assert _classify(p, tmp_path) == ACTION_WEBP


def test_classify_depth_png_is_png_recompress(tmp_path: Path):
    p = tmp_path / "mos" / "scene1" / "0" / "depth" / "00000.png"
    assert _classify(p, tmp_path) == ACTION_PNG_RECOMPRESS


def test_classify_sam2_masks_png_is_png_recompress(tmp_path: Path):
    p = tmp_path / "mos" / "scene1" / "0" / "sam2" / "masks" / "00000.png"
    assert _classify(p, tmp_path) == ACTION_PNG_RECOMPRESS


def test_classify_sam2_masks_verified_png_is_png_recompress(tmp_path: Path):
    p = tmp_path / "mos" / "scene1" / "0" / "sam2" / "masks_verified" / "00000.png"
    assert _classify(p, tmp_path) == ACTION_PNG_RECOMPRESS


def test_classify_sam2_bbox_overlay_png_is_link(tmp_path: Path):
    """Non-mask sam2 PNGs are pure visualization — left alone."""
    p = tmp_path / "mos" / "scene1" / "0" / "sam2" / "bbox_overlay" / "00000.png"
    assert _classify(p, tmp_path) == ACTION_LINK


def test_classify_cam_pose_npz_is_cam_pose(tmp_path: Path):
    p = tmp_path / "mos" / "scene1" / "0" / "cam_pose" / "00000.npz"
    assert _classify(p, tmp_path) == ACTION_CAM_POSE


def test_classify_non_image_files_are_linked(tmp_path: Path):
    for rel in (
        "cam_pose/00000.json",
        "sam2/mask_to_object.json",
        "rgb/notes.txt",
        "sam2/verified_masks.txt",
    ):
        p = tmp_path / "mos" / "scene1" / "0" / rel
        assert _classify(p, tmp_path) == ACTION_LINK, rel


def test_skip_toggles_route_to_link(tmp_path: Path):
    """Each skip_* toggle routes the affected modality to ACTION_LINK."""
    rgb = tmp_path / "mos" / "s" / "0" / "rgb" / "00000.png"
    depth = tmp_path / "mos" / "s" / "0" / "depth" / "00000.png"
    pose = tmp_path / "mos" / "s" / "0" / "cam_pose" / "00000.npz"

    base = ConvertSpec(src_root=tmp_path, out_root=tmp_path / "out")
    assert _classify(rgb, tmp_path, spec=base) == ACTION_WEBP
    assert _classify(depth, tmp_path, spec=base) == ACTION_PNG_RECOMPRESS
    assert _classify(pose, tmp_path, spec=base) == ACTION_CAM_POSE

    rgb_off = ConvertSpec(src_root=tmp_path, out_root=tmp_path / "out", skip_rgb_webp=True)
    png_off = ConvertSpec(src_root=tmp_path, out_root=tmp_path / "out", skip_png_recompress=True)
    pose_off = ConvertSpec(src_root=tmp_path, out_root=tmp_path / "out", skip_cam_pose=True)
    assert _classify(rgb, tmp_path, spec=rgb_off) == ACTION_LINK
    assert _classify(depth, tmp_path, spec=png_off) == ACTION_LINK
    assert _classify(pose, tmp_path, spec=pose_off) == ACTION_LINK


def test_constants_have_expected_names():
    assert WEBP_PARENT_DIRS == frozenset({"rgb", "fisheye"})
    assert PNG_RECOMPRESS_PARENT_DIRS == frozenset({"depth", "masks", "masks_verified"})
    assert CAM_POSE_PARENT_DIR == "cam_pose"


# --------------------------------------------------------------------------- #
# End-to-end on the mock tree (rgb + fisheye + depth + masks paths)
# --------------------------------------------------------------------------- #


def _all(root: Path, suffix: str) -> list[Path]:
    return sorted(p for p in root.rglob(f"*{suffix}") if p.is_file())


def test_end_to_end_converts_each_modality_correctly(mock_tree: Path, tmp_path: Path):
    out = tmp_path / "out"
    res = convert_capture_tree(ConvertSpec(src_root=mock_tree, out_root=out, workers=1))

    # rgb/ now contains only .webp
    for phase_dir in out.rglob("rgb"):
        if not phase_dir.is_dir():
            continue
        assert _all(phase_dir, ".png") == [], f"stray PNG in {phase_dir}"
        assert _all(phase_dir, ".webp"), f"no WebP in {phase_dir}"

    # fisheye/ now contains only .webp
    for phase_dir in out.rglob("fisheye"):
        if not phase_dir.is_dir():
            continue
        assert _all(phase_dir, ".png") == [], f"stray PNG in {phase_dir}"
        assert _all(phase_dir, ".webp"), f"no WebP in {phase_dir}"

    # depth/ stays PNG (re-encoded at level 9, same extension)
    for phase_dir in out.rglob("depth"):
        if not phase_dir.is_dir():
            continue
        assert _all(phase_dir, ".png"), f"depth lost: {phase_dir}"
        assert _all(phase_dir, ".webp") == []

    # sam2/masks/ stays PNG (re-encoded at level 9)
    for masks_dir in out.rglob("sam2/masks"):
        if not masks_dir.is_dir():
            continue
        assert _all(masks_dir, ".png"), f"masks lost: {masks_dir}"
        assert _all(masks_dir, ".webp") == []

    # Action accounting non-zero for the right paths
    assert res.by_action[ACTION_WEBP].files > 0
    assert res.by_action[ACTION_PNG_RECOMPRESS].files > 0
    assert res.by_action[ACTION_LINK].files > 0


def test_webp_round_trip_pixel_equality(mock_tree: Path, tmp_path: Path):
    import numpy as np
    from PIL import Image

    out = tmp_path / "out"
    convert_capture_tree(ConvertSpec(src_root=mock_tree, out_root=out, workers=1))

    pairs = 0
    for webp_path in out.rglob("*.webp"):
        rel = webp_path.relative_to(out)
        src_path = mock_tree / rel.with_suffix(".png")
        with Image.open(src_path) as a:
            a.load()
            src_arr = np.asarray(a)
            mode = a.mode
        with Image.open(webp_path) as b:
            b.load()
            if b.mode != mode:
                b = b.convert(mode)
            webp_arr = np.asarray(b)
        assert np.array_equal(src_arr, webp_arr), f"WebP mismatch at {rel}"
        pairs += 1
    assert pairs > 0


def test_png_recompress_preserves_mode_and_pixels(mock_tree: Path, tmp_path: Path):
    """depth (I;16) and masks (palette) must survive byte-identical
    AND keep their PIL mode."""
    import numpy as np
    from PIL import Image

    out = tmp_path / "out"
    convert_capture_tree(ConvertSpec(src_root=mock_tree, out_root=out, workers=1))

    checked = 0
    for kind in ("depth", "sam2/masks"):
        for new in out.rglob(f"{kind}/*.png"):
            old = mock_tree / new.relative_to(out)
            if not old.is_file():
                continue
            with Image.open(old) as a:
                a.load()
                src_arr = np.asarray(a)
                src_mode = a.mode
            with Image.open(new) as b:
                b.load()
                new_arr = np.asarray(b)
                new_mode = b.mode
            assert new_mode == src_mode, f"mode drifted at {new}: {src_mode}→{new_mode}"
            assert np.array_equal(src_arr, new_arr), f"PNG re-encode mismatch at {new}"
            checked += 1
    assert checked > 0


def test_non_converted_files_are_hardlinked(mock_tree: Path, tmp_path: Path):
    """Linked files share an inode with the source — zero extra disk."""
    out = tmp_path / "out"
    convert_capture_tree(ConvertSpec(src_root=mock_tree, out_root=out, workers=1))

    checked = 0
    for dst in out.rglob("*.json"):
        rel = dst.relative_to(out)
        src = mock_tree / rel
        if not src.is_file():
            continue
        assert dst.stat().st_ino == src.stat().st_ino, f"not hardlinked: {rel}"
        checked += 1
    assert checked > 0


# --------------------------------------------------------------------------- #
# cam_pose path
# --------------------------------------------------------------------------- #


def test_cam_pose_round_trip_and_layout(cam_pose_tree: Path, tmp_path: Path):
    """Per-frame .npz → per-frame .npy. The .npy is (7,) float64 packing
    [position (3,); orientation (4,)]. Both slices equal the source."""
    import numpy as np

    out = tmp_path / "out"
    res = convert_capture_tree(ConvertSpec(src_root=cam_pose_tree, out_root=out, workers=1))

    # The cam_pose action produced 3 .npy files, all in the right place.
    assert res.by_action[ACTION_CAM_POSE].files == 3
    npy_files = sorted((out / "mos" / "scene1" / "0" / "cam_pose").glob("*.npy"))
    npz_files = sorted((out / "mos" / "scene1" / "0" / "cam_pose").glob("*.npz"))
    assert len(npy_files) == 3
    assert npz_files == [], "stray .npz left in output tree"

    # Per-frame round-trip vs the source .npz arrays.
    for npy_path in npy_files:
        src_path = cam_pose_tree / npy_path.relative_to(out).with_suffix(".npz")
        src = np.load(src_path)
        new = np.load(npy_path)
        assert new.shape == (7,)
        assert new.dtype == np.float64
        assert np.array_equal(new[:3], src["position"])
        assert np.array_equal(new[3:], src["orientation"])


def test_cam_pose_skip_toggle_keeps_npz(cam_pose_tree: Path, tmp_path: Path):
    out = tmp_path / "out"
    res = convert_capture_tree(
        ConvertSpec(src_root=cam_pose_tree, out_root=out, workers=1, skip_cam_pose=True)
    )
    assert ACTION_CAM_POSE not in res.by_action or res.by_action[ACTION_CAM_POSE].files == 0
    # The .npz files are now hard-linked through verbatim
    npz_out = sorted((out / "mos" / "scene1" / "0" / "cam_pose").glob("*.npz"))
    assert len(npz_out) == 3


# --------------------------------------------------------------------------- #
# Safety rails
# --------------------------------------------------------------------------- #


def test_refuses_when_src_root_missing(tmp_path: Path):
    with pytest.raises(DatasetError, match="src_root does not exist"):
        convert_capture_tree(ConvertSpec(src_root=tmp_path / "nope", out_root=tmp_path / "out"))


def test_refuses_when_out_root_equals_src_root(mock_tree: Path):
    with pytest.raises(DatasetError, match="must differ"):
        convert_capture_tree(ConvertSpec(src_root=mock_tree, out_root=mock_tree))


def test_refuses_when_out_root_is_inside_src(mock_tree: Path):
    with pytest.raises(DatasetError, match="overlap"):
        convert_capture_tree(ConvertSpec(src_root=mock_tree, out_root=mock_tree / "child"))


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
    assert (out / "preexisting.txt").is_file()
    assert list(out.rglob("*.webp"))


# --------------------------------------------------------------------------- #
# Dry-run
# --------------------------------------------------------------------------- #


def test_dry_run_does_not_write_anything(mock_tree: Path, tmp_path: Path):
    out = tmp_path / "out"
    res = convert_capture_tree(ConvertSpec(src_root=mock_tree, out_root=out, dry_run=True))
    assert res.files_converted > 0
    assert res.files_linked > 0
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
    # webp + png-recompress + link should all appear for a mock with
    # rgb + depth + masks + ancillary files.
    assert ACTION_WEBP in actions
    assert ACTION_PNG_RECOMPRESS in actions
    assert ACTION_LINK in actions


# --------------------------------------------------------------------------- #
# Worker error paths — every encoder must abort loudly, never silently
# --------------------------------------------------------------------------- #


def test_malformed_cam_pose_missing_keys_aborts(tmp_path: Path):
    """A cam_pose .npz that lacks 'position' or 'orientation' must abort
    the run with the offending filename, not silently produce a broken
    .npy."""
    import numpy as np

    src = tmp_path / "src"
    cp = src / "mos" / "scene" / "0" / "cam_pose"
    cp.mkdir(parents=True)
    np.savez(cp / "00000.npz", wrong_key=np.zeros(3))

    with pytest.raises(DatasetError, match="missing required keys"):
        convert_capture_tree(ConvertSpec(src_root=src, out_root=tmp_path / "out", workers=1))


def test_malformed_cam_pose_wrong_shape_aborts(tmp_path: Path):
    """A cam_pose .npz whose 'position' isn't (3,) or 'orientation'
    isn't (4,) must abort the run, not silently truncate."""
    import numpy as np

    src = tmp_path / "src"
    cp = src / "mos" / "scene" / "0" / "cam_pose"
    cp.mkdir(parents=True)
    np.savez(
        cp / "00000.npz",
        position=np.zeros(5, dtype=np.float64),  # wrong: should be (3,)
        orientation=np.zeros(4, dtype=np.float64),
    )

    with pytest.raises(DatasetError, match="unexpected shapes"):
        convert_capture_tree(ConvertSpec(src_root=src, out_root=tmp_path / "out", workers=1))


def test_corrupt_webp_source_aborts(tmp_path: Path):
    """A corrupt source PNG in an rgb/ dir aborts the WebP pass with the
    file name surfaced — no silent skip, no partial output kept."""
    src = tmp_path / "src"
    rgb = src / "mos" / "scene" / "0" / "rgb"
    rgb.mkdir(parents=True)
    (rgb / "00000.png").write_bytes(b"not actually a png")

    with pytest.raises(DatasetError, match="aborted"):
        convert_capture_tree(ConvertSpec(src_root=src, out_root=tmp_path / "out", workers=1))


def test_corrupt_png_recompress_source_aborts(tmp_path: Path):
    """Same loudness contract for the PNG-recompress path."""
    src = tmp_path / "src"
    d = src / "mos" / "scene" / "0" / "depth"
    d.mkdir(parents=True)
    (d / "00000.png").write_bytes(b"not actually a png")

    with pytest.raises(DatasetError, match="aborted"):
        convert_capture_tree(ConvertSpec(src_root=src, out_root=tmp_path / "out", workers=1))


# --------------------------------------------------------------------------- #
# ConvertResult / ActionStats accounting
# --------------------------------------------------------------------------- #


def test_action_stats_saved_pct_zero_when_nothing_changed(tmp_path: Path):
    """A tree that only triggers ACTION_LINK has saved_pct == 0.0 (and
    no ZeroDivisionError)."""
    src = tmp_path / "src" / "mos" / "scene" / "0" / "etc"
    src.mkdir(parents=True)
    (src / "notes.txt").write_text("hi")
    res = convert_capture_tree(
        ConvertSpec(src_root=tmp_path / "src", out_root=tmp_path / "out", workers=1)
    )
    assert res.by_action[ACTION_LINK].files == 1
    assert res.saved_bytes == 0
    assert res.saved_pct == 0.0


def test_dry_run_reports_action_counts(mock_tree: Path, tmp_path: Path):
    """Dry-run must populate by_action counts so the operator can sanity
    check before committing to a real conversion."""
    res = convert_capture_tree(
        ConvertSpec(src_root=mock_tree, out_root=tmp_path / "out", dry_run=True)
    )
    assert ACTION_WEBP in res.by_action
    assert ACTION_PNG_RECOMPRESS in res.by_action
    assert ACTION_LINK in res.by_action
    assert res.by_action[ACTION_WEBP].files > 0
    assert res.by_action[ACTION_PNG_RECOMPRESS].files > 0
    # bytes_before / bytes_after stay 0 in dry-run (no I/O)
    assert res.bytes_before == 0
    assert res.bytes_after == 0


# --------------------------------------------------------------------------- #
# Verify=False short-circuits the per-frame round-trip but still produces
# bit-identical files (because PNG/WebP are lossless by spec — the verify
# flag is belt-and-braces, not a correctness gate). We still want test
# coverage of the non-verify branches in each worker.
# --------------------------------------------------------------------------- #


def test_verify_off_still_produces_valid_files(mock_tree: Path, tmp_path: Path):
    out = tmp_path / "out"
    res = convert_capture_tree(
        ConvertSpec(src_root=mock_tree, out_root=out, workers=1, verify=False)
    )
    # All four code paths exercised, output is on disk, no exceptions.
    assert res.by_action[ACTION_WEBP].files > 0
    assert res.by_action[ACTION_PNG_RECOMPRESS].files > 0
    assert list(out.rglob("*.webp")), "webp pass produced no output"
    assert list(out.rglob("*.png")), "png-recompress pass produced no output"


def test_verify_off_cam_pose_still_produces_valid_npy(cam_pose_tree: Path, tmp_path: Path):
    import numpy as np

    out = tmp_path / "out"
    convert_capture_tree(ConvertSpec(src_root=cam_pose_tree, out_root=out, workers=1, verify=False))
    npys = sorted((out / "mos" / "scene1" / "0" / "cam_pose").glob("*.npy"))
    assert npys, "cam_pose verify=False path produced no .npy"
    # Spot check: the .npy still round-trips even though encode didn't verify.
    for p in npys:
        arr = np.load(p)
        assert arr.shape == (7,)
        assert arr.dtype == np.float64
