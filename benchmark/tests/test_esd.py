"""Tests for ESD feature extraction (rpx_benchmark.data.esd).

Each feature is exercised against a hand-built synthetic phase dir with a
known expected value, plus edge cases (empty / single-frame / all-invalid).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rpx_benchmark.data.esd import (
    FEATURE_NAMES,
    _warn_multichannel_once,
    extract_phase_features,
    iter_phase_dirs,
)


@pytest.fixture(autouse=True)
def _reset_warning_dedupe_cache():
    """Each test starts with a clean multi-channel warning dedupe cache."""
    _warn_multichannel_once.cache_clear()
    yield
    _warn_multichannel_once.cache_clear()


@pytest.fixture(autouse=True)
def _force_propagate_for_caplog():
    """Force ``rpx_benchmark`` log propagation for the duration of each test.

    The library's :func:`configure_logging` sets ``propagate=False`` on the
    ``rpx_benchmark`` logger; if another test in the suite calls it, ``caplog``
    (which attaches to the root logger) silently stops seeing our records.
    Forcing propagate=True per-test isolates that side effect.
    """
    import logging

    logger = logging.getLogger("rpx_benchmark")
    saved = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = saved


# --------------------------------------------------------------------------- #
# Helpers — write one synthetic phase dir
# --------------------------------------------------------------------------- #


def _save_rgb(path: Path, luma: int, h: int = 8, w: int = 8) -> None:
    arr = np.full((h, w, 3), luma, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _save_depth_mm(path: Path, depth_mm: np.ndarray) -> None:
    """depth_mm: uint16 array in millimetres (0 = invalid)."""
    Image.fromarray(depth_mm.astype(np.uint16)).save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    """mask: integer array, value > 0 = instance ID."""
    Image.fromarray(mask.astype(np.uint16)).save(path)


def _save_pose(path: Path, position: np.ndarray, orientation: np.ndarray) -> None:
    np.savez(path, position=position.astype(np.float64), orientation=orientation.astype(np.float64))


def _build_phase(
    root: Path,
    scene: str,
    phase_idx: int,
    frames: list[dict],
    *,
    iter_faulty: dict[int, list[str]] | None = None,
    mask_to_object: dict[str, str] | None = None,
    h: int = 8,
    w: int = 8,
) -> Path:
    """Write a synthetic phase dir; return its path.

    ``frames`` is a list of dicts with keys: ``stem``, ``luma`` (int),
    ``depth_mm`` (np.ndarray), ``mask`` (np.ndarray), ``position`` (3,),
    ``orientation`` (4,).
    """
    pdir = root / scene / str(phase_idx)
    for sub in ("rgb", "depth", "cam_pose", "sam2/masks"):
        (pdir / sub).mkdir(parents=True, exist_ok=True)

    for fr in frames:
        s = fr["stem"]
        _save_rgb(pdir / "rgb" / f"{s}.png", fr["luma"], h, w)
        _save_depth_mm(pdir / "depth" / f"{s}.png", fr["depth_mm"])
        _save_mask(pdir / "sam2/masks" / f"{s}.png", fr["mask"])
        _save_pose(pdir / "cam_pose" / f"{s}.npz", fr["position"], fr["orientation"])

    if iter_faulty:
        for i, stems in iter_faulty.items():
            (pdir / "sam2" / f"iter{i}_faulty.txt").write_text("\n".join(stems) + "\n")

    if mask_to_object is not None:
        (pdir / "sam2" / "mask_to_object.json").write_text(json.dumps(mask_to_object))

    return pdir


def _identity_quat() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0])


def _trivial_frame(
    stem: str, h: int = 8, w: int = 8, *, luma: int = 128, position=(0.0, 0.0, 0.0)
) -> dict:
    return {
        "stem": stem,
        "luma": luma,
        "depth_mm": np.full((h, w), 1000, dtype=np.uint16),
        "mask": np.zeros((h, w), dtype=np.uint16),
        "position": np.asarray(position, dtype=np.float64),
        "orientation": _identity_quat(),
    }


# --------------------------------------------------------------------------- #
# Schema / smoke
# --------------------------------------------------------------------------- #


def test_feature_names_count_matches_appendix():
    # 19 from the new paper + 8 resurrected from old draft (RPX-overleaf.bak.pdf):
    # area_drop (§3.4 eq 15), trans_p90/rot_p90 (§3.5 eqs 20,22), and the 5
    # fisheye/stereo features (§3.7 eqs 25-29).
    assert len(FEATURE_NAMES) == 31
    assert len(set(FEATURE_NAMES)) == 31  # no duplicates


def test_extract_returns_all_18_features(tmp_path):
    pdir = _build_phase(
        tmp_path, "sceneA.lab.area", 0, [_trivial_frame("00000"), _trivial_frame("00001")]
    )
    pf = extract_phase_features(pdir)
    assert pf.scene_id == "sceneA.lab.area"
    assert pf.phase == 0
    assert pf.n_frames_total == 2
    assert pf.n_frames_used == 2
    assert set(pf.features.keys()) == set(FEATURE_NAMES)


def test_empty_phase_yields_zeroed_features(tmp_path):
    pdir = tmp_path / "scene_empty" / "0"
    (pdir / "rgb").mkdir(parents=True)
    pf = extract_phase_features(pdir)
    assert pf.n_frames_used == 0
    import math

    from rpx_benchmark.data.esd import _NAN_WHEN_ABSENT

    for k in FEATURE_NAMES:
        if k in _NAN_WHEN_ABSENT:
            assert math.isnan(pf.features[k]), f"{k} should be NaN when absent"
        else:
            assert pf.features[k] == 0.0, f"{k} should be 0.0"


# --------------------------------------------------------------------------- #
# Annotation effort
# --------------------------------------------------------------------------- #


def test_iter_counts_default_to_one(tmp_path):
    pdir = _build_phase(tmp_path, "scene1.x.y", 0, [_trivial_frame(f"{i:05d}") for i in range(3)])
    pf = extract_phase_features(pdir)
    # No iter*_faulty.txt files → every frame accepted on first pass.
    assert pf.features["iter_mean"] == 1.0
    assert pf.features["iter_max"] == 1.0


def test_iter_counts_increment_per_iter_file(tmp_path):
    """Frame 00000 is in iter1+iter2+iter3 (k=4); 00001 in iter1 (k=2); 00002 nowhere (k=1).

    Mean = (4 + 2 + 1) / 3 = 7/3; max = 4.
    """
    pdir = _build_phase(
        tmp_path,
        "s.x.y",
        0,
        [_trivial_frame(f"{i:05d}") for i in range(3)],
        iter_faulty={
            1: ["00000", "00001"],
            2: ["00000"],
            3: ["00000"],
        },
    )
    pf = extract_phase_features(pdir)
    assert pf.features["iter_mean"] == pytest.approx(7 / 3)
    assert pf.features["iter_max"] == 4.0


def test_iter_files_tolerate_paths_and_extensions(tmp_path):
    pdir = _build_phase(
        tmp_path,
        "s.x.y",
        0,
        [_trivial_frame("00000"), _trivial_frame("00001")],
        iter_faulty={1: ["00000.png", "/some/abs/path/00001.png"]},
    )
    pf = extract_phase_features(pdir)
    assert pf.features["iter_mean"] == 2.0  # both bumped to k=2
    assert pf.features["iter_max"] == 2.0


# --------------------------------------------------------------------------- #
# Scene complexity
# --------------------------------------------------------------------------- #


def test_obj_consist_is_one_when_all_objects_visible_every_frame(tmp_path):
    h, w = 8, 8
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[0:2, 0:2] = 1
    mask[2:4, 2:4] = 2
    frames = []
    for i in range(3):
        fr = _trivial_frame(f"{i:05d}")
        fr["mask"] = mask.copy()
        frames.append(fr)

    pdir = _build_phase(tmp_path, "s.x.y", 0, frames, mask_to_object={"1": "a", "2": "b"})
    pf = extract_phase_features(pdir)
    assert pf.features["obj_mean"] == 2.0
    assert pf.features["obj_std"] == 0.0
    assert pf.features["obj_consist"] == 1.0


def test_obj_consist_drops_when_one_object_disappears(tmp_path):
    """Frame 0 has both objects, frame 1 has only one → consistency = 1/2."""
    h, w = 8, 8
    full = np.zeros((h, w), dtype=np.uint16)
    full[0:2, 0:2] = 1
    full[2:4, 2:4] = 2
    half = np.zeros((h, w), dtype=np.uint16)
    half[0:2, 0:2] = 1

    frames = [_trivial_frame("00000"), _trivial_frame("00001")]
    frames[0]["mask"] = full
    frames[1]["mask"] = half

    pdir = _build_phase(tmp_path, "s.x.y", 0, frames, mask_to_object={"1": "a", "2": "b"})
    pf = extract_phase_features(pdir)
    assert pf.features["obj_mean"] == pytest.approx(1.5)
    assert pf.features["obj_consist"] == 0.5


# --------------------------------------------------------------------------- #
# Occlusion
# --------------------------------------------------------------------------- #


def test_occlusion_zero_when_objects_disjoint(tmp_path):
    h, w = 16, 16
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[0:4, 0:4] = 1
    mask[8:12, 8:12] = 2
    fr = _trivial_frame("00000", h=h, w=w)
    fr["mask"] = mask
    fr["depth_mm"] = np.full((h, w), 1000, dtype=np.uint16)
    pdir = _build_phase(tmp_path, "s.x.y", 0, [fr], h=h, w=w)
    pf = extract_phase_features(pdir)
    assert pf.features["occ_mean"] == 0.0
    assert pf.features["occ_p90"] == 0.0
    assert pf.features["occ_heavy"] == 0.0


def test_occlusion_full_overlap_yields_one(tmp_path):
    """Two objects with identical bboxes → each is fully occluded by the other."""
    h, w = 16, 16
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[0:4, 0:4] = 1  # object 1 occupies [0,4)×[0,4)
    mask[2, 2] = 2  # object 2's mask is one pixel inside object 1's bbox
    # Make object 2's bbox identical to object 1's by adding corner pixels.
    mask[0, 0] = 2
    mask[3, 3] = 2
    fr = _trivial_frame("00000", h=h, w=w)
    fr["mask"] = mask
    pdir = _build_phase(tmp_path, "s.x.y", 0, [fr], h=h, w=w)
    pf = extract_phase_features(pdir)
    assert pf.features["occ_mean"] == pytest.approx(1.0)
    assert pf.features["occ_heavy"] == 1.0


# --------------------------------------------------------------------------- #
# Depth quality
# --------------------------------------------------------------------------- #


def test_depth_invalid_fraction(tmp_path):
    h, w = 4, 4
    depth = np.full((h, w), 1000, dtype=np.uint16)
    depth[0, :] = 0  # 4/16 = 0.25 invalid
    fr = _trivial_frame("00000", h=h, w=w)
    fr["depth_mm"] = depth
    pdir = _build_phase(tmp_path, "s.x.y", 0, [fr], h=h, w=w)
    pf = extract_phase_features(pdir)
    assert pf.features["depth_invalid"] == pytest.approx(0.25)


def test_depth_invalid_mask_only_counts_inside_mask(tmp_path):
    h, w = 4, 4
    depth = np.full((h, w), 1000, dtype=np.uint16)
    depth[0, 0] = 0  # the only invalid pixel
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[0:2, 0:2] = 1  # 4 pixels in mask, 1 invalid → 0.25
    fr = _trivial_frame("00000", h=h, w=w)
    fr["depth_mm"] = depth
    fr["mask"] = mask
    pdir = _build_phase(tmp_path, "s.x.y", 0, [fr], h=h, w=w)
    pf = extract_phase_features(pdir)
    assert pf.features["depth_invalid_mask"] == pytest.approx(0.25)
    assert pf.features["depth_invalid"] == pytest.approx(1 / 16)


def test_depth_std_mask_only_uses_pixels_inside_mask(tmp_path):
    """In-mask depth std should reflect mask-region variance, not whole-image variance."""
    h, w = 8, 8
    depth = np.full((h, w), 1000, dtype=np.uint16)  # background depth = 1.0 m
    # Inside a 4×4 mask region, set varying depths so in-mask std > 0.
    depth[0:4, 0:4] = np.asarray(
        [
            [1500, 1600, 1700, 1800],
            [1500, 1600, 1700, 1800],
            [1500, 1600, 1700, 1800],
            [1500, 1600, 1700, 1800],
        ],
        dtype=np.uint16,
    )
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[0:4, 0:4] = 1
    fr = _trivial_frame("00000", h=h, w=w)
    fr["depth_mm"] = depth
    fr["mask"] = mask
    pdir = _build_phase(tmp_path, "s.x.y", 0, [fr], h=h, w=w)
    pf = extract_phase_features(pdir)

    # In-mask values are 1.5, 1.6, 1.7, 1.8 m (4× repeated rows of the same 4 values).
    # Population std of the 4 unique values: sqrt(((-0.15)² + (-0.05)² + 0.05² + 0.15²) / 4)
    # = sqrt(0.05/4) = sqrt(0.0125) ≈ 0.1118.
    expected_in_mask_std = float(np.std(np.asarray([1.5, 1.6, 1.7, 1.8] * 4, dtype=np.float32)))
    assert pf.features["depth_std_mask"] == pytest.approx(expected_in_mask_std, abs=1e-5)
    # Whole-image std should be larger (background ≠ foreground).
    assert pf.features["depth_std"] > pf.features["depth_std_mask"]


def test_depth_std_mask_zero_when_no_valid_in_mask(tmp_path):
    """If every in-mask pixel has invalid depth, depth_std_mask should be 0 (no signal)."""
    h, w = 8, 8
    depth = np.full((h, w), 1000, dtype=np.uint16)
    depth[0:4, 0:4] = 0  # invalid inside the mask region
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[0:4, 0:4] = 1
    fr = _trivial_frame("00000", h=h, w=w)
    fr["depth_mm"] = depth
    fr["mask"] = mask
    pdir = _build_phase(tmp_path, "s.x.y", 0, [fr], h=h, w=w)
    pf = extract_phase_features(pdir)
    assert pf.features["depth_std_mask"] == 0.0
    # depth_invalid_mask should be 1.0 (every in-mask pixel is invalid).
    assert pf.features["depth_invalid_mask"] == pytest.approx(1.0)


def test_depth_std_zero_for_constant_depth(tmp_path):
    pdir = _build_phase(tmp_path, "s.x.y", 0, [_trivial_frame("00000"), _trivial_frame("00001")])
    pf = extract_phase_features(pdir)
    assert pf.features["depth_std"] == 0.0


# --------------------------------------------------------------------------- #
# Photometric–depth conflict
# --------------------------------------------------------------------------- #


def test_specular_counts_bright_pixels_with_invalid_depth(tmp_path):
    h, w = 4, 4
    depth = np.zeros((h, w), dtype=np.uint16)  # all invalid
    fr_spec = _trivial_frame("00000", h=h, w=w, luma=255)
    fr_spec["depth_mm"] = depth
    fr_dark = _trivial_frame("00001", h=h, w=w, luma=0)
    fr_dark["depth_mm"] = depth
    fr_mid = _trivial_frame("00002", h=h, w=w, luma=128)
    fr_mid["depth_mm"] = depth

    pdir = _build_phase(tmp_path, "s.x.y", 0, [fr_spec, fr_dark, fr_mid], h=h, w=w)
    pf = extract_phase_features(pdir)
    # specular = mean over frames of (bright & invalid). Only frame 0 contributes (1.0).
    assert pf.features["specular"] == pytest.approx(1.0 / 3)
    assert pf.features["dark"] == pytest.approx(1.0 / 3)


# --------------------------------------------------------------------------- #
# Temporal annotation stability
# --------------------------------------------------------------------------- #


def test_area_cv_zero_when_areas_constant(tmp_path):
    h, w = 8, 8
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[0:2, 0:2] = 1
    frames = [_trivial_frame(f"{i:05d}") for i in range(3)]
    for fr in frames:
        fr["mask"] = mask.copy()
    pdir = _build_phase(tmp_path, "s.x.y", 0, frames)
    pf = extract_phase_features(pdir)
    assert pf.features["area_cv"] == 0.0
    assert pf.features["vis_instability"] == 0.0


def test_vis_instability_counts_visibility_flips(tmp_path):
    """Object 1 visible in frames 0, 2 only → 2 flips over T=3 → 2/3."""
    h, w = 8, 8
    visible = np.zeros((h, w), dtype=np.uint16)
    visible[0:2, 0:2] = 1
    invisible = np.zeros((h, w), dtype=np.uint16)
    frames = [_trivial_frame(f"{i:05d}") for i in range(3)]
    frames[0]["mask"] = visible
    frames[1]["mask"] = invisible
    frames[2]["mask"] = visible
    pdir = _build_phase(tmp_path, "s.x.y", 0, frames)
    pf = extract_phase_features(pdir)
    assert pf.features["vis_instability"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# Camera motion
# --------------------------------------------------------------------------- #


def test_zero_motion_when_pose_is_constant(tmp_path):
    pdir = _build_phase(tmp_path, "s.x.y", 0, [_trivial_frame(f"{i:05d}") for i in range(4)])
    pf = extract_phase_features(pdir)
    assert pf.features["trans_mean"] == 0.0
    assert pf.features["rot_mean"] == 0.0
    assert pf.features["jerk"] == 0.0


def test_uniform_translation_matches_step(tmp_path):
    """Camera moves +0.1 m per frame in x → trans_mean = 0.1, jerk = 0 (constant velocity)."""
    frames = []
    for i in range(4):
        fr = _trivial_frame(f"{i:05d}", position=(0.1 * i, 0.0, 0.0))
        frames.append(fr)
    pdir = _build_phase(tmp_path, "s.x.y", 0, frames)
    pf = extract_phase_features(pdir)
    assert pf.features["trans_mean"] == pytest.approx(0.1, abs=1e-6)
    assert pf.features["jerk"] == pytest.approx(0.0, abs=1e-6)


def test_rotation_angle_from_quaternion(tmp_path):
    """90° rotation about z between consecutive frames → rot_mean ≈ π/2."""
    half = np.sqrt(2) / 2  # cos(45°), sin(45°) — quat for 90° about z
    quats = [
        np.array([0.0, 0.0, 0.0, 1.0]),  # identity
        np.array([0.0, 0.0, half, half]),  # +90° about z
        np.array([0.0, 0.0, 1.0, 0.0]),  # +180° about z
    ]
    frames = []
    for i, q in enumerate(quats):
        fr = _trivial_frame(f"{i:05d}")
        fr["orientation"] = q
        frames.append(fr)
    pdir = _build_phase(tmp_path, "s.x.y", 0, frames)
    pf = extract_phase_features(pdir)
    assert pf.features["rot_mean"] == pytest.approx(np.pi / 2, abs=1e-3)


# --------------------------------------------------------------------------- #
# iter_phase_dirs
# --------------------------------------------------------------------------- #


def test_misaligned_frames_logged_and_dropped(tmp_path, caplog):
    """If a frame is missing from one modality it gets dropped; a warning fires once."""
    pdir = _build_phase(tmp_path, "s.x.y", 0, [_trivial_frame("00000"), _trivial_frame("00001")])
    # Remove the depth file for 00001 only.
    (pdir / "depth" / "00001.png").unlink()

    import logging

    with caplog.at_level(logging.WARNING, logger="rpx_benchmark.data.esd"):
        pf = extract_phase_features(pdir)
    assert pf.n_frames_total == 2  # rgb still has 2
    assert pf.n_frames_used == 1  # only the aligned frame
    assert any("modality misalignment" in rec.message for rec in caplog.records)


def test_multichannel_mask_collapses_to_first_channel(tmp_path, caplog):
    """A 3-channel mask PNG should not crash; first channel is used and a warning fires."""
    pdir = _build_phase(tmp_path, "s.x.y", 0, [_trivial_frame("00000")])
    h, w = 8, 8
    # Replicate a single-channel mask across all 3 channels — order-agnostic
    # (PIL returns RGB, cv2 returns BGR; defensive code uses channel 0 either way).
    single = np.zeros((h, w), dtype=np.uint8)
    single[0:2, 0:2] = 1
    rgb_mask = np.stack([single, single, single], axis=-1)
    Image.fromarray(rgb_mask, mode="RGB").save(pdir / "sam2/masks" / "00000.png")

    import logging

    with caplog.at_level(logging.WARNING, logger="rpx_benchmark.data.esd"):
        pf = extract_phase_features(pdir)
    multichannel_warnings = [r for r in caplog.records if "multi-channel" in r.message]
    assert len(multichannel_warnings) == 1, (
        f"multi-channel warning should fire once per directory, got {len(multichannel_warnings)}"
    )
    assert pf.features["obj_mean"] == 1.0


def test_multichannel_warning_dedupes_across_many_frames(tmp_path, caplog):
    """A phase with N RGB-saved masks should emit exactly ONE warning, not N."""
    h, w = 8, 8
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_mask[0:2, 0:2, :] = 1

    n_frames = 12
    pdir = _build_phase(tmp_path, "s.x.y", 0, [_trivial_frame(f"{i:05d}") for i in range(n_frames)])
    for i in range(n_frames):
        Image.fromarray(rgb_mask, mode="RGB").save(pdir / "sam2/masks" / f"{i:05d}.png")

    import logging

    with caplog.at_level(logging.WARNING, logger="rpx_benchmark.data.esd"):
        extract_phase_features(pdir)
    multichannel_warnings = [r for r in caplog.records if "multi-channel" in r.message]
    # Exactly one warning across 12 multi-channel frames.
    assert len(multichannel_warnings) == 1


def test_malformed_mask_to_object_json_falls_back(tmp_path):
    """A corrupt mask_to_object.json should not crash; n_total derived from observed IDs."""
    pdir = _build_phase(tmp_path, "s.x.y", 0, [_trivial_frame("00000")])
    h, w = 8, 8
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[0:2, 0:2] = 1
    mask[2:4, 2:4] = 2
    Image.fromarray(mask).save(pdir / "sam2/masks" / "00000.png")
    (pdir / "sam2" / "mask_to_object.json").write_text("{ this is not json")

    pf = extract_phase_features(pdir)
    # Fallback path → n_total = max ID = 2; both visible → consist = 1.
    assert pf.features["obj_consist"] == 1.0


def test_iter_phase_dirs_raises_on_missing_root(tmp_path):
    from rpx_benchmark.exceptions import DatasetError

    with pytest.raises(DatasetError):
        list(iter_phase_dirs(tmp_path / "does_not_exist"))


# --------------------------------------------------------------------------- #
# Old-draft features: area_drop, trans_p90/rot_p90, fisheye_*
# --------------------------------------------------------------------------- #


def test_area_drop_zero_when_areas_constant(tmp_path):
    h, w = 8, 8
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[0:2, 0:2] = 1
    frames = [_trivial_frame(f"{i:05d}") for i in range(4)]
    for fr in frames:
        fr["mask"] = mask.copy()
    pdir = _build_phase(tmp_path, "s.x.y", 0, frames)
    pf = extract_phase_features(pdir)
    assert pf.features["area_drop"] == 0.0


def test_area_drop_counts_50pct_collapse(tmp_path):
    """Object 1 has 4 px → 4 px → 1 px (75% drop): one drop event in T=3 frames."""
    h, w = 8, 8
    full = np.zeros((h, w), dtype=np.uint16)
    full[0:2, 0:2] = 1  # area = 4
    small = np.zeros((h, w), dtype=np.uint16)
    small[0, 0] = 1  # area = 1
    frames = [_trivial_frame(f"{i:05d}") for i in range(3)]
    frames[0]["mask"] = full
    frames[1]["mask"] = full
    frames[2]["mask"] = small
    pdir = _build_phase(tmp_path, "s.x.y", 0, frames)
    pf = extract_phase_features(pdir)
    # 1 drop / T=3 → 1/3 (averaged over 1 instance)
    assert pf.features["area_drop"] == pytest.approx(1 / 3)


def test_trans_p90_picks_up_motion_spikes(tmp_path):
    """Varying motion with moderate spike → p90 > mean.

    After MAD filtering, extreme single-frame pose jumps (T265 relocalization)
    are suppressed. This test uses motion that varies naturally rather than
    a single extreme outlier.
    """
    # Gradually increasing motion: 0.01, 0.01, 0.05, 0.10 m between frames
    positions = [(0, 0, 0), (0.01, 0, 0), (0.02, 0, 0), (0.07, 0, 0), (0.17, 0, 0)]
    frames = [_trivial_frame(f"{i:05d}", position=p) for i, p in enumerate(positions)]
    pdir = _build_phase(tmp_path, "s.x.y", 0, frames)
    pf = extract_phase_features(pdir)
    # delta_t = [0.01, 0.01, 0.05, 0.10] → mean=0.0425, p90 should be > mean
    assert pf.features["trans_mean"] == pytest.approx(0.0425, abs=1e-3)
    assert pf.features["trans_p90"] > pf.features["trans_mean"]


def test_rot_p90_present_when_rotation_present(tmp_path):
    """Constant per-frame rotation → mean ≈ p90 (no spike)."""
    half = math.sqrt(2) / 2
    quats = [
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, half, half]),  # +90° about z
        np.array([0.0, 0.0, 1.0, 0.0]),  # +180° about z
    ]
    frames = []
    for i, q in enumerate(quats):
        fr = _trivial_frame(f"{i:05d}")
        fr["orientation"] = q
        frames.append(fr)
    pdir = _build_phase(tmp_path, "s.x.y", 0, frames)
    pf = extract_phase_features(pdir)
    assert pf.features["rot_p90"] == pytest.approx(pf.features["rot_mean"], abs=1e-3)


def test_fisheye_features_nan_when_dir_missing(tmp_path):
    """Fisheye features are NaN (not 0) when fisheye data is absent,
    so percentile normalization maps them to 0.5 (neutral) instead of
    biasing toward low difficulty."""
    import math

    pdir = _build_phase(tmp_path, "s.x.y", 0, [_trivial_frame("00000"), _trivial_frame("00001")])
    pf = extract_phase_features(pdir)
    for key in (
        "fisheye_dark",
        "fisheye_bright",
        "fisheye_sharpness",
        "fisheye_corr",
        "fisheye_texture",
    ):
        assert math.isnan(pf.features[key]), f"{key} should be NaN when absent"


def _add_fisheye(pdir: Path, layout: str, h: int = 16, w: int = 16) -> None:
    """Attach a 3-frame fisheye block to an existing phase dir.

    layout: 'subdirs' | 'suffix' | 'single'
    """
    fish = pdir / "fisheye"
    fish.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for t in range(3):
        # Random texture so sharpness / texture / correlation are non-trivial.
        L = rng.integers(0, 256, (h, w), dtype=np.uint8)
        R = (L.astype(np.int32) + rng.integers(-5, 6, (h, w))).clip(0, 255).astype(np.uint8)
        if layout == "subdirs":
            (fish / "left").mkdir(exist_ok=True)
            (fish / "right").mkdir(exist_ok=True)
            Image.fromarray(L).save(fish / "left" / f"{t:05d}.png")
            Image.fromarray(R).save(fish / "right" / f"{t:05d}.png")
        elif layout == "suffix":
            Image.fromarray(L).save(fish / f"{t:05d}_L.png")
            Image.fromarray(R).save(fish / f"{t:05d}_R.png")
        elif layout == "single":
            Image.fromarray(L).save(fish / f"{t:05d}.png")
        else:
            raise ValueError(layout)


@pytest.mark.parametrize("layout", ["subdirs", "suffix"])
def test_fisheye_features_with_stereo_pairs(tmp_path, layout):
    pdir = _build_phase(tmp_path, "s.x.y", 0, [_trivial_frame(f"{i:05d}") for i in range(3)])
    _add_fisheye(pdir, layout)
    pf = extract_phase_features(pdir)
    # Random images: dark/bright fractions both small (uniform [0,255]); sharpness
    # and texture both non-zero; correlation high (R = L + small noise).
    assert pf.features["fisheye_sharpness"] > 0
    assert pf.features["fisheye_texture"] > 0
    assert pf.features["fisheye_corr"] > 0.9  # near-identical L/R


def test_fisheye_corr_is_zero_when_no_right_pair(tmp_path, caplog):
    pdir = _build_phase(tmp_path, "s.x.y", 0, [_trivial_frame(f"{i:05d}") for i in range(3)])
    _add_fisheye(pdir, "single")
    import logging

    with caplog.at_level(logging.WARNING, logger="rpx_benchmark.data.esd"):
        pf = extract_phase_features(pdir)
    assert pf.features["fisheye_corr"] == 0.0  # no pairs available
    assert pf.features["fisheye_sharpness"] > 0  # left-only stats still computed
    assert any("no stereo pairs" in rec.message for rec in caplog.records)


def test_iter_phase_dirs_finds_only_numeric_phase_dirs(tmp_path):
    _build_phase(tmp_path, "scene1.GDC.cse", 0, [_trivial_frame("00000")])
    _build_phase(tmp_path, "scene1.GDC.cse", 1, [_trivial_frame("00000")])
    _build_phase(tmp_path, "scene2.SBSI.lab", 0, [_trivial_frame("00000")])
    # Add a non-numeric child that should be ignored.
    (tmp_path / "scene1.GDC.cse" / "notes").mkdir()

    found = sorted(p.parent.name + "/" + p.name for p in iter_phase_dirs(tmp_path))
    assert found == [
        "scene1.GDC.cse/0",
        "scene1.GDC.cse/1",
        "scene2.SBSI.lab/0",
    ]
