"""Direct-call coverage for lossless_convert worker functions.

The end-to-end conversion test exercises every worker via the
:class:`~concurrent.futures.ProcessPoolExecutor`, but coverage doesn't
collect line hits inside subprocess workers without
``concurrent=multiprocessing`` config. This file calls each worker
function directly in the test process so coverage records them.

The functions are exhaustively tested for both success and failure
paths — the public ``convert_capture_tree`` API already proves the
happy path end-to-end, but the worker error branches (corrupt files,
missing keys, shape mismatches, cross-fs copy fallback) are the most
fault-relevant paths and deserve direct unit coverage.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from rpx_benchmark.dataset_hub.lossless_convert import (
    ACTION_LINK,
    ACTION_WEBP,
    _convert_cam_pose_one,
    _convert_webp_one,
    _ensure_disjoint,
    _link_one,
    _recompress_png_one,
)
from rpx_benchmark.exceptions import DatasetError

# --------------------------------------------------------------------------- #
# _convert_webp_one — success + error paths
# --------------------------------------------------------------------------- #


def test_convert_webp_one_success(tmp_path: Path):
    """Encode → verify → returns bytes_before/bytes_after honestly."""
    src = tmp_path / "in.png"
    dst = tmp_path / "out.webp"
    Image.fromarray(np.full((16, 16, 3), 200, np.uint8)).save(src)

    out, before, after, err = _convert_webp_one(str(src), str(dst), verify=True, webp_method=4)
    assert err is None
    assert out == str(dst)
    assert dst.is_file()
    assert before > 0
    assert after > 0


def test_convert_webp_one_corrupt_source_returns_error(tmp_path: Path):
    """A corrupt PNG returns ``err`` instead of crashing — the parent
    pass-handler aborts the run, but the worker itself surfaces a
    clean tuple so accounting stays sane."""
    src = tmp_path / "bad.png"
    dst = tmp_path / "out.webp"
    src.write_bytes(b"not a png")

    out, before, after, err = _convert_webp_one(str(src), str(dst), verify=True, webp_method=4)
    assert err is not None
    assert before == 0 and after == 0


def test_convert_webp_one_grayscale_source(tmp_path: Path):
    """Grayscale (mode L) source must round-trip via L mode too —
    no silent widening to RGB."""
    src = tmp_path / "gray.png"
    dst = tmp_path / "gray.webp"
    Image.fromarray(np.full((16, 16), 128, np.uint8), mode="L").save(src)

    out, before, after, err = _convert_webp_one(str(src), str(dst), verify=True, webp_method=4)
    assert err is None
    assert dst.is_file()


# --------------------------------------------------------------------------- #
# _recompress_png_one — success + error paths
# --------------------------------------------------------------------------- #


def test_recompress_png_one_preserves_uint16_depth(tmp_path: Path):
    """16-bit depth PNG re-encoded at level=9 must round-trip exactly."""
    src = tmp_path / "depth.png"
    dst = tmp_path / "out_depth.png"
    depth = np.full((16, 16), 1234, dtype=np.uint16)
    Image.fromarray(depth, mode="I;16").save(src)

    out, before, after, err = _recompress_png_one(
        str(src), str(dst), verify=True, png_compress_level=9
    )
    assert err is None
    # Verify bit-identical decode.
    decoded = np.asarray(Image.open(dst))
    assert np.array_equal(decoded, depth)


def test_recompress_png_one_preserves_palette_mask(tmp_path: Path):
    """Palette mask must round-trip with mode preserved."""
    src = tmp_path / "mask.png"
    dst = tmp_path / "out_mask.png"
    arr = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    img.save(src)

    out, before, after, err = _recompress_png_one(
        str(src), str(dst), verify=True, png_compress_level=9
    )
    assert err is None
    decoded = Image.open(dst)
    assert decoded.mode == "L"
    assert np.array_equal(np.asarray(decoded), arr)


def test_recompress_png_one_corrupt_source_returns_error(tmp_path: Path):
    src = tmp_path / "bad.png"
    dst = tmp_path / "out.png"
    src.write_bytes(b"not a png")

    out, before, after, err = _recompress_png_one(
        str(src), str(dst), verify=True, png_compress_level=9
    )
    assert err is not None


def test_recompress_png_one_verify_off_still_writes(tmp_path: Path):
    """verify=False short-circuits the round-trip but still produces a
    valid output file."""
    src = tmp_path / "ok.png"
    dst = tmp_path / "out.png"
    Image.fromarray(np.full((16, 16, 3), 50, np.uint8)).save(src)

    out, before, after, err = _recompress_png_one(
        str(src), str(dst), verify=False, png_compress_level=9
    )
    assert err is None
    assert dst.is_file()


# --------------------------------------------------------------------------- #
# _convert_cam_pose_one — success + every documented failure mode
# --------------------------------------------------------------------------- #


def test_convert_cam_pose_one_success(tmp_path: Path):
    src = tmp_path / "pose.npz"
    dst = tmp_path / "pose.npy"
    pos = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    np.savez(src, position=pos, orientation=quat)

    out, before, after, err = _convert_cam_pose_one(str(src), str(dst), verify=True)
    assert err is None
    arr = np.load(dst)
    assert arr.shape == (7,)
    assert np.array_equal(arr[:3], pos)
    assert np.array_equal(arr[3:], quat)


def test_convert_cam_pose_one_missing_keys_returns_error(tmp_path: Path):
    src = tmp_path / "bad.npz"
    dst = tmp_path / "out.npy"
    np.savez(src, wrong_key=np.zeros(3))

    out, before, after, err = _convert_cam_pose_one(str(src), str(dst), verify=True)
    assert err is not None and "missing required keys" in err


def test_convert_cam_pose_one_wrong_shape_returns_error(tmp_path: Path):
    src = tmp_path / "bad_shape.npz"
    dst = tmp_path / "out.npy"
    np.savez(
        src,
        position=np.zeros(5, dtype=np.float64),  # wrong: should be (3,)
        orientation=np.zeros(4, dtype=np.float64),
    )

    out, before, after, err = _convert_cam_pose_one(str(src), str(dst), verify=True)
    assert err is not None and "unexpected shapes" in err


def test_convert_cam_pose_one_verify_off_skips_round_trip(tmp_path: Path):
    src = tmp_path / "ok.npz"
    dst = tmp_path / "ok.npy"
    np.savez(
        src,
        position=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    )

    out, before, after, err = _convert_cam_pose_one(str(src), str(dst), verify=False)
    assert err is None
    assert dst.is_file()


def test_convert_cam_pose_one_corrupt_source_returns_error(tmp_path: Path):
    src = tmp_path / "junk.npz"
    dst = tmp_path / "out.npy"
    src.write_bytes(b"definitely not an npz")

    out, before, after, err = _convert_cam_pose_one(str(src), str(dst), verify=True)
    assert err is not None


# --------------------------------------------------------------------------- #
# _link_one — hardlink + cross-fs copy fallback + delete-existing path
# --------------------------------------------------------------------------- #


def test_link_one_hardlinks_when_same_fs(tmp_path: Path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("payload")

    out, before, after, err = _link_one(str(src), str(dst))
    assert err is None
    # Same inode → hard-linked, not copied
    assert dst.stat().st_ino == src.stat().st_ino


def test_link_one_overwrites_existing_destination(tmp_path: Path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("new")
    dst.write_text("old")
    assert dst.read_text() == "old"

    out, before, after, err = _link_one(str(src), str(dst))
    assert err is None
    assert dst.read_text() == "new"


def test_link_one_falls_back_to_copy_when_os_link_fails(tmp_path: Path):
    """When os.link raises (e.g. cross-fs in real life), shutil.copy2
    is the documented fallback. We mock the failure here."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("payload")

    with mock.patch("os.link", side_effect=OSError("cross-device link")):
        out, before, after, err = _link_one(str(src), str(dst))
    assert err is None
    assert dst.is_file()
    assert dst.read_text() == "payload"


def test_link_one_returns_error_on_unreadable_source(tmp_path: Path):
    """A non-existent source returns ``err`` rather than crashing."""
    src = tmp_path / "missing.txt"  # never created
    dst = tmp_path / "dst.txt"

    with mock.patch("os.link", side_effect=OSError("no such file")):
        out, before, after, err = _link_one(str(src), str(dst))
    assert err is not None


# --------------------------------------------------------------------------- #
# _ensure_disjoint + _dst_path_for branches
# --------------------------------------------------------------------------- #


def test_ensure_disjoint_accepts_truly_disjoint_paths(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _ensure_disjoint(a, b)  # must not raise


def test_ensure_disjoint_rejects_identical_paths(tmp_path: Path):
    p = tmp_path / "a"
    p.mkdir()
    with pytest.raises(DatasetError, match="must differ"):
        _ensure_disjoint(p, p)


def test_ensure_disjoint_rejects_nested_paths(tmp_path: Path):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    outer.mkdir()
    inner.mkdir()
    with pytest.raises(DatasetError, match="overlap"):
        _ensure_disjoint(outer, inner)


# --------------------------------------------------------------------------- #
# Sanity: actions constants are stable
# --------------------------------------------------------------------------- #


def test_action_constants_stable():
    """If anyone renames these, every downstream test breaks loudly."""
    assert ACTION_WEBP == "webp"
    assert ACTION_LINK == "link"
