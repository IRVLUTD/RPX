"""Adversarial / fault-proof tests for the dataset hub.

Each test deliberately tampers with one link in the publication →
download → load chain and asserts the downstream code aborts loudly
instead of silently serving corrupt or wrong-format data to a model.

Threat model
------------

The lossless-convert + manifest + extract pipeline must produce a
benchmark-correct array on the consumer's machine *or* refuse to
produce anything at all. The unacceptable outcome is "model received
a wrong number and produced a wrong score" — silent loss.

These tests cover the three points where loss could enter after
encoder-side per-frame verification has already run:

* **Tar bytes corrupted in transit / on disk** — caught by
  ``hub._extract_snapshot_tars`` via the SHA-256 in
  ``manifest/checksums.json``.
* **A decoder returns the wrong dtype/shape** — caught by the
  runtime contract assertions on every ``loader._load_*`` method.
* **An adapter calls ``cv2.imread`` without ``IMREAD_UNCHANGED`` on
  a 16-bit depth file** — silently truncates to 8-bit. We can't
  prevent the call, but the loader contract makes the symptom
  visible immediately when the result flows back through normal
  code paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow", reason="pyarrow required for manifest builder")
pytest.importorskip("pandas", reason="pandas required for split-manifest writer")
pytest.importorskip("PIL.Image", reason="Pillow required for image round-trip")

from PIL import Image

from rpx_benchmark.dataset_hub.cli import _rehydrate_pack_result
from rpx_benchmark.dataset_hub.manifest import build_frame_manifest
from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.dataset_hub.packer import PackPlan, pack_capture_tree
from rpx_benchmark.dataset_hub.scanner import scan_capture_root
from rpx_benchmark.exceptions import DatasetError, ManifestError
from rpx_benchmark.hub import _extract_snapshot_tars
from rpx_benchmark.loader import RPXDataset


# --------------------------------------------------------------------------- #
# Fixture: a packed-and-manifested mock dataset with checksums.json on disk
# --------------------------------------------------------------------------- #


@pytest.fixture
def packed(tmp_path: Path) -> dict:
    """Pack a mock dataset and build the manifest with checksums.json."""
    src = tmp_path / "src"
    generate_mock(
        src,
        MockSpec(
            multi_object_scenes=2,
            single_object_scenes=0,
            phases_per_multi=1,
            frames_per_phase=2,
            image_size=16,
        ),
    )
    staging = tmp_path / "staging"
    scan = scan_capture_root(src)
    pack_capture_tree(
        PackPlan(src_root=src, staging_root=staging, label_version="v1", overwrite=True),
        scan,
    )
    pack = _rehydrate_pack_result(scan, staging)
    splits = {s.scene_id: ("easy" if i % 2 == 0 else "hard") for i, s in enumerate(scan.scenes)}
    build_frame_manifest(scan, pack, staging, splits=splits)
    return {"src": src, "staging": staging, "scan": scan}


# --------------------------------------------------------------------------- #
# 1. checksums.json must exist and cover every shard
# --------------------------------------------------------------------------- #


def test_checksums_json_exists_and_covers_every_tar(packed: dict):
    staging: Path = packed["staging"]
    cp = staging / "manifest" / "checksums.json"
    assert cp.is_file(), "build_frame_manifest did not write checksums.json"

    payload = json.loads(cp.read_text())
    assert payload.get("algorithm") == "sha256"
    sha = payload.get("sha256") or {}
    tars = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*.tar"))
    assert tars, "no tars in staging — fixture broken"
    missing = [t for t in tars if not sha.get(t)]
    assert not missing, f"tars without SHA-256 in checksums.json: {missing[:3]}"


# --------------------------------------------------------------------------- #
# 2. Honest tar → extraction proceeds; checksum-honest run is idempotent
# --------------------------------------------------------------------------- #


def test_extract_succeeds_when_checksums_match(packed: dict):
    staging: Path = packed["staging"]
    n_new, _ = _extract_snapshot_tars(staging)
    assert n_new > 0


# --------------------------------------------------------------------------- #
# 3. Adversarial: corrupt a single tar; extractor must abort loudly
# --------------------------------------------------------------------------- #


def test_corrupted_tar_byte_aborts_extraction(packed: dict):
    """Flip a byte in a packed tar and confirm ``_extract_snapshot_tars``
    raises ``DatasetError`` with the SHA-256 mismatch surfaced — no
    bytes from this tar are allowed onto disk."""
    staging: Path = packed["staging"]
    tars = sorted(staging.rglob("*.tar"))
    assert tars, "no tars in fixture"
    target = tars[0]

    # Tamper: flip one byte deep in the file (after the tar header)
    with target.open("r+b") as f:
        f.seek(2048)
        b = f.read(1)
        f.seek(2048)
        f.write(bytes([b[0] ^ 0xFF]))

    with pytest.raises(DatasetError, match="SHA-256 mismatch"):
        _extract_snapshot_tars(staging)


def test_truncated_tar_aborts_extraction(packed: dict):
    """A tar truncated in transit (different bytes, different hash) is
    rejected before any member is extracted."""
    staging: Path = packed["staging"]
    target = next(iter(sorted(staging.rglob("*.tar"))))
    original_size = target.stat().st_size
    with target.open("r+b") as f:
        f.truncate(original_size - 64)

    with pytest.raises(DatasetError, match="SHA-256 mismatch"):
        _extract_snapshot_tars(staging)


def test_missing_checksums_json_falls_through(packed: dict):
    """Legacy datasets without checksums.json (anything uploaded before
    this PR) still extract — backward-compatible degraded-defense mode."""
    staging: Path = packed["staging"]
    (staging / "manifest" / "checksums.json").unlink()
    # No exception — falls through to the previous behaviour.
    n_new, _ = _extract_snapshot_tars(staging)
    assert n_new > 0


# --------------------------------------------------------------------------- #
# 4. Loader runtime contracts — wrong-dtype / wrong-shape files refuse load
# --------------------------------------------------------------------------- #


def test_uint8_depth_file_rejected_by_loader(tmp_path: Path):
    """An 8-bit PNG in a depth path silently masquerades as 8-bit depth
    under a naive ``cv2.imread`` adapter. The canonical loader's depth
    contract makes the symptom loud."""
    rgb_dir = tmp_path / "rgb"
    depth_dir = tmp_path / "depth"
    rgb_dir.mkdir()
    depth_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(rgb_dir / "0.png")
    # Plant an 8-bit single-channel PNG (mode L) in the depth path.
    Image.fromarray(np.full((8, 8), 42, np.uint8), mode="L").save(depth_dir / "0.png")

    manifest = {
        "task": "monocular_depth",
        "root": str(tmp_path),
        "samples": [{"id": "t", "rgb": "rgb/0.png", "depth": "depth/0.png"}],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    ds = RPXDataset.from_manifest(p, batch_size=1)
    with pytest.raises(ManifestError, match="expected uint16"):
        next(iter(ds))


def test_rgb_load_contract_holds_on_valid_input(tmp_path: Path):
    """Sanity check: the new RGB contract does not over-trigger on
    correctly-shaped uint8 RGB input."""
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(rgb_dir / "0.png")
    manifest = {
        "task": "monocular_depth",
        "root": str(tmp_path),
        "samples": [
            {
                "id": "t",
                "rgb": "rgb/0.png",
                "depth": "rgb/0.png",  # placeholder so monocular_depth parses
            }
        ],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    ds = RPXDataset.from_manifest(p, batch_size=1)
    # We don't pull a sample here — depth would fail because the
    # placeholder is RGB. The point of this test is the loader's
    # constructor + RGB load succeeds without spurious contract errors.
    # Pulling rgb directly:
    rgb = ds._load_rgb("rgb/0.png")
    assert rgb.dtype == np.uint8
    assert rgb.shape == (8, 8, 3)


# --------------------------------------------------------------------------- #
# 5. The conversion script's --no-verify flag is gone
# --------------------------------------------------------------------------- #


def test_cli_does_not_expose_no_verify_flag():
    """Argparse parser for ``lossless-convert`` must not have a
    ``--no-verify`` flag — mandatory verification is the contract."""
    from rpx_benchmark.dataset_hub.cli import build_parser

    parser = build_parser()
    # Help text walked manually for the subparser.
    sub = parser._subparsers._group_actions[0].choices["lossless-convert"]  # noqa: SLF001
    flag_strings = []
    for action in sub._actions:  # noqa: SLF001
        flag_strings.extend(action.option_strings)
    assert "--no-verify" not in flag_strings, (
        "lossless-convert CLI must not expose --no-verify — verification "
        "is the fault-proof guarantee and has no operator-facing opt-out."
    )
