"""Synthetic-fixture integration tests for ``scripts/run_nvs.py``.

The runner is verified by hand on the test mirror, but the parallel
session iterates on ``rpx_benchmark.nvs_pairs.NVSSample`` /
``rpx_benchmark.nvs_eval`` / ``rpx_benchmark.nvs_metrics`` faster than I
can re-run real-data smokes after each change. These tests lock the
runner's piecewise contracts (per-modality loaders, per-sample
evaluation, three-axis result assembly, per-object PSNR aggregation)
against a temp-dir synthetic dataset so CI catches schema drift the
moment it lands.

What is tested
--------------

* Modality loaders: shape, dtype, scaling, read-only flag.
* LRU cache: second call to the same path is a cache hit; cache size
  grows on miss.
* `_evaluate_sample`: produces the keys `evaluate_nvs` aggregates on
  (psnr / ssim / depth_absrel / depth_rmse / depth_delta1 / per_object_psnr_*).
* `_assemble_result`: emits the three RPX axis keys at the top level
  (`aggregated` / `robustness` / `compute_cost`) with no composite.
* Per-object PSNR aggregates into `aggregated` when masks are provided.

What is *not* tested here (covered by other tests / smokes):

* End-to-end `main()` with argparse — exercised by the test-mirror
  smokes.
* NVSPairGenerator path-resolution against a real parquet — covered
  by `nvs_pairs`'s own tests in the parallel session.
* Per-adapter forward passes — adapters either have their own tests
  or are out-of-process upstream code.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

# Make ./scripts importable so `import run_nvs` works.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_nvs  # noqa: E402
from nvs_models.identity_passthrough import IdentityPassthroughNVS  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fixture — minimal extracted_root with RGB + depth + cam_pose + mask
# ─────────────────────────────────────────────────────────────────────────────


def _make_pose_npz(path: Path, position: tuple, quat_xyzw: tuple) -> None:
    """Write a T265-shaped .npz (position + orientation), matching the
    schema run_nvs._load_pose reads."""
    np.savez(
        path,
        position=np.array(position, dtype=np.float64),
        orientation=np.array(quat_xyzw, dtype=np.float64),
    )


def _write_phase(
    root: Path,
    scene: str,
    phase: int,
    frame_idxs: list[int],
    h: int = 64,
    w: int = 64,
) -> None:
    """Write the four modality directories for one (scene, phase):
    rgb / depth / cam_pose / sam2/masks. Each frame's pose places the
    camera at (frame_idx, 0, 0) with identity rotation so the
    identity-baseline `closest-pose` heuristic is exercised."""
    pdir = root / "scenes" / scene / str(phase)
    (pdir / "rgb").mkdir(parents=True, exist_ok=True)
    (pdir / "depth").mkdir(parents=True, exist_ok=True)
    (pdir / "cam_pose").mkdir(parents=True, exist_ok=True)
    (pdir / "sam2" / "masks").mkdir(parents=True, exist_ok=True)

    for idx in frame_idxs:
        # RGB: constant gray * frame_idx so closest-pose pick is verifiable.
        rgb_val = (idx * 30) % 255
        Image.fromarray(
            np.full((h, w, 3), rgb_val, dtype=np.uint8)
        ).save(pdir / "rgb" / f"{idx:05d}.png")

        # Depth: idx metres in mm = idx*1000 mm uint16.
        Image.fromarray(
            np.full((h, w), idx * 1000, dtype=np.uint16),
            mode="I;16",
        ).save(pdir / "depth" / f"{idx:05d}.png")

        # Pose: position (idx, 0, 0), identity orientation.
        _make_pose_npz(
            pdir / "cam_pose" / f"{idx:05d}.npz",
            position=(float(idx), 0.0, 0.0),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        )

        # Mask: half background (id 0), half instance 1.
        m = np.zeros((h, w), dtype=np.uint16)
        m[:, w // 2:] = 1
        Image.fromarray(m, mode="I;16").save(pdir / "sam2" / "masks" / f"{idx:05d}.png")


@pytest.fixture
def synthetic_nvs_root(tmp_path: Path) -> Path:
    """Build a temp extracted_root with one scene, two phases, 4 frames each."""
    scene = "scene00.synth"
    _write_phase(tmp_path, scene, phase=0, frame_idxs=[0, 1, 2, 3])
    _write_phase(tmp_path, scene, phase=2, frame_idxs=[0, 1, 2, 3])
    # Clear loader caches so each test starts from a clean slate.
    run_nvs._load_rgb.cache_clear()
    run_nvs._load_depth.cache_clear()
    run_nvs._load_pose.cache_clear()
    run_nvs._load_mask.cache_clear()
    return tmp_path


def _make_sample(
    scene_id: str = "scene00.synth",
    phase: int = 0,
    phase_target: int = 0,
    n_context: int = 2,
    sample_type: str = "interpolation",
    context_frame_idxs: list[int] | None = None,
    target_frame_idx: int = 2,
) -> SimpleNamespace:
    """SimpleNamespace duck-typed to look like nvs_pairs.NVSSample."""
    if context_frame_idxs is None:
        context_frame_idxs = [0, 1]

    def _rel(modality: str, p: int, idx: int) -> str:
        ext = "npz" if modality == "cam_pose" else "png"
        return f"scenes/{scene_id}/{p}/{modality}/{idx:05d}.{ext}"

    return SimpleNamespace(
        id=f"{scene_id}_{phase}_{target_frame_idx}",
        scene_id=scene_id,
        phase=phase,
        phase_target=phase_target,
        n_context=n_context,
        sample_type=sample_type,
        context_frame_idxs=context_frame_idxs,
        target_frame_idx=target_frame_idx,
        context_rgb_paths=[_rel("rgb", phase, i) for i in context_frame_idxs],
        context_depth_paths=[_rel("depth", phase, i) for i in context_frame_idxs],
        context_pose_paths=[_rel("cam_pose", phase, i) for i in context_frame_idxs],
        target_rgb_path=_rel("rgb", phase_target, target_frame_idx),
        target_depth_path=_rel("depth", phase_target, target_frame_idx),
        target_pose_path=_rel("cam_pose", phase_target, target_frame_idx),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Modality loaders
# ─────────────────────────────────────────────────────────────────────────────


class TestModalityLoaders:
    def test_rgb_shape_dtype_readonly(self, synthetic_nvs_root: Path) -> None:
        arr = run_nvs._load_rgb(synthetic_nvs_root / "scenes/scene00.synth/0/rgb/00001.png")
        assert arr.shape == (64, 64, 3)
        assert arr.dtype == np.uint8
        assert not arr.flags.writeable
        # Frame 1 was written with constant value 30.
        assert arr[0, 0, 0] == 30

    def test_depth_mm_to_m_scaling_readonly(self, synthetic_nvs_root: Path) -> None:
        arr = run_nvs._load_depth(synthetic_nvs_root / "scenes/scene00.synth/0/depth/00003.png")
        assert arr.shape == (64, 64)
        assert arr.dtype == np.float32
        assert not arr.flags.writeable
        # Frame 3 was written at 3000 mm → 3.0 m.
        assert np.allclose(arr, 3.0)

    def test_pose_quat_to_se3_readonly(self, synthetic_nvs_root: Path) -> None:
        T = run_nvs._load_pose(synthetic_nvs_root / "scenes/scene00.synth/0/cam_pose/00002.npz")
        assert T.shape == (4, 4)
        assert T.dtype == np.float64
        assert not T.flags.writeable
        # Frame 2 was written at position (2, 0, 0) with identity orientation.
        assert np.allclose(T[:3, :3], np.eye(3))
        assert np.allclose(T[:3, 3], [2.0, 0.0, 0.0])
        # Bottom row of a homogeneous SE(3) matrix.
        assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0])

    def test_mask_shape_dtype_readonly(self, synthetic_nvs_root: Path) -> None:
        m = run_nvs._load_mask(
            synthetic_nvs_root / "scenes/scene00.synth/0/sam2/masks/00000.png"
        )
        assert m.shape == (64, 64)
        assert m.dtype == np.int32
        assert not m.flags.writeable
        # Half background (0), half instance 1.
        assert set(np.unique(m).tolist()) == {0, 1}


# ─────────────────────────────────────────────────────────────────────────────
# LRU cache
# ─────────────────────────────────────────────────────────────────────────────


class TestLoaderCache:
    def test_repeated_calls_are_cache_hits(self, synthetic_nvs_root: Path) -> None:
        p = synthetic_nvs_root / "scenes/scene00.synth/0/rgb/00000.png"
        # First call: cold, all misses.
        run_nvs._load_rgb(p)
        info0 = run_nvs._load_rgb.cache_info()
        assert info0.hits == 0
        assert info0.misses == 1

        # Second call to the same path: cache hit, no extra miss.
        run_nvs._load_rgb(p)
        info1 = run_nvs._load_rgb.cache_info()
        assert info1.hits == 1
        assert info1.misses == 1
        assert info1.currsize == 1

    def test_cache_stats_dict_shape(self, synthetic_nvs_root: Path) -> None:
        run_nvs._load_rgb(synthetic_nvs_root / "scenes/scene00.synth/0/rgb/00000.png")
        stats = run_nvs._loader_cache_stats()
        assert set(stats) == {"rgb", "depth", "mask", "pose"}
        for v in stats.values():
            assert {"hits", "misses", "hit_rate", "size", "maxsize"} <= set(v)


# ─────────────────────────────────────────────────────────────────────────────
# _evaluate_sample — keys consumed by evaluate_nvs
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluateSample:
    def _run_one(
        self,
        synthetic_nvs_root: Path,
        with_mask: bool = False,
    ) -> dict:
        sample = _make_sample()
        adapter = IdentityPassthroughNVS(device="cpu")
        context_rgbs = [run_nvs._load_rgb(synthetic_nvs_root / p) for p in sample.context_rgb_paths]
        context_depths = [run_nvs._load_depth(synthetic_nvs_root / p) for p in sample.context_depth_paths]
        context_poses = [run_nvs._load_pose(synthetic_nvs_root / p) for p in sample.context_pose_paths]
        target_pose = run_nvs._load_pose(synthetic_nvs_root / sample.target_pose_path)
        gt_rgb = run_nvs._load_rgb(synthetic_nvs_root / sample.target_rgb_path)
        gt_depth = run_nvs._load_depth(synthetic_nvs_root / sample.target_depth_path)
        gt_mask = None
        if with_mask:
            gt_mask = run_nvs._load_mask(
                synthetic_nvs_root / run_nvs._mask_path_for_target(sample)
            )
        rendered = adapter(context_rgbs, context_depths, context_poses, target_pose)
        return run_nvs._evaluate_sample(sample, rendered, gt_rgb, gt_depth, gt_mask=gt_mask)

    def test_metric_keys_present_no_mask(self, synthetic_nvs_root: Path) -> None:
        row = self._run_one(synthetic_nvs_root, with_mask=False)
        # Sample-metadata keys for stratification by evaluate_nvs.
        for k in ("sample_id", "scene_id", "phase", "n_context", "sample_type"):
            assert k in row
        # Standard metrics evaluate_nvs aggregates on.
        for k in ("psnr", "ssim", "depth_absrel", "depth_rmse", "depth_delta1"):
            assert k in row, f"missing metric key: {k}"
            assert row[k] is not None
        # Per-object should NOT appear without a mask.
        assert "per_object_psnr_mean" not in row

    def test_per_object_keys_when_mask_provided(self, synthetic_nvs_root: Path) -> None:
        row = self._run_one(synthetic_nvs_root, with_mask=True)
        # Per-object metrics surface from evaluate_single_sample.
        assert "per_object_psnr_mean" in row
        assert "per_object_psnr_min" in row
        assert "n_objects_evaluated" in row
        # Synthetic mask has exactly one instance (id 1) > 10 pixels.
        assert row["n_objects_evaluated"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# _assemble_result — three-axis schema
# ─────────────────────────────────────────────────────────────────────────────


class TestAssembleResult:
    def _per_sample_rows(
        self, with_per_object: bool = False, n: int = 4,
    ) -> list[dict]:
        rows = []
        for i in range(n):
            row = {
                "sample_id":    f"s{i}",
                "scene_id":     "scene00",
                "phase":        0,
                "n_context":    2 if i % 2 == 0 else 4,
                "sample_type":  "interpolation" if i < n // 2 else "cross_phase",
                "psnr":         20.0 + i,
                "ssim":         0.5 + 0.1 * i,
                "depth_absrel": 0.1 - 0.01 * i,
                "depth_rmse":   0.3,
                "depth_delta1": 0.85,
            }
            if with_per_object:
                row["per_object_psnr_mean"] = 15.0 + i
                row["per_object_psnr_min"]  = 10.0 + i
                row["n_objects_evaluated"]  = 5.0
            rows.append(row)
        return rows

    def test_top_level_three_axis_schema(self) -> None:
        result = run_nvs._assemble_result(
            model_key="identity_passthrough",
            display_name="IdentityPassthrough",
            split="easy",
            per_sample=self._per_sample_rows(),
            cost_block={
                "params_m": 0.0, "flops_g": None, "macs_g": None,
                "memory_traffic_gb": None, "arithmetic_intensity": None,
                "roofline": None, "latency_ms_per_sample": 0.5,
                "peak_memory_mb": None, "system_card": None,
                "operating_point": {"precision": "fp32", "params_m": 0.0, "flops_g": None},
            },
            latencies_ms=[0.5, 0.6, 0.4, 0.7],
            wall_seconds=1.2,
        )

        # Three RPX axes, no composite.
        assert "aggregated" in result      # Axis 1
        assert "robustness" in result      # Axis 2
        assert "compute_cost" in result    # Axis 3
        assert "deployment_readiness" not in result, (
            "composite key should not be present — three-axis policy"
        )
        assert "drs" not in result

        # Axis-2 sub-blocks present.
        rb = result["robustness"]
        for k in ("by_sample_type", "by_context_count", "cross_phase_delta"):
            assert k in rb

        # Cross-phase Δ exercised by the half-cross_phase fixture rows.
        assert rb["cross_phase_delta"], "expected cross-phase Δ values with mixed sample_types"

    def test_per_object_aggregation_surfaces_when_present(self) -> None:
        result = run_nvs._assemble_result(
            model_key="identity_passthrough",
            display_name="IdentityPassthrough",
            split="easy",
            per_sample=self._per_sample_rows(with_per_object=True),
            cost_block={
                "params_m": 0.0, "flops_g": None, "macs_g": None,
                "memory_traffic_gb": None, "arithmetic_intensity": None,
                "roofline": None, "latency_ms_per_sample": 0.5,
                "peak_memory_mb": None, "system_card": None,
                "operating_point": {"precision": "fp32", "params_m": 0.0, "flops_g": None},
            },
            latencies_ms=[0.5, 0.6, 0.4, 0.7],
            wall_seconds=1.2,
        )

        a = result["aggregated"]
        assert "per_object_psnr_mean" in a
        assert "per_object_psnr_min" in a
        assert "n_objects_evaluated" in a
        # np.mean of [15, 16, 17, 18] = 16.5
        assert a["per_object_psnr_mean"] == pytest.approx(16.5)
        # np.mean of [10, 11, 12, 13] = 11.5
        assert a["per_object_psnr_min"] == pytest.approx(11.5)
        # All four rows had n_objects_evaluated = 5.
        assert a["n_objects_evaluated"] == pytest.approx(5.0)

    def test_per_object_keys_absent_when_no_masks(self) -> None:
        result = run_nvs._assemble_result(
            model_key="identity_passthrough",
            display_name="IdentityPassthrough",
            split="easy",
            per_sample=self._per_sample_rows(with_per_object=False),
            cost_block={
                "params_m": 0.0, "flops_g": None, "macs_g": None,
                "memory_traffic_gb": None, "arithmetic_intensity": None,
                "roofline": None, "latency_ms_per_sample": 0.5,
                "peak_memory_mb": None, "system_card": None,
                "operating_point": {"precision": "fp32", "params_m": 0.0, "flops_g": None},
            },
            latencies_ms=[0.5, 0.6, 0.4, 0.7],
            wall_seconds=1.2,
        )
        a = result["aggregated"]
        for k in ("per_object_psnr_mean", "per_object_psnr_min", "n_objects_evaluated"):
            assert k not in a
