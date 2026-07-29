"""True-batched depth model wrapper.

The stock :class:`rpx_benchmark.adapters.base.BenchmarkableModel` iterates
samples one at a time inside ``predict``: even with ``--batch-size 8``,
the underlying adapter still sees one image per call, so frameworks that
batch internally (HuggingFace pipelines, DataLoader-driven PyTorch
inference) get no benefit. ``BatchedDepthBenchmarkModel`` fixes that for
monocular metric depth: it hands the whole list of RGBs to the adapter
in one call, lets the adapter do a single batched forward, then unwraps
the depth predictions one-per-sample.

The model itself is untouched — only the dispatch shape changes — so
metric values are bit-identical to the per-sample path. Only throughput
improves (≈1.3× at batch 8 for ZoeDepth on an RTX 5070 Laptop;
typically larger for smaller models).

Adapter contract
----------------
The wrapped adapter must accept either:

* a single ``np.ndarray`` of shape ``(H, W, 3)`` uint8 → return depth ``(H', W')`` float
* a ``list[np.ndarray]`` → return ``list[np.ndarray]`` (one depth array per RGB,
  same order)

If the adapter returns a depth shape different from its input RGB, we
bilinearly resize back to the RGB's HW so the depth aligns with the GT
mask (PIL-only, no OpenCV dep).

Save layout
-----------
When ``save_dir`` is provided, predictions are written as
``<save_dir>/<scene>/<phase>/<frame>.npz`` (key ``"depth"``, compressed
float32 in the model's native output space) — the same scene/phase shape as
the on-disk dataset, so downstream analytics and per-scene plotting share one
navigation pattern. Writes are atomic and compression is outside inference
timing.

Example
-------
::

    from rpx_benchmark.adapters import BatchedDepthBenchmarkModel
    from depth_models.zoedepth import ZoeDepth

    adapter = ZoeDepth(device="cuda", batch_size=8)
    model = BatchedDepthBenchmarkModel(
        adapter, name="ZoeDepth_NK",
        save_dir="rpx_results/ZoeDepth_NK/easy/predictions",
    )
    runner = BenchmarkRunner(model=model, dataset=dataset, ...)
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from PIL import Image

from ..api import DepthPrediction, Sample, TaskType

__all__ = ["BatchedDepthBenchmarkModel"]


def _resize_bilinear_2d(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2D float array to ``(H, W)``. PIL-only — no OpenCV dependency."""
    img = Image.fromarray(src.astype(np.float32), mode="F")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


class BatchedDepthBenchmarkModel:
    """Depth model wrapper that batches predictions across a sample list.

    Parameters
    ----------
    adapter
        Callable: ``adapter(rgb_list_or_single) -> depth_list_or_single``.
        Single-image: ``np.ndarray (H, W, 3) uint8`` → ``np.ndarray (H', W') float``.
        Batched: ``list[np.ndarray]`` → ``list[np.ndarray]``.
    name
        Display name surfaced in result.json and Box paths.
    save_dir
        If provided, writes per-frame ``.npz`` predictions under
        ``<save_dir>/<scene>/<phase>/<frame>.npz``. Created lazily on the
        first save (no side effect at construction time).
    native_alignment
        Declared by the adapter (``"none"`` for metric models like ZoeDepth /
        UniDepth V2 / Depth Pro; ``"ls_affine"`` for up-to-scale models
        like Marigold / Lotus-2). The runner reads this to apply the
        right alignment by default. Falls back to ``"none"`` if the
        adapter doesn't declare it. The wrapper returns raw predictions;
        the runner applies this mode once per complete scene/phase cell.
    native_precision
        Declared by the adapter (``"fp32"``, ``"fp16"``, ``"bf16"``).
        Used by the runner to record the OperatingPoint precision
        tag. Falls back to ``"fp32"`` if not declared.
    allow_nonpositive_predictions
        Permit finite, non-degenerate raw depth maps containing zero or
        negative values. This is opt-in for protocols that preserve raw model
        output and apply a documented evaluation-domain transform later.
    """

    task = TaskType.MONOCULAR_DEPTH

    def __init__(
        self,
        adapter: Any,
        *,
        name: str,
        save_dir: Optional[str | Path] = None,
        native_alignment: Optional[str] = None,
        native_precision: Optional[str] = None,
        resume_predictions: bool = False,
        allow_nonpositive_predictions: bool = False,
    ) -> None:
        self._adapter = adapter
        self.name = name
        self._save_dir = Path(save_dir) if save_dir else None
        if resume_predictions and self._save_dir is None:
            from ..exceptions import ConfigError

            raise ConfigError("resume_predictions requires save_dir")
        self.resume_predictions = bool(resume_predictions)
        self.allow_nonpositive_predictions = bool(allow_nonpositive_predictions)
        self.resume_stats = {
            "cache_hits": 0,
            "inferred_new": 0,
            "invalid_recomputed": 0,
        }
        self.last_cache_hits: list[bool] = []
        self._last_save_required: list[bool] = []
        # Read alignment off the adapter if it declared one, else fall
        # back to the constructor arg, else "none" (metric assumption).
        self.native_alignment: str = (
            native_alignment or getattr(adapter, "native_alignment", None) or "none"
        )
        self.native_precision: str = (
            native_precision or getattr(adapter, "native_precision", None) or "fp32"
        )
        self.depth_output_kind: str = "relative" if self.native_alignment != "none" else "metric"
        # Profiler walker reaches the underlying nn.Module via this attr.
        self.model = adapter

    # No-op: the adapter loads its own weights at construction.
    def setup(self) -> None:
        return None

    def predict(self, batch: Sequence[Sample]) -> List[DepthPrediction]:
        cached: dict[int, np.ndarray] = {}
        missing_indices: list[int] = []
        invalid_existing: set[int] = set()
        for index, sample in enumerate(batch):
            path = self._prediction_path(sample)
            if self.resume_predictions and path is not None and path.exists():
                depth = self._load_valid_prediction(
                    path,
                    np.asarray(sample.rgb).shape[:2],
                    require_positive=not self.allow_nonpositive_predictions,
                )
                if depth is not None:
                    cached[index] = depth
                    self.resume_stats["cache_hits"] += 1
                    continue
                invalid_existing.add(index)
            missing_indices.append(index)

        inferred: list[np.ndarray] = []
        if missing_indices:
            rgbs = [np.asarray(batch[i].rgb, dtype=np.uint8) for i in missing_indices]
            raw_depths = self._adapter(rgbs)
            if not isinstance(raw_depths, (list, tuple)):
                raw_depths = [raw_depths]
            inferred = [np.asarray(depth, dtype=np.float32) for depth in raw_depths]
        if len(inferred) != len(missing_indices):
            from ..exceptions import AdapterError

            raise AdapterError(
                f"adapter returned {len(inferred)} depths for {len(missing_indices)} inputs",
                hint="Check the adapter's batched contract: it must return "
                "one depth map per input RGB.",
            )

        inferred_by_index = dict(zip(missing_indices, inferred, strict=True))
        preds: List[DepthPrediction] = []
        self.last_cache_hits = []
        self._last_save_required = []
        for index, sample in enumerate(batch):
            is_cached = index in cached
            d_raw = cached[index] if is_cached else inferred_by_index[index]
            target_hw = np.asarray(sample.rgb).shape[:2]
            if d_raw.shape != target_hw:
                d_raw = _resize_bilinear_2d(d_raw, target_hw)
            if not is_cached:
                if index in invalid_existing:
                    self.resume_stats["invalid_recomputed"] += 1
                else:
                    self.resume_stats["inferred_new"] += 1

            # Return the raw prediction. Relative outputs are deliberately
            # aligned later by BenchmarkRunner, after every frame in the
            # same (scene, phase) cell is available for one pooled solve.
            preds.append(DepthPrediction(depth_map=d_raw))
            self.last_cache_hits.append(is_cached)
            self._last_save_required.append(not is_cached)
        return preds

    def persist_predictions(
        self,
        batch: Sequence[Sample],
        predictions: Sequence[DepthPrediction],
    ) -> None:
        """Persist newly inferred predictions after the runner stops timing."""
        pending = [
            (sample, prediction.depth_map)
            for sample, prediction, required in zip(
                batch,
                predictions,
                self._last_save_required,
                strict=True,
            )
            if required
        ]
        if len(pending) <= 1:
            for sample, depth in pending:
                self._maybe_save(sample, depth)
            return

        # Compression is CPU-bound and each frame has a distinct atomic target.
        # Parallelize it within the batch so a fast GPU is not followed by a
        # serial chain of np.savez_compressed calls.
        workers = min(len(pending), 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._maybe_save, sample, depth)
                for sample, depth in pending
            ]
            for future in futures:
                future.result()

    def _prediction_path(self, sample: Sample) -> Path | None:
        if self._save_dir is None:
            return None
        # Prefer the manifest's metadata block (set by local_manifest.py
        # and dataset_hub.split_manifests so the loader rides it through
        # to Sample.metadata). Fall back to id-parsing.
        meta = getattr(sample, "metadata", None) or {}
        scene = meta.get("scene_id")
        phase = meta.get("phase_idx")
        frame = meta.get("frame")
        if not (scene and frame is not None and phase is not None):
            try:
                scene, phase, frame = str(sample.id).split("__", 2)
            except ValueError:
                scene, phase, frame = "unknown", "0", str(sample.id)
        return self._save_dir / str(scene) / str(phase) / f"{frame}.npz"

    @staticmethod
    def _load_valid_prediction(
        path: Path,
        target_hw: tuple[int, int],
        *,
        require_positive: bool = True,
    ) -> np.ndarray | None:
        try:
            with np.load(path, allow_pickle=False) as payload:
                if payload.files != ["depth"]:
                    return None
                depth = np.asarray(payload["depth"])
            return BatchedDepthBenchmarkModel._validated_depth(
                depth,
                target_hw,
                require_positive=require_positive,
            )
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return None

    @staticmethod
    def _validated_depth(
        depth: np.ndarray,
        target_hw: tuple[int, int],
        *,
        require_positive: bool = True,
    ) -> np.ndarray | None:
        depth = np.asarray(depth)
        if depth.ndim != 2 or depth.shape != target_hw:
            return None
        if depth.dtype != np.dtype(np.float32):
            return None
        if not np.isfinite(depth).all():
            return None
        if require_positive and np.any(depth <= 0):
            return None
        if float(np.ptp(depth)) <= 1e-6:
            return None
        return depth

    def _maybe_save(self, sample: Sample, depth: np.ndarray) -> None:
        out = self._prediction_path(sample)
        if out is None:
            return
        canonical = np.asarray(depth, dtype=np.float32)
        target_hw = np.asarray(sample.rgb).shape[:2]
        if (
            self._validated_depth(
                canonical,
                target_hw,
                require_positive=not self.allow_nonpositive_predictions,
            )
            is None
        ):
            from ..exceptions import AdapterError

            raise AdapterError(
                f"Refusing to save an invalid depth prediction for sample {sample.id!r}: "
                f"expected a finite, non-degenerate float32 map with shape {target_hw}"
                + (
                    " and strictly positive values."
                    if not self.allow_nonpositive_predictions
                    else "."
                )
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                suffix=".npz.part",
                dir=out.parent,
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                np.savez_compressed(handle, depth=canonical)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, out)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
