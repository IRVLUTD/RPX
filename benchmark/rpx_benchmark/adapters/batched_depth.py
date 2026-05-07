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
``<save_dir>/<scene>/<phase>/<frame>.npz`` (key ``"depth"``, float32
metres) — the same scene/phase shape as the on-disk dataset, so Box
mirroring, downstream analytics, and per-scene plotting all share one
navigation pattern.

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

from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from PIL import Image

from ..api import DepthPrediction, Sample, TaskType


__all__ = ["BatchedDepthBenchmarkModel"]


from ..metrics.depth_alignment import align_pred_to_gt as _align_pred_to_gt


def _resize_bilinear_2d(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2D float array to ``(H, W)``. PIL-only — no OpenCV dependency."""
    img = Image.fromarray(src.astype(np.float32), mode="F")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _extract_gt_depth(sample: Sample) -> Optional[np.ndarray]:
    """Pull GT depth (metres, float32) off a Sample, or None if missing.

    Sample.ground_truth is the task-specific GT dataclass; for monocular
    depth that's a ``DepthGroundTruth`` with a ``depth_map`` field.
    """
    gt_obj = getattr(sample, "ground_truth", None)
    if gt_obj is None:
        return None
    arr = getattr(gt_obj, "depth_map", None)
    if arr is None:
        arr = getattr(gt_obj, "depth", None)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32)


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
        adapter doesn't declare it.
    native_precision
        Declared by the adapter (``"fp32"``, ``"fp16"``, ``"bf16"``).
        Used by the runner to record the operating point for DRS
        computation. Falls back to ``"fp32"`` if not declared.
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
    ) -> None:
        self._adapter = adapter
        self.name = name
        self._save_dir = Path(save_dir) if save_dir else None
        # Read alignment off the adapter if it declared one, else fall
        # back to the constructor arg, else "none" (metric assumption).
        self.native_alignment: str = (
            native_alignment
            or getattr(adapter, "native_alignment", None)
            or "none"
        )
        self.native_precision: str = (
            native_precision
            or getattr(adapter, "native_precision", None)
            or "fp32"
        )
        # Profiler walker reaches the underlying nn.Module via this attr.
        self.model = adapter

    # No-op: the adapter loads its own weights at construction.
    def setup(self) -> None:
        return None

    def predict(self, batch: Sequence[Sample]) -> List[DepthPrediction]:
        rgbs = [np.asarray(s.rgb, dtype=np.uint8) for s in batch]
        depths = self._adapter(rgbs)                  # one batched forward
        if not isinstance(depths, (list, tuple)):
            depths = [depths]
        if len(depths) != len(batch):
            from ..exceptions import AdapterError
            raise AdapterError(
                f"adapter returned {len(depths)} depths for a batch of "
                f"{len(batch)}",
                hint="Check the adapter's batched contract: it must return "
                     "one depth map per input RGB.",
            )

        preds: List[DepthPrediction] = []
        for sample, depth in zip(batch, depths):
            d_raw = np.asarray(depth, dtype=np.float32)
            target_hw = np.asarray(sample.rgb).shape[:2]
            if d_raw.shape != target_hw:
                d_raw = _resize_bilinear_2d(d_raw, target_hw)

            # Persist the raw model output (preserves what the model
            # actually predicted) — same on disk regardless of alignment.
            self._maybe_save(sample, d_raw)

            # For the runner's primary metric (AbsRel et al), apply the
            # adapter's declared native alignment. Without this,
            # relative-depth models report nonsensical raw-space numbers
            # in result.json["aggregated"] (e.g. AbsRel ≈ 100+ for MiDaS)
            # while the comprehensive post-processor's aligned numbers
            # disagree wildly. This keeps result.json's headline numbers
            # honest under the model's declared alignment policy.
            d_for_runner = d_raw
            if self.native_alignment != "none":
                gt_arr = _extract_gt_depth(sample)
                if gt_arr is not None:
                    d_for_runner = _align_pred_to_gt(
                        d_raw, gt_arr, mode=self.native_alignment,
                    )
            preds.append(DepthPrediction(depth_map=d_for_runner))
        return preds

    def _maybe_save(self, sample: Sample, depth: np.ndarray) -> None:
        if self._save_dir is None:
            return
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
        out = self._save_dir / str(scene) / str(phase) / f"{frame}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        # np.savez_compressed appends '.npz' if the path doesn't end in
        # it, so we write directly to the final path (no .part rename
        # dance — that bug bit us earlier).
        np.savez_compressed(out, depth=depth.astype(np.float32))
