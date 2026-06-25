"""Sequence-shaped loader for video tasks (D1-V today; future video tasks).

Skeleton — wiring contract is final, metric integration and the actual
clip iterator are TODOs that land once Feynman's depth-metric study
(:file:`benchmark/docs/depth_metric_decisions.md`) finalises the
temporal metric tuple.

Design contract (already agreed with the user):

* Iteration unit is one ``(scene, phase)`` clip — yields
  :class:`~rpx_benchmark.api.VideoSample`, NOT per-frame
  :class:`~rpx_benchmark.api.Sample`.
* Cell-log key for the runner is ``(model, task, scene, phase)``;
  D1-V models produce a sequence prediction in one inference call, so
  per-frame keys would over-count.
* RGB/depth files are decoded through the canonical
  :mod:`~rpx_benchmark.decode_contracts` helpers — sequence-shaped
  arrays come out the other side stacked along axis 0.
* Memory budget per clip at D435 resolution: ~537 MB
  (250 × 640 × 480 × 3 B RGB + 250 × 640 × 480 × 4 B depth). Workable
  on a 64 GB host with one clip in flight per worker.
* Video models that emit affine-invariant (relative) depth get
  per-clip (not per-frame) scale-and-shift alignment before metric
  computation — aligning per-frame would defeat the temporal
  consistency metrics. Per-clip alignment is the canonical Ranftl
  et al. 2020 procedure for video depth evaluation.

What is NOT yet implemented (deliberately):

* The actual ``__iter__`` walk over the parquet manifest, grouping
  by ``(scene, phase)``. The contract is fixed but the
  implementation needs the manifest reader pattern from
  :class:`~rpx_benchmark.loader.RPXDataset` extended to consume
  grouped rows.
* Temporal metric computation. Waiting on Feynman's recommendation
  for the temporal metric tuple (TGM / TAE / OPW / TCC / scale-drift).
* Per-clip scale-and-shift solver for affine-invariant models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import numpy as np

from .api import TaskType, VideoSample
from .exceptions import ManifestError
from .logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class D1VDataset:
    """Iterates per ``(scene, phase)`` clip for the video-depth task.

    Yielded value is :class:`~rpx_benchmark.api.VideoSample` whose
    ``rgb_seq`` is a ``(T, H, W, 3) uint8`` stack and whose
    ``ground_truth`` is :class:`~rpx_benchmark.api.VideoDepthGroundTruth`
    with the same ``T``. ``T`` is whatever the phase actually has on
    disk — no padding, no truncation. Adapters must handle variable
    ``T`` (the alternative — fixed-length sub-clips — would break the
    temporal metrics by hiding any frame-drop boundary).

    Parameters
    ----------
    manifest_path : str or Path
        Path to a per-task per-split manifest JSON (the kind
        :func:`rpx_benchmark.dataset_hub.split_manifests.write_split_manifests`
        emits, but with rows grouped by ``(scene, phase)`` rather than
        per-frame). The manifest writer currently does NOT produce
        video-task manifests; that wiring is part of this work.
    """

    # The fields below mirror :class:`RPXDataset`'s shape so adapter
    # registration and the runner's iteration protocol can stay
    # uniform across single-frame and video tasks. The downstream
    # iteration just yields VideoSample instead of Sample.
    samples: List[dict]  # grouped manifest entries, one per (scene, phase)
    task: TaskType
    root: Path
    batch_size: int = 1  # always 1 for D1-V: clips are not stackable

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[List[VideoSample]]:
        # NOTE: not yet implemented. The intended shape is:
        #
        #   for group in self.samples:                     # one (scene, phase)
        #       rgb_seq = np.stack([safe_load_rgb(...) for ...])
        #       depth_seq = np.stack([safe_load_depth(...) / 1000.0 for ...])
        #       valid_seq = depth_seq > 0   # D435 sentinel
        #       gt = VideoDepthGroundTruth(
        #           depth_map_seq=depth_seq.astype(np.float32),
        #           valid_mask_seq=valid_seq,
        #           frame_indices=np.array(group["frame_indices"], dtype=np.int32),
        #       )
        #       sample = VideoSample(
        #           id=f"{group['scene_id']}_{group['phase']}",
        #           rgb_seq=rgb_seq,
        #           ground_truth=gt,
        #           metadata={...},
        #           phase=...,
        #           camera_pose_seq=...,
        #       )
        #       yield [sample]
        #
        # The blocker is the per-task per-split manifest format: the
        # writer in ``split_manifests.py`` currently emits flat-per-frame
        # JSONs. Either (a) extend the writer to emit
        # ``manifests/video_depth/<split>.json`` with rows grouped by
        # (scene, phase) and frame_indices arrays, or (b) regroup on
        # read here. We will go with (a) when the metric tuple lands.
        raise NotImplementedError(
            "D1VDataset.__iter__ is not yet wired. The contract is "
            "stable (yields lists of VideoSample, one per (scene, phase) "
            "clip), but implementation is deferred until "
            "benchmark/docs/depth_metric_decisions.md finalises the "
            "temporal metric tuple — that pins down which fields the "
            "clip must carry (e.g. whether we need ego_motion_seq for "
            "optical-flow-warped error)."
        )

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        batch_size: int = 1,
    ) -> "D1VDataset":
        """Build a D1VDataset from a per-task per-split JSON manifest.

        Not yet operational — depends on the writer producing a
        per-clip manifest shape. The shape this constructor expects is
        documented in the module docstring. Raises ``NotImplementedError``
        until both ends are wired.
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise ManifestError(
                f"Manifest file not found: {manifest_path}",
                hint=(
                    "D1VDataset requires a per-clip manifest produced by a "
                    "future version of write_split_manifests (the current "
                    "writer emits flat per-frame JSONs)."
                ),
            )
        raise NotImplementedError(
            "D1VDataset.from_manifest is not yet wired — see the open "
            "TODO in benchmark/docs/depth_metric_decisions.md."
        )


# --------------------------------------------------------------------------- #
# Per-clip scale-and-shift alignment (deferred)
# --------------------------------------------------------------------------- #


def align_scale_and_shift_per_clip(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
    valid_mask_seq: np.ndarray,
) -> np.ndarray:
    """Solve a single per-clip scale + shift for affine-invariant models.

    For D1-V, the alignment is solved *once per clip*, not per frame —
    aligning per frame would zero out the temporal-consistency signal
    the video metrics are designed to measure. See Ranftl et al. 2020
    for the closed-form least-squares solution; the per-clip variant
    pools every valid pixel across all frames.

    Not yet implemented. Signature is fixed so adapters can stub
    against it.
    """
    raise NotImplementedError(
        "align_scale_and_shift_per_clip is not yet implemented. "
        "The plan is the closed-form Ranftl 2020 least-squares solver "
        "pooled across (T, H, W) valid pixels, returning (s, t) such "
        "that ``s * pred + t`` matches gt in the L2 sense over the clip."
    )


__all__ = ["D1VDataset", "align_scale_and_shift_per_clip"]
