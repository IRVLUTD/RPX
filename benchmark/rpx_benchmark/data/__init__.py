"""HuggingFace `datasets` integration for RPX.

This subpackage provides the canonical bridge between the RPX benchmark
and the `HuggingFace datasets library
<https://huggingface.co/docs/datasets/>`_. It complements
:mod:`rpx_benchmark.hub` (the snapshot-download path that works against
raw JSON manifests and the file tree on the Hub): where ``hub`` is
optimised for modality-aware bulk downloads, this module is optimised
for the "one-line load" UX most ML practitioners expect.

Three public entry points:

- :func:`rpx_benchmark.data.load_hf` — thin wrapper over
  ``datasets.load_dataset`` returning an :class:`RPXDataset` typed for
  the requested task.
- :class:`rpx_benchmark.data.hf_bridge.RPXHFBridge` — converts any
  pre-loaded ``datasets.Dataset`` row into a :class:`Sample`. Useful
  when users have their own dataset construction pipeline.
- :func:`rpx_benchmark.data.torch_dataloader.to_torch_dataloader` —
  PyTorch ``DataLoader`` adapter with a collate function that
  preserves the RPX sample contract.

Install
-------
``datasets`` is an optional dependency::

    pip install 'rpx-benchmark[hf-datasets]'

It lands under the ``[hf-datasets]`` extra rather than ``[hub]`` so
users who only want modality-aware ``huggingface_hub`` snapshot
downloads don't have to pull the ~100MB Arrow / PyArrow runtime.
"""

from __future__ import annotations

from .esd import (
    FEATURE_NAMES,
    PhaseFeatures,
    extract_phase_features,
    iter_phase_dirs,
)
from .features import RPX_FEATURES, features_for_task
from .hf_bridge import RPXHFBridge, row_to_sample
from .load import load_hf

__all__ = [
    "RPX_FEATURES",
    "features_for_task",
    "RPXHFBridge",
    "row_to_sample",
    "load_hf",
    # ESD feature extraction
    "FEATURE_NAMES",
    "PhaseFeatures",
    "extract_phase_features",
    "iter_phase_dirs",
]
