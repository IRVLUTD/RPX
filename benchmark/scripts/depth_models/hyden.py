"""HyDen adapter — Meta's Hybrid Density-aware Depth (ICLR'26).

Two checkpoints share this adapter via ``model_id``:

* ``facebook/hyden-da2-metric-depth``    — metric (metres).
* ``facebook/hyden-da2-relative-depth``  — affine-invariant.

The metric variant declares ``native_alignment="none"``; the relative
variant declares ``native_alignment="ls_affine"`` (set by the registry
builders, not on the class itself).

Tracker reference: MonocularMetricDepth — HyDen-{metric, relative}.

Install
-------
    pip install transformers torch pillow

Notes
-----
HyDen ships under the standard ``DepthEstimation`` head, so the HF
auto-pipeline path works out of the box. We use ``HFDepthEstimationAdapter``
indirectly here; this thin class just pins the metric/relative
declarations so it's discoverable from the registry alongside the
other depth models.

Falls back to a clear AdapterError if the checkpoint hasn't shipped
yet (HyDen is a 2026 release; the public repo ids may shift).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .hf_pipeline import HFDepthEstimationAdapter


class HyDen(HFDepthEstimationAdapter):
    """HyDen depth: rgb → depth.

    Subclass of the generic HF-pipeline adapter — pins the model_id
    defaults but inherits the batched dispatch + path-resolution logic.
    """

    METRIC_MODEL_ID   = "facebook/hyden-da2-metric-depth"
    RELATIVE_MODEL_ID = "facebook/hyden-da2-relative-depth"

    def __init__(
        self,
        model_id: str = METRIC_MODEL_ID,
        device: str = "cuda",
        batch_size: int = 1,
        native_alignment: str = "none",
        native_precision: str = "fp16",
        dtype: Optional[str] = None,
    ) -> None:
        try:
            super().__init__(
                model_id=model_id,
                device=device,
                batch_size=batch_size,
                native_alignment=native_alignment,
                native_precision=native_precision,
                dtype=dtype,
            )
        except Exception as e:
            from rpx_benchmark.exceptions import AdapterError
            raise AdapterError(
                f"HyDen load failed for {model_id!r}: {e}",
                hint="HyDen is a 2026 release; the published HF repo id may "
                     "have shifted. Check https://huggingface.co/facebook for "
                     "the latest hyden-* checkpoint and pass `model_id=...`.",
            ) from e
