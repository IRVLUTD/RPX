"""D4RT — paper roster entry; UNVERIFIED upstream weights.

The paper roster lists D4RT (CVPR'26). As of 2026-06-25 there is no
verified-official HF or GitHub release: the only candidate found on
HF is ``AlysonIrene/D4RT_checkpoint`` (author personal handle, no
README, no documentation linking it to the paper authors).

This adapter ships behind the
:class:`~scripts.video_depth_models._unverified.UnverifiedAdapterMixin`
safety rail: building succeeds, but ``setup()`` raises
:class:`UnverifiedAdapterError` unless the caller explicitly passes
``acknowledge_unverified=True``. Once the team confirms the candidate
weights are the paper's actual model (or replaces them with the
verified-official release), the safety rail can be removed.
"""

from __future__ import annotations

import numpy as np

from rpx_benchmark.api import VideoSample
from rpx_benchmark.exceptions import AdapterError

from ._unverified import UnverifiedAdapterMixin
from ._video_adapter_base import VideoDepthAdapterBase

_CANDIDATE_HF = "AlysonIrene/D4RT_checkpoint"


class D4RTAdapter(VideoDepthAdapterBase, UnverifiedAdapterMixin):
    """D4RT adapter behind the unverified-weights safety rail."""

    DISPLAY_NAME = "D4RT"
    OUTPUT_KIND = "metric"  # canonical roster says metric
    UNVERIFIED = True
    UNVERIFIED_NOTE = (
        "Candidate weights at huggingface.co/AlysonIrene/D4RT_checkpoint "
        "are an author personal upload with no README and no documented "
        "lineage to the CVPR'26 D4RT paper authors. Lineage must be "
        "verified before publishing benchmark numbers under this model name."
    )
    UNVERIFIED_CANDIDATE_HF = _CANDIDATE_HF

    def __init__(
        self,
        device: str = "cuda",
        *,
        acknowledge_unverified: bool = False,
    ) -> None:
        super().__init__(device=device)
        self._acknowledged = bool(acknowledge_unverified)
        self._model = None

    def setup(self) -> None:
        if self._loaded:
            return
        self._assert_acknowledged()
        raise NotImplementedError(
            "D4RTAdapter: safety rail cleared via acknowledge_unverified, "
            "but no upstream model class is wired yet. Locate the D4RT "
            "paper's official model code, import it here "
            "(e.g. `from d4rt.model import D4RT`), download weights via "
            f"`hf_hub_download(repo_id='{_CANDIDATE_HF}', filename=...)`, "
            "load the state_dict, and replace this raise with the model "
            "instantiation. Then drop the UnverifiedAdapterMixin once "
            "the paper-authors lineage is confirmed."
        )

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        raise AdapterError(
            "D4RTAdapter.predict() unreachable — setup() did not "
            "complete (no upstream model class wired). See setup() "
            "docstring for the team task."
        )


def build(device: str = "cuda", **kwargs):
    return D4RTAdapter(device=device, **kwargs)


__all__ = ["D4RTAdapter", "build"]
