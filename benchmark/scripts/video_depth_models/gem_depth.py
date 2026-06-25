"""GemDepth — paper roster entry; UNVERIFIED upstream weights.

The paper roster lists GemDepth (ICML'26). As of 2026-06-25 there is
no verified-official HF or GitHub release: the only candidate found
on HF is ``YuechengLiu/GemDepth`` (author personal handle, empty
README, no documented lineage to the paper authors).

Ships behind the
:class:`~scripts.video_depth_models._unverified.UnverifiedAdapterMixin`
safety rail — same pattern as :mod:`d4rt`.
"""

from __future__ import annotations

import numpy as np

from rpx_benchmark.api import VideoSample
from rpx_benchmark.exceptions import AdapterError

from ._unverified import UnverifiedAdapterMixin
from ._video_adapter_base import VideoDepthAdapterBase


_CANDIDATE_HF = "YuechengLiu/GemDepth"


class GemDepthAdapter(VideoDepthAdapterBase, UnverifiedAdapterMixin):
    """GemDepth adapter behind the unverified-weights safety rail."""

    DISPLAY_NAME = "GemDepth"
    OUTPUT_KIND = "metric"
    UNVERIFIED = True
    UNVERIFIED_NOTE = (
        "Candidate weights at huggingface.co/YuechengLiu/GemDepth are an "
        "author personal upload with an empty README and no documented "
        "lineage to the ICML'26 GemDepth paper authors. Lineage must be "
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
            "GemDepthAdapter: safety rail cleared via "
            "acknowledge_unverified, but no upstream model class is "
            "wired yet. Locate the GemDepth paper's official model "
            "code, import it here, download weights via "
            f"`hf_hub_download(repo_id='{_CANDIDATE_HF}', filename=...)`, "
            "load the state_dict, and replace this raise with the model "
            "instantiation. Then drop the UnverifiedAdapterMixin once "
            "the paper-authors lineage is confirmed."
        )

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        raise AdapterError(
            "GemDepthAdapter.predict() unreachable — setup() did not "
            "complete (no upstream model class wired). See setup() "
            "docstring for the team task."
        )


def build(device: str = "cuda", **kwargs):
    return GemDepthAdapter(device=device, **kwargs)


__all__ = ["GemDepthAdapter", "build"]
