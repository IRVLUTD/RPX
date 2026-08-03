"""EdgeTAM video-object-segmentation adapter for the RPX D3 protocol."""

from __future__ import annotations

from .sam2_tracker import SAM2Tracker

EDGETAM_MODEL_ID = "facebook/EdgeTAM"
EDGETAM_MODEL_REVISION = "14d7ecc48c656b94e5184519f698cd5386c5a2bf"
EDGETAM_CHECKPOINT = "edgetam.pt"
EDGETAM_CONFIG = "configs/edgetam.yaml"


class EdgeTAMTracker(SAM2Tracker):
    """Run official EdgeTAM from ground-truth masks on the first frame."""

    model_name = "edgetam"
    model_id = EDGETAM_MODEL_ID
    model_revision = EDGETAM_MODEL_REVISION
    checkpoint_filename = EDGETAM_CHECKPOINT
    config_name = EDGETAM_CONFIG
    adapter_label = "EdgeTAM"
