"""EdgeTAM video-object-segmentation adapter for the RPX D3 protocol."""

from __future__ import annotations

import os
from pathlib import Path

from .sam2_tracker import SAM2Tracker

EDGETAM_MODEL_ID = "facebook/EdgeTAM"
EDGETAM_MODEL_REVISION = "14d7ecc48c656b94e5184519f698cd5386c5a2bf"
EDGETAM_CHECKPOINT = "edgetam.pt"
EDGETAM_CONFIG = "edgetam.yaml"
EDGETAM_CONFIG_DIR = "/opt/rpx-models/edgetam/sam2/configs"


class EdgeTAMTracker(SAM2Tracker):
    """Run official EdgeTAM from ground-truth masks on the first frame."""

    model_name = "edgetam"
    model_id = EDGETAM_MODEL_ID
    model_revision = EDGETAM_MODEL_REVISION
    checkpoint_filename = EDGETAM_CHECKPOINT
    config_name = EDGETAM_CONFIG
    config_directory = EDGETAM_CONFIG_DIR
    adapter_label = "EdgeTAM"

    def __init__(self, device: str = "cuda") -> None:
        # EdgeTAM's setup metadata omits the nested Hydra YAML files from the
        # installed wheel. The cumulative image retains the commit-pinned
        # source checkout, so make that authoritative config search root.
        config_dir = Path(
            os.environ.get("EDGETAM_CONFIG_DIR", self.config_directory)
        )
        config_path = config_dir / self.config_name
        if not config_path.is_file():
            raise RuntimeError(
                f"EdgeTAM config is missing: {config_path}. "
                "Use the pinned cumulative EdgeTAM Docker image."
            )

        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=str(config_dir.resolve()), version_base="1.2")
        super().__init__(device=device)
