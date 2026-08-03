from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from tracking_models import EdgeTAMTracker, SAM2Tracker, TRACKER_CLASSES  # noqa: E402


def test_tracking_registry_contains_production_adapters() -> None:
    assert TRACKER_CLASSES == {
        "edgetam": EdgeTAMTracker,
        "sam2": SAM2Tracker,
    }


def test_edgetam_uses_pinned_official_checkpoint_and_config() -> None:
    assert EdgeTAMTracker.model_id == "facebook/EdgeTAM"
    assert EdgeTAMTracker.model_revision == "14d7ecc48c656b94e5184519f698cd5386c5a2bf"
    assert EdgeTAMTracker.checkpoint_filename == "edgetam.pt"
    assert EdgeTAMTracker.config_name == "edgetam.yaml"


def test_sam2_metadata_is_unchanged() -> None:
    assert SAM2Tracker.model_name == "sam2"
    assert SAM2Tracker.checkpoint_filename == "sam2_hiera_large.pt"
    assert SAM2Tracker.config_name == "configs/sam2/sam2_hiera_l.yaml"
