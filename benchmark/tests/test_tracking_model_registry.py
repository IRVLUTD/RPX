from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from tracking_models import (  # noqa: E402
    TRACKER_CLASSES,
    CutieTracker,
    EdgeTAMTracker,
    MOTIPTracker,
    SAM2LongTracker,
    SAM2PlusTracker,
    SAM2Tracker,
)


def test_tracking_registry_contains_production_adapters() -> None:
    assert TRACKER_CLASSES == {
        "cutie": CutieTracker,
        "edgetam": EdgeTAMTracker,
        "motip": MOTIPTracker,
        "sam2": SAM2Tracker,
        "sam2-plus": SAM2PlusTracker,
        "sam2long": SAM2LongTracker,
    }


def test_edgetam_uses_pinned_official_checkpoint_and_config() -> None:
    assert EdgeTAMTracker.model_id == "facebook/EdgeTAM"
    assert EdgeTAMTracker.model_revision == "14d7ecc48c656b94e5184519f698cd5386c5a2bf"
    assert EdgeTAMTracker.checkpoint_filename == "edgetam.pt"
    assert EdgeTAMTracker.config_name == "edgetam.yaml"
    assert EdgeTAMTracker.config_directory.endswith("/rpx-models/edgetam/sam2/configs")


def test_sam2_metadata_is_unchanged() -> None:
    assert SAM2Tracker.model_name == "sam2"
    assert SAM2Tracker.checkpoint_filename == "sam2_hiera_large.pt"
    assert SAM2Tracker.config_name == "configs/sam2/sam2_hiera_l.yaml"


def test_cutie_uses_official_base_mega_release() -> None:
    assert CutieTracker.model_id == "hkchengrex/Cutie"
    assert CutieTracker.model_revision == "v1.0"
    assert CutieTracker.checkpoint_filename == "cutie-base-mega.pth"
    assert CutieTracker.config_directory == "/opt/rpx-models/cutie/cutie/config"


def test_sam2long_uses_official_source_and_sam21_large_checkpoint() -> None:
    assert SAM2LongTracker.model_name == "sam2long"
    assert SAM2LongTracker.model_id == "facebook/sam2.1-hiera-large"
    assert SAM2LongTracker.model_revision == "665f8e2ad61cf5f53d65644ff27c8ee525124610"
    assert SAM2LongTracker.checkpoint_filename == "sam2.1_hiera_large.pt"
    assert SAM2LongTracker.config_name == "sam2.1_hiera_l.yaml"
    assert SAM2LongTracker.config_directory == ("/opt/rpx-models/sam2long/sam2/configs/sam2.1")


def test_sam2long_consolidates_independent_object_pathways() -> None:
    logits = torch.tensor(
        [
            [[2.0, -1.0], [0.1, -2.0]],
            [[-1.0, 3.0], [0.2, -3.0]],
        ]
    )
    result = SAM2LongTracker._combine_masks([7, 19], logits, (2, 2))
    np.testing.assert_array_equal(result, np.asarray([[7, 19], [19, 0]]))


def test_sam2_plus_uses_official_unified_checkpoint_and_source_config() -> None:
    assert SAM2PlusTracker.model_name == "sam2-plus"
    assert SAM2PlusTracker.model_id == "MCG-NJU/SAM2-Plus"
    assert SAM2PlusTracker.model_revision == "c3c534e30469d8788123287a484488567c5115d4"
    assert SAM2PlusTracker.checkpoint_filename == "checkpoint_phase123.pt"
    assert SAM2PlusTracker.prompt_type == "box"
    assert SAM2PlusTracker.config_name == (
        "sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml"
    )
    assert SAM2PlusTracker.config_directory == (
        "/opt/rpx-models/sam2_plus/sam2_plus/configs/sam2.1"
    )


def test_sam2_plus_consolidates_unified_decoder_masks() -> None:
    logits = torch.tensor(
        [
            [[[2.0, -1.0], [0.1, -2.0]]],
            [[[-1.0, 3.0], [0.2, -3.0]]],
        ]
    )
    result = SAM2PlusTracker._combine_masks([7, 19], logits, (2, 2))
    np.testing.assert_array_equal(result, np.asarray([[7, 19], [19, 0]]))


def test_sam2_plus_derives_tight_xyxy_boxes_from_first_frame_instances() -> None:
    mask = np.zeros((7, 9), dtype=np.int32)
    mask[1:5, 2:7] = 7
    mask[4:7, 0:2] = 19

    boxes = SAM2PlusTracker._object_boxes(mask, [7, 19])

    np.testing.assert_array_equal(boxes[7], np.asarray([2, 1, 6, 4], np.float32))
    np.testing.assert_array_equal(boxes[19], np.asarray([0, 4, 1, 6], np.float32))


def test_motip_uses_official_dancetrack_detector_checkpoint() -> None:
    assert MOTIPTracker.model_name == "motip"
    assert MOTIPTracker.model_id == "MCG-NJU/MOTIP"
    assert MOTIPTracker.model_revision == "v0.1"
    assert MOTIPTracker.checkpoint_filename == (
        "r50_deformable_detr_motip_dancetrack.pth"
    )
    assert MOTIPTracker.prompt_type == "detector"
    assert MOTIPTracker.tracking_mode == "native-joint-mot"


def test_motip_rasterizes_higher_confidence_boxes_last() -> None:
    boxes = np.asarray([[0.0, 0.0, 3.0, 3.0], [2.0, 2.0, 3.0, 3.0]])
    result = MOTIPTracker._rasterize_boxes(
        boxes,
        np.asarray([4, 9]),
        np.asarray([0.4, 0.9]),
        (5, 5),
    )
    assert result[0, 0] == 5
    assert result[2, 2] == 10
    assert result[4, 4] == 10
