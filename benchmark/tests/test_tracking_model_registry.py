from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from tracking_models import (  # noqa: E402
    TRACKER_CLASSES,
    CutieTracker,
    EdgeTAMTracker,
    MITSTracker,
    OVTRTracker,
    SAM2LongTracker,
    SAM2PlusTracker,
    SAM2Tracker,
    XMemTracker,
)
from tracking_models.masa_tracker import MASATracker  # noqa: E402
from tracking_models.motip_tracker import MOTIPTracker  # noqa: E402


def test_tracking_registry_contains_production_adapters() -> None:
    assert TRACKER_CLASSES == {
        "cutie": CutieTracker,
        "edgetam": EdgeTAMTracker,
        "mits": MITSTracker,
        "ovtr": OVTRTracker,
        "sam2": SAM2Tracker,
        "sam2-plus": SAM2PlusTracker,
        "sam2long": SAM2LongTracker,
        "xmem": XMemTracker,
    }


def test_tracking_registry_declares_initialization_protocols() -> None:
    assert {name: tracker.prompt_type for name, tracker in TRACKER_CLASSES.items()} == {
        "cutie": "mask",
        "edgetam": "mask",
        "mits": "box",
        "ovtr": "detector",
        "sam2": "mask",
        "sam2-plus": "box",
        "sam2long": "mask",
        "xmem": "mask",
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


def test_mits_uses_official_full_checkpoint_and_box_protocol() -> None:
    assert MITSTracker.model_name == "mits"
    assert MITSTracker.model_id == "yoxu515/MITS"
    assert MITSTracker.source_revision == "462ebee2c995818998d5f96ab6b615dd86c42688"
    assert MITSTracker.model_revision == "gdrive-1Db9DxXc-gyRkxhKs0AMXJ2RH6DyHWCfq"
    assert MITSTracker.checkpoint_filename == "mits.pth"
    assert MITSTracker.prompt_type == "box"
    assert MITSTracker.tracking_mode == "multi-object-box-initialized-vos"


def test_mits_rasterizes_tight_boxes_and_preserves_original_id_mapping() -> None:
    mask = np.zeros((8, 10), dtype=np.int32)
    mask[1:7, 1:9] = 7
    mask[3:5, 4:6] = 19

    prompt, local_by_original = MITSTracker._box_prompt(mask, [7, 19])

    assert local_by_original == {7: 1, 19: 2}
    assert prompt[1, 1] == 1
    assert prompt[3, 4] == 2
    assert prompt[4, 5] == 2
    assert prompt[0, 0] == 0


def test_mits_passes_the_complete_augmentation_list_to_multi_to_tensor(
    tmp_path, monkeypatch
) -> None:
    class DummyTensor:
        def unsqueeze(self, dimension):
            assert dimension == 0
            return self

        def cuda(self, non_blocking):
            assert non_blocking is True
            return self

    class FakeRestrictSize:
        def __init__(self, *args):
            pass

        def __call__(self, sample):
            return [sample]

    class FakeToTensor:
        def __call__(self, samples):
            assert isinstance(samples, list)
            samples[0]["current_img"] = DummyTensor()
            return samples

    package = ModuleType("dataloaders")
    transforms = ModuleType("dataloaders.video_transforms")
    transforms.MultiRestrictSize = FakeRestrictSize
    transforms.MultiToTensor = FakeToTensor
    monkeypatch.setitem(sys.modules, "dataloaders", package)
    monkeypatch.setitem(sys.modules, "dataloaders.video_transforms", transforms)

    image_path = tmp_path / "00000.jpg"
    Image.new("RGB", (8, 6)).save(image_path)
    tracker = MITSTracker.__new__(MITSTracker)
    tracker.cfg = SimpleNamespace(
        TEST_MAX_SHORT_EDGE=None,
        TEST_MAX_LONG_EDGE=1040,
        MODEL_ALIGN_CORNERS=True,
    )

    assert isinstance(tracker._transform_image(image_path), DummyTensor)


def test_xmem_uses_official_v1_release_and_mask_protocol() -> None:
    assert XMemTracker.model_name == "xmem"
    assert XMemTracker.model_id == "hkchengrex/XMem"
    assert XMemTracker.source_revision == "f3b841d50df058910bbf690229ddc15fb1aef7d6"
    assert XMemTracker.model_revision == "v1.0"
    assert XMemTracker.checkpoint_filename == "XMem.pth"
    assert XMemTracker.prompt_type == "mask"
    assert XMemTracker.tracking_mode == "multi-object-long-term-memory-vos"


def test_xmem_uses_official_short_side_480_resize() -> None:
    assert XMemTracker._target_size(480, 640) == (480, 640)
    assert XMemTracker._target_size(720, 1280) == (480, 853)
    assert XMemTracker._target_size(640, 480) == (640, 480)


def test_ovtr_uses_official_full_open_vocabulary_model() -> None:
    assert OVTRTracker.model_name == "ovtr"
    assert OVTRTracker.model_id == "jinyanglii/OVTR"
    assert OVTRTracker.source_revision == "500e72c19bf5f7f8717546911a5639fdc26bfee5"
    assert OVTRTracker.model_revision == "gdrive-10GKAIBxAseTiXnJXV1MnxnJBTmOHVFh5"
    assert OVTRTracker.checkpoint_filename == "ovtr_5_frame.pth"
    assert OVTRTracker.prompt_type == "detector"
    assert OVTRTracker.tracking_mode == "native-open-vocabulary-mot"
    assert OVTRTracker.vocabulary_size == 1203


def test_ovtr_rasterizes_higher_confidence_open_vocab_boxes_last() -> None:
    boxes = np.asarray([[0.0, 0.0, 3.0, 3.0], [2.0, 2.0, 5.0, 5.0]])
    result = OVTRTracker._rasterize_boxes(
        boxes,
        np.asarray([4, 9]),
        np.asarray([0.4, 0.9]),
        (5, 5),
    )
    assert result[0, 0] == 5
    assert result[2, 2] == 10
    assert result[4, 4] == 10


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


def test_masa_uses_pinned_unified_detic_checkpoint() -> None:
    assert MASATracker.model_name == "masa"
    assert MASATracker.model_id == "dereksiyuanli/masa"
    assert MASATracker.model_revision == (
        "25ed372c47f2c46cf36fd446d1b657b656bc7ea9"
    )
    assert MASATracker.checkpoint_filename == "detic_masa.pth"
    assert MASATracker.prompt_type == "detector"
    assert MASATracker.tracking_mode == "detector-association-mot"


def test_masa_rasterizes_xyxy_boxes_and_applies_score_threshold() -> None:
    boxes = np.asarray([[0.0, 0.0, 3.0, 3.0], [2.0, 2.0, 5.0, 5.0]])
    result = MASATracker._rasterize_boxes(
        boxes,
        np.asarray([4, 9]),
        np.asarray([0.1, 0.9]),
        (5, 5),
    )
    assert result[0, 0] == 0
    assert result[2, 2] == 10
    assert result[4, 4] == 10
