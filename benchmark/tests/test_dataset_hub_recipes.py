"""Tests for ``rpx_benchmark.dataset_hub.recipes``."""

from __future__ import annotations

import pytest

from rpx_benchmark.dataset_hub import (
    MULTI_OBJECT_TASK_RECIPES,
    SINGLE_OBJECT_TASK_RECIPES,
    SceneType,
    resolve_recipe,
)
from rpx_benchmark.dataset_hub.recipes import (
    DEPTH,
    MASKS,
    RGB,
    all_recipe_names,
)
from rpx_benchmark.exceptions import ConfigError


def test_multi_object_recipes_have_expected_fields():
    rec = MULTI_OBJECT_TASK_RECIPES["segmentation"]
    assert rec.name == "segmentation"
    assert rec.scene_type is SceneType.MULTI_OBJECT
    assert RGB in rec.inputs
    assert MASKS in rec.labels


def test_single_object_recipes_isolated_from_multi():
    """Single-object recipes are *not* in the multi-object table."""
    multi_keys = set(MULTI_OBJECT_TASK_RECIPES)
    single_keys = set(SINGLE_OBJECT_TASK_RECIPES)
    assert multi_keys.isdisjoint(single_keys)


def test_resolve_finds_multi_first_then_single():
    assert resolve_recipe("segmentation").scene_type is SceneType.MULTI_OBJECT
    assert resolve_recipe("object_templates").scene_type is SceneType.SINGLE_OBJECT


def test_resolve_respects_explicit_scene_type():
    rec = resolve_recipe("object_templates", scene_type=SceneType.SINGLE_OBJECT)
    assert rec.scene_type is SceneType.SINGLE_OBJECT


def test_resolve_unknown_task_raises_with_helpful_message():
    with pytest.raises(ConfigError, match="Unknown task 'banana'") as exc:
        resolve_recipe("banana")
    assert "single-object" in exc.value.hint


def test_all_modalities_unions_inputs_and_labels():
    rec = MULTI_OBJECT_TASK_RECIPES["rgbd_segmentation"]
    assert rec.all_modalities() == {RGB, DEPTH, MASKS}


def test_recipes_only_reference_known_modalities():
    """Sanity: every modality in every recipe is from the known vocabulary."""
    known = {"rgb", "depth", "fisheye", "cam_pose", "masks", "masks_aux", "questionnaire", "vqa"}
    for table in (MULTI_OBJECT_TASK_RECIPES, SINGLE_OBJECT_TASK_RECIPES):
        for name, rec in table.items():
            unknown = rec.all_modalities() - known
            assert not unknown, f"{name} references unknown modalities {unknown}"


def test_all_recipe_names_filtering():
    multi_names = list(all_recipe_names(SceneType.MULTI_OBJECT))
    assert "segmentation" in multi_names
    assert "object_templates" not in multi_names

    single_names = list(all_recipe_names(SceneType.SINGLE_OBJECT))
    assert "object_templates" in single_names
    assert "segmentation" not in single_names

    ego_names = list(all_recipe_names(SceneType.EGO))
    assert "ego_segmentation" in ego_names
    assert "segmentation" not in ego_names

    everything = list(all_recipe_names())
    assert set(everything) == set(multi_names) | set(single_names) | set(ego_names)


def test_vqa_recipe_reserved_slot():
    """VQA recipe exists so the downloader can resolve it once labels land."""
    rec = MULTI_OBJECT_TASK_RECIPES["vqa"]
    assert "vqa" in rec.labels
    assert "v1.x" in rec.notes
