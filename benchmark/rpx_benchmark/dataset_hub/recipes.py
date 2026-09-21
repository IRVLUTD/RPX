"""Task → modality recipes for selective dataset downloads.

A *recipe* answers the question "what subset of files does the dataset need
to materialise on disk so a given task can run?". The downloader takes the
recipe, expands it to a list of glob patterns, and hands them to
``huggingface_hub.snapshot_download(allow_patterns=...)``.

Two scene families have separate recipe tables because they have different
on-disk shapes:

* ``MULTI_OBJECT_TASK_RECIPES`` — scenes named ``scene<N>.<building>.<area>/``
  with three phases ``0/`` ``1/`` ``2/``. Used for the bulk of perception
  benchmarks (segmentation, depth, pose, tracking, ...).

* ``SINGLE_OBJECT_TASK_RECIPES`` — scenes named ``object<N>.<name>/`` with
  one collection ``0/`` (a 360° turntable view). Used for in-context tasks
  (object templates, few-shot reference views, retrieval).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, Optional

from ..exceptions import ConfigError

DEFAULT_REPO_ID = "anonymous/RPX"


class SceneType(str, Enum):
    """Top-level partition: which scene family a recipe targets."""

    MULTI_OBJECT = "multi_object"
    SINGLE_OBJECT = "single_object"
    EGO = "ego"


@dataclass(frozen=True)
class TaskRecipe:
    """Declarative spec: 'task X needs these modalities for inputs and labels'.

    The split between ``inputs`` and ``labels`` is purely a documentation
    aid — the downloader unions them when expanding to glob patterns.
    Storing them separately makes it obvious *why* each modality is in the
    recipe (which matters when modalities are added or renamed).
    """

    name: str
    scene_type: SceneType
    inputs: FrozenSet[str]
    labels: FrozenSet[str] = field(default_factory=frozenset)
    notes: str = ""

    def all_modalities(self) -> FrozenSet[str]:
        return self.inputs | self.labels


# --------------------------------------------------------------------- #
# Modality vocabulary (matches the on-disk capture layout under
# `test_dataset_aggregated/<scene>/<phase>/<modality>/`)
# --------------------------------------------------------------------- #

#: Raw modalities — captured once, never re-released.
RGB = "rgb"
DEPTH = "depth"
FISHEYE = "fisheye"
CAM_POSE = "cam_pose"

#: Label modalities — versioned (``labels/<name>/v1``, ``v2``, ...).
MASKS = "masks"  # essential SAM2 output (object masks)
MASKS_AUX = "masks_aux"  # opt-in viz/derived (bbox_overlay, dino_output, palette, ...)
QUESTIONNAIRE = "questionnaire"  # FewSOL-style per-scene/object text
VQA = "vqa"  # generated Q&A pairs (lands in v1.x)


# --------------------------------------------------------------------- #
# Recipe tables
# --------------------------------------------------------------------- #

# Multi-object scenes (3 phases each, ESD-stratified).
MULTI_OBJECT_TASK_RECIPES: Dict[str, TaskRecipe] = {
    "monocular_depth": TaskRecipe(
        name="monocular_depth",
        scene_type=SceneType.MULTI_OBJECT,
        inputs=frozenset({RGB}),
        labels=frozenset({DEPTH}),
        notes=(
            "Paper task Image Depth. Frame-level: model receives one RGB frame, "
            "outputs one depth map. Depth captured by D435 is the GT."
        ),
    ),
    "video_depth": TaskRecipe(
        name="video_depth",
        scene_type=SceneType.MULTI_OBJECT,
        # Same modalities as monocular_depth — Video Depth differs only in
        # iteration unit (per-(scene, phase) clip) and metric set
        # (adds temporal metrics on top of Image Depth's per-frame ones).
        # Downloading via `download_for_task("video_depth", ...)` pulls
        # the exact same tar shards as monocular_depth; the runner
        # decides how to feed them to the model.
        inputs=frozenset({RGB}),
        labels=frozenset({DEPTH}),
        notes=(
            "Paper task Video Depth. Video-level: model receives the full "
            "~250-frame phase clip as RGB sequence, outputs a per-frame "
            "depth sequence. Same scenes and GT as monocular_depth; "
            "differs in iteration unit and metric set (per-frame metrics "
            "+ temporal metrics — TODO: finalise from Feynman's research "
            "in benchmark/README.md)."
        ),
    ),
    "rgbd_segmentation": TaskRecipe(
        name="rgbd_segmentation",
        scene_type=SceneType.MULTI_OBJECT,
        inputs=frozenset({RGB, DEPTH}),
        labels=frozenset({MASKS}),
    ),
    "segmentation": TaskRecipe(
        name="segmentation",
        scene_type=SceneType.MULTI_OBJECT,
        inputs=frozenset({RGB}),
        labels=frozenset({MASKS}),
    ),
    "relative_pose": TaskRecipe(
        name="relative_pose",
        scene_type=SceneType.MULTI_OBJECT,
        inputs=frozenset({RGB}),
        labels=frozenset({CAM_POSE}),
    ),
    "rgbd_relative_pose": TaskRecipe(
        name="rgbd_relative_pose",
        scene_type=SceneType.MULTI_OBJECT,
        inputs=frozenset({RGB, DEPTH}),
        labels=frozenset({CAM_POSE}),
    ),
    "stereo_depth": TaskRecipe(
        name="stereo_depth",
        scene_type=SceneType.MULTI_OBJECT,
        inputs=frozenset({FISHEYE}),
        labels=frozenset({DEPTH}),
        notes="T265 stereo fisheye → depth. D435 depth is the GT after rectification.",
    ),
    "object_tracking": TaskRecipe(
        name="object_tracking",
        scene_type=SceneType.MULTI_OBJECT,
        inputs=frozenset({RGB}),
        labels=frozenset({MASKS}),
    ),
    "vqa": TaskRecipe(
        name="vqa",
        scene_type=SceneType.MULTI_OBJECT,
        inputs=frozenset({RGB}),
        labels=frozenset({VQA, QUESTIONNAIRE}),
        notes="Reserved slot — VQA labels land in v1.x.",
    ),
}

# Ego (egocentric/GoPro) scenes — one continuous capture per scene, no
# phase subdivision (mirrors the single-object shape: one collection "0").
# Same scene_id as the sibling MOS scene (e.g. "scene20.su.checkerboard"),
# so downloaders can pair an ego clip with its static-camera MOS phases.
# Only rgb + masks exist for ego today — no depth/fisheye/cam_pose.
EGO_TASK_RECIPES: Dict[str, TaskRecipe] = {
    "ego_segmentation": TaskRecipe(
        name="ego_segmentation",
        scene_type=SceneType.EGO,
        inputs=frozenset({RGB}),
        labels=frozenset({MASKS}),
        notes="Frame-level instance segmentation from the egocentric (head/hand-worn) viewpoint.",
    ),
    "ego_object_tracking": TaskRecipe(
        name="ego_object_tracking",
        scene_type=SceneType.EGO,
        inputs=frozenset({RGB}),
        labels=frozenset({MASKS}),
        notes=(
            "Same modalities as ego_segmentation — separate recipe because the "
            "task (per-frame masks vs. track-consistent masks across the clip) "
            "differs, matching the mos segmentation/object_tracking split."
        ),
    ),
    "ego_vqa": TaskRecipe(
        name="ego_vqa",
        scene_type=SceneType.EGO,
        inputs=frozenset({RGB}),
        labels=frozenset({VQA, QUESTIONNAIRE}),
        notes="Reserved slot, mirrors mos vqa — VQA labels land in v1.x.",
    ),
}

# Single-object scenes (1 collection, used for in-context / template tasks).
SINGLE_OBJECT_TASK_RECIPES: Dict[str, TaskRecipe] = {
    "object_templates": TaskRecipe(
        name="object_templates",
        scene_type=SceneType.SINGLE_OBJECT,
        inputs=frozenset({RGB}),
        labels=frozenset({MASKS}),
        notes="360° turntable RGB + mask — feed as reference views for in-context tasks.",
    ),
    "object_templates_rgbd": TaskRecipe(
        name="object_templates_rgbd",
        scene_type=SceneType.SINGLE_OBJECT,
        inputs=frozenset({RGB, DEPTH}),
        labels=frozenset({MASKS}),
    ),
    "object_pose_library": TaskRecipe(
        name="object_pose_library",
        scene_type=SceneType.SINGLE_OBJECT,
        inputs=frozenset({RGB, DEPTH}),
        labels=frozenset({CAM_POSE, MASKS}),
        notes="Per-object pose graph for retrieval / pose initialisation.",
    ),
}


def resolve_recipe(
    task: str,
    scene_type: Optional[SceneType] = None,
) -> TaskRecipe:
    """Look up a recipe by ``task`` name.

    If ``scene_type`` is given, only that table is searched (use this when a
    name collides between multi- and single-object tables). Otherwise we
    search multi-object first, then single-object, and raise on miss.
    """
    if scene_type is SceneType.MULTI_OBJECT:
        return MULTI_OBJECT_TASK_RECIPES[task]
    if scene_type is SceneType.SINGLE_OBJECT:
        return SINGLE_OBJECT_TASK_RECIPES[task]
    if scene_type is SceneType.EGO:
        return EGO_TASK_RECIPES[task]
    if task in MULTI_OBJECT_TASK_RECIPES:
        return MULTI_OBJECT_TASK_RECIPES[task]
    if task in SINGLE_OBJECT_TASK_RECIPES:
        return SINGLE_OBJECT_TASK_RECIPES[task]
    if task in EGO_TASK_RECIPES:
        return EGO_TASK_RECIPES[task]
    raise ConfigError(
        f"Unknown task {task!r}.",
        hint=(
            "Known multi-object recipes: "
            f"{sorted(MULTI_OBJECT_TASK_RECIPES)}. "
            "Known single-object recipes: "
            f"{sorted(SINGLE_OBJECT_TASK_RECIPES)}. "
            "Known ego recipes: "
            f"{sorted(EGO_TASK_RECIPES)}. "
            "Pass `scene_type=SceneType.SINGLE_OBJECT` if a name collides "
            "between tables."
        ),
    )


def all_recipe_names(scene_type: Optional[SceneType] = None) -> Iterable[str]:
    """Iterator over recipe names, optionally filtered by scene type."""
    if scene_type is None or scene_type is SceneType.MULTI_OBJECT:
        yield from MULTI_OBJECT_TASK_RECIPES
    if scene_type is None or scene_type is SceneType.SINGLE_OBJECT:
        yield from SINGLE_OBJECT_TASK_RECIPES
    if scene_type is None or scene_type is SceneType.EGO:
        yield from EGO_TASK_RECIPES
