"""Strict, typed manifest schemas for the RPX benchmark.

This module defines Pydantic v2 models that mirror the JSON manifest
format consumed by :class:`rpx_benchmark.loader.RPXDataset`. It is an
**additive, opt-in** validation layer: the loader's hand-rolled checks
remain the default path and continue to work without Pydantic
installed. Import this module when you want:

- Field-level validation errors (e.g. "samples[3].boxes: Field
  required") instead of the loader's coarser ``KeyError`` surface.
- JSON Schema export for third-party tooling or docs generation, via
  :func:`dump_json_schema`.
- Pre-upload validation in release automation that catches malformed
  shards before they reach the HuggingFace repo.

Stability
---------
The model classes here are part of the public API at the
``rpx_benchmark.schemas`` surface. Field additions are non-breaking;
renames or removals require a major version bump. ``TaskType`` and
the enum values re-used here are defined in :mod:`rpx_benchmark.api`.

Design
------
One :class:`Manifest` model holds the top-level fields (``task``,
``root``, ``samples``). Per-task sample schemas live in
:data:`SAMPLE_MODELS` and are dispatched on the manifest's
``task`` field by :func:`validate_manifest`. This keeps the model
class count small while still producing precise errors per task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Type, Union

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except ImportError as e:  # pragma: no cover — fail at import time with a clear hint
    raise ImportError(
        "rpx_benchmark.schemas requires pydantic>=2.5. "
        "Install with: pip install 'rpx-benchmark[schemas]'"
    ) from e

from .api import Difficulty, Phase, TaskType
from .exceptions import ManifestError

__all__ = [
    "BaseSampleEntry",
    "DepthSampleEntry",
    "DetectionSampleEntry",
    "SegmentationSampleEntry",
    "TrackingSampleEntry",
    "RelativePoseSampleEntry",
    "VisualGroundingSampleEntry",
    "SparseDepthSampleEntry",
    "NovelViewSynthesisSampleEntry",
    "KeypointMatchingSampleEntry",
    "Manifest",
    "SAMPLE_MODELS",
    "validate_manifest",
    "dump_json_schema",
]


# --------------------------------------------------------------------------- #
# Base configuration
# --------------------------------------------------------------------------- #


class _StrictBase(BaseModel):
    """Base for all manifest models.

    - ``extra='allow'`` keeps forward-compatibility: unknown keys on a
      sample do not fail validation (they are preserved in metadata by
      the loader).
    - ``str_strip_whitespace`` normalises pasted paths.
    - ``frozen=False`` — we allow downstream tooling to mutate fields
      (e.g. rewriting relative paths to absolute).
    """

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        populate_by_name=True,
    )


# --------------------------------------------------------------------------- #
# Sample schemas (one per task)
# --------------------------------------------------------------------------- #


class BaseSampleEntry(_StrictBase):
    """Fields common to every per-task sample entry."""

    id: str = Field(..., description="Unique sample id, typically '{scene}_{phase}_{frame}'.")
    rgb: str = Field(..., description="Relative or absolute path to the RGB image.")
    scene: str | None = None
    phase: Phase | int | None = Field(
        default=None,
        description="Capture phase label ('clutter'/'interaction'/'clean') or 0/1/2.",
    )
    difficulty: Difficulty | None = None
    pose: str | None = Field(
        default=None,
        description="Path to a T265 NPZ pose file (camera-to-world SE(3)).",
    )
    fisheye_left: str | None = None
    fisheye_right: str | None = None
    rgb_b: str | None = Field(
        default=None,
        description="Secondary RGB frame path (pair tasks: keypoint matching, pose).",
    )
    metadata: Dict[str, Any] | None = None


class DepthSampleEntry(BaseSampleEntry):
    depth: str = Field(..., description="Path to a 16-bit PNG depth map in millimetres.")


class VideoDepthSampleEntry(BaseSampleEntry):
    """Per-clip sample entry for the Video Depth task.

    Differs from :class:`DepthSampleEntry` in that ``rgb`` and ``depth``
    fields refer to entire frame sequences:

    * ``rgb`` carries the *first* frame's path (kept for compatibility
      with :class:`BaseSampleEntry`'s required field); the full clip
      is in ``rgb_seq``.
    * ``rgb_seq``, ``depth_seq`` are lists of relative-or-absolute
      paths in temporal order.
    * ``frame_indices`` records which original phase-relative indices
      these paths correspond to (for adapters that downsample, this
      lets the metric runner reconcile predictions to GT).
    """

    rgb_seq: List[str] = Field(
        ...,
        description="Ordered list of RGB frame paths for one (scene, phase) clip.",
    )
    depth_seq: List[str] = Field(
        ...,
        description="Ordered list of 16-bit PNG depth maps in millimetres.",
    )
    frame_indices: List[int] = Field(
        ...,
        description="Phase-relative integer indices for each frame in the clip.",
    )


class DetectionSampleEntry(BaseSampleEntry):
    boxes: str = Field(..., description="Path to a JSON list of {bbox: [x1,y1,x2,y2], label: str}.")


class SegmentationSampleEntry(BaseSampleEntry):
    mask: str = Field(..., description="Path to a single-channel PNG with per-pixel instance IDs.")


class TrackingSampleEntry(BaseSampleEntry):
    tracks: str = Field(..., description="Path to a JSON list of tracklets.")


class RelativePoseSampleEntry(BaseSampleEntry):
    pose_a: str
    pose_b: str
    rgb_b: str = Field(..., description="Secondary RGB frame (required for pose pairs).")


class VisualGroundingSampleEntry(BaseSampleEntry):
    text: str
    boxes: str | List[List[float]] | None = None
    labels: List[str] | None = None


class SparseDepthSampleEntry(BaseSampleEntry):
    coordinates: str | List[List[float]]
    depths: str | List[float]


class NovelViewSynthesisSampleEntry(BaseSampleEntry):
    target_rgb: str
    target_pose: str | None = None
    camera_pose: str | None = None


class KeypointMatchingSampleEntry(BaseSampleEntry):
    points0: str | List[List[float]]
    points1: str | List[List[float]]
    visibility: str | List[bool] | None = None
    rgb_b: str = Field(..., description="Secondary RGB frame (required for keypoint matching).")


#: Dispatch table: ``TaskType → SampleEntry`` model class.
SAMPLE_MODELS: Dict[TaskType, Type[BaseSampleEntry]] = {
    TaskType.MONOCULAR_DEPTH: DepthSampleEntry,
    TaskType.VIDEO_DEPTH: VideoDepthSampleEntry,
    TaskType.OBJECT_DETECTION: DetectionSampleEntry,
    TaskType.OPEN_VOCAB_DETECTION: DetectionSampleEntry,
    TaskType.OBJECT_SEGMENTATION: SegmentationSampleEntry,
    TaskType.OBJECT_TRACKING: TrackingSampleEntry,
    TaskType.RELATIVE_CAMERA_POSE: RelativePoseSampleEntry,
    TaskType.VISUAL_GROUNDING: VisualGroundingSampleEntry,
    TaskType.SPARSE_DEPTH: SparseDepthSampleEntry,
    TaskType.NOVEL_VIEW_SYNTHESIS: NovelViewSynthesisSampleEntry,
    TaskType.KEYPOINT_MATCHING: KeypointMatchingSampleEntry,
}


# --------------------------------------------------------------------------- #
# Top-level manifest
# --------------------------------------------------------------------------- #

AnySampleEntry = Union[
    DepthSampleEntry,
    DetectionSampleEntry,
    SegmentationSampleEntry,
    TrackingSampleEntry,
    RelativePoseSampleEntry,
    VisualGroundingSampleEntry,
    SparseDepthSampleEntry,
    NovelViewSynthesisSampleEntry,
    KeypointMatchingSampleEntry,
]


class Manifest(_StrictBase):
    """Typed manifest describing a slice of the RPX dataset.

    Attributes
    ----------
    task : TaskType
        Task this manifest targets. Determines which
        :class:`BaseSampleEntry` subclass each sample is validated
        against.
    root : str, optional
        Root the sample paths resolve against. Absent manifests must
        carry absolute paths in every sample.
    samples : list of SampleEntry
        Per-sample records. Parsed against the task-specific model in
        :data:`SAMPLE_MODELS`.
    scenes : list, optional
        Alternate representation enumerating ``(scene, phase)`` pairs.
        Preserved verbatim so hub download logic can read it without
        round-tripping through per-sample records.
    """

    task: TaskType
    root: str | None = None
    samples: List[Dict[str, Any]] = Field(default_factory=list)
    scenes: List[Dict[str, Any]] | None = None


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #


def validate_manifest(source: str | Path | Dict[str, Any]) -> Manifest:
    """Validate ``source`` against the :class:`Manifest` schema.

    Parameters
    ----------
    source : str, Path, or dict
        Path to a manifest JSON, or an already-parsed dict.

    Returns
    -------
    Manifest
        The validated model.

    Raises
    ------
    ManifestError
        If the JSON is malformed or fails any schema validation. The
        wrapped ``ValidationError`` message is surfaced in the
        exception's ``details`` dict for programmatic access.
    """
    import json

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ManifestError(f"Manifest file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ManifestError(f"Manifest at {path} is not valid JSON: {e}") from e
    else:
        data = source

    try:
        manifest = Manifest.model_validate(data)
    except ValidationError as e:
        raise ManifestError(
            "Manifest failed schema validation.",
            hint="Run `rpx validate-manifest <path>` to see field-level errors.",
            details={"pydantic_errors": e.errors()},
        ) from e

    # Per-sample validation against the task-specific schema. Kept
    # outside the model_validator so per-sample errors surface as a
    # single ManifestError with the full pydantic error list rather
    # than being collapsed into one ValueError.
    sample_model = SAMPLE_MODELS[manifest.task]
    sample_errors: list[Dict[str, Any]] = []
    rebuilt: List[Dict[str, Any]] = []
    for idx, raw in enumerate(manifest.samples):
        try:
            rebuilt.append(sample_model.model_validate(raw).model_dump(exclude_none=True))
        except ValidationError as e:
            for err in e.errors():
                # Prefix each error's location with the sample index so
                # downstream tooling can point at the offending entry.
                err = dict(err)
                err["loc"] = ("samples", idx, *err.get("loc", ()))
                sample_errors.append(err)
    if sample_errors:
        raise ManifestError(
            f"{len(sample_errors)} sample(s) failed {sample_model.__name__} validation.",
            hint="Each 'loc' entry points at the offending sample index and field.",
            details={"pydantic_errors": sample_errors},
        )
    manifest.samples = rebuilt
    return manifest


def dump_json_schema(task: TaskType | None = None) -> Dict[str, Any]:
    """Return the JSON Schema for a manifest (optionally task-scoped).

    When ``task`` is ``None``, returns the generic
    :class:`Manifest` schema. When a task is passed, returns the
    per-task :class:`BaseSampleEntry` schema — useful for emitting
    per-task documentation tables.

    Examples
    --------
    >>> from rpx_benchmark.schemas import dump_json_schema
    >>> from rpx_benchmark.api import TaskType
    >>> schema = dump_json_schema(TaskType.MONOCULAR_DEPTH)
    >>> "depth" in schema["properties"]
    True
    """
    if task is None:
        return Manifest.model_json_schema()
    return SAMPLE_MODELS[task].model_json_schema()
