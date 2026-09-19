"""Tests for :mod:`rpx_benchmark.schemas`.

Covers:

- Happy-path validation for every TaskType (one minimal sample each).
- Error surfacing: missing required fields → ``ManifestError`` with
  ``pydantic_errors`` in ``details``.
- JSON Schema export shape is stable enough to generate docs from.
- Opt-in validation wiring on :meth:`RPXDataset.from_dict`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic", minversion="2.5")

from rpx_benchmark.api import TaskType
from rpx_benchmark.exceptions import ManifestError
from rpx_benchmark.loader import RPXDataset
from rpx_benchmark.schemas import (
    SAMPLE_MODELS,
    Manifest,
    dump_json_schema,
    validate_manifest,
)

# Minimal representative sample entry per task. Only the fields the
# per-task Pydantic model marks as required are populated — anything
# else flows through ``extra='allow'``.
_MIN_SAMPLES = {
    TaskType.MONOCULAR_DEPTH: {"id": "s0", "rgb": "rgb/0.png", "depth": "depth/0.png"},
    TaskType.VIDEO_DEPTH: {
        "id": "s0",
        "rgb": "rgb/0.png",
        "rgb_seq": ["rgb/0.png", "rgb/1.png"],
        "depth_seq": ["depth/0.png", "depth/1.png"],
        "frame_indices": [0, 1],
    },
    TaskType.OBJECT_DETECTION: {"id": "s0", "rgb": "rgb/0.png", "boxes": "boxes/0.json"},
    TaskType.OPEN_VOCAB_DETECTION: {"id": "s0", "rgb": "rgb/0.png", "boxes": "boxes/0.json"},
    TaskType.OBJECT_SEGMENTATION: {"id": "s0", "rgb": "rgb/0.png", "mask": "mask/0.png"},
    TaskType.OBJECT_TRACKING: {"id": "s0", "rgb": "rgb/0.png", "tracks": "tracks/0.json"},
    TaskType.RELATIVE_CAMERA_POSE: {
        "id": "s0",
        "rgb": "rgb/0.png",
        "rgb_b": "rgb/1.png",
        "pose_a": "pose/0.npz",
        "pose_b": "pose/1.npz",
    },
    TaskType.VISUAL_GROUNDING: {"id": "s0", "rgb": "rgb/0.png", "text": "the red mug"},
    TaskType.SPARSE_DEPTH: {
        "id": "s0",
        "rgb": "rgb/0.png",
        "coordinates": [[1.0, 2.0]],
        "depths": [0.5],
    },
    TaskType.KEYPOINT_MATCHING: {
        "id": "s0",
        "rgb": "rgb/0.png",
        "rgb_b": "rgb/1.png",
        "points0": [[1.0, 2.0]],
        "points1": [[1.1, 2.1]],
    },
}


@pytest.mark.parametrize("task", list(TaskType))
def test_happy_path_all_tasks(task: TaskType) -> None:
    """Every TaskType parses its minimal sample without error."""
    m = Manifest.model_validate(
        {"task": task.value, "root": "/tmp", "samples": [_MIN_SAMPLES[task]]}
    )
    assert m.task == task
    assert len(m.samples) == 1
    assert m.samples[0]["id"] == "s0"


def test_missing_required_field_is_reported_per_sample() -> None:
    """Missing ``depth`` on a depth sample must surface as a ManifestError."""
    with pytest.raises(ManifestError) as excinfo:
        validate_manifest(
            {
                "task": "monocular_depth",
                "root": "/tmp",
                "samples": [{"id": "s0", "rgb": "rgb/0.png"}],
            }
        )
    errors = excinfo.value.details["pydantic_errors"]
    locations = [".".join(str(p) for p in err["loc"]) for err in errors]
    # Per-sample errors carry a full location path into the offending
    # field, e.g. "samples.0.depth".
    assert any(loc.endswith(".depth") for loc in locations), locations


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ManifestError):
        validate_manifest({"task": "not_a_task", "samples": []})


def test_dump_json_schema_generic() -> None:
    schema = dump_json_schema(None)
    assert schema["type"] == "object"
    assert "task" in schema["properties"]
    assert "samples" in schema["properties"]


@pytest.mark.parametrize("task", list(TaskType))
def test_dump_json_schema_per_task(task: TaskType) -> None:
    schema = dump_json_schema(task)
    assert schema["type"] == "object"
    # All per-task models inherit id/rgb from BaseSampleEntry.
    assert "id" in schema["properties"]
    assert "rgb" in schema["properties"]


def test_sample_models_cover_every_task() -> None:
    """Guardrail: adding a new TaskType must also add a schema."""
    missing = [t for t in TaskType if t not in SAMPLE_MODELS]
    assert not missing, f"Missing SAMPLE_MODELS entries for: {missing}"


def test_rpxdataset_validate_opt_in_accepts_good_manifest() -> None:
    """``RPXDataset.from_dict(..., validate=True)`` returns a dataset."""
    ds = RPXDataset.from_dict(
        {
            "task": "monocular_depth",
            "root": "/tmp",
            "samples": [_MIN_SAMPLES[TaskType.MONOCULAR_DEPTH]],
        },
        validate=True,
    )
    assert ds.task == TaskType.MONOCULAR_DEPTH
    assert len(ds.samples) == 1


def test_rpxdataset_validate_opt_in_rejects_bad_manifest() -> None:
    with pytest.raises(ManifestError):
        RPXDataset.from_dict(
            {
                "task": "monocular_depth",
                "root": "/tmp",
                # Missing required ``depth``.
                "samples": [{"id": "s0", "rgb": "rgb/0.png"}],
            },
            validate=True,
        )
