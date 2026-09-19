"""Canonical, model-independent contract for RPX VQA benchmark rows.

Supports both one-image ("normal") and two-image ("in-context") samples.
Normal rows carry a single target image and are unchanged from the original
single-image contract, including the strict locator round-trip check in
``VQASample.validate``. In-context rows additionally carry an ordered
reference image (Image 1) ahead of the target image (Image 2); the answer
bbox always belongs to the target image's own ``img_w``/``img_h``, never the
reference's. In-context locators are taken verbatim from the in-context
Parquets (``target_image``/``reference_image``) rather than recomputed by
``image_locator`` -- those Parquets pin an immutable Hub revision that is
unrelated to the mutable ``main``/``ego-preview-v2`` revisions the normal
contract uses, and reconstructing them would silently disconnect a sample
from the exact asset it was generated against.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..exceptions import ManifestError

SCHEMA_VERSION = "rpx-vqa-1.0"

# Pinned immutable Hub revision baked into every in-context Parquet locator
# (data/mask_annotation/visual_grounding_gt/pilot/out/release_candidate_v3).
# Never replace this with "main" or any other mutable ref.
IN_CONTEXT_REVISION = "93e31d378f1f98eca18a7aa01a2279c9f332440c"

BINARY_TYPES = frozenset({"spatial_lr_binary", "spatial_ud_binary"})
GENERAL_BBOX_TYPES = frozenset(
    {
        "attr_single_color",
        "attr_single_material",
        "attr_single_function",
        "attr_composition",
        "attr_odd_one_out",
    }
)
SPATIAL_BBOX_TYPES = frozenset(
    {"spatial_lr_extreme", "depth_closest", "spatial_farthest"}
)
# In-context (two-image) types. Note attr_odd_one_out has no normal-contract
# bbox counterpart in GENERAL_BBOX_TYPES above (the single-image odd-one-out
# question is answered as an object name, not a bbox); the in-context task
# always answers with a bbox, so it is only ever in this in-context set.
INCONTEXT_GENERAL_BBOX_TYPES = frozenset(
    {
        "inctx_attr_single_color",
        "inctx_attr_single_material",
        "inctx_attr_single_function",
        "inctx_attr_composition",
        "inctx_attr_odd_one_out",
    }
)
INCONTEXT_SPATIAL_BBOX_TYPES = frozenset({"inctx_spatial_farthest"})
INCONTEXT_BBOX_TYPES = INCONTEXT_GENERAL_BBOX_TYPES | INCONTEXT_SPATIAL_BBOX_TYPES
BBOX_TYPES = GENERAL_BBOX_TYPES | SPATIAL_BBOX_TYPES | INCONTEXT_BBOX_TYPES
ATTRIBUTE_TYPES: frozenset[str] = frozenset()
TASK_TYPES = BINARY_TYPES | BBOX_TYPES | ATTRIBUTE_TYPES

_REQUIRED_LOCATOR_KEYS = frozenset({"repo_id", "revision", "shard", "member"})
_FLOAT_PATH_SEGMENT_RE = re.compile(r"/(\d+)\.0(?=/)")


def normalize_shard(shard: str) -> str:
    """Fix a known upstream defect: some in-context ``target_image`` shard
    paths serialize an integer MOS phase as a float path segment (e.g.
    ``scenes/scene001/0.0/rgb.tar``), which does not exist on the Hub (the
    real path is ``scenes/scene001/0/rgb.tar``). This is purely a URL-
    construction normalization -- the stored locator dict itself is never
    mutated, so provenance/hashing over the raw manifest is unaffected."""
    return _FLOAT_PATH_SEGMENT_RE.sub(r"/\1", shard)


def stable_sample_id(row: dict[str, Any]) -> str:
    """Return an ID stable across file moves and parquet rewrites."""
    identity = "\x1f".join(
        str(row.get(key, ""))
        for key in ("scene_id", "kind", "phase", "frame", "type", "question")
    )
    return "vqa_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def image_locator(row: dict[str, Any]) -> dict[str, str]:
    """Build a portable pointer to one RGB member in an uncompressed Hub tar."""
    kind = str(row["kind"])
    if kind == "ego":
        revision = "ego-preview-v2"
        partition = "ego"
    elif kind == "mos":
        revision = "main"
        partition = str(row["phase"])
    else:
        raise ManifestError(f"unsupported VQA capture kind: {kind!r}", hint="use 'mos' or 'ego'")
    scene = str(row["scene_id"])
    frame = str(row["frame"])
    return {
        "repo_id": "IRVLUTD/RPX",
        "revision": revision,
        "shard": f"scenes/{scene}/{partition}/rgb.tar",
        "member": f"rgb/{frame}.webp",
    }


def _bbox_tuple(values: Any) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ManifestError("Bounding boxes must contain exactly four coordinates")
    return int(values[0]), int(values[1]), int(values[2]), int(values[3])


@dataclass(frozen=True)
class VQASample:
    sample_id: str
    scene_id: str
    kind: str
    phase: int | None
    frame: str
    image: dict[str, str]
    img_w: int
    img_h: int
    question_type: str
    question: str
    answer: str
    answer_bbox: tuple[int, int, int, int] | None = None
    evidence: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION
    # In-context (two-image) fields. None on every normal row.
    reference_image: dict[str, str] | None = None
    reference_crop_bbox: tuple[int, int, int, int] | None = None
    reference_crop_sha256: str | None = None
    source_sample_id: str | None = None

    @property
    def is_in_context(self) -> bool:
        return self.question_type in INCONTEXT_BBOX_TYPES

    @property
    def images(self) -> tuple[dict[str, str], ...]:
        """Ordered locators to send to a model: [reference, target] for
        in-context samples, [target] alone for normal samples."""
        if self.is_in_context:
            assert self.reference_image is not None
            return (self.reference_image, self.image)
        return (self.image,)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "VQASample":
        bbox = row.get("answer_bbox")
        evidence = row.get("evidence")
        if isinstance(evidence, str):
            evidence = json.loads(evidence)
        question_type = str(row.get("question_type") or row["type"])
        in_context = question_type in INCONTEXT_BBOX_TYPES
        # In-context Parquet rows store their target/reference locators under
        # target_image/reference_image; a round-tripped manifest row (written
        # by write_manifest, reloaded by load_manifest) stores them under the
        # canonical image/reference_image keys. Accept either spelling.
        target_locator = row.get("image") or row.get("target_image")
        if target_locator is None:
            target_locator = image_locator(row)
        target_locator = {key: str(target_locator[key]) for key in _REQUIRED_LOCATOR_KEYS}
        reference_locator = row.get("reference_image")
        reference_crop_bbox = row.get("reference_crop_bbox")
        if reference_locator is not None:
            if reference_crop_bbox is None and "crop_bbox" in reference_locator:
                # The raw in-context Parquet's reference_image struct also
                # carries a redundant crop_bbox key; prefer the top-level
                # reference_crop_bbox column when present (it always is in
                # the current release), falling back to the nested copy.
                reference_crop_bbox = reference_locator["crop_bbox"]
            reference_locator = {key: str(reference_locator[key]) for key in _REQUIRED_LOCATOR_KEYS}
        raw_phase = row.get("phase")
        is_null_phase = raw_phase is None or (isinstance(raw_phase, float) and raw_phase != raw_phase)
        sample = cls(
            sample_id=str(row.get("sample_id") or stable_sample_id(row)),
            scene_id=str(row["scene_id"]),
            kind=str(row["kind"]),
            phase=None if raw_phase is None or is_null_phase else int(raw_phase),
            frame=str(row["frame"]),
            image=dict(target_locator),
            img_w=int(row["img_w"]),
            img_h=int(row["img_h"]),
            question_type=question_type,
            question=str(row["question"]),
            answer=str(row["answer"]),
            answer_bbox=None if bbox is None else _bbox_tuple(bbox),
            evidence=evidence,
            schema_version=str(row.get("schema_version", SCHEMA_VERSION)),
            reference_image=dict(reference_locator) if reference_locator is not None else None,
            reference_crop_bbox=(
                None if reference_crop_bbox is None else _bbox_tuple(reference_crop_bbox)
            ),
            reference_crop_sha256=(
                None
                if row.get("reference_crop_sha256") is None
                else str(row["reference_crop_sha256"])
            ),
            source_sample_id=(
                None if row.get("source_sample_id") is None else str(row["source_sample_id"])
            ),
        )
        if in_context and reference_locator is None:
            raise ManifestError(
                f"in-context VQA type {question_type!r} is missing reference_image",
                hint="regenerate from the in-context Parquet, not the single-image one",
            )
        sample.validate()
        return sample

    def validate(self) -> None:
        if self.schema_version not in {SCHEMA_VERSION, "incontext_v1"}:
            raise ManifestError(
                f"unsupported VQA schema_version: {self.schema_version}",
                hint=f"regenerate with schema {SCHEMA_VERSION} or incontext_v1",
            )
        if self.question_type not in TASK_TYPES:
            raise ManifestError(
                f"out-of-scope VQA type: {self.question_type}", hint="filter to the frozen task set"
            )
        if self.kind not in {"mos", "ego"}:
            raise ManifestError(f"invalid VQA kind: {self.kind}", hint="use 'mos' or 'ego'")
        if self.kind == "mos" and self.phase not in {0, 1, 2}:
            raise ManifestError("MOS sample has an invalid phase", hint="use phase 0, 1, or 2")
        if self.kind == "ego" and self.phase is not None:
            raise ManifestError("ego sample has a phase", hint="set phase to null for ego")
        if self.question_type in BINARY_TYPES and self.answer not in {"yes", "no"}:
            raise ManifestError("invalid binary answer", hint="use exactly 'yes' or 'no'")
        if self.question_type in BBOX_TYPES and self.answer_bbox is None:
            raise ManifestError("bbox task is missing answer_bbox", hint="regenerate its ground truth")
        if self.answer_bbox is not None:
            # Always checked against the TARGET image's own img_w/img_h --
            # for in-context rows that is Image 2, never Image 1 (reference).
            x0, y0, x1, y1 = self.answer_bbox
            if not (0 <= x0 <= x1 < self.img_w and 0 <= y0 <= y1 < self.img_h):
                raise ManifestError(
                    f"invalid inclusive xyxy bbox: {self.answer_bbox}",
                    hint="keep all coordinates ordered and inside the original image",
                )
        if self.is_in_context:
            self._validate_in_context_locators()
        else:
            expected = image_locator(
                {
                    "scene_id": self.scene_id,
                    "kind": self.kind,
                    "phase": self.phase,
                    "frame": self.frame,
                }
            )
            if self.image != expected:
                raise ManifestError(
                    f"non-canonical VQA image locator: {self.image!r}",
                    hint=f"use {expected!r}",
                )

    def _validate_in_context_locators(self) -> None:
        # In-context locators are taken verbatim from the in-context Parquet
        # (see module docstring) -- never recomputed via image_locator. What
        # IS enforced: both locators are structurally complete and pinned to
        # an immutable (non-"main"/non-branch-name) revision, and the
        # reference crop has the bbox+hash needed to reconstruct Image 1.
        if self.reference_image is None:
            raise ManifestError(
                "in-context sample is missing reference_image", hint="regenerate from the in-context Parquet"
            )
        for label, locator in (("target_image", self.image), ("reference_image", self.reference_image)):
            missing = _REQUIRED_LOCATOR_KEYS - locator.keys()
            if missing:
                raise ManifestError(
                    f"{label} locator is missing keys: {sorted(missing)}",
                    hint=f"expected keys {sorted(_REQUIRED_LOCATOR_KEYS)}",
                )
            revision = str(locator["revision"])
            if revision in {"main", "master"} or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ManifestError(
                    f"{label} is not pinned to an immutable revision: {revision!r}",
                    hint="use the exact revision stored in the in-context Parquet",
                )
        if self.reference_crop_bbox is None:
            raise ManifestError(
                "in-context sample is missing reference_crop_bbox",
                hint="regenerate from the in-context Parquet",
            )
        if self.reference_crop_sha256 is None or not re.fullmatch(
            r"[0-9a-f]{64}", self.reference_crop_sha256
        ):
            raise ManifestError(
                f"invalid reference_crop_sha256: {self.reference_crop_sha256!r}",
                hint="expected a 64-char lowercase hex sha256",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "scene_id": self.scene_id,
            "kind": self.kind,
            "phase": self.phase,
            "frame": self.frame,
            "image": self.image,
            "img_w": self.img_w,
            "img_h": self.img_h,
            "question_type": self.question_type,
            "question": self.question,
            "answer": self.answer,
            "answer_bbox": list(self.answer_bbox) if self.answer_bbox is not None else None,
            "bbox_format": "xyxy_pixel_inclusive" if self.answer_bbox is not None else None,
            "evidence": self.evidence,
            "reference_image": self.reference_image,
            "reference_crop_bbox": (
                list(self.reference_crop_bbox) if self.reference_crop_bbox is not None else None
            ),
            "reference_crop_sha256": self.reference_crop_sha256,
            "source_sample_id": self.source_sample_id,
        }


def load_manifest(path: str | Path) -> list[VQASample]:
    records: list[VQASample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(VQASample.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ManifestError(
                    f"invalid VQA manifest row at {path}:{line_number}: {exc}",
                    hint="regenerate the manifest from the source parquets",
                ) from exc
    if not records:
        raise ManifestError(f"empty VQA manifest: {path}", hint="provide at least one sample")
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise ManifestError(
            "duplicate sample_id in VQA manifest", hint="deduplicate source questions"
        )
    return records


def write_manifest(samples: Iterable[VQASample], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
