"""Canonical, model-independent contract for RPX VQA benchmark rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..exceptions import ManifestError

SCHEMA_VERSION = "rpx-vqa-1.0"
BINARY_TYPES = frozenset({"spatial_lr_binary", "spatial_ud_binary"})
BBOX_TYPES = frozenset({"spatial_lr_extreme", "depth_closest", "spatial_farthest"})
ATTRIBUTE_TYPES = frozenset({"attr_composition"})
TASK_TYPES = BINARY_TYPES | BBOX_TYPES | ATTRIBUTE_TYPES


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

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "VQASample":
        bbox = row.get("answer_bbox")
        evidence = row.get("evidence")
        if isinstance(evidence, str):
            evidence = json.loads(evidence)
        sample = cls(
            sample_id=str(row.get("sample_id") or stable_sample_id(row)),
            scene_id=str(row["scene_id"]),
            kind=str(row["kind"]),
            phase=None if row.get("phase") is None else int(row["phase"]),
            frame=str(row["frame"]),
            image=dict(row.get("image") or image_locator(row)),
            img_w=int(row["img_w"]),
            img_h=int(row["img_h"]),
            question_type=str(row.get("question_type") or row["type"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            answer_bbox=None if bbox is None else tuple(int(v) for v in bbox),
            evidence=evidence,
            schema_version=str(row.get("schema_version", SCHEMA_VERSION)),
        )
        sample.validate()
        return sample

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported VQA schema_version: {self.schema_version}",
                hint=f"regenerate with schema {SCHEMA_VERSION}",
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
            x0, y0, x1, y1 = self.answer_bbox
            if not (0 <= x0 <= x1 < self.img_w and 0 <= y0 <= y1 < self.img_h):
                raise ManifestError(
                    f"invalid inclusive xyxy bbox: {self.answer_bbox}",
                    hint="keep all coordinates ordered and inside the original image",
                )
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
