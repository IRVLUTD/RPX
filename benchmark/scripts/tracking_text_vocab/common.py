"""Shared constants, catalog loading, and normalization for the tracking
text-initialization vocabulary. Self-contained: does not depend on any
local checkout of questionnaire data or on the (separate, VQA-only)
data/mask_annotation/visual_grounding_gt/vqa_gt/sos_catalog.py module --
every input is downloaded from the Hub at a pinned, immutable revision.

Three distinct immutable revisions are involved, each for a different
purpose (do not conflate them):

- CATALOG_REVISION: manifest/object_catalog_v1.json and each object's
  objects_meta/<id>/questionnaire.json (color/material/function/category/
  name). Resolved from the dataset's "main" branch, which is where this
  session's earlier in-context VQA work found complete, non-empty
  color/material/function questions for all 70 objects. The tracking-pinned
  revisions below predate those questions being authored -- their catalog
  copies have category/name only, so they must NOT be used as the
  attribute source.
- MOS_TRACKING_REVISION: scenes/<scene>/<phase>/labels/sam2_meta/v1.tar and
  .../labels/masks/v1.tar. Taken verbatim from
  benchmark/scripts/run_tracking.py's PINNED_DATASET_REVISION on the
  docker/tracking-all-yoloe branch -- the actual revision the D3 tracking
  benchmark runner already pins for MOS.
- EGO_TRACKING_REVISION: the ego equivalent, from the same file's
  PINNED_EGO_DATASET_REVISION.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ID = "IRVLUTD/RPX"

CATALOG_REVISION = "93e31d378f1f98eca18a7aa01a2279c9f332440c"
MOS_TRACKING_REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"
EGO_TRACKING_REVISION = "f082723002bad5800dd85e583115b4ea05734d31"

SCHEMA_VERSION = "tracking-text-vocab-1.0"

ATTR_FIELDS = ("name", "category", "material", "function", "color")

# Matched against questionnaire.json question text. Order matters: "made
# of"/"used for" must be checked before the generic "color"/"category"/
# "name" patterns since some question strings contain more than one
# keyword-ish substring. Identical to
# data/mask_annotation/visual_grounding_gt/vqa_gt/sos_catalog.py's
# _QUESTION_KEY_PATTERNS -- ported, not imported, to keep this package
# free of a cross-branch dependency.
_QUESTION_KEY_PATTERNS = [
    (re.compile(r"made of", re.I), "material"),
    (re.compile(r"used for", re.I), "function"),
    (re.compile(r"\bcolor\b", re.I), "color"),
    (re.compile(r"\bcategory\b", re.I), "category"),
    (re.compile(r"\bname\b", re.I), "name"),
]

_COLOR_ALIASES = {"gray": "grey"}

# A real, documented sentinel found in Ego mask_to_object.json entries:
# {"id": "unknown", "name": "unlabeled"} marks a tracked mask instance that
# was never assigned a real object identity. Confirmed empirically: 60 of
# 2,799 raw entries across all 400 scene-conditions, 100% kind=ego, 100%
# exactly this id/name pair, never seen in MOS, never any other unresolved
# id. Not a catalog gap -- excluded from scene_condition_vocab by design
# (there is no color/name to build a prompt from), reported, never silently
# dropped.
SENTINEL_SOURCE_CATALOG_IDS = frozenset({"unknown"})


def normalize_value(s: str) -> str:
    """Lowercase, strip, collapse internal whitespace. The only
    normalization ever applied to authored attribute strings."""
    return " ".join(s.strip().lower().split())


def canon_color(value: str) -> str:
    """normalize_value plus the one known spelling-variant canonicalization
    (gray -> grey), matching gt_incontext.py's _canon_color rule."""
    norm = normalize_value(value)
    return _COLOR_ALIASES.get(norm, norm)


def display_name_of(canonical_name: str) -> str:
    return canonical_name.strip().lower().replace("_", " ")


def _field_for_question(question_text: str) -> str | None:
    for pattern, name in _QUESTION_KEY_PATTERNS:
        if pattern.search(question_text):
            return name
    return None


def _parse_questionnaire(payload: dict) -> dict[str, list[str]]:
    raw: dict[str, list[str]] = {field_name: [] for field_name in ATTR_FIELDS}
    for question_text, values in payload.get("questions", {}).items():
        field_name = _field_for_question(question_text)
        if field_name is None:
            continue
        raw[field_name] = list(values)
    return raw


@dataclass(frozen=True)
class CatalogObject:
    source_catalog_id: str
    global_object_id: int
    object_id: str
    canonical_name: str
    display_name: str
    colors_raw: list[str] = field(default_factory=list)
    colors_normalized: list[str] = field(default_factory=list)
    materials_raw: list[str] = field(default_factory=list)
    functions_raw: list[str] = field(default_factory=list)
    categories_raw: list[str] = field(default_factory=list)
    primary_color: str | None = None
    prompt_text: str | None = None
    questionnaire_path: str = ""
    sos_data_path: str = ""

    def to_row(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_catalog_id": self.source_catalog_id,
            "global_object_id": self.global_object_id,
            "object_id": self.object_id,
            "canonical_name": self.canonical_name,
            "display_name": self.display_name,
            "colors_raw": list(self.colors_raw),
            "colors_normalized": list(self.colors_normalized),
            "primary_color": self.primary_color,
            "materials_raw": list(self.materials_raw),
            "functions_raw": list(self.functions_raw),
            "categories_raw": list(self.categories_raw),
            "prompt_text": self.prompt_text,
            "questionnaire_path": self.questionnaire_path,
            "sos_data_path": self.sos_data_path,
            "catalog_revision": CATALOG_REVISION,
        }


def load_catalog(revision: str = CATALOG_REVISION, repo_id: str = REPO_ID) -> list[CatalogObject]:
    """Download manifest/object_catalog_v1.json and every object's
    questionnaire.json (small JSON only, no image/tar assets), and build one
    CatalogObject per published identity. primary_color is always the FIRST
    entry of colors_raw, normalized+canonicalized -- deterministic by
    construction, never chosen by frequency or randomness."""
    from huggingface_hub import hf_hub_download

    catalog_path = hf_hub_download(
        repo_id, repo_type="dataset", filename="manifest/object_catalog_v1.json", revision=revision
    )
    catalog = json.loads(Path(catalog_path).read_text())
    if catalog.get("schema_version") != "v1":
        raise ValueError(f"unexpected catalog schema_version: {catalog.get('schema_version')!r}")

    objects: list[CatalogObject] = []
    seen_source_ids: set[str] = set()
    for entry in catalog["objects"]:
        source_catalog_id = str(entry["source_catalog_id"])
        if source_catalog_id in seen_source_ids:
            raise ValueError(f"duplicate source_catalog_id in official catalog: {source_catalog_id!r}")
        seen_source_ids.add(source_catalog_id)

        q_path = hf_hub_download(
            repo_id, repo_type="dataset", filename=entry["questionnaire_path"], revision=revision
        )
        questionnaire = json.loads(Path(q_path).read_text())
        raw = _parse_questionnaire(questionnaire)

        colors_raw = raw["color"]
        colors_normalized = [canon_color(v) for v in colors_raw]
        canonical_name = entry["object_name"]
        display_name = display_name_of(canonical_name)
        primary_color = colors_normalized[0] if colors_normalized else None
        prompt_text = f"{primary_color} {display_name}".strip() if primary_color else None

        objects.append(
            CatalogObject(
                source_catalog_id=source_catalog_id,
                global_object_id=int(entry["global_object_id"]),
                object_id=entry["object_id"],
                canonical_name=canonical_name,
                display_name=display_name,
                colors_raw=colors_raw,
                colors_normalized=colors_normalized,
                materials_raw=raw["material"],
                functions_raw=raw["function"],
                categories_raw=raw["category"],
                primary_color=primary_color,
                prompt_text=prompt_text,
                questionnaire_path=entry["questionnaire_path"],
                sos_data_path=entry.get("sos_data_path", ""),
            )
        )
    return objects


def by_source_catalog_id(objects: list[CatalogObject]) -> dict[str, CatalogObject]:
    out: dict[str, CatalogObject] = {}
    for obj in objects:
        if obj.source_catalog_id in out:
            raise ValueError(f"ambiguous source_catalog_id {obj.source_catalog_id!r} in catalog")
        out[obj.source_catalog_id] = obj
    return out


def sort_key_scene_condition(row: dict):
    """scene_id, kind, phase-null-last, mask_index, source_catalog_id."""
    phase = row.get("phase")
    phase_rank = (1, 0) if phase is None else (0, int(phase))
    return (row["scene_id"], row["kind"], phase_rank, int(row["mask_index"]), row["source_catalog_id"])
