"""Authoritative SOS reference catalog + scene mask identity joins, sourced
from the reviewer-provided dataset manifests --
deliberately NOT the local single_objects/sos_wrapped/ scan lib.py uses for
the existing (non-in-context) generator.

Why a separate catalog module: the local sos_wrapped/ scan covers 220
objects, but only 70 of those are actually published under objects/<id>/...
on HF with real rgb.tar/masks tars (confirmed: the other 150 have bare
{object_id, fewsol_id, fewsol_dir} local metadata.json and no HF assets at
all). An in-context reference image (Image 1) can only ever be one of the 70
-- so the reference-selection pool here is intentionally the strict subset
manifest/object_catalog_v1.json defines, never the wider local catalog.

Identity fields, verbatim from manifest/object_catalog_v1.json's own
id_policy block:
  source_catalog_id  - original PDF catalog ID, STRING (decimal suffix =
                        selected instance, not a numeric code -- "63.2" must
                        never be coerced to a float)
  object_id           - actual SOS folder name under objects/ and objects_meta/
  global_object_id    - 1-based contiguous int over the 70-object subset
This is also, confirmed empirically, exactly the string every scene phase's
sam2/mask_to_object.json stores as object.id (loaded by
generate_spatial_gt.load_mapping() as mapping[mask_id]["oid"]) -- i.e.
source_catalog_id is the join key between a scene mask and this catalog,
NOT object_id (the folder-name form) and NOT global_object_id.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ATTR_FIELDS = ("name", "category", "material", "function", "color")

# Matched against the questionnaire.json question text (order matters --
# "made of"/"used for" must be checked before the generic "name"/"category"
# patterns since some question strings contain more than one keyword-ish
# substring).
_QUESTION_KEY_PATTERNS = [
    (re.compile(r"made of", re.I), "material"),
    (re.compile(r"used for", re.I), "function"),
    (re.compile(r"\bcolor\b", re.I), "color"),
    (re.compile(r"\bcategory\b", re.I), "category"),
    (re.compile(r"\bname\b", re.I), "name"),
]


def normalize_value(s: str) -> str:
    """The ONLY normalization applied anywhere in the in-context pipeline:
    lowercase, strip leading/trailing whitespace, collapse internal
    whitespace runs to a single space. Deliberately does NOT:
      - split on '/' (e.g. "plastic/metal" stays one atomic value, matching
        how the existing gt_attributes.py already treats it -- it keys
        owners on the literal list-item string, never splits it)
      - split or normalize '&'
      - strip plurals
      - map synonyms not already present verbatim in some object's own
        questionnaire
    Any of those would be a semantic judgment call this module has no
    authority to make silently; two values are only considered "the same
    fact" when their normalize_value() strings are equal, full stop."""
    return " ".join(s.strip().lower().split())


@dataclass(frozen=True)
class CatalogObject:
    global_object_id: int
    object_id: str
    source_catalog_id: str
    source_category_id: Optional[str]
    source_instance_id: Optional[str]
    object_name: str
    class_name: str
    sos_data_path: str
    questionnaire_path: str
    attrs_raw: dict   # field -> [raw strings, exactly as authored]
    attrs_norm: dict  # field -> [normalize_value(s) for s in attrs_raw[field]]


class AmbiguousMappingError(ValueError):
    pass


def _field_for_question(q: str) -> Optional[str]:
    for pat, name in _QUESTION_KEY_PATTERNS:
        if pat.search(q):
            return name
    return None


def _parse_questionnaire_json(d: dict) -> tuple[dict, dict]:
    raw = {f: [] for f in ATTR_FIELDS}
    norm = {f: [] for f in ATTR_FIELDS}
    for q_text, values in d.get("questions", {}).items():
        field_name = _field_for_question(q_text)
        if field_name is None:
            continue
        raw[field_name] = list(values)
        norm[field_name] = [normalize_value(v) for v in values]
    return raw, norm


def load_catalog(revision: str = "main", repo_id: Optional[str] = None) -> dict[str, CatalogObject]:
    """object_id -> CatalogObject, for exactly the officially published SOS
    objects (measured: 70). Downloads manifest/object_catalog_v1.json plus
    each object's objects_meta/<id>/questionnaire.json -- small JSON only
    (measured total <200KB); never touches objects/<id>/*.tar (the actual
    RGB/mask archives -- see sos_reference.py)."""
    from huggingface_hub import hf_hub_download

    repo_id = repo_id or os.environ.get("RPX_HF_REPO", "anonymous/RPX")
    catalog_path = hf_hub_download(repo_id, repo_type="dataset",
                                    filename="manifest/object_catalog_v1.json", revision=revision)
    catalog = json.loads(Path(catalog_path).read_text())
    assert catalog["schema_version"] == "v1", f"unexpected catalog schema_version: {catalog['schema_version']!r}"

    out: dict[str, CatalogObject] = {}
    for entry in catalog["objects"]:
        object_id = entry["object_id"]
        q_path = hf_hub_download(repo_id, repo_type="dataset",
                                  filename=entry["questionnaire_path"], revision=revision)
        q = json.loads(Path(q_path).read_text())
        raw, norm = _parse_questionnaire_json(q)
        if object_id in out:
            raise AmbiguousMappingError(f"duplicate object_id in official catalog: {object_id!r}")
        out[object_id] = CatalogObject(
            global_object_id=int(entry["global_object_id"]),
            object_id=object_id,
            source_catalog_id=str(entry["source_catalog_id"]),
            source_category_id=(str(entry["source_category_id"])
                                 if entry.get("source_category_id") is not None else None),
            source_instance_id=(str(entry["source_instance_id"])
                                 if entry.get("source_instance_id") is not None else None),
            object_name=entry["object_name"],
            class_name=entry["class_name"],
            sos_data_path=entry["sos_data_path"],
            questionnaire_path=entry["questionnaire_path"],
            attrs_raw=raw, attrs_norm=norm,
        )
    return out


def by_source_catalog_id(catalog: dict[str, CatalogObject]) -> dict[str, CatalogObject]:
    """source_catalog_id -> CatalogObject -- the actual join key from a
    scene's mask_to_object.json (see module docstring). Raises rather than
    silently picking one if the official catalog ever contained a duplicate
    source_catalog_id (would mean the catalog itself is broken)."""
    out: dict[str, CatalogObject] = {}
    for obj in catalog.values():
        if obj.source_catalog_id in out:
            raise AmbiguousMappingError(
                f"ambiguous source_catalog_id {obj.source_catalog_id!r}: "
                f"{out[obj.source_catalog_id].object_id} vs {obj.object_id}")
        out[obj.source_catalog_id] = obj
    return out


def load_mos_mask_map(revision: str = "main", repo_id: Optional[str] = None):
    """The officially published scene_id+phase+local_mask_id ->
    object_id/global_object_id/source_catalog_id join table for MOS, used
    here ONLY as an independent verification cross-check against the
    per-scene mask_to_object.json join every row's identity actually comes
    from (see join_scene_mask below) -- not as the generator's primary join
    path, so a missing/stale manifest never blocks generation, only
    verification. source_catalog_id is forced to string dtype: pandas'
    numeric type inference on a column that's mostly integer-looking would
    otherwise silently mangle "63.2"-style values."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo_id = repo_id or os.environ.get("RPX_HF_REPO", "anonymous/RPX")
    p = hf_hub_download(repo_id, repo_type="dataset",
                         filename="manifest/mos_mask_object_map_v1.parquet", revision=revision)
    df = pd.read_parquet(p)
    # pandas 3 infers ``StringDtype`` for ``astype(str)``.  Keep this join key
    # as ordinary Python strings in an object column so dotted catalogue IDs
    # retain the same public dataframe contract across pandas 2 and 3.
    df["source_catalog_id"] = df["source_catalog_id"].astype(str).astype(object)
    df["local_mask_id"] = df["local_mask_id"].astype(int)
    return df


def verify_against_mos_manifest(mos_map_df, scene_id: str, phase: int, local_mask_id: int,
                                 resolved_source_catalog_id: str) -> Optional[bool]:
    """None if the manifest has no row for this cell (can't verify, not a
    failure), True/False if it does and agrees/disagrees with the
    Path-A-resolved source_catalog_id. Raises AmbiguousMappingError if the
    manifest itself has more than one row for this exact cell."""
    phase_str = f"phase{phase}"
    rows = mos_map_df[(mos_map_df["scene_id"] == scene_id) &
                       (mos_map_df["phase"] == phase_str) &
                       (mos_map_df["local_mask_id"] == local_mask_id)]
    if len(rows) == 0:
        return None
    if len(rows) > 1:
        raise AmbiguousMappingError(f"{scene_id}/{phase_str}/mask{local_mask_id}: {len(rows)} manifest rows")
    return str(rows.iloc[0]["source_catalog_id"]) == resolved_source_catalog_id


@dataclass(frozen=True)
class Identity:
    """A resolved scene-mask identity. global_object_id/object_id are None
    when the local mask's source_catalog_id doesn't join to the official
    70-object catalog (a real, expected case for non-reference/target
    objects -- most scene objects are NOT part of the 70-object published
    subset; only the *reference* side requires catalog membership)."""
    local_mask_id: int
    source_catalog_id: str
    name: str
    object_id: Optional[str]
    global_object_id: Optional[int]


def join_scene_mask(mapping: dict, mask_id: int, catalog_by_scid: dict[str, CatalogObject]) -> Optional[Identity]:
    """mapping: generate_spatial_gt.load_mapping() output (mask_id ->
    {"name","oid"}), works identically for MOS and Ego since both are
    sourced from the same kind of per-phase sam2/mask_to_object.json."""
    if mask_id not in mapping:
        return None
    scid = mapping[mask_id]["oid"]
    cat_obj = catalog_by_scid.get(scid)
    return Identity(
        local_mask_id=mask_id,
        source_catalog_id=scid,
        name=mapping[mask_id]["name"],
        object_id=cat_obj.object_id if cat_obj else None,
        global_object_id=cat_obj.global_object_id if cat_obj else None,
    )
