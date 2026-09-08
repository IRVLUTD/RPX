"""Canonicalization for detecting semantic (not just literal-string)
duplicate facts at *sampling* time -- used only by the acceptance-manifest
and benchmark-plan builders to avoid spending two scarce slots on what is,
after normalization, the same underlying question. Mirrors the alias tables
data/mask_annotation/visual_grounding_gt/vqa_gt/gt_incontext.py applies
during ground-truth reference matching (``_canon_color``/``_canon_function``)
-- never used to alter a stored question, answer, or fact field."""

from __future__ import annotations

import re

_COLOR_ALIASES = {"gray": "grey"}
_FUNCTION_ALIASES = {"to drink": "drink", "to eat": "eat", "to snack": "snack"}


def normalize_value(value: str) -> str:
    return " ".join(value.strip().lower().split())


def canon_color(value: str | None) -> str | None:
    if value is None:
        return None
    norm = normalize_value(value)
    return _COLOR_ALIASES.get(norm, norm)


def canon_function(value: str | None) -> str | None:
    if value is None:
        return None
    norm = normalize_value(value)
    return _FUNCTION_ALIASES.get(norm, norm)


def canon_material(value: str | None) -> str | None:
    return None if value is None else normalize_value(value)


_COMPOSITION_RE = re.compile(r"made of (.+?) that is used for (.+?)\??$", re.IGNORECASE)
_ODD_ONE_OUT_RE = re.compile(r"NOT made of (.+?)\??$", re.IGNORECASE)


def parse_composition_question(question: str) -> tuple[str, str] | None:
    match = _COMPOSITION_RE.search(question)
    return None if match is None else (match.group(1).strip(), match.group(2).strip())


def parse_odd_one_out_question(question: str) -> str | None:
    match = _ODD_ONE_OUT_RE.search(question)
    return None if match is None else match.group(1).strip()


def semantic_key(row: dict) -> tuple:
    """A (scene, kind, phase, frame, type, target, reference, canon-value...)
    tuple such that two rows sharing a key are the same underlying question
    after alias normalization, even if their literal answer/question text
    spells the shared attribute value differently (e.g. "drink" vs
    "to drink"). Works for both normal rows (attr_value / question-derived
    material+function) and in-context rows (attribute_value / attribute_
    material / attribute_function columns)."""
    qtype = row["type"]
    target = row.get("target_source_catalog_id") or row.get("target_oid")
    reference = row.get("reference_source_catalog_id")
    base = (row["scene_id"], row["kind"], row.get("phase"), row["frame"], qtype, target, reference)
    if qtype in {"attr_single_color", "inctx_attr_single_color"}:
        return base + (canon_color(row.get("attribute_value") or row.get("attr_value")),)
    if qtype in {"attr_single_function", "inctx_attr_single_function"}:
        return base + (canon_function(row.get("attribute_value") or row.get("attr_value")),)
    if qtype in {"attr_single_material", "inctx_attr_single_material"}:
        return base + (canon_material(row.get("attribute_value") or row.get("attr_value")),)
    if qtype in {"attr_composition", "inctx_attr_composition"}:
        material = row.get("attribute_material")
        function = row.get("attribute_function")
        if material is None or function is None:
            parsed = parse_composition_question(row["question"])
            material, function = parsed if parsed else (material, function)
        return base + (canon_material(material), canon_function(function))
    if qtype in {"attr_odd_one_out", "inctx_attr_odd_one_out"}:
        material = row.get("attribute_material")
        if material is None:
            material = parse_odd_one_out_question(row["question"])
        return base + (canon_material(material),)
    # Spatial and any other type: no alias-prone attribute value, identity
    # is already the base tuple plus the raw answer.
    return base + (row.get("answer"),)
