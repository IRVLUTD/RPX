"""SOS (single-object-scan) catalog lookup: fewsol_id -> parsed attributes
(name/synonyms, category, material, function, color), read from each
object's local questionnaire.txt.

SOS_ROOT must point at a local checkout of the SOS catalog (the
`single_objects/sos_wrapped/<object>/questionnaire.txt` layout). Set it via
the --sos-root CLI flag on generate_gt.py, or the RPX_SOS_ROOT environment
variable -- there is no safe machine-independent default.
"""
import os
import re
from pathlib import Path

SOS_ROOT = Path(os.environ.get("RPX_SOS_ROOT", "")) if os.environ.get("RPX_SOS_ROOT") else None

_FIELD_RE = re.compile(
    r"\d+\.\s*What (?:is|can be) the (name|category|object.*made of|object.*used for|color)"
    r".*?\n(.*?)(?=\n\d+\.|\Z)",
    re.S,
)


def load_fewsol_lookup(sos_root: Path):
    """fewsol_id -> object folder path, scanning every object's metadata.json."""
    import json
    lookup = {}
    for d in sos_root.iterdir():
        if not d.is_dir():
            continue
        meta = d / "metadata.json"
        if not meta.exists():
            continue
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        fid = m.get("fewsol_id")
        if fid:
            lookup[fid] = str(d)
    return lookup


def _parse_questionnaire(text: str):
    fields = {"name": [], "category": [], "material": [], "function": [], "color": []}
    key_map = {
        "name": "name", "category": "category",
        "object.*made of": "material", "object.*used for": "function", "color": "color",
    }
    for m in _FIELD_RE.finditer(text):
        raw_key, raw_val = m.group(1), m.group(2)
        for pat, out_key in key_map.items():
            if re.match(pat, raw_key):
                vals = [v.strip() for v in raw_val.strip().split(",") if v.strip()]
                fields[out_key] = vals
                break
    return fields


_ATTRS_CACHE: dict = {}


def object_attrs(fewsol_id: str, lookup: dict):
    """Cached: a catalog object's questionnaire never changes within one
    run, but this gets called for the same ~220 catalog objects repeatedly
    -- every present object in every frame, plus up to distractor_scan
    catalog scans per frame for attr_synonym's "no" case and attr_absent.
    Without caching that's thousands of redundant file reads + regex
    parses of the same static files per scene."""
    if fewsol_id in _ATTRS_CACHE:
        return _ATTRS_CACHE[fewsol_id]
    folder = lookup.get(fewsol_id)
    if not folder:
        _ATTRS_CACHE[fewsol_id] = None
        return None
    q = Path(folder) / "questionnaire.txt"
    if not q.exists():
        _ATTRS_CACHE[fewsol_id] = None
        return None
    result = _parse_questionnaire(q.read_text())
    _ATTRS_CACHE[fewsol_id] = result
    return result
