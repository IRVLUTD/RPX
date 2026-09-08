"""Strict parsing and normalization of VQA model outputs."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

from ..exceptions import AdapterError, RPXError
from .contract import ATTRIBUTE_TYPES, BBOX_TYPES, BINARY_TYPES, VQASample

_LOC_RE = re.compile(
    r"<loc(?P<y0>\d{4})><loc(?P<x0>\d{4})><loc(?P<y1>\d{4})><loc(?P<x1>\d{4})>"
)


def normalize_label(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().lower().replace("_", " ")
    value = re.sub(r"[^\w\s-]", "", value)
    return " ".join(value.split())


@dataclass(frozen=True)
class ParsedOutput:
    valid: bool
    label: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    error: str | None = None
    coordinate_format: str | None = None


def _json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise AdapterError("missing JSON object", hint="return the requested bbox JSON only")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise AdapterError("output is not a JSON object", hint="return one JSON object")
    return value


def parse_output(sample: VQASample, raw: str, model_key: str) -> ParsedOutput:
    if sample.question_type in BINARY_TYPES:
        token = normalize_label(raw)
        if token not in {"yes", "no"}:
            return ParsedOutput(False, error="expected exactly yes or no")
        return ParsedOutput(True, label=token)
    if sample.question_type in ATTRIBUTE_TYPES:
        label = normalize_label(raw)
        if not label or len(label.split()) > 8:
            return ParsedOutput(False, error="invalid object-name answer")
        return ParsedOutput(True, label=label)
    if sample.question_type in BBOX_TYPES and model_key.startswith("paligemma2-"):
        match = _LOC_RE.search(raw)
        if not match:
            return ParsedOutput(False, error="missing PaliGemma loc tokens")
        vals = {key: int(value) for key, value in match.groupdict().items()}
        bbox = (
            vals["x0"] * sample.img_w / 1024,
            vals["y0"] * sample.img_h / 1024,
            vals["x1"] * sample.img_w / 1024,
            vals["y1"] * sample.img_h / 1024,
        )
        label = normalize_label(raw[match.end() :].split("<", 1)[0]) or None
        return ParsedOutput(True, label=label, bbox=bbox)
    if sample.question_type in BBOX_TYPES:
        try:
            value = _json_object(raw)
            raw_bbox = value["bbox"]
            # Some otherwise-correct VLM responses add one redundant list
            # dimension.  Unwrap exactly one singleton; do not guess any more
            # complicated structure.
            if (
                isinstance(raw_bbox, list)
                and len(raw_bbox) == 1
                and isinstance(raw_bbox[0], list)
            ):
                raw_bbox = raw_bbox[0]
            if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
                raise AdapterError(
                    "bbox must contain four numbers", hint="return normalized xyxy coordinates"
                )
            bbox = tuple(float(v) for v in raw_bbox)
            if not all(float("-inf") < value < float("inf") for value in bbox):
                raise AdapterError("bbox contains a non-finite number", hint="return finite pixels")
            if not all(0 <= coordinate <= 1000 for coordinate in bbox):
                raise AdapterError(
                    "bbox is outside normalized 0-1000 coordinates",
                    hint="return normalized xyxy coordinates",
                )
            # Decode the two widespread normalized conventions without using
            # model identity. Values containing a genuine fraction and entirely
            # within [0,1] are unit-normalized; everything else follows the
            # requested [0,1000] protocol. This fixes the previous silent bug
            # where [0.5,...] became a sub-pixel box at the top-left.
            unit_normalized = all(0 <= coordinate <= 1 for coordinate in bbox)
            denominator = 1 if unit_normalized else 1000
            coordinate_format = "normalized_0_1" if unit_normalized else "normalized_0_1000"
            bbox = (
                bbox[0] * (sample.img_w - 1) / denominator,
                bbox[1] * (sample.img_h - 1) / denominator,
                bbox[2] * (sample.img_w - 1) / denominator,
                bbox[3] * (sample.img_h - 1) / denominator,
            )
            x0, y0, x1, y1 = bbox
            if not (0 <= x0 <= x1 < sample.img_w and 0 <= y0 <= y1 < sample.img_h):
                raise AdapterError(
                    "bbox is outside the original image", hint="undo processor resizing first"
                )
            label = normalize_label(str(value.get("label", ""))) or None
            return ParsedOutput(
                True, label=label, bbox=bbox, coordinate_format=coordinate_format
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, RPXError) as exc:
            return ParsedOutput(False, error=str(exc))
    return ParsedOutput(False, error="unsupported task")
