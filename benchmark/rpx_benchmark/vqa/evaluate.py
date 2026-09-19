"""Run any callable VQA model with the published RPX prompts and scoring."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from ..exceptions import AdapterError, ConfigError, ManifestError
from .contract import load_manifest
from .hub_rgb import fetch_images
from .metrics import score_predictions
from .outputs import ParsedOutput, parse_output
from .prompts import build_prompt


def evaluate_vqa(
    manifest_path: str | Path,
    predict: Callable[[tuple[Path, ...], str, int, str], str],
    *,
    model_name: str,
    output_dir: str | Path,
    image_cache: str | Path,
    model_revision: str,
) -> dict:
    """Evaluate a local model or API callable, without changing the model roster.

    ``predict(image_paths, prompt, max_new_tokens, output_kind)`` returns raw
    text. Only input images and the public prompt reach the callable; answers,
    evidence and target boxes remain inside the evaluator. Two-image requests
    preserve reference-then-target order and use the verified reference crop.

    The generic bbox contract is one JSON object with ``label`` and ``bbox``
    (XYXY, normalized 0–1000). Binary questions require ``yes`` or ``no``.
    Malformed outputs and inference exceptions score as failures and remain
    in the denominator. Dataset download errors abort rather than dropping rows.
    Use a fresh output directory for each run; no silent overwrites or resume.
    """
    if not model_name.strip() or not model_revision.strip():
        raise ConfigError("model_name and model_revision must identify the evaluated model")
    manifest_path = Path(manifest_path)
    samples = load_manifest(manifest_path)
    if not samples:
        raise ManifestError("VQA manifest contains no samples")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ConfigError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "model": model_name,
        "model_revision": model_revision,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "num_samples": len(samples),
        "protocol": "generic_single_call",
        "timing": "end_to_end_callable_wall_ms_including_API_transport",
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    scored = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            images = fetch_images(sample, image_cache)
            prompt = build_prompt(sample, "custom")
            started = time.perf_counter()
            raw = None
            error = None
            try:
                raw = predict(images, prompt.text, prompt.max_new_tokens, prompt.output_kind)
                if not isinstance(raw, str):
                    raise AdapterError("The VQA callable must return raw text")
                parsed = parse_output(sample, raw, "custom")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                parsed = ParsedOutput(False, error=error)
            elapsed_ms = (time.perf_counter() - started) * 1000
            scored.append((sample, parsed))
            row = {
                "sample_id": sample.sample_id,
                "raw_output": raw if isinstance(raw, str) else None,
                "parsed": asdict(parsed),
                "inference_error": error,
                "callable_wall_ms": elapsed_ms,
            }
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
    report = {**config, "metrics": score_predictions(scored)}
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
