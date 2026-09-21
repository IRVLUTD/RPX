"""Result serialisation: JSON + markdown summary for benchmark runs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

from .deployment import DeploymentReadinessReport
from .evaluators import BenchmarkResult


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj


def write_json(
    path: str | Path,
    *,
    task: str,
    model_name: str,
    split: str,
    repo_id: str,
    result: BenchmarkResult,
    dr_report: DeploymentReadinessReport | None = None,
    extra: Dict[str, Any] | None = None,
) -> Path:
    """Serialise a benchmark result + deployment report to JSON.

    Parameters
    ----------
    path : str or Path
        Output file path. Parent directories are created if missing.
    task : str
        Task name string (e.g. ``"monocular_depth"``).
    model_name : str
        Display name of the model under test.
    split : str
        ESD difficulty split (``"easy"``, ``"medium"``, ``"hard"``).
    repo_id : str
        HuggingFace dataset repo id the samples came from.
    result : BenchmarkResult
        Per-sample + aggregated metric container returned by
        :class:`~rpx_benchmark.runner.BenchmarkRunner`.
    dr_report : DeploymentReadinessReport, optional
        Weighted Phase Score, STR, TS, efficiency metadata. Omitted
        when ``None``.
    extra : dict, optional
        Arbitrary free-form extra fields to embed under the ``extra``
        key of the payload. Useful for run-specific metadata a caller
        wants preserved alongside the standard keys.

    Returns
    -------
    Path
        The resolved output path, for chaining.

    Notes
    -----
    Dataclasses are converted via :func:`dataclasses.asdict`, enums
    are lowered to their ``.value``, and unknown objects pass through
    unchanged. The output is pretty-printed with indent=2 for
    diff-friendliness.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "task": task,
        "model": model_name,
        "split": split,
        "repo_id": repo_id,
        "num_samples": result.num_samples,
        "aggregated": result.aggregated,
        "per_sample": result.per_sample,
    }
    if dr_report is not None:
        # Split the report into the three top-level RPX axes (no
        # composite score; see benchmark/README.md for the policy).
        dr_json = _to_jsonable(dr_report)
        payload["robustness"] = {
            "weighted_phase_score": dr_json.get("weighted_phase_score"),
            "temporal_stability":   dr_json.get("temporal_stability"),
            "state_transition":     dr_json.get("state_transition"),
            "geometric_coherence":  dr_json.get("geometric_coherence"),
        }
        payload["compute_cost"] = {
            "params_m":              dr_json.get("params_m"),
            "flops_g":               dr_json.get("flops_g"),
            "macs_g":                dr_json.get("macs_g"),
            "actmem_gb_fp16":        dr_json.get("actmem_gb_fp16"),
            "memory_traffic_gb":     dr_json.get("memory_traffic_gb"),
            "arithmetic_intensity":  dr_json.get("arithmetic_intensity"),
            "roofline":              dr_json.get("roofline"),
            "latency_ms_per_sample": dr_json.get("latency_ms_per_sample"),
            "peak_memory_mb":        dr_json.get("peak_memory_mb"),
            "system_card":           dr_json.get("system_card"),
            "operating_point":       dr_json.get("operating_point"),
        }
    if extra:
        payload["extra"] = extra
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)
    return path


def format_markdown_summary(
    *,
    task: str,
    model_name: str,
    split: str,
    repo_id: str,
    result: BenchmarkResult,
    dr_report: DeploymentReadinessReport | None = None,
) -> str:
    """Render a benchmark result as a human-readable markdown report.

    Parameters
    ----------
    task : str
    model_name : str
    split : str
    repo_id : str
    result : BenchmarkResult
    dr_report : DeploymentReadinessReport, optional
        When provided, the output includes the Weighted Phase Score
        table, State-Transition Robustness deltas, Temporal Stability
        score, and an Efficiency table (params, FLOPs, latency).

    Returns
    -------
    str
        A Markdown-formatted report. Matches the terminal UI tables
        the CLI prints so on-disk reports and terminal output stay in
        sync.

    Examples
    --------
    >>> from rpx_benchmark.reports import format_markdown_summary  # doctest: +SKIP
    >>> md = format_markdown_summary(                              # doctest: +SKIP
    ...     task="monocular_depth", model_name="depth_pro",
    ...     split="hard", repo_id="IRVLUTD/RPX",
    ...     result=result, dr_report=report,
    ... )
    """
    lines = [
        f"# RPX benchmark — {task}",
        "",
        f"- **Model:** `{model_name}`",
        f"- **Split:** `{split}`",
        f"- **Repo:** `{repo_id}`",
        f"- **Samples:** {result.num_samples}",
        "",
        "## Aggregated metrics",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k, v in result.aggregated.items():
        lines.append(f"| {k} | {v:.4f} |")

    if dr_report is not None and dr_report.weighted_phase_score is not None:
        wps = dr_report.weighted_phase_score
        lines += [
            "",
            "## Weighted Phase Score",
            "",
            "| phase | score |",
            "|---|---|",
            f"| clutter | {wps.s_clutter:.4f} |",
            f"| interaction | {wps.s_interaction:.4f} |",
            f"| clean | {wps.s_clean:.4f} |",
            f"| **overall** | **{wps.s_overall:.4f}** |",
            f"| Δ interaction (S_I − S_C) | {wps.delta_int:+.4f} |",
            f"| Δ recovery    (S_L − S_I) | {wps.delta_rec:+.4f} |",
        ]
        if dr_report.state_transition is not None:
            st = dr_report.state_transition
            lines += [
                "",
                f"- **STR C→I (interaction drop):** {st.str_c_to_i:+.4f}",
                f"- **STR I→L (recovery):**         {st.str_i_to_l:+.4f}",
            ]
        if dr_report.temporal_stability is not None:
            lines += [
                f"- **Temporal stability (TS):** {dr_report.temporal_stability.ts_score:.4f}",
            ]

    if dr_report is not None:
        eff_rows = []
        if dr_report.params_m is not None:
            eff_rows.append(("params (M)", f"{dr_report.params_m:.2f}"))
        if dr_report.flops_g is not None:
            eff_rows.append(("FLOPs (G)", f"{dr_report.flops_g:.2f}"))
        if dr_report.latency_ms_per_sample is not None:
            eff_rows.append(("latency (ms/sample)", f"{dr_report.latency_ms_per_sample:.1f}"))
        if dr_report.peak_memory_mb is not None:
            eff_rows.append(("peak memory (MB)", f"{dr_report.peak_memory_mb:.1f}"))
        if eff_rows:
            lines += ["", "## Efficiency", "", "| metric | value |", "|---|---|"]
            for k, v in eff_rows:
                lines.append(f"| {k} | {v} |")

    return "\n".join(lines) + "\n"
