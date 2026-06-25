"""Shared pipeline: download -> load -> run -> report.

Every task runner calls :func:`run_pipeline`. The task module only
owns the task-specific bits: a config subclass, a ``TASK_SPEC``, and
a one-line ``run_<task>`` wrapper around :func:`run_pipeline`.

The toolkit does not ship models. Users supply a
:class:`BenchmarkableModel` (usually built via
``rpx.make_numpy_<task>_model(fn)`` or a custom ``InputAdapter`` /
``OutputAdapter`` pair) and set it on the config's ``model`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..adapters import BenchmarkableModel
from ..api import Difficulty, TaskType
from ..cell_log import cells_from_per_sample, write_cells
from ..deployment import DeploymentReadinessReport
from ..exceptions import ConfigError
from ..hub import DEFAULT_REPO_ID, download_split
from ..loader import RPXDataset
from ..logging_utils import get_logger
from ..metrics.registry import BenchmarkResult, MetricSuite
from ..profiler import EfficiencyMetadata, SystemCard, count_parameters
from ..reports import format_markdown_summary, write_json
from ..runner import BenchmarkRunner, ProgressCallback

log = get_logger(__name__)

PipelineResult = Tuple[BenchmarkResult, DeploymentReadinessReport, Dict[str, Path]]


@dataclass
class TaskRunConfig:
    """Base config for every task runner.

    Subclass with ``@dataclass`` when a task wants extra fields. The
    user supplies ``model`` — the toolkit doesn't ship any.
    """

    model: Optional[BenchmarkableModel] = None

    split: Difficulty | str = Difficulty.HARD
    repo_id: Optional[str] = None
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    batch_size: int = 1

    #: When set, the pipeline skips :func:`download_split` and loads
    #: the manifest from disk. Use for runs against a locally-staged
    #: lossless dataset (e.g. ``dataset_hub.cli manifest`` output)
    #: before the v2-webp tree has been uploaded to HF. Path must point
    #: at the JSON file (typically ``<staging>/manifests/<task>/<split>.json``);
    #: the ``root`` inside that JSON resolves modality file paths.
    manifest_path: Optional[str] = None

    device: str = "cuda"
    output_dir: Optional[str] = None
    progress: Optional[ProgressCallback] = None

    #: When True, mirror the per-run output directory to UTD Box at
    #: ``<box_folder_id>/<task>/<model>/<split>/`` after the run finishes.
    #: Requires ``BOX_DEVELOPER_TOKEN`` in the environment. Idempotent
    #: re-uploads (size-matched skip).
    upload_to_box: bool = False
    #: Box folder id to root the upload under. Defaults to the team's
    #: RPX-Outputs folder. Unused unless ``upload_to_box=True``.
    box_folder_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.model is None:
            raise ConfigError(
                f"{type(self).__name__}.model is required. Wrap your "
                "model via rpx.make_numpy_<task>_model(fn) or build a "
                "BenchmarkableModel with your own input/output adapters.",
            )
        if isinstance(self.split, str):
            try:
                self.split = Difficulty(self.split)
            except ValueError as e:
                raise ConfigError(
                    f"Unknown split {self.split!r}. Use one of: {[d.value for d in Difficulty]}",
                ) from e
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}")


def resolve_device(requested: str) -> str:
    """Fall back to CPU when CUDA was requested but isn't available."""
    if requested != "cuda":
        return requested
    try:
        import torch
    except ImportError:
        return requested
    if torch.cuda.is_available():
        return requested
    log.warning("CUDA requested but unavailable; falling back to CPU.")
    return "cpu"


def _count_params(model: BenchmarkableModel) -> EfficiencyMetadata:
    raw = getattr(model, "model", None)
    if raw is None or not hasattr(raw, "parameters"):
        return EfficiencyMetadata(model_type="local", notes="non-torch callable")
    return EfficiencyMetadata(params_m=count_parameters(raw), model_type="local")


def run_pipeline(
    *,
    task: TaskType,
    primary_metric: str,
    cfg: TaskRunConfig,
    compute_ts: bool = True,
    compute_sgc: bool = False,
) -> PipelineResult:
    """Run a task end-to-end: download, load, predict, score, report.

    Parameters
    ----------
    task : TaskType
    primary_metric : str
        Metric key that drives ESD-weighted phase scoring.
    cfg : TaskRunConfig
        Must already have ``model`` set.
    compute_ts : bool
        Compute Temporal Stability. Disable for tasks where adjacent
        samples are unrelated (e.g. keypoint matching).
    compute_sgc : bool
        Compute Stack Geometric Coherence. Only meaningful for
        segmentation with depth available.
    """
    cfg.device = resolve_device(cfg.device)
    split_name = cfg.split.value if isinstance(cfg.split, Difficulty) else str(cfg.split)
    repo_id = cfg.repo_id or DEFAULT_REPO_ID

    log.info("pipeline: task=%s split=%s device=%s", task.value, split_name, cfg.device)

    if cfg.manifest_path:
        manifest_path = Path(cfg.manifest_path)
        if not manifest_path.exists():
            raise ConfigError(
                f"manifest_path does not exist: {manifest_path}",
                hint="Generate per-task manifests with "
                "`python -m rpx_benchmark.dataset_hub.cli manifest "
                "--src <raw> --staging <out> --splits "
                "benchmark/data/splits/scene_splits.json` and pass the "
                "resulting <out>/manifests/<task>/<split>.json path.",
            )
        log.info("using local manifest: %s", manifest_path)
    else:
        manifest_path = download_split(
            task=task,
            split=cfg.split,
            repo_id=repo_id,
            cache_dir=cfg.cache_dir,
            revision=cfg.revision,
        )
    dataset = RPXDataset.from_manifest(manifest_path, batch_size=cfg.batch_size)
    log.info("loaded %d samples from %s", len(dataset), manifest_path)

    model = cfg.model
    name = getattr(model, "name", "model")

    model.setup()
    efficiency = _count_params(model)
    if efficiency.params_m is not None:
        log.info("model %s: %.2f M params", name, efficiency.params_m)
    # SystemCard auto-detects GPU / precision / batch_size and is
    # stamped onto every cell row so cross-host aggregation can
    # disambiguate cells produced on different hardware.
    efficiency.system_card = SystemCard.auto_detect(
        precision=getattr(model, "native_precision", "fp32"),
        batch_size=cfg.batch_size,
    )

    runner = BenchmarkRunner(
        model=model,
        dataset=dataset,
        metric_suite=MetricSuite.for_task(task),
        call_setup=False,
    )
    result, dr_report = runner.run_with_report(
        primary_metric=primary_metric,
        model_name=name,
        efficiency=efficiency,
        compute_ts=compute_ts,
        compute_sgc_flag=compute_sgc,
        progress=cfg.progress,
    )

    safe_name = name.replace("/", "__")
    out_dir = Path(cfg.output_dir or f"./rpx_results/{safe_name}/{split_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "result.json"
    md_path = out_dir / "summary.md"
    cells_path = out_dir / "cells.parquet"

    # cells.parquet is the canonical artefact downstream Φ / 𝒥 / paper-
    # table fills read from. Bucket per_sample (one row per frame) into
    # per-(scene, phase) cells with the mean of every numeric metric.
    # Mirrors the per-clip pipeline (_video_pipeline.py) so both tasks
    # feed the same downstream path.
    cells = cells_from_per_sample(
        result.per_sample,
        model_name=name,
        task=task.value,
        system_card=efficiency.system_card,
    )
    if cells:
        write_cells(cells, cells_path)
        log.info("wrote %d cell rows to %s", len(cells), cells_path)
    else:
        log.warning(
            "no cells emitted for %s/%s — per_sample lacks scene/phase fields. "
            "This usually means the dataset manifest's samples don't carry "
            "scene_id + phase metadata, or the runner's metric_suite stripped "
            "them. Downstream Φ/J aggregation will be unable to bucket.",
            name, split_name,
        )

    write_json(
        json_path,
        task=task.value,
        model_name=name,
        split=split_name,
        repo_id=repo_id,
        result=result,
        dr_report=dr_report,
    )
    md_path.write_text(
        format_markdown_summary(
            task=task.value,
            model_name=name,
            split=split_name,
            repo_id=repo_id,
            result=result,
            dr_report=dr_report,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s and %s", json_path, md_path)

    paths: Dict[str, Path] = {
        "json": json_path,
        "markdown": md_path,
        "out_dir": out_dir,
    }
    if cells:
        paths["cells"] = cells_path

    if cfg.upload_to_box:
        # Lazy import keeps Box deps (requests) out of the import path
        # for users who never upload.
        from ..box_upload import DEFAULT_BOX_FOLDER_ID, upload_run_dir

        try:
            summary = upload_run_dir(
                out_dir,
                task=task.value,
                model_name=name,
                split=split_name,
                root_folder_id=cfg.box_folder_id or DEFAULT_BOX_FOLDER_ID,
            )
            log.info(
                "box: %d uploaded, %d skipped → box:%s",
                summary["uploaded"],
                summary["skipped"],
                summary["remote_path"],
            )
            paths["box_remote"] = summary["remote_path"]
        except Exception as e:  # noqa: BLE001
            # Don't lose the result.json on the floor just because the
            # token expired — log loudly and continue.
            log.error(
                "Box upload failed (%s: %s) — local artefacts at %s are intact; "
                "use `scripts/sync_results_to_box.py` to recover with a fresh token.",
                type(e).__name__,
                e,
                out_dir,
            )

    return result, dr_report, paths


__all__ = ["TaskRunConfig", "PipelineResult", "resolve_device", "run_pipeline"]
