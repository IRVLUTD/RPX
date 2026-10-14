"""Two calls for any RPX paper task: download its data, then benchmark a model on it.

>>> import rpx_benchmark as rpx
>>> data = rpx.download("T1", split="easy", max_samples=6)       # doctest: +SKIP
>>> result = rpx.benchmark("T1", data, rpx.dummy_models.image_depth)  # doctest: +SKIP

``task`` is the paper's task ID: T1 image depth, T2 video depth, T3 tracking,
T4 relative camera pose, T5 VQA, T6 in-context VQA. ``predict`` is a
plain function; :mod:`rpx_benchmark.examples.dummy_models` shows the inputs and
outputs for each task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .exceptions import ConfigError

PAPER_TASKS = ("T1", "T2", "T3", "T4", "T5", "T6")
_VQA = {"T5", "T6"}


def _check_task(task: str) -> str:
    task = task.upper()
    if task not in PAPER_TASKS:
        raise ConfigError(
            f"Unknown task {task!r}",
            hint="Use a paper task ID: T1, T2, T3, T4, T5 or T6.",
        )
    return task


def download(
    task: str,
    split: str = "easy",
    max_samples: int | None = None,
    cache_dir: str | Path | None = None,
    full: bool = False,
) -> Path:
    """Download the data one paper task needs and return its local manifest.

    Only that task's files are fetched (a 6-sample T1 trial is about 160 MB).
    T5/T6 use the packaged, revision-pinned questions and fetch just their
    images; ``split`` does not apply to them. ``full=True`` uses the paper's
    complete T5/T6 list instead (21,000 / 10,500 questions), read from
    ``hub.VQA_MANIFEST_REVISION``. ``cache_dir`` relocates the T1-T4 dataset
    cache and the T5/T6 image folder (default ``rpx-data``).
    """
    from .examples import benchmark_tasks as bt
    from .hub import download_split
    from .vqa.contract import load_manifest, write_manifest
    from .vqa.hub_rgb import fetch_images_many

    task = _check_task(task)
    if full and task not in _VQA:
        raise ConfigError(
            f"full=True applies to T5 and T6 only, not {task}",
            hint="T1-T4 already download the full split when max_samples is None.",
        )
    if task in _VQA:
        if full:
            from huggingface_hub import hf_hub_download

            from .hub import DEFAULT_REPO_ID, VQA_BENCHMARK_LIST, VQA_MANIFEST_REVISION

            source = hf_hub_download(
                DEFAULT_REPO_ID,
                VQA_BENCHMARK_LIST,
                repo_type="dataset",
                revision=VQA_MANIFEST_REVISION,
            )
            samples = [s for s in load_manifest(source) if s.is_in_context == (task == "T6")]
        else:
            samples = load_manifest(bt.VQA_QUICKSTART / f"{task.lower()}.jsonl")
        samples = samples[:max_samples] if max_samples else samples
        root = Path(cache_dir or "rpx-data")
        print(f"Downloading {bt.TASK_LABELS[task]} images ({len(samples)} questions) ...")
        fetch_images_many(samples, root / "vqa-images")
        manifest = root / f"{task}_{'full' if full else 'quickstart'}.jsonl"
        write_manifest(samples, manifest)
        return manifest
    name, hub_name = bt.HUB_MANIFESTS[task]
    amount = f"{max_samples} samples" if max_samples else "full split"
    print(f"Downloading {bt.TASK_LABELS[task]} data ({split}, {amount}) ...")
    return Path(
        download_split(
            name, split, cache_dir=cache_dir, max_samples=max_samples, manifest_name=hub_name
        )
    )


def benchmark(
    task: str,
    data: str | Path,
    predict: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    output: str | Path | None = None,
    device: str = "cpu",
    depth_output_kind: str = "metric",
) -> dict[str, Any]:
    """Score ``predict`` on downloaded data with the paper's metrics.

    ``data`` is what :func:`download` returned. Without ``predict`` the task's
    dummy model is scored. Results go to ``rpx_results/<task>/<name>``
    (numbered if taken) unless ``output`` is given; the returned report's
    ``metrics`` holds the aggregate scores.
    """
    from .examples import benchmark_tasks as bt

    task = _check_task(task)
    data = Path(data)
    if predict is None:
        predict, name = bt.DUMMY_MODELS[task], name or "dummy"
    model_name = str(name or getattr(predict, "__name__", "custom"))
    out = Path(output) if output else bt.default_output(task, model_name)
    split = "hard"  # the runner's default; T5/T6 ignore it
    image_cache = None
    if task in _VQA:
        image_cache = data.parent / "vqa-images"
    else:
        manifest = json.loads(data.read_text())
        samples = manifest.get("samples") or [{}]
        split = manifest.get("split") or samples[0].get("difficulty") or split
    print(f"Scoring {bt.TASK_LABELS[task]} with {model_name} ...")
    report = bt.run_task(
        task,
        data,
        predict,
        out,
        device=device,
        split=split,
        image_cache=image_cache,
        model_name=model_name,
        depth_output_kind=depth_output_kind,
    )
    report.setdefault("metrics", report.get("aggregated", {}))
    print(bt.summary_line(task, report, out))
    return report
