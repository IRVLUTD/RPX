"""Backend-neutral VQA checkpoint and runner registry."""

from __future__ import annotations

from pathlib import Path

from vqa_models.florence_backend import CHECKPOINTS as FLORENCE_CHECKPOINTS
from vqa_models.florence_backend import FlorenceVQARunner
from vqa_models.vllm_backend import CHECKPOINTS as VLLM_CHECKPOINTS
from vqa_models.vllm_backend import VLLMVQARunner

CHECKPOINTS = {**VLLM_CHECKPOINTS, **FLORENCE_CHECKPOINTS}


def backend_name(model_key: str) -> str:
    return (
        "transformers-florence2"
        if model_key in FLORENCE_CHECKPOINTS
        else "vllm"
    )


def backend_version(model_key: str) -> str:
    if model_key in FLORENCE_CHECKPOINTS:
        import transformers

        return transformers.__version__
    import vllm

    return vllm.__version__


def create_runner(
    model_key: str,
    image_root: str | Path,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 1,
):
    runner_type = (
        FlorenceVQARunner if model_key in FLORENCE_CHECKPOINTS else VLLMVQARunner
    )
    return runner_type(
        model_key,
        image_root,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
    )


def provenance(model_key: str) -> dict[str, str]:
    checkpoint = CHECKPOINTS[model_key]
    values = {
        "backend": backend_name(model_key),
        "backend_version": backend_version(model_key),
        "checkpoint": checkpoint.repo_id,
        "revision": checkpoint.revision,
    }
    values[
        "transformers_version" if model_key in FLORENCE_CHECKPOINTS else "vllm_version"
    ] = values["backend_version"]
    return values
