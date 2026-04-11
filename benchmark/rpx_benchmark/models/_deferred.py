"""Stub factories for depth models that are *intentionally* not yet
wired into the monocular absolute depth pipeline.

Each stub raises ``NotImplementedError`` with a concrete explanation so
users running ``rpx models`` see the intended slate and understand why
these entries do not yet run.

Remove this file's entries from the registry once the corresponding
constraints are addressed.
"""

from __future__ import annotations

from ..adapters import BenchmarkableModel


def _deferred(name: str, reason: str) -> BenchmarkableModel:
    raise NotImplementedError(
        f"{name} is registered but not yet runnable in monocular absolute "
        f"depth mode.\n    Reason: {reason}\n"
        "    Remove this stub and add a proper factory once the constraint "
        "is addressed."
    )


def video_depth_anything_large(*, device: str = "cuda", **_) -> BenchmarkableModel:
    """Defer: Video Depth Anything is a *video* model.

    Evaluating it per-frame throws away its core contribution (temporal
    consistency), and benchmarking it properly needs a sequence-level
    evaluation mode over consecutive RPX frames within a phase. Wire
    that mode first, then add a real factory.
    """
    return _deferred(
        "video_depth_anything_large",
        "Sequence-level model; per-frame eval is meaningless. Add a "
        "temporal benchmark mode before wiring this factory.",
    )


def prompt_depth_anything_vits(*, device: str = "cuda", **_) -> BenchmarkableModel:
    """Defer: PromptDA needs a sparse depth *prompt* for metric output.

    Without a prompt the model falls back to relative depth, which fails
    the "monocular *absolute*" contract of this pipeline. PromptDA
    belongs in a separate "prompted depth completion" task where we can
    feed D435 sparse samples as the prompt.
    """
    return _deferred(
        "prompt_depth_anything_vits",
        "Requires sparse depth prompt to produce metric output. "
        "Belongs in a prompted-depth task, not monocular absolute depth.",
    )


def depth_anything_3(*, device: str = "cuda", **_) -> BenchmarkableModel:
    """Defer: Depth Anything 3 is not yet available via transformers.

    The native repo ships ``.pt`` weights for multi-view geometry; once
    an ``AutoModelForDepthEstimation`` checkpoint lands on the HF hub,
    replace this stub with a 10-line factory that calls
    :func:`make_hf_depth_model`.
    """
    return _deferred(
        "depth_anything_3",
        "Not yet in the transformers AutoModelForDepthEstimation registry. "
        "Wire via make_hf_depth_model once the HF checkpoint is published.",
    )
