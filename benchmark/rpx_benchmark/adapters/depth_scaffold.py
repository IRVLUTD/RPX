"""Shared scaffold for depth-model adapters (D1-F and D1-V).

Every adapter in ``rpx_benchmark/adapters/depth/`` and
``rpx_benchmark/adapters/video_depth/`` follows the same shape:

* Declare ``task`` (``TaskType.MONOCULAR_DEPTH`` for D1-F,
  ``TaskType.VIDEO_DEPTH`` for D1-V).
* Declare ``depth_output_kind`` (``"metric"`` or ``"relative"``) so
  the runner knows whether to apply per-scene-phase scale-and-shift
  alignment before metric computation.
* Declare ``install_hint`` so a contributor who hits the
  ``NotImplementedError`` knows exactly which dependency to install.
* Implement ``setup()`` (lazy weight loading) and ``predict()``
  (forward call returning the right Prediction dataclass).

The scaffold provides:

* A clean :class:`DepthAdapterSkeleton` base class with sensible
  defaults and a ``predict`` stub that raises a descriptive
  ``NotImplementedError`` until a real implementation lands.
* A central :data:`DEPTH_MODEL_CARDS` table that documents every
  model in the paper rosters (D1-F + D1-V) — name, task, output
  kind, paper reference, install hint. Tests assert the table covers
  every roster entry so we can't accidentally drop a model.
* :func:`available_depth_adapters` — programmatic introspection of
  which adapters are importable in the current environment (a model
  whose package is uninstalled raises at adapter-class definition
  time, so the caller can know up-front).

The skeleton approach is deliberate. The 19 models in the rosters
have wildly different installation requirements (different CUDA
versions, gated HF access, custom forks, multi-GB checkpoints). We
ship the registration surface so the harness "knows" about every
model from day one, and contributors fill in the forward call when
they install each model — no PR-per-model needed once the skeleton
exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Sequence

from ..api import BenchmarkModel, Sample, TaskType
from ..exceptions import ConfigError, ModelError
from ..logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Model cards — the canonical roster
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DepthModelCard:
    """Static metadata about one depth model in the roster.

    Used by tests to verify the adapter package covers every model
    the paper claims to evaluate, and by tooling to render
    "available models" tables.
    """

    name: str
    """Display name used in result tables — matches the paper's
    Table 2 / Table 3 spelling exactly (e.g. ``"DA3 Metric-L"``)."""

    task: TaskType
    """``MONOCULAR_DEPTH`` (D1-F) or ``VIDEO_DEPTH`` (D1-V)."""

    depth_output_kind: Literal["metric", "relative"]
    """Determines whether the runner applies per-scene-phase scale +
    shift alignment before metrics. ``"metric"`` for models that
    output meters directly; ``"relative"`` for affine-invariant
    models (MoGe-2, HyDen, Lotus-2, FE2E, DepthLM, DA-V2 relative)."""

    install_hint: str
    """One-line `pip install ...` (or equivalent) the contributor
    runs to make ``setup()`` work. Surfaced in the
    ``NotImplementedError`` raised by skeleton ``predict()`` so the
    next contributor doesn't have to grep."""

    paper_ref: str = ""
    """Optional citation key — matches an entry in
    ``paper-submission/overleaf/root.bib`` so the model card can
    cross-link to its primary source."""


#: Canonical roster of every depth model the paper benchmarks.
#: Tests assert that every entry here has a corresponding adapter
#: skeleton importable from the adapters package.
#:
#: Source: paper-submission/overleaf/text/neurips_v2/04_tasks.tex
#: (the D1-F and D1-V rows of Table 2).
DEPTH_MODEL_CARDS: dict[str, DepthModelCard] = {
    # -------------------------------------------------------------- D1-F
    "da3-metric-l": DepthModelCard(
        name="DA3 Metric-L",
        task=TaskType.MONOCULAR_DEPTH,
        depth_output_kind="metric",
        install_hint="pip install depth-anything-3 (or HF transformers + the depth-anything-v3 weights)",
        paper_ref="yang2024depth",
    ),
    "da-v2-large": DepthModelCard(
        name="DA-V2 Large",
        task=TaskType.MONOCULAR_DEPTH,
        depth_output_kind="metric",
        install_hint="pip install depth-anything-v2 transformers; metric variant from HF",
        paper_ref="yang2024depthanythingv2",
    ),
    "depth-pro": DepthModelCard(
        name="Depth Pro",
        task=TaskType.MONOCULAR_DEPTH,
        depth_output_kind="metric",
        install_hint="pip install depth-pro (Apple); or `pip install git+https://github.com/apple/ml-depth-pro`",
        paper_ref="bochkovskii2025depthpro",
    ),
    "unidepth-v2": DepthModelCard(
        name="UniDepth V2",
        task=TaskType.MONOCULAR_DEPTH,
        depth_output_kind="metric",
        install_hint="pip install unidepth (or `pip install git+https://github.com/lpiccinelli-eth/UniDepth`)",
        paper_ref="piccinelli2024unidepth",
    ),
    "metric3d-v2": DepthModelCard(
        name="Metric3D V2",
        task=TaskType.MONOCULAR_DEPTH,
        depth_output_kind="metric",
        install_hint="pip install metric3d (or `pip install git+https://github.com/YvanYin/Metric3D`)",
        paper_ref="metric3d_v2",
    ),
    "moge-2-vit-l": DepthModelCard(
        name="MoGe-2 ViT-L",
        task=TaskType.MONOCULAR_DEPTH,
        depth_output_kind="relative",
        install_hint="pip install moge (Microsoft); HF: microsoft/moge-2",
        paper_ref="wang2024moge",
    ),
    "hyden": DepthModelCard(
        name="HyDen",
        task=TaskType.MONOCULAR_DEPTH,
        depth_output_kind="relative",
        install_hint="pip install hyden (TBD — research code, expect to clone the upstream repo)",
        paper_ref="hyden",
    ),
    "lotus-2": DepthModelCard(
        name="Lotus-2",
        task=TaskType.MONOCULAR_DEPTH,
        depth_output_kind="relative",
        install_hint="pip install diffusers transformers; HF: jingheya/lotus-depth-d-v2",
        paper_ref="he2024lotus",
    ),
    "fe2e": DepthModelCard(
        name="FE2E",
        task=TaskType.MONOCULAR_DEPTH,
        depth_output_kind="relative",
        install_hint="pip install fe2e (TBD — clone upstream repo)",
        paper_ref="fe2e",
    ),
    "depthlm": DepthModelCard(
        name="DepthLM",
        task=TaskType.MONOCULAR_DEPTH,
        depth_output_kind="relative",
        install_hint="pip install depthlm (TBD — clone upstream repo)",
        paper_ref="depthlm",
    ),
    # -------------------------------------------------------------- D1-V
    "da3-video": DepthModelCard(
        name="DA3",
        task=TaskType.VIDEO_DEPTH,
        depth_output_kind="metric",
        install_hint=(
            "pip install depth-anything-3; same weights as da3-metric-l, "
            "fed per-clip instead of per-frame"
        ),
        paper_ref="yang2024depth",
    ),
    "depth-crafter": DepthModelCard(
        name="DepthCrafter",
        task=TaskType.VIDEO_DEPTH,
        depth_output_kind="relative",
        install_hint="pip install depthcrafter (Tencent); HF: tencent/DepthCrafter",
        paper_ref="hu2024depthcrafter",
    ),
    "video-da": DepthModelCard(
        name="Video DA",
        task=TaskType.VIDEO_DEPTH,
        depth_output_kind="metric",
        install_hint="pip install video-depth-anything (or upstream repo)",
        paper_ref="video_depth_anything",
    ),
    "chrono-depth": DepthModelCard(
        name="ChronoDepth",
        task=TaskType.VIDEO_DEPTH,
        depth_output_kind="relative",
        install_hint="pip install chronodepth (TBD — clone upstream repo)",
        paper_ref="shao2024chronodepth",
    ),
    "rolling-depth": DepthModelCard(
        name="RollingDepth",
        task=TaskType.VIDEO_DEPTH,
        depth_output_kind="relative",
        install_hint="pip install rollingdepth (TBD — clone upstream repo)",
        paper_ref="rollingdepth",
    ),
    "d4rt": DepthModelCard(
        name="D4RT",
        task=TaskType.VIDEO_DEPTH,
        depth_output_kind="metric",
        install_hint="pip install d4rt (TBD — clone upstream repo)",
        paper_ref="d4rt",
    ),
    "gem-depth": DepthModelCard(
        name="GemDepth",
        task=TaskType.VIDEO_DEPTH,
        depth_output_kind="metric",
        install_hint="pip install gemdepth (TBD — clone upstream repo)",
        paper_ref="gemdepth",
    ),
    "vigeo": DepthModelCard(
        name="ViGeo",
        task=TaskType.VIDEO_DEPTH,
        depth_output_kind="metric",
        install_hint="pip install vigeo (TBD — clone upstream repo)",
        paper_ref="vigeo",
    ),
    "monst3r": DepthModelCard(
        name="MonST3R",
        task=TaskType.VIDEO_DEPTH,
        depth_output_kind="metric",
        install_hint="pip install monst3r (or `pip install git+https://github.com/Junyi42/monst3r`)",
        paper_ref="zhang2024monst3r",
    ),
    "vggt-omega": DepthModelCard(
        name="VGGT-Ω",
        task=TaskType.VIDEO_DEPTH,
        depth_output_kind="metric",
        install_hint="pip install vggt (or upstream Meta repo); Ω = the large variant",
        paper_ref="vggt_omega",
    ),
}


# --------------------------------------------------------------------------- #
# Skeleton base class
# --------------------------------------------------------------------------- #


class DepthAdapterSkeleton(BenchmarkModel):
    """Base class for every depth-model adapter skeleton.

    Subclasses set the class-level :attr:`MODEL_KEY` to the
    ``DEPTH_MODEL_CARDS`` entry they wrap; everything else flows from
    that card automatically. Override ``setup()`` and ``predict()``
    with real implementations once the model is installed.

    The default ``predict()`` raises a ``NotImplementedError`` whose
    message includes the install hint, so any contributor who tries
    to run an unimplemented adapter sees exactly what they need to do.
    """

    #: Key into :data:`DEPTH_MODEL_CARDS`. Must be set by subclasses.
    MODEL_KEY: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.MODEL_KEY:
            # The abstract scaffold itself doesn't have a key; subclasses
            # without a key are an error.
            if cls.__name__ != "DepthAdapterSkeleton":
                raise ConfigError(
                    f"{cls.__name__}: MODEL_KEY must be set on every DepthAdapterSkeleton subclass",
                    hint="Set MODEL_KEY to the matching DEPTH_MODEL_CARDS key.",
                )
            return
        if cls.MODEL_KEY not in DEPTH_MODEL_CARDS:
            raise ConfigError(
                f"{cls.__name__}: MODEL_KEY {cls.MODEL_KEY!r} is not in DEPTH_MODEL_CARDS",
                hint=(
                    f"Either add a card for {cls.MODEL_KEY!r} to "
                    f"DEPTH_MODEL_CARDS, or fix the typo. Known keys: "
                    f"{sorted(DEPTH_MODEL_CARDS)}"
                ),
            )
        card = DEPTH_MODEL_CARDS[cls.MODEL_KEY]
        # Inherit task + output kind from the card so adapter authors
        # can't accidentally diverge from the canonical roster.
        cls.task = card.task
        cls.depth_output_kind = card.depth_output_kind
        cls.name = card.name
        cls.install_hint = card.install_hint

    name: ClassVar[str] = ""
    install_hint: ClassVar[str] = ""

    def setup(self) -> None:
        """Load weights, warm CUDA, etc. Skeleton: no-op.

        Override in concrete subclasses. The runner calls this once
        before iteration; the override should be idempotent so
        repeated calls (e.g. when warming a multi-process pool) are
        cheap.
        """
        return None

    def predict(self, batch: Sequence[Sample]) -> Sequence[Any]:
        """Skeleton ``predict`` — raises ``NotImplementedError`` with
        the install hint baked in so a contributor sees exactly which
        package to install / which forward call to wire up.
        """
        raise NotImplementedError(
            f"Adapter for {self.name!r} ({self.MODEL_KEY}) is a skeleton. "
            f"Install the model and implement `predict`: {self.install_hint}",
        )


# --------------------------------------------------------------------------- #
# Discovery helpers
# --------------------------------------------------------------------------- #


def available_depth_adapters() -> dict[str, type[DepthAdapterSkeleton]]:
    """Return every DepthAdapterSkeleton subclass currently importable.

    Walks the registered subclasses of :class:`DepthAdapterSkeleton`
    and returns ``{MODEL_KEY: class}``. Useful for CLI tooling and
    tests that want to enumerate "every model we've at least
    skeletoned".
    """
    out: dict[str, type[DepthAdapterSkeleton]] = {}
    for cls in DepthAdapterSkeleton.__subclasses__():
        if cls.MODEL_KEY:
            out[cls.MODEL_KEY] = cls
    return out


def lazy_import(module_name: str, *, install_hint: str) -> Any:
    """Best-effort import with a friendly error.

    Skeleton ``setup()`` overrides call this to defer heavy deps
    until the adapter is actually run; if the dep is missing, the
    contributor gets the install hint inline.
    """
    try:
        import importlib

        return importlib.import_module(module_name)
    except ImportError as e:
        raise ModelError(
            f"failed to import {module_name!r}",
            hint=install_hint,
        ) from e


__all__ = [
    "DepthAdapterSkeleton",
    "DepthModelCard",
    "DEPTH_MODEL_CARDS",
    "available_depth_adapters",
    "lazy_import",
]
