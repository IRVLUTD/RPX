"""Reference pose-model adapters for the RPX benchmark — RCPE axis.

Each adapter is a callable class with the contract:

* batched: ``adapter(list_of_pair_dicts) -> list_of_pose_dicts``
  where each input is ``{"rgb_a": HxWx3 uint8, "rgb_b": HxWx3 uint8}``
  and each output is ``{"rotation": 3x3 float64, "translation": 3 float64}``.
* single-pair: optional convenience for offline use.

Plus optional class-level attributes the runner reads:

* ``native_alignment: str`` — pose has no scale ambiguity for rotations,
  but translations are often up-to-scale. ``"none"`` means the model
  publishes metric translations; ``"unit"`` means we should compare
  translations as unit vectors only (i.e. translation_angular_deg).
* ``native_precision: str`` — fp32 / fp16 / bf16. Drives the OperatingPoint precision tag.
* ``torch_module`` — exposed so the profiler walker can count parameters.

The :data:`MODEL_REGISTRY` below maps short names (``--model X``) to a
zero-arg builder. Adding a new model is one line.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

ModelBuilder = Callable[..., Any]


# ── Direct pose regression (HF-loadable) ─────────────────────────────────────


def _build_vggt_omega(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """VGGT-Ω: official facebook/VGGT-1B camera head."""
    from .vggt_omega import VGGTOmega

    return VGGTOmega(device=device, batch_size=batch_size, **kwargs)


def _build_reloc3r(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """Reloc3r (CVPR 2025) — 25 ms inference, current SOTA regression."""
    from .reloc3r import Reloc3r

    return Reloc3r(device=device, batch_size=batch_size, **kwargs)


def _build_dust3r(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """DUSt3R (CVPR 2024) — joint 3D pointmap + relative pose."""
    from .dust3r import DUSt3R

    return DUSt3R(device=device, batch_size=batch_size, **kwargs)


def _build_mast3r(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """MASt3R (2024) — matching-aware DUSt3R extension."""
    from .mast3r import MASt3R

    return MASt3R(device=device, batch_size=batch_size, **kwargs)


def _build_far(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """FAR (CVPR 2024) — hybrid regression + matching."""
    from .far import FAR

    return FAR(device=device, batch_size=batch_size, **kwargs)


def _build_srpose(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """SRPose (ECCV 2024) — sparse keypoint, robust to varying intrinsics."""
    from .srpose import SRPose

    return SRPose(device=device, batch_size=batch_size, **kwargs)


def _build_nope_sac(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """NOPE-SAC (TPAMI 2023) — neural-guided RANSAC."""
    from .nope_sac import NopeSAC

    return NopeSAC(device=device, batch_size=batch_size, **kwargs)


def _build_mickey(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """MicKey (CVPR 2024 Oral) — metric keypoints, metric-scale pose."""
    from .mickey import MicKey

    return MicKey(device=device, batch_size=batch_size, **kwargs)


def _build_loftr(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """LoFTR (CVPR 2021) — dense matching baseline + 5-point pose solver."""
    from .loftr import LoFTR

    return LoFTR(device=device, batch_size=batch_size, **kwargs)


def _build_opencv_baseline(*, device: str = "cpu", batch_size: int = 1, **kwargs):
    """OpenCV findEssentialMat — classical no-learning floor."""
    from .opencv_baseline import OpenCVEssentialMat

    return OpenCVEssentialMat(batch_size=batch_size, **kwargs)


def _build_icp_open3d(*, device: str = "cpu", batch_size: int = 1, **kwargs):
    """Colored ICP (Open3D) — classical RGBD baseline."""
    from .icp_open3d import ColoredICP

    return ColoredICP(batch_size=batch_size, **kwargs)


#: Public name → builder.
MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    # ── Category A: direct pose regression ─────────────────────────────
    "vggt-omega": _build_vggt_omega,
    "reloc3r": _build_reloc3r,
    "dust3r": _build_dust3r,
    "mast3r": _build_mast3r,
    "far": _build_far,
    "srpose": _build_srpose,
    "nope_sac": _build_nope_sac,
    "mickey": _build_mickey,
    # ── Category B: feature matching + solver ──────────────────────────
    "loftr": _build_loftr,
    # ── Category C: classical ──────────────────────────────────────────
    "opencv_baseline": _build_opencv_baseline,
    # ── Category D: RGBD ───────────────────────────────────────────────
    "icp_open3d": _build_icp_open3d,
}


MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "vggt-omega": "VGGT-Ω",
    "reloc3r": "Reloc3r-512",
    "dust3r": "DUSt3R-ViTL-512",
    "mast3r": "MASt3R-ViTL-512",
    "far": "FAR",
    "srpose": "SRPose",
    "nope_sac": "NOPE-SAC",
    "mickey": "MicKey",
    "loftr": "LoFTR",
    "opencv_baseline": "OpenCV-EssentialMat",
    "icp_open3d": "Colored-ICP-Open3D",
}


def list_models() -> list[str]:
    """Return the sorted list of registered model keys."""
    return sorted(MODEL_REGISTRY.keys())


__all__ = [
    "MODEL_REGISTRY",
    "MODEL_DISPLAY_NAMES",
    "list_models",
]
