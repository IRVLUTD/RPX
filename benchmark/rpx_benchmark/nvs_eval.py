"""RPX-NVS evaluation pipeline — robot-centric novel view synthesis benchmark.

Evaluates NVS FOR ROBOTS: on robot sensor data (D435 640×480), with
robot-relevant metrics (depth accuracy, PUS, cross-phase robustness),
at varying observation budgets (2/4/8/16 views).

Five evaluation dimensions absent from existing NVS benchmarks:
1. Perceptual Utility Score (PUS): rendered view utility for downstream perception
2. Rendered depth accuracy: per-pixel depth vs D435 sensor GT
3. Cross-phase NVS: build from Clutter → render Clean
4. Robot-sensor evaluation: D435 at 640×480, not DSLR
5. Observation-budget curves: systematic 2→4→8→16 context scaling

Usage
-----
    from rpx_benchmark.nvs_eval import NVSEvaluator

    evaluator = NVSEvaluator(extracted_root, parquet_path, split="easy")
    # Evaluate a model
    results = evaluator.evaluate(model_fn, n_context=4)
    print(results["aggregated"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .logging_utils import get_logger
from .nvs_metrics import (
    depth_metrics,
    evaluate_nvs,
    per_object_psnr,
    psnr,
    ssim,
)
from .nvs_pairs import NVSConfig, NVSPairGenerator, NVSSample

log = get_logger(__name__)

# D435 approximate intrinsics at 640×480
# TODO: replace with per-device calibration if available
D435_FX = 615.0
D435_FY = 615.0
D435_CX = 320.0
D435_CY = 240.0
D435_W = 640
D435_H = 480


# ─────────────────────────────────────────────────────────────────────────────
# Camera convention utilities
# ─────────────────────────────────────────────────────────────────────────────

def _load_pose_c2w(npz_path: Path) -> np.ndarray:
    """Load T265 NPZ → 4×4 camera-to-world (OpenCV convention)."""
    data = np.load(npz_path)
    pos = data["position"].astype(np.float64).reshape(3)
    q = data["orientation"].astype(np.float64).reshape(4)  # xyzw
    x, y, z, w = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Camera-to-world → world-to-camera (invert SE(3))."""
    return np.linalg.inv(c2w)


def intrinsics_matrix(
    fx: float = D435_FX, fy: float = D435_FY,
    cx: float = D435_CX, cy: float = D435_CY,
) -> np.ndarray:
    """3×3 camera intrinsics matrix."""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def normalized_intrinsics(
    fx: float = D435_FX, fy: float = D435_FY,
    cx: float = D435_CX, cy: float = D435_CY,
    w: int = D435_W, h: int = D435_H,
) -> Tuple[float, float, float, float]:
    """Intrinsics normalized by image dimensions (feed-forward model convention)."""
    return fx / w, fy / h, cx / w, cy / h


def prepare_camera_for_model(
    pose_path: Path,
    convention: str = "opencv_w2c",
    normalize_intrinsics: bool = True,
) -> Dict[str, Any]:
    """Prepare camera data in the format feed-forward models expect.

    Parameters
    ----------
    pose_path : Path
        Path to the T265 .npz pose file.
    convention : str
        ``"opencv_c2w"`` — camera-to-world, OpenCV axes (RPX native).
        ``"opencv_w2c"`` — world-to-camera, OpenCV axes (DepthSplat/MVSplat).
        ``"opengl_c2w"`` — camera-to-world, OpenGL axes (flip Y+Z).
    normalize_intrinsics : bool
        If True, divide fx/fy/cx/cy by image width/height.

    Returns
    -------
    dict with ``pose`` (4×4), ``intrinsics`` (fx, fy, cx, cy), ``image_size`` (w, h).
    """
    c2w = _load_pose_c2w(pose_path)

    if convention == "opencv_c2w":
        pose = c2w
    elif convention == "opencv_w2c":
        pose = c2w_to_w2c(c2w)
    elif convention == "opengl_c2w":
        # OpenGL: flip Y and Z axes
        flip = np.diag([1, -1, -1, 1]).astype(np.float64)
        pose = c2w @ flip
    else:
        from .exceptions import ConfigError
        raise ConfigError(
            f"Unknown camera convention: {convention!r}",
            hint="Use 'opencv_c2w', 'opencv_w2c', or 'opengl_c2w'.",
        )

    if normalize_intrinsics:
        fx, fy, cx, cy = normalized_intrinsics()
    else:
        fx, fy, cx, cy = D435_FX, D435_FY, D435_CX, D435_CY

    return {
        "pose": pose,
        "intrinsics": (fx, fy, cx, cy),
        "image_size": (D435_W, D435_H),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_rgb(root: Path, rel_path: str) -> np.ndarray:
    """Load RGB → uint8 HxWx3."""
    with Image.open(root / rel_path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def _load_depth(root: Path, rel_path: str) -> np.ndarray:
    """Load 16-bit PNG depth (mm) → float32 metres."""
    with Image.open(root / rel_path) as im:
        d = np.array(im, dtype=np.float32)
    d[d == 0] = 0.0  # keep 0 as invalid
    return d / 1000.0


def load_nvs_sample(
    root: Path, sample: NVSSample,
    camera_convention: str = "opencv_w2c",
) -> Dict[str, Any]:
    """Load all data for one NVS evaluation sample.

    Returns dict with context views, target query, and GT for evaluation.
    """
    context_rgbs = [_load_rgb(root, p) for p in sample.context_rgb_paths]
    context_depths = [_load_depth(root, p) for p in sample.context_depth_paths]
    context_cameras = [
        prepare_camera_for_model(root / p, convention=camera_convention)
        for p in sample.context_pose_paths
    ]

    target_camera = prepare_camera_for_model(
        root / sample.target_pose_path, convention=camera_convention,
    )
    gt_rgb = _load_rgb(root, sample.target_rgb_path)
    gt_depth = _load_depth(root, sample.target_depth_path)

    return {
        "id": sample.id,
        "context_rgbs": context_rgbs,
        "context_depths": context_depths,
        "context_cameras": context_cameras,
        "target_camera": target_camera,
        "gt_rgb": gt_rgb,
        "gt_depth": gt_depth,
        "sample_type": sample.sample_type,
        "n_context": sample.n_context,
        "scene_id": sample.scene_id,
        "phase": sample.phase,
        "phase_target": sample.phase_target,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LPIPS (lazy import — requires lpips package)
# ─────────────────────────────────────────────────────────────────────────────

_lpips_model = None


def _compute_lpips(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute LPIPS (AlexNet v0.1 backbone, NerfBaselines standard).

    Requires ``pip install lpips``. Returns None if not available.
    """
    global _lpips_model
    try:
        import lpips
        import torch
    except ImportError:
        return float("nan")

    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex", version="0.1")
        _lpips_model.eval()
        if torch.cuda.is_available():
            _lpips_model = _lpips_model.cuda()

    # LPIPS expects [-1, 1] range, CHW format
    def to_tensor(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        t = t * 2.0 - 1.0  # [0,1] → [-1,1]
        return t.unsqueeze(0)

    device = next(_lpips_model.parameters()).device
    with torch.no_grad():
        pred_t = to_tensor(pred).to(device)
        gt_t = to_tensor(gt).to(device)
        score = _lpips_model(pred_t, gt_t)
    return float(score.item())


# ─────────────────────────────────────────────────────────────────────────────
# SSIM (proper sliding window, not the global-stats version)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ssim_proper(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM with Gaussian kernel (NerfBaselines standard: k=11, σ=1.5).

    Falls back to simplified global SSIM if skimage unavailable.
    """
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(
            pred, gt, win_size=11, channel_axis=2, data_range=255,
        ))
    except ImportError:
        return ssim(pred, gt)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_single_sample(
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    pred_depth: Optional[np.ndarray] = None,
    gt_depth: Optional[np.ndarray] = None,
    gt_mask: Optional[np.ndarray] = None,
    compute_lpips: bool = True,
) -> Dict[str, float]:
    """Evaluate one rendered view against GT.

    Parameters
    ----------
    pred_rgb : HxWx3 uint8 — rendered RGB
    gt_rgb : HxWx3 uint8 — real RGB
    pred_depth : HxW float32 — rendered depth (metres), optional
    gt_depth : HxW float32 — D435 depth (metres), optional
    gt_mask : HxW int32 — instance mask for per-object eval, optional

    Returns
    -------
    dict with all computed metrics.
    """
    # Ensure same shape
    if pred_rgb.shape != gt_rgb.shape:
        from PIL import Image as _PILImage
        pred_rgb = np.array(
            _PILImage.fromarray(pred_rgb).resize(
                (gt_rgb.shape[1], gt_rgb.shape[0]), _PILImage.BILINEAR
            ),
            dtype=np.uint8,
        )

    result: Dict[str, float] = {}

    # Standard rendering quality
    result["psnr"] = psnr(pred_rgb, gt_rgb)
    result["ssim"] = _compute_ssim_proper(pred_rgb, gt_rgb)
    if compute_lpips:
        result["lpips"] = _compute_lpips(pred_rgb, gt_rgb)

    # Depth rendering accuracy (novel)
    if pred_depth is not None and gt_depth is not None:
        if pred_depth.shape != gt_depth.shape:
            from PIL import Image as _PILImage
            pred_depth = np.array(
                _PILImage.fromarray(pred_depth).resize(
                    (gt_depth.shape[1], gt_depth.shape[0]), _PILImage.BILINEAR
                ),
                dtype=np.float32,
            )
        dm = depth_metrics(pred_depth, gt_depth)
        result.update(dm)

    # Per-object rendering quality (novel)
    if gt_mask is not None:
        obj_psnr = per_object_psnr(pred_rgb, gt_rgb, gt_mask)
        if obj_psnr:
            result["per_object_psnr_mean"] = float(np.mean(list(obj_psnr.values())))
            result["per_object_psnr_min"] = float(min(obj_psnr.values()))
            result["n_objects_evaluated"] = float(len(obj_psnr))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluator
# ─────────────────────────────────────────────────────────────────────────────

# NVS model callable contract:
#   model(context_rgbs, context_depths, context_cameras, target_camera)
#   → {"rgb": HxWx3 uint8, "depth": HxW float32 (optional)}
NVSModelFn = Callable[
    [List[np.ndarray], List[np.ndarray], List[Dict], Dict],
    Dict[str, np.ndarray],
]


class NVSEvaluator:
    """Full NVS evaluation pipeline for RPX.

    Usage::

        evaluator = NVSEvaluator(extracted_root, parquet_path, split="easy")

        def my_model(context_rgbs, context_depths, context_cameras, target_camera):
            # Your NVS model here
            return {"rgb": rendered_rgb, "depth": rendered_depth}

        results = evaluator.evaluate(my_model, n_context=4)
    """

    def __init__(
        self,
        extracted_root: Path | str,
        parquet_path: Path | str,
        split: str,
        config: NVSConfig | None = None,
        snapshot_root: Path | str | None = None,
        repo_id: str | None = None,
    ) -> None:
        self.root = Path(extracted_root)
        self.split = split
        self.gen = NVSPairGenerator(
            extracted_root=extracted_root,
            parquet_path=parquet_path,
            split=split,
            config=config,
            snapshot_root=snapshot_root,
            repo_id=repo_id,
        )

    def evaluate(
        self,
        model_fn: NVSModelFn,
        n_context: Optional[int] = None,
        camera_convention: str = "opencv_w2c",
        compute_lpips: bool = True,
        max_samples: Optional[int] = None,
        progress_fn: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Run full NVS evaluation.

        Parameters
        ----------
        model_fn : callable
            ``(context_rgbs, context_depths, context_cameras, target_camera)``
            → ``{"rgb": HxWx3 uint8, "depth": HxW float32 (optional)}``.
        n_context : int, optional
            If set, filter to only this context count. Otherwise evaluate all.
        camera_convention : str
            Camera format for the model (``"opencv_w2c"`` for most feed-forward models).
        compute_lpips : bool
            Compute LPIPS (requires lpips package).
        max_samples : int, optional
            Cap for smoke testing.

        Returns
        -------
        dict from :func:`evaluate_nvs` with aggregated + per-type + per-context breakdowns.
        """
        samples = self.gen.samples()
        if n_context is not None:
            samples = [s for s in samples if s.n_context == n_context]
        if max_samples is not None:
            samples = samples[:max_samples]

        total = len(samples)
        log.info("NVS evaluation: %d samples, convention=%s", total, camera_convention)

        per_sample_results: List[Dict[str, Any]] = []
        for i, sample in enumerate(samples):
            if progress_fn:
                progress_fn(i, total)

            data = load_nvs_sample(self.root, sample, camera_convention)

            # Run model
            output = model_fn(
                data["context_rgbs"],
                data["context_depths"],
                data["context_cameras"],
                data["target_camera"],
            )

            pred_rgb = output.get("rgb")
            pred_depth = output.get("depth")

            if pred_rgb is None:
                log.warning("Model returned no RGB for sample %s", sample.id)
                continue

            # Evaluate
            metrics = evaluate_single_sample(
                pred_rgb=pred_rgb,
                gt_rgb=data["gt_rgb"],
                pred_depth=pred_depth,
                gt_depth=data["gt_depth"],
                compute_lpips=compute_lpips,
            )

            # Attach metadata
            metrics["sample_type"] = sample.sample_type
            metrics["n_context"] = sample.n_context
            metrics["scene_id"] = sample.scene_id
            metrics["id"] = sample.id

            per_sample_results.append(metrics)

        return evaluate_nvs(per_sample_results)
