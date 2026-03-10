"""
rpx.utils.model_profiler
------------------------
Hardware-agnostic model complexity profiling.

Collects all available metrics so they can be reported selectively in papers.

Requires (all optional — graceful fallback if not installed):
    pip install fvcore thop
"""

import os
import time
import tempfile
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
    _FVCORE_AVAILABLE = True
except ImportError:
    _FVCORE_AVAILABLE = False

try:
    from thop import profile as thop_profile
    _THOP_AVAILABLE = True
except ImportError:
    _THOP_AVAILABLE = False


@dataclass
class ModelComplexityProfile:
    """
    Hardware-agnostic model complexity profile.

    All count-based metrics are hardware-independent.
    Latency is hardware-dependent but included for completeness
    with device annotation.
    """
    # ── Parameter counts ──────────────────────────────────────────────────────
    total_params: Optional[int] = None          # all parameters
    trainable_params: Optional[int] = None      # parameters with grad
    non_trainable_params: Optional[int] = None  # frozen parameters

    # ── Compute complexity ────────────────────────────────────────────────────
    flops_fvcore: Optional[float] = None   # FLOPs via fvcore (GFLOPs)
    macs_thop: Optional[float] = None      # MACs via thop (GMACs)
    activations: Optional[float] = None    # activation count via fvcore (M)
    flops_per_pixel: Optional[float] = None  # GFLOPs / (H*W): resolution-normalised

    # ── Model size ────────────────────────────────────────────────────────────
    model_size_mb: Optional[float] = None   # weight file size on disk (MB)
    param_size_mb: Optional[float] = None   # theoretical FP32 weight memory (MB)

    # ── Structural ────────────────────────────────────────────────────────────
    num_layers: Optional[int] = None        # number of named modules
    model_depth: Optional[int] = None       # max nesting depth of modules
    input_shape: Optional[Tuple] = None     # (C, H, W) used for profiling

    # ── Latency ───────────────────────────────────────────────────────────────
    latency_ms_mean: Optional[float] = None   # mean inference time per sample (ms)
    latency_ms_std: Optional[float] = None    # std of inference time (ms)
    latency_ms_p95: Optional[float] = None    # 95th percentile latency (ms)
    fps: Optional[float] = None               # frames per second
    latency_device: Optional[str] = None      # device on which latency was measured

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def summary(self) -> str:
        lines = ["── Model Complexity Profile ──────────────────────"]
        d = self.to_dict()
        fmt = {
            'total_params':        ('Parameters (total)',    lambda v: f"{v/1e6:.2f} M"),
            'trainable_params':    ('Parameters (trainable)',lambda v: f"{v/1e6:.2f} M"),
            'flops_fvcore':        ('FLOPs (fvcore)',        lambda v: f"{v:.3f} GFLOPs"),
            'macs_thop':           ('MACs (thop)',           lambda v: f"{v:.3f} GMACs"),
            'activations':         ('Activations',           lambda v: f"{v:.2f} M"),
            'flops_per_pixel':     ('FLOPs/pixel',           lambda v: f"{v*1e6:.1f} KFLOPs/px"),
            'model_size_mb':       ('Model size (disk)',     lambda v: f"{v:.1f} MB"),
            'param_size_mb':       ('Param memory (FP32)',   lambda v: f"{v:.1f} MB"),
            'num_layers':          ('# Modules',             lambda v: str(v)),
            'model_depth':         ('Depth (max nesting)',   lambda v: str(v)),
            'latency_ms_mean':     ('Latency (mean)',        lambda v: f"{v:.2f} ms"),
            'latency_ms_std':      ('Latency (std)',         lambda v: f"{v:.2f} ms"),
            'latency_ms_p95':      ('Latency (p95)',         lambda v: f"{v:.2f} ms"),
            'fps':                 ('Throughput',            lambda v: f"{v:.1f} FPS"),
            'latency_device':      ('Latency device',        str),
        }
        for key, (label, fn) in fmt.items():
            if key in d:
                lines.append(f"  {label:<28} {fn(d[key])}")
        lines.append("─" * 50)
        return "\n".join(lines)


def profile_model(
    model: Any,
    dummy_input,
    warmup_runs: int = 5,
    timing_runs: int = 50,
    device: Optional[str] = None,
) -> ModelComplexityProfile:
    """
    Profile a model comprehensively.

    Args:
        model: Any callable model (PyTorch nn.Module recommended).
        dummy_input: A single representative input tensor (or tuple of tensors).
        warmup_runs: Number of warmup forward passes before timing.
        timing_runs: Number of timed forward passes.
        device: Device string (e.g. 'cpu', 'cuda'). Auto-detected if None.

    Returns:
        ModelComplexityProfile
    """
    profile = ModelComplexityProfile()

    if not _TORCH_AVAILABLE:
        warnings.warn("PyTorch not available — skipping model profiling.")
        return profile

    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = 'cpu'
    device = str(device)

    # Ensure input is a tuple
    if not isinstance(dummy_input, tuple):
        dummy_input = (dummy_input,)

    # ── 1. Parameter counts ──────────────────────────────────────────────────
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        profile.total_params = total
        profile.trainable_params = trainable
        profile.non_trainable_params = total - trainable
        profile.param_size_mb = total * 4 / (1024 ** 2)  # FP32 = 4 bytes
    except Exception as e:
        warnings.warn(f"Parameter count failed: {e}")

    # ── 2. Input shape ───────────────────────────────────────────────────────
    try:
        profile.input_shape = tuple(dummy_input[0].shape)
    except Exception:
        pass

    # ── 3. FLOPs + Activations via fvcore ───────────────────────────────────
    if _FVCORE_AVAILABLE:
        try:
            flop_counter = FlopCountAnalysis(model, dummy_input)
            flop_counter.unsupported_ops_warnings(False)
            flop_counter.uncalled_modules_warnings(False)
            flops = flop_counter.total()
            profile.flops_fvcore = flops / 1e9  # GFLOPs

            act_counter = ActivationCountAnalysis(model, dummy_input)
            act_counter.unsupported_ops_warnings(False)
            profile.activations = act_counter.total() / 1e6  # M activations

            # FLOPs per pixel (resolution-normalised)
            try:
                _, h, w = dummy_input[0].shape[-3], dummy_input[0].shape[-2], dummy_input[0].shape[-1]
                profile.flops_per_pixel = (flops / 1e9) / (h * w)
            except Exception:
                pass
        except Exception as e:
            warnings.warn(f"fvcore FLOPs/activations failed: {e}")
    else:
        warnings.warn("fvcore not installed. Run: pip install fvcore")

    # ── 4. MACs via thop ────────────────────────────────────────────────────
    if _THOP_AVAILABLE:
        try:
            macs, _ = thop_profile(model, inputs=dummy_input, verbose=False)
            profile.macs_thop = macs / 1e9  # GMACs
        except Exception as e:
            warnings.warn(f"thop MACs failed: {e}")
    else:
        warnings.warn("thop not installed. Run: pip install thop")

    # ── 5. Model size on disk ───────────────────────────────────────────────
    try:
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            tmp_path = f.name
        import torch
        torch.save(model.state_dict(), tmp_path)
        profile.model_size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
        os.remove(tmp_path)
    except Exception as e:
        warnings.warn(f"Model size on disk failed: {e}")

    # ── 6. Structural metrics ────────────────────────────────────────────────
    try:
        modules = list(model.named_modules())
        profile.num_layers = len(modules)

        def _depth(module, current=0):
            children = list(module.children())
            if not children:
                return current
            return max(_depth(c, current + 1) for c in children)

        profile.model_depth = _depth(model)
    except Exception as e:
        warnings.warn(f"Structural metrics failed: {e}")

    # ── 7. Latency profiling ─────────────────────────────────────────────────
    try:
        model.eval()
        import torch

        with torch.no_grad():
            # Warmup
            for _ in range(warmup_runs):
                _ = model(*dummy_input)

            # Sync before timing (CUDA)
            if 'cuda' in device and torch.cuda.is_available():
                torch.cuda.synchronize()

            timings = []
            for _ in range(timing_runs):
                t0 = time.perf_counter()
                _ = model(*dummy_input)
                if 'cuda' in device and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000)  # ms

        timings = np.array(timings)
        profile.latency_ms_mean = float(np.mean(timings))
        profile.latency_ms_std = float(np.std(timings))
        profile.latency_ms_p95 = float(np.percentile(timings, 95))
        profile.fps = 1000.0 / profile.latency_ms_mean
        profile.latency_device = device
    except Exception as e:
        warnings.warn(f"Latency profiling failed: {e}")

    return profile
