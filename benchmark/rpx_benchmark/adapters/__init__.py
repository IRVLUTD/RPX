"""Formal adapter framework for bringing models into the RPX benchmark.

The contract is intentionally minimal: preprocess the RPX ``Sample`` into
whatever the model wants, invoke the model, postprocess the output into
the RPX prediction contract. Users composing their own adapter stack
only need to touch the *model* -- the input and output adapters shipped
here handle the plumbing for common model families.

See :mod:`rpx_benchmark.adapters.base` for the core types and
the repository script ``scripts/depth_models/hf_pipeline.py`` for a
Hugging Face depth-pipeline wrapper.
"""

from .base import (
    BenchmarkableModel,
    InputAdapter,
    ModelInvoker,
    OutputAdapter,
    PreparedInput,
    default_invoker,
    make_numpy_depth_model,
    make_numpy_detection_model,
    make_numpy_grounding_model,
    make_numpy_keypoint_model,
    make_numpy_mask_model,
    make_numpy_pose_model,
    make_numpy_sparse_depth_model,
    make_numpy_tracking_model,
    make_numpy_video_depth_model,
)
from .batched_depth import BatchedDepthBenchmarkModel
from .batched_multimodal import (
    BatchedRelativePoseBenchmarkModel,
    BatchedSegmentationBenchmarkModel,
    BatchedTaskBenchmarkModel,
)

__all__ = [
    "BatchedDepthBenchmarkModel",
    "BatchedRelativePoseBenchmarkModel",
    "BatchedSegmentationBenchmarkModel",
    "BatchedTaskBenchmarkModel",
    "BenchmarkableModel",
    "InputAdapter",
    "ModelInvoker",
    "OutputAdapter",
    "PreparedInput",
    "default_invoker",
    # Per-task numpy fast paths
    "make_numpy_depth_model",
    "make_numpy_detection_model",
    "make_numpy_grounding_model",
    "make_numpy_keypoint_model",
    "make_numpy_mask_model",
    "make_numpy_pose_model",
    "make_numpy_sparse_depth_model",
    "make_numpy_tracking_model",
    "make_numpy_video_depth_model",
]
