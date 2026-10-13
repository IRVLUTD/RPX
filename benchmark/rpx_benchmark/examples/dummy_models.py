"""Dummy models for the six RPX paper tasks: copy one, replace its body with your model.

Each function shows exactly what RPX passes in and expects back for one task.
They run as-is, so you can try any task before writing code::

    python -m rpx_benchmark.examples.benchmark_tasks --task T2 --split easy --max-samples 2 \\
        --model rpx_benchmark.examples.dummy_models:video_depth

The outputs are placeholders, not predictions; their scores mean nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def image_depth(rgb: np.ndarray) -> np.ndarray:
    """T1 image depth: RGB image ``(H, W, 3)`` uint8 -> depth ``(H, W)`` in metres.

    For relative (up-to-scale) depth, also pass ``--depth-output-kind relative``.
    """
    return np.full(rgb.shape[:2], 2.0, np.float32)


def video_depth(clip: np.ndarray) -> np.ndarray:
    """T2 video depth: RGB clip ``(T, H, W, 3)`` uint8 -> depth ``(T, H, W)`` in metres.

    The evaluator rejects a constant map as degenerate, so this placeholder is a
    left-to-right ramp from 1.5 m to 2.5 m.
    """
    t, h, w = clip.shape[:3]
    ramp = np.linspace(1.5, 2.5, w, dtype=np.float32)
    return np.broadcast_to(ramp, (t, h, w)).copy()


def tracking(rgb: np.ndarray) -> list[dict]:
    """T3 object tracking: one RGB frame ``(H, W, 3)`` -> the objects tracked so far.

    Called once per frame, in order, so a real tracker keeps its state between
    calls. Each track has a stable ``track_id`` and its ``boxes`` as
    ``[x1, y1, x2, y2]`` pixel rows.
    """
    h, w = rgb.shape[:2]
    box = np.array([[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], np.float32)
    return [{"track_id": "0", "boxes": box}]


def camera_pose(rgb_a: np.ndarray, rgb_b: np.ndarray) -> dict:
    """T4 relative camera pose: two RGB images -> pose of camera B relative to camera A.

    ``rotation`` is a 3x3 matrix (or a quaternion) and ``translation`` a 3-vector in metres.
    """
    return {"rotation": np.eye(3), "translation": np.zeros(3)}


def vqa(image_paths: tuple[Path, ...], prompt: str, max_new_tokens: int, output_kind: str) -> str:
    """T5 VQA: one image and a question -> your model's raw text answer.

    ``image_paths`` holds the target image. Answer with one JSON object: the
    object's ``label`` and its ``bbox`` as ``[x1, y1, x2, y2]`` normalized to 0-1000.
    Works with local models and hosted APIs alike.
    """
    return json.dumps({"label": "object", "bbox": [250, 250, 750, 750]})


def incontext_vqa(
    image_paths: tuple[Path, ...], prompt: str, max_new_tokens: int, output_kind: str
) -> str:
    """T6 in-context VQA: a reference crop and a target image -> the same JSON answer as T5.

    ``image_paths`` is ``(reference_crop, target)``; the box refers to the target.
    """
    return vqa(image_paths, prompt, max_new_tokens, output_kind)
