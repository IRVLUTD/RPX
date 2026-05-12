"""Farthest-point NBV baseline.

Geometric heuristic: pick the candidate pose whose camera centre is
L2-farthest from the centroid of the K context camera centres. The
intuition is that more distant viewpoints maximise coverage of
previously unseen scene content — without any learned model.

A real learned NBV should beat this on Coverage-NBV and PUS_active,
but often *not* on `pose_geodesic_deg` to a Task-NBV oracle (the
task-aware oracle may prefer a close-up of one object, not the
farthest pose). The disagreement across oracle objectives is exactly
what the paper's three-axis framing surfaces.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


class FarthestPointNBV:
    """Farthest-from-context-centroid heuristic."""

    name = "farthest_point"
    native_precision = "fp64"
    torch_module = None

    def __init__(self, device: str = "cpu", **_kwargs: Any) -> None:
        self.device = device

    def __call__(
        self,
        context_poses: List[np.ndarray],
        candidate_poses: Optional[List[np.ndarray]] = None,
        context_rgbs: Optional[List[np.ndarray]] = None,
        task_goal: Optional[str] = None,
    ) -> np.ndarray:
        if not context_poses:
            raise ValueError("farthest_point received zero context views")
        if not candidate_poses:
            # Degenerate fallback: return the last context pose. The
            # runner should always supply the candidate pool — this
            # branch only fires for edge-case smoke tests.
            return np.asarray(context_poses[-1], dtype=np.float64)

        centres = np.asarray(
            [np.asarray(p)[:3, 3] for p in context_poses], dtype=np.float64,
        )
        centroid = centres.mean(axis=0)  # (3,)

        cand_centres = np.asarray(
            [np.asarray(p)[:3, 3] for p in candidate_poses], dtype=np.float64,
        )
        dists = np.linalg.norm(cand_centres - centroid, axis=1)
        return np.asarray(candidate_poses[int(np.argmax(dists))], dtype=np.float64)
