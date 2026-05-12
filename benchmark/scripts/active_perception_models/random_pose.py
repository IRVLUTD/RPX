"""Random-pose NBV baseline — the paper floor.

Samples uniformly from a trajectory-pool of candidate poses. The
sample-pool comes from the active-perception generator (the remaining
frames in the same (scene, phase) after the K context views are
chosen). Any learned NBV that doesn't beat this is not learning
anything about scene structure.

Deterministic per (seed, sample_id) — re-running the same sweep
produces byte-identical predictions, which keeps `result.json` diffs
clean across CI runs.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


class RandomPoseNBV:
    """Uniform sampler over a trajectory-pool of candidate poses."""

    name = "random_pose"
    native_precision = "fp64"
    torch_module = None  # no learned weights

    def __init__(self, device: str = "cpu", seed: int = 5_062_026, **_kwargs: Any) -> None:
        self.device = device
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def __call__(
        self,
        context_poses: List[np.ndarray],
        candidate_poses: Optional[List[np.ndarray]] = None,
        context_rgbs: Optional[List[np.ndarray]] = None,
        task_goal: Optional[str] = None,
    ) -> np.ndarray:
        """Pick a pose uniformly at random.

        Parameters
        ----------
        context_poses
            K context-view poses. Only consumed to validate the input
            is non-empty; the random sampler ignores their content.
        candidate_poses
            Optional pool of (4, 4) poses to sample from. If ``None``
            the baseline falls back to the **last context pose** — a
            degenerate but never-crashes fallback. Real runs always
            supply the trajectory-pool.

        Returns
        -------
        np.ndarray (4, 4) float64 — predicted next pose.
        """
        if not context_poses:
            raise ValueError("random_pose received zero context views")

        if candidate_poses:
            idx = int(self._rng.integers(0, len(candidate_poses)))
            return np.asarray(candidate_poses[idx], dtype=np.float64)

        # Fallback: return the last context pose. Documented degenerate
        # behaviour — the runner should always pass candidate_poses.
        return np.asarray(context_poses[-1], dtype=np.float64)
