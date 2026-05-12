"""Reference active-perception adapters for the RPX benchmark — Task #11.

Each adapter is a callable class with the contract:

* ``adapter(context_poses, context_rgbs=None, task_goal=None) -> np.ndarray``
  - ``context_poses``: ``list[np.ndarray (4, 4) float64]`` (camera-to-world)
  - ``context_rgbs``:  ``list[np.ndarray (H, W, 3) uint8]`` (optional; for
    semantic baselines)
  - ``task_goal``:     ``str`` (optional; for VLM-style baselines)
  - returns ``np.ndarray (4, 4) float64`` — predicted ``T̂_next``

Plus the same class-level attributes the NVS / pose registries use:
``native_precision``, optional ``torch_module``.

Two baselines wired in this PR (no upstream deps):

* ``random_pose``    — sample uniformly from the trajectory pool the
  generator passed; provides the FLOOR every learned NBV must beat.
* ``farthest_point`` — pose with the largest translation distance
  from the context centroid; a geometric heuristic that often beats
  random by 1-2× on coverage-oriented oracles.

Pending (learned NBVs / VLM baselines) — added in follow-up PRs once
the upstreams are wired (see ``docs/methods/active_perception.md``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

ModelBuilder = Callable[..., Any]


def _build_random_pose(*, device: str = "cpu", **kwargs: Any) -> Any:
    """Uniform sampler over the trajectory pool — paper floor."""
    from .random_pose import RandomPoseNBV

    return RandomPoseNBV(device=device, **kwargs)


def _build_farthest_point(*, device: str = "cpu", **kwargs: Any) -> Any:
    """Farthest-from-context-centroid heuristic."""
    from .farthest_point import FarthestPointNBV

    return FarthestPointNBV(device=device, **kwargs)


MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    "random_pose":     _build_random_pose,
    "farthest_point":  _build_farthest_point,
}


MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "random_pose":    "RandomPose",
    "farthest_point": "FarthestPoint",
}


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())


__all__ = ["MODEL_REGISTRY", "MODEL_DISPLAY_NAMES", "list_models"]
