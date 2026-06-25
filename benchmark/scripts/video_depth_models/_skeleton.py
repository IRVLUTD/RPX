"""Shared skeleton for true-video-depth adapters that aren't installed yet.

Every roster entry under :data:`~rpx_benchmark.adapters.depth_scaffold.DEPTH_MODEL_CARDS`
with ``task == VIDEO_DEPTH`` should have an importable module under this
package — otherwise the runner's
:func:`scripts.run_video_depth._load_model` raises a confusing
``ImportError`` instead of a clean "you need to install model X" message.

The skeleton here lets a contributor stand up the module file with one
line of code:

.. code-block:: python

    # scripts/video_depth_models/depth_crafter.py
    from ._skeleton import build_skeleton
    build = build_skeleton("depth-crafter")

The resulting ``build(device)`` factory raises a
:class:`NotImplementedError` whose message includes the install hint
from the model card, so the team member who tries to run the model
sees exactly what to install. When the package is available, the
contributor replaces this one-liner with a real ``build`` body.
"""

from __future__ import annotations

from typing import Callable

from rpx_benchmark.adapters.depth_scaffold import DEPTH_MODEL_CARDS
from rpx_benchmark.api import BenchmarkModel


def build_skeleton(model_key: str) -> Callable[[str], BenchmarkModel]:
    """Return a ``build(device)`` factory that raises a friendly
    "not implemented yet" with the install hint from the model card.

    Parameters
    ----------
    model_key
        Canonical roster key from
        :data:`~rpx_benchmark.adapters.depth_scaffold.DEPTH_MODEL_CARDS`,
        e.g. ``"depth-crafter"`` or ``"monst3r"``.

    Raises
    ------
    KeyError
        At skeleton-construction time if ``model_key`` is not in
        :data:`DEPTH_MODEL_CARDS` (typo in the wrapper file).
    """
    if model_key not in DEPTH_MODEL_CARDS:
        raise KeyError(
            f"build_skeleton({model_key!r}): not in DEPTH_MODEL_CARDS. "
            f"Known keys: {sorted(DEPTH_MODEL_CARDS)}"
        )
    card = DEPTH_MODEL_CARDS[model_key]

    def build(device: str = "cuda", **kwargs) -> BenchmarkModel:
        raise NotImplementedError(
            f"{card.name!r} ({model_key}) is a video-depth adapter skeleton. "
            f"Install the model and replace `build` with a real factory in "
            f"scripts/video_depth_models/{model_key.replace('-', '_')}.py: "
            f"{card.install_hint}",
        )

    return build


__all__ = ["build_skeleton"]
