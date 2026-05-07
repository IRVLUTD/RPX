"""Deterministic seeding for reproducible benchmark runs.

Seeds every stochastic source the toolkit is likely to touch:
``random``, ``numpy``, ``torch`` (CPU + CUDA + MPS), and
HuggingFace ``transformers`` / ``datasets`` where present. Safe to
call even when those libraries aren't installed — the function
touches only the backends that import successfully.

Example
-------

Imperative::

    from rpx_benchmark.determinism import seed_all
    seed_all(42)

Context manager (restores RNG state on exit)::

    from rpx_benchmark.determinism import deterministic

    with deterministic(42):
        result = runner.run()

Design notes
------------
- Seeding is *best-effort across backends*. Calling code should treat
  this as "best chance at determinism", not a hard guarantee —
  non-deterministic CUDA kernels, cuDNN algorithm selection, and
  multi-threaded data loaders can still introduce run-to-run variance.
- The context manager snapshots and restores the standard-library and
  numpy RNG state so nested determinism blocks don't leak state.
  Torch / transformers / datasets expose no cheap state-snapshot
  primitive, so their post-exit state is the sequence advanced during
  the block (same as re-seeding without a context manager).
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Iterator

__all__ = ["seed_all", "deterministic", "RPX_SEED"]


#: Canonical project-wide seed. MMDDYYYY-encoded 05/06/2026 — the date
#: the team committed to a single deterministic seed across the whole
#: benchmark (bootstrap CI, test data generation, sparse-depth /
#: keypoint-pair samplers, ORD pixel-pair sampler, etc.). Stored here so
#: there's exactly one place to change if the project policy shifts.
#: Python's int literal grammar disallows a leading zero, so the
#: storage form is ``5_062_026``; the underscores keep the
#: month-day-year segmentation visible at a glance.
RPX_SEED: int = 5_062_026


def _seed_python(seed: int) -> None:
    random.seed(seed)
    # PYTHONHASHSEED affects dict ordering in child subprocesses; the
    # running interpreter snapshots it at startup so setting it here
    # only helps processes spawned later (e.g. DataLoader workers).
    os.environ["PYTHONHASHSEED"] = str(seed)


def _seed_numpy(seed: int) -> None:
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return
    np.random.seed(seed)


def _seed_torch(seed: int) -> None:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "manual_seed"):
        try:
            mps.manual_seed(seed)
        except Exception:
            pass


def _seed_hf(seed: int) -> None:
    """Seed transformers / datasets when they're importable.

    Both libraries expose a ``set_seed`` helper that internally seeds
    Python, NumPy, and PyTorch — we still call our own seeders first
    so the HF helpers only fill gaps we might have missed.
    """
    try:
        from transformers import set_seed as _tf_set_seed  # noqa: PLC0415
        _tf_set_seed(seed)
    except ImportError:
        pass


def seed_all(seed: int) -> None:
    """Seed every stochastic backend detectable in the current env.

    Parameters
    ----------
    seed : int
        Seed forwarded to Python ``random``, NumPy, PyTorch (CPU +
        CUDA + MPS), and ``transformers.set_seed`` when available.
    """
    _seed_python(seed)
    _seed_numpy(seed)
    _seed_torch(seed)
    _seed_hf(seed)


@contextmanager
def deterministic(seed: int) -> Iterator[None]:
    """Run a block with deterministic RNG and restore on exit.

    Python ``random`` and NumPy RNG state are snapshotted and
    restored. Torch / transformers state is merely re-seeded; see the
    module docstring for the full scope.

    Parameters
    ----------
    seed : int

    Yields
    ------
    None
    """
    py_state = random.getstate()
    np_state = None
    try:
        import numpy as np  # noqa: PLC0415
        np_state = np.random.get_state()
    except ImportError:
        np = None  # type: ignore[assignment]

    seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        if np_state is not None:
            import numpy as _np  # noqa: PLC0415
            _np.random.set_state(np_state)
