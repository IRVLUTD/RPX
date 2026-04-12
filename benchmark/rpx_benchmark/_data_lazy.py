"""Lazy facade for :mod:`rpx_benchmark.data`.

Mirrors :mod:`rpx_benchmark._schemas_lazy`: any attribute access
triggers the real import, so ``import rpx_benchmark`` does not force a
``datasets`` / ``pyarrow`` import on users who only want the default
loader or the ``huggingface_hub`` snapshot path.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:  # noqa: D401 — module-level __getattr__
    from . import data as _real  # noqa: PLC0415

    return getattr(_real, name)


def __dir__() -> list[str]:
    try:
        from . import data as _real

        return sorted(getattr(_real, "__all__", []))
    except ImportError:
        return []
