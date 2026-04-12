"""Lazy facade for :mod:`rpx_benchmark.schemas`.

Keeps ``rpx_benchmark.schemas`` accessible as an attribute of the
top-level package without forcing a hard pydantic import at
``import rpx_benchmark`` time.

Any attribute access triggers the real import; if pydantic is not
installed the resulting :class:`ImportError` carries the standard
``[schemas]`` extra hint.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:  # noqa: D401 — module-level __getattr__
    from . import schemas as _real  # noqa: PLC0415 — lazy by design

    return getattr(_real, name)


def __dir__() -> list[str]:
    try:
        from . import schemas as _real

        return sorted(getattr(_real, "__all__", []))
    except ImportError:
        return []
