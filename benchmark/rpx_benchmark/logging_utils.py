"""Structured logging for the RPX benchmark toolkit.

Library modules do **not** configure logging themselves. They only call
:func:`get_logger` at module top::

    from rpx_benchmark.logging_utils import get_logger

    log = get_logger(__name__)

    def do_the_thing():
        log.info("starting the thing")
        log.debug("parameter x=%s", x)

The CLI entrypoint (and any application code driving the toolkit)
calls :func:`configure_logging` exactly once to attach a handler. If
``rich`` is installed, a pretty :class:`rich.logging.RichHandler` is
used so library log lines render with colour, level icons, and path
markers alongside the benchmark progress UI. Otherwise we fall back to
the standard library handler.

The logger hierarchy mirrors the package structure
(``rpx_benchmark``, ``rpx_benchmark.hub``, ``rpx_benchmark.runner``...)
so power users can turn individual modules up or down with one call::

    import logging
    logging.getLogger("rpx_benchmark.hub").setLevel(logging.DEBUG)

Log level conventions
---------------------

- ``DEBUG`` — internal state useful for bug reports. Off by default.
- ``INFO`` — user-facing progress ("downloading split", "loaded model").
- ``WARNING`` — degradations the user should know about but that do
  not abort the run ("CUDA unavailable; falling back to CPU",
  "profiler unavailable; FLOPs not reported").
- ``ERROR`` — recoverable failures that abort one operation.
- ``CRITICAL`` — reserved for unrecoverable state.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_ROOT_LOGGER_NAME = "rpx_benchmark"
_CONFIGURED = False


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger nested under the ``rpx_benchmark`` root.

    Parameters
    ----------
    name : str
        Usually ``__name__``. If it starts with ``rpx_benchmark`` it is
        returned as-is; otherwise it is attached as a child of the root
        logger so external integrations can still funnel output.

    Returns
    -------
    logging.Logger
        A logger ready to use. Call ``log.info(...)`` etc.

    Examples
    --------
    >>> from rpx_benchmark.logging_utils import get_logger
    >>> log = get_logger("rpx_benchmark.hub")
    >>> log.name
    'rpx_benchmark.hub'
    """
    if not name:
        name = _ROOT_LOGGER_NAME
    if name == _ROOT_LOGGER_NAME or name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def configure_logging(
    level: str | int = "INFO",
    *,
    force: bool = False,
    use_rich: Optional[bool] = None,
) -> logging.Logger:
    """Install a handler on the root ``rpx_benchmark`` logger.

    Safe to call multiple times: subsequent calls are no-ops unless
    ``force=True``. The CLI calls this once at the start of ``main``;
    library users can call it from their own entrypoint.

    Parameters
    ----------
    level : str or int
        Logging level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``) or numeric level. Can be overridden at runtime
        via the ``RPX_LOG_LEVEL`` environment variable.
    force : bool
        If True, re-install the handler even if one is already
        present. Use with care — multiple handlers cause duplicate
        output.
    use_rich : bool, optional
        Force rich or plain handler selection. When omitted, auto-
        detects: rich if the ``rich`` package is importable and stderr
        is a TTY, otherwise plain.

    Returns
    -------
    logging.Logger
        The configured root logger, for chaining.

    Examples
    --------
    >>> from rpx_benchmark.logging_utils import configure_logging
    >>> log = configure_logging("DEBUG")  # doctest: +SKIP
    >>> log.info("benchmark starting")    # doctest: +SKIP
    """
    global _CONFIGURED

    env_level = os.environ.get("RPX_LOG_LEVEL")
    if env_level:
        level = env_level

    if isinstance(level, str):
        level_value = logging.getLevelName(level.upper())
        if not isinstance(level_value, int):
            # Unknown name -> default to INFO with a warning emitted
            # the first time.
            level_value = logging.INFO
    else:
        level_value = int(level)

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level_value)
    root.propagate = False

    if _CONFIGURED and not force:
        return root

    # Wipe any prior handlers so re-configuration is idempotent.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = _build_handler(level_value, use_rich=use_rich)
    root.addHandler(handler)
    _CONFIGURED = True
    return root


def set_level(level: str | int) -> None:
    """Shortcut for ``logging.getLogger('rpx_benchmark').setLevel(level)``.

    Useful for quick toggles without re-running :func:`configure_logging`.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    logging.getLogger(_ROOT_LOGGER_NAME).setLevel(level)


# --------------------------------------------------------------------------- #
# Internal handler factory
# --------------------------------------------------------------------------- #

def _rich_available() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def _build_handler(
    level: int,
    *,
    use_rich: Optional[bool],
) -> logging.Handler:
    """Build a log handler, preferring rich if available and stderr is a tty."""
    if use_rich is None:
        use_rich = _rich_available() and sys.stderr.isatty()

    if use_rich:
        try:
            from rich.logging import RichHandler
            handler: logging.Handler = RichHandler(
                level=level,
                show_time=False,
                show_path=False,
                rich_tracebacks=True,
                markup=False,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            return handler
        except ImportError:
            pass  # fall through to plain

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(levelname)s] %(name)s: %(message)s",
        )
    )
    return handler


__all__ = [
    "get_logger",
    "configure_logging",
    "set_level",
]
