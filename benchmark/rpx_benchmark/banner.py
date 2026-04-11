"""RPX startup banner.

Printed at the top of every long-running RPX operation so users know
which tool they're looking at and which version is running.
Rendered via :mod:`rich` when available for a Claude-Code-style
panel with the RPX logo, tagline, version, and links; falls back to
plain text otherwise.

Usage
-----

From Python::

    from rpx_benchmark.banner import show_banner
    show_banner(context="task=monocular_depth  split=hard")

From the shell::

    RPX_NO_BANNER=1 rpx bench monocular_depth --split hard

The banner goes to ``stderr`` so piping stdout to ``jq`` / ``grep`` is
unaffected.

Suppression rules (checked in order):

1. ``enabled=False`` argument wins.
2. ``RPX_NO_BANNER`` environment variable set to any non-empty value.
3. ``--quiet`` CLI flag (handled by the CLI entrypoint).
4. Otherwise: the banner prints.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, TextIO

from .logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# ASCII logo — ANSI Shadow font, RPX
# --------------------------------------------------------------------------- #

_RPX_LOGO = r"""██████╗ ██████╗ ██╗  ██╗
██╔══██╗██╔══██╗╚██╗██╔╝
██████╔╝██████╔╝ ╚███╔╝
██╔══██╗██╔═══╝  ██╔██╗
██║  ██║██║     ██╔╝ ██╗
╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝"""

_TAGLINE = "Choose and rank perception models for robot learning"
_TAGLINE_SUB = "Bring your model — we bring the dataset, splits, metrics, and tables."
_DOCS_URL = "https://irvlutd.github.io/RPX/"
_REPO_URL = "https://github.com/IRVLUTD/RPX"


# --------------------------------------------------------------------------- #
# Version discovery
# --------------------------------------------------------------------------- #

def _get_version() -> str:
    """Return the installed package version, falling back to ``'dev'``."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover — ancient Python
        return "dev"
    try:
        return version("rpx-benchmark")
    except PackageNotFoundError:
        return "dev"


# --------------------------------------------------------------------------- #
# Banner rendering
# --------------------------------------------------------------------------- #

def _should_show(enabled: Optional[bool]) -> bool:
    """Resolve the banner's on/off state from the argument + env var."""
    if enabled is False:
        return False
    if enabled is True:
        return True
    # Auto mode — honour RPX_NO_BANNER.
    if os.environ.get("RPX_NO_BANNER"):
        return False
    return True


def show_banner(
    *,
    context: Optional[str] = None,
    subtitle: Optional[str] = None,
    enabled: Optional[bool] = None,
    file: Optional[TextIO] = None,
) -> None:
    """Print the RPX startup banner to a stream (default ``sys.stderr``).

    Parameters
    ----------
    context : str, optional
        Short status line appended under the links (e.g.
        ``"task=monocular_depth  split=hard  device=cuda"``).
    subtitle : str, optional
        Secondary line directly under the RPX tagline. Use this for
        the current script name or a per-operation label.
    enabled : bool, optional
        Force banner on or off. When ``None`` (default), the banner
        prints unless ``RPX_NO_BANNER`` is set in the environment.
    file : file-like, optional
        Output stream. Defaults to :data:`sys.stderr` so the banner
        does not pollute ``stdout``-parsing pipelines.

    Examples
    --------
    >>> from rpx_benchmark.banner import show_banner
    >>> show_banner(context="my smoke run")  # doctest: +SKIP
    """
    if not _should_show(enabled):
        return
    target = file if file is not None else sys.stderr

    try:
        _render_rich(target, context=context, subtitle=subtitle)
    except ImportError:
        _render_plain(target, context=context, subtitle=subtitle)
    except Exception as e:  # pragma: no cover — never break the caller
        log.debug("banner render failed (%s); falling back to plain", e)
        _render_plain(target, context=context, subtitle=subtitle)


# --------------------------------------------------------------------------- #
# Rich backend
# --------------------------------------------------------------------------- #

#: 6-step true-colour gradient from electric cyan through hot pink to
#: warm red, applied one colour per logo row for a smooth vertical
#: fade. These values were picked to maintain perceived luminance so
#: the whole logo reads at once on both light and dark terminals.
_LOGO_GRADIENT = (
    "rgb(0,229,255)",    # electric cyan
    "rgb(94,179,255)",   # sky blue
    "rgb(156,122,255)",  # periwinkle
    "rgb(214,94,255)",   # violet
    "rgb(255,86,196)",   # hot pink
    "rgb(255,117,117)",  # coral
)

#: Background accent colours used for the inline context pill badges.
#: Each `key=value` token in ``context`` is assigned the next colour
#: in this cycle so multi-value contexts stay readable.
_PILL_COLOURS = (
    "rgb(0,229,255)",    # cyan
    "rgb(255,86,196)",   # pink
    "rgb(255,214,0)",    # amber
    "rgb(101,255,161)",  # mint
    "rgb(214,94,255)",   # violet
)


def _render_rich(
    target: TextIO,
    *,
    context: Optional[str],
    subtitle: Optional[str],
) -> None:
    from rich import box
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text

    version = _get_version()
    console = Console(file=target, highlight=False, soft_wrap=False)

    # Logo — vertical RGB gradient, one colour per row.
    logo = Text()
    for line, colour in zip(_RPX_LOGO.splitlines(), _LOGO_GRADIENT):
        logo.append(line + "\n", style=f"bold {colour}")

    # Header row: name + coloured version pill + licence pill.
    header = Text()
    header.append("🤖 ", style="")
    header.append("Robot Perception X", style="bold white")
    header.append("   ")
    header.append(f" v{version} ", style="bold black on rgb(0,229,255)")
    header.append(" ")
    header.append(" MIT ", style="bold black on rgb(255,86,196)")

    # Tagline — highlight the domain-specific words in colour.
    tagline = Text()
    tagline.append("✨ ", style="")
    tagline.append("Choose and rank ", style="bold white")
    tagline.append("perception models ", style="bold rgb(0,229,255)")
    tagline.append("for ", style="bold white")
    tagline.append("robot learning", style="bold rgb(255,86,196)")

    sub = Text()
    sub.append("   Bring your model — we bring the ", style="dim italic")
    sub.append("dataset", style="italic rgb(0,229,255)")
    sub.append(", ", style="dim italic")
    sub.append("splits", style="italic rgb(0,229,255)")
    sub.append(", ", style="dim italic")
    sub.append("metrics", style="italic rgb(0,229,255)")
    sub.append(", and ", style="dim italic")
    sub.append("tables", style="italic rgb(0,229,255)")
    sub.append(".", style="dim italic")

    # Links with emoji icons.
    links = Text()
    links.append("📖 docs    ", style="bold rgb(0,229,255)")
    links.append(_DOCS_URL + "\n", style="underline rgb(94,179,255)")
    links.append("⭐ source  ", style="bold rgb(0,229,255)")
    links.append(_REPO_URL, style="underline rgb(94,179,255)")

    parts: list[Text] = [logo, header, tagline, sub]
    if subtitle:
        parts.append(Text(f"   ⚡ {subtitle}", style="dim"))
    parts.append(Text(""))
    parts.append(links)

    if context:
        parts.append(Text(""))
        parts.append(_render_context_pills(context))

    panel = Panel(
        Group(*parts),
        border_style="rgb(0,229,255)",
        box=box.HEAVY,
        padding=(1, 3),
        title="[bold rgb(0,229,255)]◆ RPX Benchmark[/]",
        title_align="left",
        subtitle="[dim italic]· robot perception · benchmarked ·[/]",
        subtitle_align="right",
    )
    console.print(panel)


def _render_context_pills(context: str):
    """Render ``key=value`` tokens as coloured pill badges.

    Pills wrap to a new row at ``_PILLS_PER_ROW`` tokens so a long
    context string still renders cleanly inside the panel instead of
    tearing mid-badge. Each token gets the next colour in the
    :data:`_PILL_COLOURS` cycle.
    """
    from rich.text import Text

    tokens = context.split()
    pills = Text()
    for i, token in enumerate(tokens):
        colour = _PILL_COLOURS[i % len(_PILL_COLOURS)]
        # Start a new row every _PILLS_PER_ROW tokens so long context
        # strings don't tear across panel edges.
        if i > 0:
            if i % _PILLS_PER_ROW == 0:
                pills.append("\n   ")
            else:
                pills.append("   ")
        else:
            pills.append("   ")

        if "=" in token:
            key, val = token.split("=", 1)
            pills.append(f" {key.upper()} ", style=f"bold black on {colour}")
            pills.append(" ")
            pills.append(val, style=f"bold {colour}")
        else:
            pills.append(f" {token} ", style=f"bold black on {colour}")
    return pills


#: Maximum number of context pills per line before wrapping to a new
#: row. Kept low so each row fits comfortably inside an 80-column
#: terminal even with the widest pill values.
_PILLS_PER_ROW = 3


# --------------------------------------------------------------------------- #
# Plain backend
# --------------------------------------------------------------------------- #

def _render_plain(
    target: TextIO,
    *,
    context: Optional[str],
    subtitle: Optional[str],
) -> None:
    version = _get_version()
    bar = "=" * 68
    print(bar, file=target)
    print(_RPX_LOGO, file=target)
    print(f"Robot Perception X  v{version}  ·  MIT", file=target)
    print(_TAGLINE, file=target)
    print(_TAGLINE_SUB, file=target)
    if subtitle:
        print(subtitle, file=target)
    print(f"docs:   {_DOCS_URL}", file=target)
    print(f"source: {_REPO_URL}", file=target)
    if context:
        print(f"> {context}", file=target)
    print(bar, file=target)


__all__ = ["show_banner"]
