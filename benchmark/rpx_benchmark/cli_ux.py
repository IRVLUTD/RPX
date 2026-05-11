"""Industry-grade colored logging + progress UI for the RPX CLI scripts.

Built on `rich`. One canonical helper module so every runnable script
(``run_depth.py``, ``run_relative_pose.py``, ``sync_results_to_box.py``,
``build_esd_splits.py``, …) produces a consistent, scannable terminal
experience: sectioned headers, spinners for long ops, progress bars
with rate + ETA, and a final summary panel.

Public surface
--------------
* :func:`setup(name)` — return a configured ``logging.Logger`` with a
  ``RichHandler`` attached. Idempotent; call once at the top of ``main()``.
* :func:`banner(title, subtitle)` — splash header at the top of a run.
* :func:`section(title, subtitle)` — colored section divider.
* :func:`step(msg)`, :func:`note(msg)`, :func:`warn(msg)`, :func:`error(msg)`,
  :func:`success(msg)`, :func:`bullet(msg)` — short status lines.
* :func:`kv(key, value)` — inline key-value (renders as ``key: value``).
* :func:`config(rows)` — multi-row config echo (used right after argparse).
* :func:`working(msg)` — context manager: spinner while a long op runs,
  ✓ + elapsed time on exit.
* :func:`progress(description, total)` — wrapped ``rich.progress.Progress``.
* :func:`summary(rows, title)` — final 2-column table panel.
* :func:`fmt_duration(s)`, :func:`fmt_rate(n, s)`, :func:`fmt_bytes(n)` —
  human-friendly formatters.

Environment
-----------
- ``NO_COLOR=1`` — strip all colour escapes (respects the universal
  no-color convention).
- ``CI=true`` (set by GitHub Actions, GitLab CI, etc.) — auto-degrade
  spinners + live progress to plain log lines, so captured run output
  stays readable in CI consoles.
- ``RPX_QUIET=1`` — suppress all decoration; only ERROR-level logging
  lines reach stdout. For piping into scripts.
- ``RPX_NO_ANIMATION=1`` — render the splash logo statically (no
  letter-by-letter reveal). Useful when piping into tools that don't
  handle the live-redraw escape sequences gracefully.
- ``RPX_ANIM=fast|normal|slow|off`` — fine-grained reveal speed knob.
  ``off`` is equivalent to ``RPX_NO_ANIMATION=1``. ``fast`` ≈ 550ms
  total, ``slow`` ≈ 1.4s total. Default ``normal`` ≈ 900ms. Honoured
  only when the terminal supports the animation in the first place
  (i.e. not in CI or with ``NO_COLOR``).

Brand palette
-------------
Matches the LaTeX draft (see ``paper-submission/overleaf/root.tex``):
``rpxR = #4F46E5`` (indigo-600), ``rpxP = #DB2777`` (pink-600),
``rpxX = #C2410C`` (orange-700).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Iterator, Mapping, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

__all__ = [
    "setup",
    "banner",
    "section",
    "step",
    "note",
    "warn",
    "error",
    "success",
    "bullet",
    "kv",
    "config",
    "working",
    "progress",
    "summary",
    "fmt_duration",
    "fmt_rate",
    "fmt_bytes",
]


# ─────────────────────────────────────────────────────────────────────────────
# Brand palette (kept in sync with paper-submission/overleaf/root.tex)
# ─────────────────────────────────────────────────────────────────────────────

BRAND_R = "#4F46E5"   # rpxR — indigo-600
BRAND_P = "#DB2777"   # rpxP — pink-600
BRAND_X = "#C2410C"   # rpxX — orange-700


# ─────────────────────────────────────────────────────────────────────────────
# Console singleton + environment detection
# ─────────────────────────────────────────────────────────────────────────────


_console: Optional[Console] = None


def _is_ci() -> bool:
    """Detect common CI environments. Suppresses spinners (which become
    ANSI cursor-junk in captured CI logs)."""
    return any(os.environ.get(k) for k in ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE"))


def _is_quiet() -> bool:
    return bool(os.environ.get("RPX_QUIET"))


def _get_console() -> Console:
    global _console
    if _console is None:
        # Force colors when stdout is captured but the user wants them
        # (helpful when running `python script.py | tee out.log`); NO_COLOR
        # still takes precedence and strips everything.
        no_color = bool(os.environ.get("NO_COLOR"))
        force = not no_color and not _is_ci()
        _console = Console(force_terminal=force, no_color=no_color, highlight=False)
    return _console


# ─────────────────────────────────────────────────────────────────────────────
# Logger setup
# ─────────────────────────────────────────────────────────────────────────────


def setup(name: str = "rpx", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a single ``RichHandler``. Idempotent — safe to
    call from every script's ``main()``.

    In CI environments and under ``RPX_QUIET=1`` the handler still works
    but spinner/progress helpers degrade to plain log lines.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_rich_configured", False):
        return logger
    if _is_quiet():
        level = logging.ERROR
    logger.setLevel(level)
    logger.propagate = False
    handler = RichHandler(
        console=_get_console(),
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        log_time_format="[%H:%M:%S]",
    )
    handler.setLevel(level)
    # Replace, don't append, so calling setup twice doesn't double-log.
    logger.handlers = [handler]
    logger._rich_configured = True  # type: ignore[attr-defined]
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Splash logo (ANSI Shadow R / P / X coloured to match the LaTeX brand palette)
# ─────────────────────────────────────────────────────────────────────────────

# Each letter is 6 rows tall. Rows are constant-width within a letter so the
# three blocks line up cleanly when concatenated column-wise.
_RPX_GLYPHS: tuple[tuple[str, ...], ...] = (
    # R — indigo
    (
        " ██████╗ ",
        " ██╔══██╗",
        " ██████╔╝",
        " ██╔══██╗",
        " ██║  ██║",
        " ╚═╝  ╚═╝",
    ),
    # P — pink
    (
        " ██████╗ ",
        " ██╔══██╗",
        " ██████╔╝",
        " ██╔═══╝ ",
        " ██║     ",
        " ╚═╝     ",
    ),
    # X — orange (leading space added so the inter-letter gap matches R→P)
    (
        " ██╗  ██╗",
        " ╚██╗██╔╝",
        "  ╚███╔╝ ",
        "  ██╔██╗ ",
        " ██╔╝ ██╗",
        " ╚═╝  ╚═╝",
    ),
)
_RPX_COLORS = (BRAND_R, BRAND_P, BRAND_X)
_RPX_ROWS = 6


_RPX_LETTER_WIDTHS: tuple[int, ...] = tuple(len(g[0]) for g in _RPX_GLYPHS)
_RPX_TOTAL_COLS: int = sum(_RPX_LETTER_WIDTHS)


def _logo_text(n_cols: int = _RPX_TOTAL_COLS) -> Text:
    """Compose the RPX logo revealed up to ``n_cols`` columns from the left.

    The block height stays at ``_RPX_ROWS`` rows regardless of how much is
    revealed — unfilled columns are padded with blanks so the surrounding
    layout never reflows mid-animation.
    """
    n_cols = max(0, min(n_cols, _RPX_TOTAL_COLS))
    out = Text()
    for row in range(_RPX_ROWS):
        consumed = 0
        for li in range(3):
            w = _RPX_LETTER_WIDTHS[li]
            if consumed + w <= n_cols:
                # Letter fully revealed on this row.
                out.append(_RPX_GLYPHS[li][row], style=f"bold {_RPX_COLORS[li]}")
            elif consumed < n_cols:
                # Letter partially revealed — show prefix, pad the rest.
                visible = n_cols - consumed
                out.append(_RPX_GLYPHS[li][row][:visible], style=f"bold {_RPX_COLORS[li]}")
                out.append(" " * (w - visible))
            else:
                # Letter still hidden — full-width blank pad.
                out.append(" " * w)
            consumed += w
        if row != _RPX_ROWS - 1:
            out.append("\n")
    return out


_ANIM_SPEEDS: dict[str, float] = {
    "fast":   0.015,
    "normal": 0.025,
    "slow":   0.045,
}


def _splash_already_seen_this_shell() -> bool:
    """Per-shell first-run gate. First call within a shell session returns
    False (and records a marker); subsequent calls within the same parent
    PID return True so the animation is skipped — first-time delight, never
    annoying on the 5th invocation.

    Disabled under CI / RPX_QUIET (where it's already disabled by other
    means) and under ``RPX_FORCE_SPLASH=1`` for screencast capture.
    """
    if _is_ci() or _is_quiet() or os.environ.get("RPX_FORCE_SPLASH"):
        return False
    from pathlib import Path

    try:
        marker = Path(f"/tmp/rpx-splash-{os.getppid()}")  # noqa: S108
    except Exception:
        return False
    fresh = marker.exists() and time.time() - marker.stat().st_mtime < 3600
    if not fresh:
        try:
            marker.touch()
        except OSError:
            return False  # can't write marker → animate every time
    return fresh


def _animate_logo() -> None:
    """Smooth column-wipe reveal of the RPX logo using ``rich.live``.

    Each frame extends the reveal by one column, so the logo appears to be
    "drawn" left-to-right at ~40 fps (≈900ms total budget by default).
    Falls back to a single static print under CI / NO_COLOR /
    RPX_NO_ANIMATION / RPX_ANIM=off / second-or-later invocation in the
    same shell session.
    """
    console = _get_console()
    speed = (os.environ.get("RPX_ANIM") or "normal").lower()
    static = (
        _is_ci()
        or _is_quiet()
        or bool(os.environ.get("NO_COLOR"))
        or bool(os.environ.get("RPX_NO_ANIMATION"))
        or speed == "off"
        or _splash_already_seen_this_shell()
    )
    if static:
        if not _is_quiet():
            console.print(_logo_text())
        return

    from rich.live import Live  # local import: rich.live is a heavier submodule

    frame_delay = _ANIM_SPEEDS.get(speed, _ANIM_SPEEDS["normal"])
    with Live(
        _logo_text(0),
        console=console,
        refresh_per_second=60,
        transient=False,
    ) as live:
        for cols in range(1, _RPX_TOTAL_COLS + 1):
            live.update(_logo_text(cols))
            time.sleep(frame_delay)
        time.sleep(0.18)  # brief settle on full logo before handing back


# ─────────────────────────────────────────────────────────────────────────────
# Headers / sectioning
# ─────────────────────────────────────────────────────────────────────────────


def banner(title: str, subtitle: Optional[str] = None) -> None:
    """Top-of-run splash: animated RPX logo + a tightly-set tagline row.

    Designed to be the single visual element at the top of every script —
    no second bordered panel competing with the logo. The script title
    appears in bold directly under the logo, followed by an optional dim
    subtitle, separated by a thin indigo rule that visually grounds the
    splash.

    Honours ``RPX_QUIET=1`` (suppress entirely), ``RPX_NO_ANIMATION=1``
    (render the logo statically), and ``RPX_ANIM=off`` (alias).
    """
    if _is_quiet():
        return
    console = _get_console()
    console.print()
    _animate_logo()
    # Tagline row — bold title + dim subtitle, sitting directly under the
    # logo. No bordered panel: the logo itself carries the visual weight.
    console.print()
    console.print(Text(f"  {title}", style=f"bold {BRAND_R}"))
    if subtitle:
        console.print(Text(f"  {subtitle}", style="dim"))
    console.rule(style=BRAND_R)


def section(title: str, subtitle: Optional[str] = None) -> None:
    """Mid-run colored section divider."""
    if _is_quiet():
        return
    console = _get_console()
    text = Text(title, style=f"bold {BRAND_R}")
    if subtitle:
        text.append("  ", style="")
        text.append(subtitle, style="dim")
    console.print()
    console.rule(text, style=BRAND_R)


# ─────────────────────────────────────────────────────────────────────────────
# Inline status lines
# ─────────────────────────────────────────────────────────────────────────────


def step(msg: str) -> None:
    """Green ✓ — a completed step."""
    if _is_quiet():
        return
    _get_console().print(f"[green]✓[/] {msg}")


def note(msg: str) -> None:
    """Dim two-space-indented info line (file paths, config echoes)."""
    if _is_quiet():
        return
    _get_console().print(f"  [dim]{msg}[/]")


def bullet(msg: str) -> None:
    """List bullet, slightly stronger than ``note``."""
    if _is_quiet():
        return
    _get_console().print(f"  [{BRAND_R}]•[/] {msg}")


def warn(msg: str) -> None:
    """Yellow ! — non-fatal warning."""
    _get_console().print(f"[yellow]![/] {msg}")


def error(msg: str) -> None:
    """Red ✗ — fatal-class error (caller still decides whether to raise)."""
    _get_console().print(f"[red]✗[/] {msg}", style="red")


def success(msg: str) -> None:
    """Bold green ✓ — a milestone, louder than ``step``."""
    if _is_quiet():
        return
    _get_console().print(f"[bold green]✓[/] {msg}")


def kv(key: str, value: object) -> None:
    """Inline key-value pair, right-aligned key column."""
    if _is_quiet():
        return
    _get_console().print(f"  [bold]{key:>18}[/]  [dim]:[/]  {value}")


def config(rows: Mapping[str, object], title: str = "Configuration") -> None:
    """Echo a config dict (typically the parsed argparse Namespace)
    as a labelled section followed by aligned ``key: value`` rows."""
    if _is_quiet():
        return
    section(title)
    for k, v in rows.items():
        kv(k, v)


# ─────────────────────────────────────────────────────────────────────────────
# Long-running operations
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def working(msg: str, success_msg: Optional[str] = None) -> Iterator[object]:
    """Spinner + live message while a long op runs; on exit replaces the
    spinner with a green ✓ line that includes elapsed time. In CI / quiet
    mode the spinner is omitted and the message is logged plainly.

    Usage::

        with cli_ux.working("Loading checkpoint"):
            model.load_weights(...)

        with cli_ux.working("Evaluating") as status:
            for i in range(N):
                status.update(f"[cyan]Evaluating ({i}/{N})…[/]")
    """
    console = _get_console()
    t0 = time.time()
    if _is_ci() or _is_quiet():
        # Plain mode — no live spinner.
        if not _is_quiet():
            console.print(f"… {msg}")

        class _NullStatus:
            def update(self, _msg: str) -> None:
                pass

        try:
            yield _NullStatus()
        except Exception:
            console.print(f"[red]✗[/] {msg} (failed after {fmt_duration(time.time() - t0)})")
            raise
        else:
            if not _is_quiet():
                console.print(f"[green]✓[/] {success_msg or msg} ({fmt_duration(time.time() - t0)})")
        return

    status = console.status(f"[{BRAND_R}]{msg}…[/]", spinner="dots")
    status.start()
    try:
        yield status
    except Exception:
        status.stop()
        console.print(f"[red]✗[/] {msg} (failed after {fmt_duration(time.time() - t0)})")
        raise
    else:
        status.stop()
        dt = time.time() - t0
        console.print(f"[green]✓[/] {success_msg or msg} ({fmt_duration(dt)})")


@contextmanager
def progress(description: str = "working", total: Optional[int] = None) -> Iterator[tuple[object, int]]:
    """Context manager yielding ``(Progress, task_id)`` for a tracked loop.

    Usage::

        with cli_ux.progress("Frames", total=N) as (p, task):
            for i in range(N):
                ...
                p.update(task, advance=1)

    In CI / quiet mode the live bar is replaced with periodic plain log
    lines (every 10% of progress).
    """
    if _is_ci() or _is_quiet():
        # Plain mode — emit a log line every ~10% of progress.
        console = _get_console()
        state = {"done": 0, "last_pct": -10}

        class _PlainProgress:
            def update(self, _task: int, advance: int = 1) -> None:
                state["done"] += advance
                if total:
                    pct = int(100 * state["done"] / total)
                    if pct >= state["last_pct"] + 10:
                        state["last_pct"] = pct
                        if not _is_quiet():
                            console.print(f"  [{description}] {state['done']}/{total} ({pct}%)")
                elif state["done"] % 50 == 0 and not _is_quiet():
                    console.print(f"  [{description}] {state['done']}")

        yield _PlainProgress(), 0
        return

    p = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_get_console(),
        transient=False,
    )
    p.start()
    try:
        task = p.add_task(description, total=total)
        yield p, task
    finally:
        p.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Summaries
# ─────────────────────────────────────────────────────────────────────────────


def summary(rows: Mapping[str, object], title: str = "Summary") -> None:
    """Final 2-column table panel — last thing a CLI run prints."""
    if _is_quiet():
        return
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("k", style=f"bold {BRAND_R}")
    table.add_column("v")
    for k, v in rows.items():
        table.add_row(str(k), str(v))
    panel = Panel(table, title=f"[bold]{title}[/]", border_style=BRAND_R, expand=False)
    console = _get_console()
    console.print()
    console.print(panel)


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────


def fmt_duration(seconds: float) -> str:
    """``23 ms`` / ``1.4 s`` / ``2 m 13 s`` / ``1 h 02 m``."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m} m {s:02d} s"
    h, rem = divmod(int(seconds), 3600)
    m, _ = divmod(rem, 60)
    return f"{h} h {m:02d} m"


def fmt_rate(count: int, seconds: float) -> str:
    """``42.3/s``, with em-dash for zero/negative durations."""
    if seconds <= 0:
        return "—"
    return f"{count / seconds:.1f}/s"


def fmt_bytes(n: float) -> str:
    """``1.4 KB`` / ``128.0 MB`` / ``3.2 GB``."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ─────────────────────────────────────────────────────────────────────────────
# Side effects when run as a smoke test
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":  # pragma: no cover
    # Force the animation on every smoke-test invocation, regardless of
    # whether this shell already saw the splash in a real script today.
    os.environ["RPX_FORCE_SPLASH"] = "1"
    log = setup("demo")
    banner("rpx_benchmark.cli_ux — smoke test", "verify every helper renders")
    section("Imports", "verify rich + helper functions wired correctly")
    step("module imported")
    note("import path: rpx_benchmark.cli_ux")
    bullet("setup, banner, section, step, note, kv, working, progress, summary")

    config({"split": "easy", "batch_size": 4, "device": "cuda"})

    section("Live demo")
    with working("Pretend-loading model"):
        time.sleep(0.4)

    with progress("Pretend frames", total=20) as (p, task):
        for _ in range(20):
            time.sleep(0.02)
            p.update(task, advance=1)  # type: ignore[attr-defined]

    warn("Pretend warning: token expires in 5 min")
    success("Demo complete")

    summary(
        {
            "frames": "20",
            "elapsed": fmt_duration(0.42),
            "rate": fmt_rate(20, 0.42),
        },
        title="Demo summary",
    )
