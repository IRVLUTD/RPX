"""Terminal UX for the RPX CLI.

Degrades gracefully: if ``rich`` is installed we emit Claude-Code-style
panels, spinners, and a live progress bar. Without ``rich`` we fall
back to plain ``print`` lines so the CLI is still usable over bare ssh,
CI logs, and in scripts.

Usage::

    from rpx_benchmark.ui import ConsoleUI
    ui = ConsoleUI.auto()
    ui.header("Monocular Depth", model="depth_pro", split="hard")
    with ui.stage("Downloading split"):
        ...
    ui.result_table(result.aggregated)
    ui.phase_score_table(dr_report.weighted_phase_score)
    ui.progress_cb(total_samples)   # returns a ProgressCallback

The module is import-safe even when rich is missing.
"""

from __future__ import annotations

import contextlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from .runner import ProgressCallback

# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #

def _rich_available() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Plain backend (always available)
# --------------------------------------------------------------------------- #

class _PlainUI:
    """Minimal no-dependency fallback."""

    def header(self, title: str, **fields: Any) -> None:
        bar = "=" * 64
        print(bar)
        print(f"RPX — {title}")
        print(bar)
        for k, v in fields.items():
            print(f"  {k:<10} {v}")

    @contextlib.contextmanager
    def stage(self, label: str) -> Iterator[None]:
        t0 = time.time()
        print(f"[..] {label}")
        try:
            yield
        except Exception:
            print(f"[!!] {label}  failed")
            raise
        else:
            print(f"[ok] {label}  ({time.time() - t0:.1f}s)")

    def info(self, message: str) -> None:
        print(f"[info] {message}")

    def warn(self, message: str) -> None:
        print(f"[warn] {message}", file=sys.stderr)

    def result_table(self, metrics: Dict[str, float], title: str = "Metrics") -> None:
        print()
        print(f"-- {title} --")
        width = max((len(k) for k in metrics), default=6)
        for k, v in metrics.items():
            print(f"  {k:<{width}}  {v:.4f}")

    def phase_score_table(self, wps: Any) -> None:
        if wps is None:
            return
        print()
        print("-- Weighted Phase Score --")
        print(f"  clutter      {wps.s_clutter:.4f}")
        print(f"  interaction  {wps.s_interaction:.4f}")
        print(f"  clean        {wps.s_clean:.4f}")
        print(f"  overall      {wps.s_overall:.4f}")
        print(f"  Δ_int (I-C)  {wps.delta_int:+.4f}")
        print(f"  Δ_rec (L-I)  {wps.delta_rec:+.4f}")

    def efficiency_table(self, dr_report: Any) -> None:
        if dr_report is None:
            return
        rows = []
        if dr_report.params_m is not None:
            rows.append(("params (M)", f"{dr_report.params_m:.2f}"))
        if dr_report.flops_g is not None:
            rows.append(("FLOPs (G)", f"{dr_report.flops_g:.2f}"))
        if dr_report.latency_ms_per_sample is not None:
            rows.append(
                ("latency (ms/sample)", f"{dr_report.latency_ms_per_sample:.1f}")
            )
        if not rows:
            return
        print()
        print("-- Efficiency --")
        width = max(len(k) for k, _ in rows)
        for k, v in rows:
            print(f"  {k:<{width}}  {v}")

    def footer(self, **paths: Any) -> None:
        print()
        for k, v in paths.items():
            print(f"[{k}] {v}")

    def progress_cb(self, total: Optional[int]) -> ProgressCallback:
        state = {"last": 0, "stage": None}

        def cb(done: int, total_: Optional[int], stage: str) -> None:
            if stage != state["stage"]:
                state["stage"] = stage
                print(f"[..] {stage}")
            t = total_ or total
            if t and done and done - state["last"] >= max(1, t // 20):
                pct = 100 * done / t
                print(f"     {done}/{t}  ({pct:.0f}%)")
                state["last"] = done

        return cb


# --------------------------------------------------------------------------- #
# Rich backend (pretty, Claude-Code-style)
# --------------------------------------------------------------------------- #

class _RichUI:
    def __init__(self) -> None:
        from rich.console import Console
        self.console = Console()
        self._progress = None
        self._progress_task = None

    def header(self, title: str, **fields: Any) -> None:
        from rich.panel import Panel
        from rich.table import Table
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold cyan")
        t.add_column()
        for k, v in fields.items():
            t.add_row(f"{k}", f"[white]{v}[/]")
        self.console.print(
            Panel(t, title=f"[bold]RPX — {title}[/]",
                  border_style="cyan", padding=(0, 2))
        )

    @contextlib.contextmanager
    def stage(self, label: str) -> Iterator[None]:
        t0 = time.time()
        with self.console.status(f"[cyan]{label}[/]", spinner="dots"):
            try:
                yield
            except Exception:
                self.console.print(f"[red]✗[/] {label} [dim]failed[/]")
                raise
        self.console.print(
            f"[green]✓[/] {label} [dim]({time.time() - t0:.1f}s)[/]"
        )

    def info(self, message: str) -> None:
        self.console.print(f"[cyan]ℹ[/] {message}")

    def warn(self, message: str) -> None:
        self.console.print(f"[yellow]![/] {message}")

    def result_table(self, metrics: Dict[str, float], title: str = "Metrics") -> None:
        from rich.table import Table
        t = Table(title=title, title_style="bold", show_header=True,
                  header_style="bold magenta", border_style="dim")
        t.add_column("metric", style="cyan")
        t.add_column("value", justify="right", style="white")
        for k, v in metrics.items():
            t.add_row(k, f"{v:.4f}")
        self.console.print()
        self.console.print(t)

    def phase_score_table(self, wps: Any) -> None:
        if wps is None:
            return
        from rich.table import Table
        t = Table(title="Weighted Phase Score (ESD-weighted)",
                  title_style="bold", show_header=True,
                  header_style="bold magenta", border_style="dim")
        t.add_column("phase", style="cyan")
        t.add_column("S_p", justify="right", style="white")
        t.add_row("clutter", f"{wps.s_clutter:.4f}")
        t.add_row("interaction", f"{wps.s_interaction:.4f}")
        t.add_row("clean", f"{wps.s_clean:.4f}")
        t.add_row("[bold]overall[/]", f"[bold]{wps.s_overall:.4f}[/]")
        t.add_row("Δ interaction (I-C)",
                  f"[{'red' if wps.delta_int < 0 else 'green'}]{wps.delta_int:+.4f}[/]")
        t.add_row("Δ recovery    (L-I)",
                  f"[{'green' if wps.delta_rec > 0 else 'red'}]{wps.delta_rec:+.4f}[/]")
        self.console.print()
        self.console.print(t)

    def efficiency_table(self, dr_report: Any) -> None:
        if dr_report is None:
            return
        rows = []
        if dr_report.params_m is not None:
            rows.append(("params (M)", f"{dr_report.params_m:.2f}"))
        if dr_report.flops_g is not None:
            rows.append(("FLOPs (G)", f"{dr_report.flops_g:.2f}"))
        if dr_report.latency_ms_per_sample is not None:
            rows.append(
                ("latency (ms/sample)", f"{dr_report.latency_ms_per_sample:.1f}")
            )
        if not rows:
            return
        from rich.table import Table
        t = Table(title="Efficiency", title_style="bold", show_header=True,
                  header_style="bold magenta", border_style="dim")
        t.add_column("metric", style="cyan")
        t.add_column("value", justify="right", style="white")
        for k, v in rows:
            t.add_row(k, v)
        self.console.print()
        self.console.print(t)

    def footer(self, **paths: Any) -> None:
        from rich.table import Table
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim")
        t.add_column()
        for k, v in paths.items():
            t.add_row(f"{k}:", f"[white]{v}[/]")
        self.console.print()
        self.console.print(t)

    def progress_cb(self, total: Optional[int]) -> ProgressCallback:
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
            TextColumn, TimeElapsedColumn, TimeRemainingColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}[/]"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )
        progress.start()
        task_id = progress.add_task("inference", total=total)
        self._progress = progress
        self._progress_task = task_id

        state = {"stage": None}

        def cb(done: int, total_: Optional[int], stage: str) -> None:
            if stage != state["stage"]:
                if stage == "predict":
                    progress.update(task_id, description="inference")
                elif stage == "setup":
                    progress.update(task_id, description="model setup")
                else:
                    progress.update(task_id, description=stage)
                state["stage"] = stage
            if total_ is not None:
                progress.update(task_id, total=total_, completed=done)

        return cb

    def close_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None


# --------------------------------------------------------------------------- #
# Public factory
# --------------------------------------------------------------------------- #

@dataclass
class ConsoleUI:
    """Thin wrapper delegating to either the rich or plain backend."""

    _backend: Any

    @classmethod
    def auto(cls, force_plain: bool = False) -> "ConsoleUI":
        if force_plain or not _rich_available():
            return cls(_backend=_PlainUI())
        return cls(_backend=_RichUI())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)
