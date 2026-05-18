"""Graceful-shutdown plumbing for long-running CLI scripts.

Problem this solves
-------------------
Top-level scripts (``run_depth.py``, ``visualize_rerun.py``,
``dataset_hub.cli``, ...) spawn subprocesses: HF download workers, the
Rerun viewer, DataLoader workers, ffmpeg encoders. When the user hits
Ctrl+C or the parent receives SIGTERM, Python's default behaviour
raises ``KeyboardInterrupt`` only in the main thread — child processes
keep running and accumulate as orphans (visible as background CPU /
memory the user has to hunt down later).

What this module does
---------------------
:func:`install_signal_cleanup` installs three guarantees in one call:

1. The current process becomes its **own process group leader** via
   ``os.setpgrp()``. Every child / grandchild spawned afterwards
   inherits the group id by default.
2. ``SIGINT`` and ``SIGTERM`` deliver a ``SIGTERM`` to the **whole
   group**, wait up to ``timeout`` seconds for graceful exit, then send
   ``SIGKILL`` to anything still alive. The signalled process exits via
   ``sys.exit(128 + signum)`` so its own exit code reflects the signal.
3. A double-tap (Ctrl+C twice within the timeout) skips the wait and
   goes straight to SIGKILL of the group.

Idempotent: calling it twice is a no-op. Linux-only — on macOS / BSD
the process-group dance still works, on Windows the call is a no-op
(no equivalent group semantics, no harm in importing).
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import time
from types import FrameType
from typing import Optional

_INSTALLED = False
_KILL_TIMEOUT = 5.0
_DOUBLE_TAP_AT: Optional[float] = None


def _signal_to_group(sig: int) -> None:
    """Send ``sig`` to our process group, ignoring failures (group already
    empty, no permission, race with a child exiting normally)."""
    try:
        os.killpg(os.getpgrp(), sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _on_signal(signum: int, _frame: Optional[FrameType]) -> None:
    global _DOUBLE_TAP_AT
    now = time.monotonic()
    name = signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else str(signum)

    # Double-tap (e.g. two Ctrl+C in quick succession) → escalate immediately.
    if _DOUBLE_TAP_AT is not None and now - _DOUBLE_TAP_AT < _KILL_TIMEOUT:
        sys.stderr.write(f"\n[cleanup] {name} (double-tap) — SIGKILL process group\n")
        _signal_to_group(signal.SIGKILL)
        os._exit(128 + signum)
    _DOUBLE_TAP_AT = now

    sys.stderr.write(
        f"\n[cleanup] {name} received — SIGTERM process group, "
        f"waiting up to {_KILL_TIMEOUT:.0f}s for graceful exit "
        "(press Ctrl+C again to SIGKILL)\n"
    )
    # SIGTERM the children. We don't SIGTERM ourselves — we want to
    # complete shutdown and exit normally.
    pgid = os.getpgrp()
    for pid in _list_children(pgid):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    # Wait for the group to drain.
    deadline = now + _KILL_TIMEOUT
    while time.monotonic() < deadline and _children_still_alive(pgid):
        time.sleep(0.1)

    if _children_still_alive(pgid):
        sys.stderr.write("[cleanup] timeout — SIGKILL stragglers\n")
        for pid in _list_children(pgid):
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    sys.exit(128 + signum)


def _list_children(pgid: int) -> list[int]:
    """Return PIDs in our process group (Linux /proc walk; falls back to []
    on platforms without /proc)."""
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    out: list[int] = []
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as f:
                fields = f.read().split()
            # /proc/<pid>/stat field 5 (1-indexed) is pgid
            if len(fields) >= 5 and int(fields[4]) == pgid:
                out.append(int(name))
        except (OSError, ValueError):
            continue
    return out


def _children_still_alive(pgid: int) -> bool:
    return any(pid != os.getpid() for pid in _list_children(pgid))


def _on_atexit() -> None:
    """On normal interpreter shutdown, SIGTERM any lingering descendants.

    Don't escalate to SIGKILL here — normal exit usually means children
    have already cleaned themselves up; this is a safety net.
    """
    pgid = os.getpgrp()
    for pid in _list_children(pgid):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def install_signal_cleanup(timeout: float = 5.0) -> None:
    """Install signal handlers + process group setup.

    Safe to call multiple times — only the first call takes effect.
    No-op on Windows (no process-group semantics).

    Parameters
    ----------
    timeout
        Seconds to wait between sending SIGTERM and escalating to
        SIGKILL. The default (5s) gives child PyTorch / HF workers
        enough time to flush state without making Ctrl+C feel slow.
    """
    global _INSTALLED, _KILL_TIMEOUT
    if _INSTALLED:
        return
    if sys.platform.startswith("win"):
        _INSTALLED = True
        return
    _KILL_TIMEOUT = float(timeout)
    # New process group → killing the leader kills every descendant.
    try:
        os.setpgrp()
    except OSError:
        # Already a group leader, or unsupported on this platform — fine.
        pass
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    atexit.register(_on_atexit)
    _INSTALLED = True
