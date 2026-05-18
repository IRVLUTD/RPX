"""Tests for the graceful-shutdown shim (``rpx_benchmark.cleanup``).

Every test runs the shim in a SUBPROCESS — installing handlers + a new
process group in the pytest worker itself would mutate the test
runner's own state and cascade into other tests. The subprocess
isolation also lets us observe real signal behaviour (SIGTERM
delivery, child reaping, exit codes).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _run_child(src: str, **popen_kwargs) -> subprocess.Popen:
    """Spawn `python -c <src>` with the repo on sys.path, in a new session."""
    return subprocess.Popen(
        [sys.executable, "-c", src],
        start_new_session=True,
        **popen_kwargs,
    )


def test_install_is_idempotent_in_subprocess():
    """Second install() call must reuse the first call's handler."""
    src = textwrap.dedent(
        f"""
        import signal, sys
        sys.path.insert(0, {REPO_ROOT!r})
        from rpx_benchmark.cleanup import install_signal_cleanup
        install_signal_cleanup()
        first = signal.getsignal(signal.SIGTERM)
        install_signal_cleanup()
        second = signal.getsignal(signal.SIGTERM)
        assert first is second, "second install replaced the handler"
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("OK")


def test_install_sets_own_process_group():
    """After install(), the process is its own group leader (pgrp == pid)."""
    src = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {REPO_ROOT!r})
        from rpx_benchmark.cleanup import install_signal_cleanup
        install_signal_cleanup()
        assert os.getpgrp() == os.getpid(), f"pgrp={{os.getpgrp()}} pid={{os.getpid()}}"
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("OK")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX process group semantics")
def test_sigterm_reaps_child_subprocess(tmp_path):
    """End-to-end: parent installs shim, spawns a long-running child, the
    parent receives SIGTERM, and the child is gone within the timeout —
    no orphan left behind."""
    pidfile = tmp_path / "pids.txt"
    parent_src = textwrap.dedent(
        f"""
        import os, subprocess, sys
        sys.path.insert(0, {REPO_ROOT!r})
        from rpx_benchmark.cleanup import install_signal_cleanup
        install_signal_cleanup(timeout=2.0)

        child = subprocess.Popen(["sleep", "30"])
        open({str(pidfile)!r}, "w").write(f"{{os.getpid()}} {{child.pid}}\\n")
        try:
            child.wait()
        except KeyboardInterrupt:
            pass
        """
    )
    parent = _run_child(parent_src)

    # Wait for parent to spawn the child and record both PIDs
    deadline = time.monotonic() + 5
    while not pidfile.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pidfile.exists(), "parent never wrote the sentinel"
    parent_pid, child_pid = (int(x) for x in pidfile.read_text().split())

    # Sanity: child exists before we signal
    os.kill(child_pid, 0)

    parent.send_signal(signal.SIGTERM)
    parent.wait(timeout=10)

    # Child should be reaped within shim_timeout (2s) + small buffer
    deadline = time.monotonic() + 4
    child_gone = False
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            child_gone = True
            break
        time.sleep(0.05)

    if not child_gone:
        # Cleanup so we don't leak; then fail the test
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        pytest.fail(f"child pid {child_pid} survived parent SIGTERM (orphan leak)")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX process group semantics")
def test_normal_exit_atexit_sigterms_descendants(tmp_path):
    """If the parent exits normally (not via signal), atexit should still
    clean up any descendant the script left running."""
    pidfile = tmp_path / "child.pid"
    parent_src = textwrap.dedent(
        f"""
        import os, subprocess, sys, time
        sys.path.insert(0, {REPO_ROOT!r})
        from rpx_benchmark.cleanup import install_signal_cleanup
        install_signal_cleanup()
        child = subprocess.Popen(["sleep", "30"])
        open({str(pidfile)!r}, "w").write(str(child.pid))
        time.sleep(0.5)  # let the child get scheduled
        # Exit normally — atexit hook should SIGTERM the child.
        """
    )
    parent = _run_child(parent_src)
    parent.wait(timeout=10)
    assert pidfile.exists()
    child_pid = int(pidfile.read_text())

    deadline = time.monotonic() + 3
    child_gone = False
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            child_gone = True
            break
        time.sleep(0.05)

    if not child_gone:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        pytest.fail(f"child pid {child_pid} survived parent normal exit")
