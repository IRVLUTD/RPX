"""Integration tests for the ``rpx`` command-line interface.

Every test invokes :func:`rpx_benchmark.cli.main` with an explicit
``argv`` list so the tests are hermetic and do not depend on the
current process's sys.argv or an actual terminal.
"""

from __future__ import annotations

import pytest

from rpx_benchmark import cli
from rpx_benchmark.exceptions import ConfigError


# --------------------------------------------------------------------------- #
# Help + listing subcommands
# --------------------------------------------------------------------------- #

def test_top_level_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    # The description mentions "RPX" and the tagline; the exact wording
    # has changed over time but both should always appear together.
    assert "RPX" in captured.out
    assert "rank" in captured.out or "robot learning" in captured.out
    assert "bench" in captured.out


def test_missing_subcommand_exits_two(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_rpx_ls_lists_all_task_types(capsys):
    rc = cli.main(["ls"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "monocular_depth" in out
    assert "object_segmentation" in out
    assert "Splits:" in out
    assert "easy" in out
    assert "hard" in out


def test_rpx_models_lists_runnable_and_deferred(capsys):
    rc = cli.main(["models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Runnable models:" in out
    assert "depth_pro" in out
    assert "Deferred" in out
    assert "depth_anything_3" in out


def test_rpx_bench_help_shows_monocular_depth_subcommand(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["bench", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "monocular_depth" in out


def test_rpx_bench_monocular_depth_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["bench", "monocular_depth", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--split" in out
    assert "--hf-checkpoint" in out or "--model" in out


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #

def test_bench_without_model_or_hf_checkpoint_is_argparse_error(capsys):
    """Both selectors omitted — argparse itself should error (exit 2)."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["bench", "monocular_depth", "--split", "hard"])
    assert exc.value.code == 2


def test_main_exits_1_on_rpx_error(monkeypatch):
    """Any RPXError bubbling out of a command should map to exit code 1."""
    def _boom(_args):
        raise ConfigError("kaboom", hint="do the thing")

    # Replace the `ls` subcommand body to simulate an RPXError.
    monkeypatch.setattr(cli, "_cmd_ls", _boom)
    rc = cli.main(["ls"])
    assert rc == 1


def test_main_keyboard_interrupt_returns_130(monkeypatch):
    def _boom(_args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "_cmd_ls", _boom)
    rc = cli.main(["ls"])
    assert rc == 130


def test_verbose_flag_sets_debug_level():
    """--verbose should configure logging at DEBUG."""
    import logging
    cli.main(["--verbose", "ls"])
    assert logging.getLogger("rpx_benchmark").level == logging.DEBUG


def test_quiet_flag_sets_warning_level():
    import logging
    cli.main(["--quiet", "ls"])
    assert logging.getLogger("rpx_benchmark").level == logging.WARNING
