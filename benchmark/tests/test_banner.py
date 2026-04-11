"""Tests for the RPX startup banner."""

from __future__ import annotations

import io

from rpx_benchmark.banner import show_banner


def test_banner_writes_to_stderr_by_default(capsys):
    show_banner()
    captured = capsys.readouterr()
    assert "RPX" in captured.err or "Robot Perception X" in captured.err
    # Nothing on stdout — banner must never pollute scripting pipelines.
    assert captured.out == ""


def test_banner_respects_disabled():
    buf = io.StringIO()
    show_banner(enabled=False, file=buf)
    assert buf.getvalue() == ""


def test_banner_respects_env_var(monkeypatch):
    monkeypatch.setenv("RPX_NO_BANNER", "1")
    buf = io.StringIO()
    show_banner(file=buf)
    assert buf.getvalue() == ""


def test_banner_forced_on_ignores_env_var(monkeypatch):
    monkeypatch.setenv("RPX_NO_BANNER", "1")
    buf = io.StringIO()
    show_banner(enabled=True, file=buf)
    assert "RPX" in buf.getvalue() or "Robot Perception" in buf.getvalue()


def test_banner_writes_to_provided_file():
    buf = io.StringIO()
    show_banner(file=buf)
    assert "RPX" in buf.getvalue() or "Robot Perception" in buf.getvalue()


def test_banner_includes_context_and_subtitle():
    buf = io.StringIO()
    show_banner(
        subtitle="rpx bench monocular_depth",
        context="task=monocular_depth split=hard",
        file=buf,
    )
    text = buf.getvalue()
    # Either backend (rich or plain) should surface the context string.
    assert "monocular_depth" in text
    assert "hard" in text


def test_banner_plain_backend_fallback(monkeypatch):
    """When rich is unavailable the plain backend still writes something."""
    import sys as _sys

    # Force-pretend rich is missing by stashing a fake loader.
    real_rich = _sys.modules.get("rich")
    real_rich_submods = {k: v for k, v in _sys.modules.items() if k.startswith("rich")}
    for k in real_rich_submods:
        _sys.modules[k] = None  # type: ignore[assignment]
    try:
        buf = io.StringIO()
        show_banner(file=buf)
        assert "RPX" in buf.getvalue() or "Robot Perception" in buf.getvalue()
    finally:
        # Restore so later tests can still use rich.
        for k, v in real_rich_submods.items():
            _sys.modules[k] = v
