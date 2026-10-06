#!/usr/bin/env python3
"""Build searchable RPX guides and API pages under ``toolkit-docs/``."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("_site"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / ".nojekyll").touch()
    api_output = output / "toolkit-docs"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "benchmark") + os.pathsep + env.get("PYTHONPATH", "")
    # Explicit discovery bypasses package __all__, which otherwise hides most
    # task, metric and loader modules from pdoc's recursive traversal.
    modules = []
    for source in sorted((ROOT / "benchmark/rpx_benchmark").rglob("*.py")):
        parts = list(source.relative_to(ROOT / "benchmark").with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules.append(".".join(parts))
    with tempfile.TemporaryDirectory(prefix="rpx-api-docs-") as tmp:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/docs/render_api.py"),
                "--templates",
                str(ROOT / "tools/docs/templates"),
                "--output",
                tmp,
                *modules,
            ],
            check=True,
            cwd=ROOT,
            env=env,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "--config-file",
                str(ROOT / "mkdocs.yml"),
                "--site-dir",
                str(api_output),
            ],
            check=True,
            cwd=ROOT,
        )
        # Merge API pages after MkDocs builds (MkDocs clears its destination).
        # Keep its existing module URLs and connect them back to the new guides.
        for source in Path(tmp).rglob("*"):
            if not source.is_file() or source.name == "index.html":
                continue
            relative = source.relative_to(tmp)
            target = api_output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".html":
                depth = len(relative.parts) - 1
                guide_root = "../" * depth or "./"
                toolbar = (
                    '<nav class="rpx-guide-nav" aria-label="Toolkit guides">'
                    f'<a href="{guide_root}">RPX Toolkit</a>'
                    f'<a href="{guide_root}metrics/">Add metrics</a>'
                    f'<a href="{guide_root}tasks/">Add tasks</a>'
                    f'<a href="{guide_root}api/">API overview</a></nav>'
                )
                style = (
                    "<style>" + (ROOT / "tools/docs/api.css").read_text() + "</style>"
                )
                html = source.read_text()
                html = html.replace("</head>", style + "</head>", 1)
                html = re.sub(r"(<body[^>]*>)", lambda m: m[1] + toolbar, html, count=1)
                target.write_text(html)
            else:
                target.write_bytes(source.read_bytes())
    for module in modules:
        name = module.replace(".", "/") + ".html"
        if not (api_output / name).is_file():
            raise RuntimeError(f"Documentation build missing public module {module}")
    subprocess.run(
        [sys.executable, str(ROOT / "tools/docs/check_links.py"), str(api_output)],
        check=True,
    )
    for name in [
        "index.html",
        "metrics/index.html",
        "tasks/index.html",
        "capabilities/index.html",
        "annotation/index.html",
        "profiling/index.html",
        "analysis/index.html",
        "benchmarks/index.html",
        "benchmarks/t1/index.html",
        "benchmarks/t2/index.html",
        "benchmarks/t3/index.html",
        "benchmarks/t4/index.html",
        "benchmarks/t5/index.html",
        "benchmarks/t6/index.html",
        "citation/index.html",
        "rpx_benchmark/hub.html",
        "rpx_benchmark/api.html",
        "rpx_benchmark/adapters.html",
        "rpx_benchmark/metrics/registry.html",
        "rpx_benchmark/tasks/registry.html",
    ]:
        if not (api_output / name).is_file():
            raise RuntimeError(f"Documentation build missing {name}")


if __name__ == "__main__":
    main()
