#!/usr/bin/env python3
"""Build searchable RPX guides and API pages under ``toolkit-docs/``."""

from __future__ import annotations

import argparse
import os
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
    with tempfile.TemporaryDirectory(prefix="rpx-api-docs-") as tmp:
        subprocess.run(
            [sys.executable, "-m", "pdoc", "rpx_benchmark", "--output-directory", tmp],
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
                    "<style>.rpx-guide-nav{display:flex;flex-wrap:wrap;gap:1.2rem;"
                    "padding:1rem 1.5rem;background:#087f8c;color:white;"
                    "position:sticky;top:0;z-index:1000;box-sizing:border-box;"
                    "font:600 14px system-ui}.rpx-guide-nav a{color:white;"
                    "text-decoration:none}.rpx-guide-nav a:hover{text-decoration:underline}"
                    "html{scroll-padding-top:1rem}body{font-family:system-ui,sans-serif;}"
                    "nav.pdoc{border-right:1px solid #087f8c25;top:52px;height:calc(100vh - 52px);}"
                    "main.pdoc{max-width:1100px;}main.pdoc h1{letter-spacing:-.035em;}"
                    "main.pdoc h2{border-bottom:1px solid #087f8c25;padding-bottom:.4rem;}"
                    "main.pdoc a{color:#087f8c;}main.pdoc pre{border-radius:.5rem;}"
                    "main.pdoc .docstring{line-height:1.7;}</style>"
                )
                html = source.read_text()
                html = html.replace("</head>", style + "</head>", 1)
                import re

                html = re.sub(r"(<body[^>]*>)", lambda m: m[1] + toolbar, html, count=1)
                target.write_text(html)
            else:
                target.write_bytes(source.read_bytes())
    for name in [
        "index.html",
        "metrics/index.html",
        "tasks/index.html",
        "rpx_benchmark/hub.html",
    ]:
        if not (api_output / name).is_file():
            raise RuntimeError(f"Documentation build missing {name}")


if __name__ == "__main__":
    main()
