#!/usr/bin/env python3
"""Build the RPX project landing page and Python API documentation."""

from __future__ import annotations

import argparse
import html
import subprocess
import sys
from pathlib import Path

TASKS = (
    ("Image depth", "AbsRel", "10 reference models"),
    ("Video depth", "AbsRel + temporal", "10 reference models"),
    ("Relative pose", "rotation / translation", "10 reference models"),
    ("Object tracking", "MOTA / HOTA", "mask, box and text init"),
    ("VQA + grounding", "accuracy / bbox", "20 VLM entries"),
    ("Six callable APIs", "task-specific", "bring any model or service"),
)


def landing_page() -> str:
    cards = "\n".join(
        f"<article><h3>{html.escape(name)}</h3><b>{html.escape(metric)}</b>"
        f"<p>{html.escape(scope)}</p></article>"
        for name, metric, scope in TASKS
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>RPX — Robot Perception X</title><meta name="description" content="A real-world RGB-D benchmark for robot perception.">
<style>
:root{{--ink:#14213d;--muted:#596579;--blue:#2563eb;--pale:#eef4ff;--line:#dce4ef}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif;background:#fff}}
nav,main,footer{{max-width:1120px;margin:auto;padding:1rem 1.5rem}}nav{{display:flex;justify-content:space-between;align-items:center}}
nav a{{color:var(--ink);text-decoration:none;margin-left:1.2rem}}.brand{{font-weight:800;font-size:1.3rem;margin:0}}
.hero{{padding:6rem 1.5rem 5rem;background:linear-gradient(135deg,#f7faff,#e8f0ff);text-align:center}}
.hero h1{{font-size:clamp(2.6rem,8vw,5.8rem);line-height:1;margin:.2rem}}.hero p{{font-size:1.3rem;color:var(--muted);max-width:760px;margin:1.5rem auto}}
.button{{display:inline-block;padding:.75rem 1.1rem;border-radius:.55rem;background:var(--blue);color:white;text-decoration:none;font-weight:700;margin:.25rem}}
.button.alt{{background:white;color:var(--blue);border:1px solid var(--line)}}
.stats,.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin:2.5rem 0}}
.stats div,.cards article{{border:1px solid var(--line);border-radius:.8rem;padding:1.2rem;background:white}}.stats strong{{display:block;font-size:2rem;color:var(--blue)}}
h2{{font-size:2rem;margin-top:3rem}}h3{{margin:.1rem 0}}article p{{color:var(--muted);margin:.4rem 0}}
pre{{overflow:auto;background:#111827;color:#e5e7eb;padding:1.2rem;border-radius:.7rem}}code{{font-family:ui-monospace,SFMono-Regular,monospace}}
.flow{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}}.flow div{{background:var(--pale);padding:1.2rem;border-radius:.8rem}}
footer{{border-top:1px solid var(--line);color:var(--muted);margin-top:4rem}}@media(max-width:650px){{.flow{{grid-template-columns:1fr}}nav span{{display:none}}}}
</style></head><body>
<nav><a class="brand" href="./">RPX</a><span><a href="toolkit-docs/">Toolkit API</a><a href="https://huggingface.co/datasets/IRVLUTD/RPX">Dataset</a><a href="https://github.com/IRVLUTD/RPX">GitHub</a></span></nav>
<header class="hero"><h1>RPX</h1><p>A real-world RGB-D benchmark for choosing robot-perception models by task quality, scene-change robustness, and compute cost.</p>
<a class="button" href="toolkit-docs/">Explore the toolkit API</a><a class="button alt" href="https://github.com/IRVLUTD/RPX#readme">Run the benchmark</a></header>
<main><section class="stats"><div><strong>100</strong>indoor scenes</div><div><strong>3</strong>states per scene</div><div><strong>10</strong>core task APIs</div><div><strong>75k+</strong>RGB-D frames</div></section>
<section><h2>One benchmark, three views of deployment</h2><div class="flow"><div><h3>1. Task performance</h3>Use the metric that belongs to the task, stratified by difficulty.</div><div><h3>2. Robustness</h3>Measure behavior across clutter, interaction, and clean states.</div><div><h3>3. Compute cost</h3>Record parameters and observed runtime with hardware provenance.</div></div></section>
<section><h2>Tasks and tools</h2><div class="cards">{cards}</div></section>
<section><h2>Bring a model in a few lines</h2><pre><code>import rpx_benchmark as rpx

model = rpx.make_numpy_depth_model(predict_depth, name="my-depth")
result, report, paths = rpx.run_monocular_depth(
    rpx.MonocularDepthRunConfig(model=model, split="easy")
)</code></pre><p>The same evaluator accepts a local checkpoint or an API-backed callable. Dataset downloads default to the immutable consolidated RPX release.</p></section>
<section><h2>Reproduce or extend</h2><p>Use the supplied Docker environments for reference sweeps, or connect your own model through the public Python and VQA callable contracts. Every route writes comparable task metrics and provenance.</p><a class="button" href="toolkit-docs/">Read API documentation</a> <a class="button alt" href="https://github.com/IRVLUTD/RPX/blob/naren/all/benchmark/README.md">Usage examples</a></section></main>
<footer>RPX is maintained by IRVL at The University of Texas at Dallas. Code: MIT. Dataset: CC BY 4.0.</footer></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("_site"))
    parser.add_argument("--project-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(landing_page(), encoding="utf-8")
    (output / ".nojekyll").touch()
    if args.project_only:
        return
    api_output = output / "toolkit-docs"
    subprocess.run(
        [sys.executable, "-m", "pdoc", "rpx_benchmark", "--output-directory", str(api_output)],
        check=True,
    )
    if not (api_output / "index.html").is_file():
        raise RuntimeError("pdoc did not create toolkit-docs/index.html")


if __name__ == "__main__":
    main()
