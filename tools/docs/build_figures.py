#!/usr/bin/env python3
"""Rebuild vector workflow figures from the bundled, sourced RPX scene images.

No generated model predictions or invented annotations are used. Each SVG is
self-contained and can be opened or reused independently of the website.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "benchmark/rpx_benchmark/guides/assets"
INK = "#183b38"
MUTED = "#50665f"
PURPLE = "#6254ce"
TEAL = "#137f79"


def text(x, y, label, *, size=25, color=INK, weight=400):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{html.escape(label)}</text>'


def image(name, x, y, width, height):
    mime = "image/png" if name.endswith("png") else "image/jpeg"
    encoded = base64.b64encode((ASSETS / name).read_bytes()).decode()
    return f'<image x="{x}" y="{y}" width="{width}" height="{height}" preserveAspectRatio="xMidYMid meet" href="data:{mime};base64,{encoded}"/>'


def frame(title, subtitle, height=670):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="{height}" viewBox="0 0 1440 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(subtitle)}</desc>',
            '<rect width="100%" height="100%" fill="#fff"/>',
            '<g font-family="Arial, Helvetica, sans-serif">',
            text(40, 49, "RPX / TOOLKIT WORKFLOWS", size=17, color=MUTED, weight=700),
            text(40, 104, title, size=40, weight=700), text(40, 143, subtitle, size=23, color=MUTED),
            '<path d="M40 168H1400" stroke="#d5dfd9"/>']


def panel(x, y, width, height, label, *, accent=TEAL):
    return [f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#f8faf8" stroke="#d5dfd9"/>',
            f'<rect x="{x}" y="{y}" width="{width}" height="5" fill="{accent}"/>',
            text(x + 22, y + 44, label, size=26, weight=700, color=accent)]


def arrow(x1, x2, y):
    return f'<path d="M{x1} {y}H{x2}l-10-7m10 7-10 7" fill="none" stroke="{PURPLE}" stroke-width="3"/>'


def save(name, parts, caption):
    parts += [text(40, 640, caption, size=18, color=MUTED), '</g></svg>']
    (ASSETS / name).write_text("\n".join(parts) + "\n")


def workflow(name, title, subtitle, middle_title, middle, right_title, right,
             *, photo="hero-interaction.jpg", input_label="Scene012 / Interaction", caption=None):
    parts = frame(title, subtitle)
    parts += panel(40, 200, 370, 392, "01 / INPUT")
    if name in {"benchmark-t2.svg", "benchmark-t4.svg", "benchmark-t6.svg"}:
        first = "scene012-sos-reference.png" if name == "benchmark-t6.svg" else "scene012-rgb.png"
        second = "scene012-rgb.png" if name == "benchmark-t6.svg" else "scene012-next.png"
        parts += [image(first, 62, 282, 155, 156), image(second, 233, 282, 155, 156)]
        labels = ("SOS reference", "Target / 00113") if name == "benchmark-t6.svg" else ("RGB / 00113", "RGB / 00114")
        parts += [text(62, 471, labels[0], size=20), text(233, 471, labels[1], size=20)]
        parts += [text(62, 530, "Ordered inputs; same target scene", size=21)]
    else:
        parts += [image(photo, 62, 270, 326, 244), text(62, 558, input_label, size=22)]
    parts += panel(455, 200, 415, 392, f"02 / {middle_title}", accent=PURPLE)
    for index, label in enumerate(middle):
        parts += [text(477, 288 + index * 54, label, size=24)]
    parts += panel(915, 200, 485, 392, f"03 / {right_title}")
    for index, label in enumerate(right):
        parts += [text(937, 288 + index * 54, label, size=24)]
    parts += [arrow(420, 443, 397), arrow(880, 903, 397)]
    save(name, parts, caption or "Real RPX imagery. Output labels describe contracts; no model predictions are fabricated.")


def main():
    parts = frame("Bring your model. Keep the benchmark.",
                  "Shared scenes → task adapters → raw metrics → robustness and deployment evidence.")
    parts += panel(40, 200, 370, 392, "RPX TASK DATA")
    parts += [image("hero-interaction.jpg", 62, 266, 326, 230),
              text(62, 533, "Scene012 + released annotations", size=21),
              text(62, 565, "Manifest, phases, persistent IDs", size=21, color=MUTED)]
    parts += panel(455, 200, 415, 392, "YOUR MODEL / SIX TASKS", accent=PURPLE)
    for i, label in enumerate(["T1 Image depth · T2 Video depth", "T3 Tracking · T4 Camera pose",
                               "T5 Grounding · T6 In-context", "NumPy callable or custom adapters",
                               "Local weights or your API client", "+ Extend tasks and metric calculators"]):
        parts += [text(477, 283 + i * 49, label, size=21)]
    parts += panel(915, 200, 485, 392, "RPX EVALUATION")
    for i, label in enumerate(["Raw task metrics + cell records", "Φ  Phase robustness", "JEDI  Joint desirability",
                               "Jmin  Worst-phase mean quality", "Latency + memory + system card", "JSON reports for reproducible analysis"]):
        parts += [text(937, 283 + i * 49, label, size=22)]
    parts += [arrow(420, 443, 397), arrow(880, 903, 397)]
    save("toolkit-overview.svg", parts, "Annotation tools create GT in a source checkout. Benchmarking consumes the released GT.")
    workflow("getting-started.svg", "A small run, from installation to evidence.",
             "Validate the interface on synthetic inputs, then run a pinned real-data subset.",
             "RUN", ["Install rpx-benchmark", "Discover tasks and metrics", "Run six offline smoke examples", "Connect your actual model", "Select a task-specific manifest"],
             "CHECK", ["Prediction shapes, units and IDs", "Every requested sample accounted for", "Per-sample + aggregate metrics", "Reports and cell records", "Expand to the full protocol"])
    workflow("bring-your-model.svg", "Your inference pipeline. RPX's scoring contract.",
             "Input adapter → local model or API → output adapter → shared task metrics.",
             "ADAPT", ["Preprocess inputs for your model", "Run local or remote inference", "Decode task-specific predictions", "Preserve frame/object identities", "Keep T6 reference → target order"],
             "EVALUATE", ["Ground truth stays in the evaluator", "Failures remain visible", "Raw scores and aggregate reports", "Φ/JEDI from scene-phase records", "Add tasks and add metrics"])
    workflow("mask-pipeline.svg", "Masks: automate, review, correct, repeat.",
             "Annotation creates persistent integer labels; benchmarking reads those labels.",
             "ANNOTATE", ["Suggest and curate bounding boxes", "SAM2: initialize and propagate", "Human: verify frame by frame", "Refine faulty runs from neighbors", "Redraw stubborn frames"],
             "RELEASE", ["Re-review every corrected frame", "Preserve per-scene instance IDs", "Map IDs to object identities", "Package masks with version metadata", "Tracking and grounding consume GT"],
             photo="scene012-rgb.png", caption="Scene012 RGB is the walkthrough input. Right-hand labels describe the annotation workflow, not a new GT release.")
    workflow("hardware-profiler.svg", "Measure the same inference boundary.",
             "Preload inputs. Exclude warmup. Synchronize CUDA. Record the complete configuration.",
             "TIME", ["Fixed inputs + explicit device", "Warmup calls, outside statistics", "One timed call per measurement", "Synchronize every visible GPU", "Keep batch size and scope fixed"],
             "REPORT", ["p50 / p95 / p99 / mean latency", "Raw per-call timing samples", "CPU RSS + per-GPU memory peaks", "Hardware/software system card", "Parameters and roofline via APIs"],
             caption="Measured wall time, model-cost estimates and roofline lower bounds are distinct outputs.")
    workflow("phi-jedi.svg", "Raw metrics → phase robustness and quality.",
             "Use paired scene/phase observations from one model and one task at a time.",
             "CALCULATE", ["Raw rows → scene/phase means", "Three phases in canonical order", "Explicit metric directions/bounds", "Complete-case Φ analysis", "JEDI geometric desirability"],
             "INTERPRET", ["Φ: stability across phase changes", "JEDI: mean cell desirability", "Jmin: minimum phase-mean JEDI", "Dropped scenes + statistical notes", "No Φ reconstruction from table means"],
             caption="Scene012 illustrates a cell. Statistical Φ needs multiple complete scenes with N_eff > K.")
    specifications = [
        ("T1", "Image depth", "One RGB image", ["RGB → your depth estimator", "Return H × W depth", "Metric metres or declared relative", "Preserve the sample identity"], ["RGB-D error metrics", "Validity rules + alignment policy", "Per-image scores → scene/phase cells", "Φ and JEDI from raw cells"]),
        ("T2", "Video depth", "A phase RGB sequence", ["T × H × W × 3 RGB clip", "Return T × H × W depth", "One prediction per input frame", "Declare metric/relative depth"], ["Framewise depth accuracy", "TGM / TGSE temporal errors", "Per-clip scene/phase cell records", "Use the same frame budget"]),
        ("T3", "Object tracking", "Frames + initialization", ["Track objects across frames", "Keep persistent track IDs", "Return boxes with frame identity", "Reset state at sequence boundaries"], ["Generic per-frame tracking scores", "Paper sequence evaluator separately", "MOTA / IDF1 / HOTA per protocol", "Coverage and ID consistency"]),
        ("T4", "Relative camera pose", "Two RGB frames", ["Image A + image B", "Return rotation + translation", "Honor the frame convention", "Use task-defined pose pairs"], ["Rotation / translation errors", "Threshold AUC under the protocol", "Pair IDs and phase eligibility", "Metrics → raw cell records"]),
        ("T5", "Visual grounding", "Target RGB + question", ["One image + canonical prompt", "Return JSON label + bbox", "XYXY normalized to 0–1000", "No answer or target box in inputs"], ["Box IoU / localization accuracy", "Parser validity and question types", "Failures stay in the denominator", "Raw predictions + scored JSON"]),
        ("T6", "In-context grounding", "Reference crop + target", ["Image 1: verified SOS crop", "Image 2: target scene012 frame", "Question → JSON label + bbox", "Box belongs to the TARGET image"], ["Reference hash and crop checks", "Ordered two-image requests", "Shared grounding metrics", "Distinct in-context protocol"]),
    ]
    for task, title, input_kind, middle, right in specifications:
        workflow(f"benchmark-{task.lower()}.svg", f"{task} / {title}", f"{input_kind}. Same scene012 walkthrough, task-specific inputs and scoring.",
                 "MODEL CONTRACT", middle, "BENCHMARK", right,
                 photo="scene012-rgb.png" if task != "T3" else "hero-interaction.jpg",
                 caption="Matched SOS teapot (global ID65) + scene012 target. Illustration only; real requests use their own stored crop and hash."
                 if task == "T6" else None)
    workflow("add-metrics.svg", "Add a metric. Specify what the score means.",
             "Extend a task's calculator registry and, when needed, its raw-metric analysis specs.",
             "REGISTER", ["MetricCalculator.compute(pred, GT)", "Unique calculator and scalar keys", "Explicit validity and units", "@register_metric(TaskType...)", "Import before constructing the suite"],
             "VALIDATE", ["Hand-computed expected scores", "Invalid predictions and empty GT", "Normal runner emits the new score", "MetricSpec for Φ/JEDI inclusion", "Frozen normalization bounds"])
    workflow("add-tasks.svg", "Define a task that other people can reproduce.",
             "A new identity needs source changes; an existing task runner can be replaced at runtime.",
             "IMPLEMENT", ["TaskType + prediction/GT contract", "Loader + task-specific manifest", "Input and output adapters", "Metric suite + public run config", "TaskSpec + public exports"],
             "TEST", ["Synthetic end-to-end pipeline", "Missing inputs and invalid outputs", "Per-sample IDs survive scoring", "A documented real-data smoke run", "A versioned, repeatable protocol"])
    workflow("data-contract.svg", "One capture. Multiple aligned modalities.",
             "Scene012 / Interaction / frame00113. Metric depth shown here is ground truth.",
             "LOAD", ["RGB for model input", "D435 metric depth as GT", "Instance masks and global IDs", "Calibrated stereo and camera pose", "Ego has an independent clock"],
             "JOIN", ["Stream + scene + phase + frame", "Task-specific validity rules", "Depth in metres; masks are IDs", "Pinned manifests and data revision", "Use eligible samples per task"],
             photo="scene012-depth.png", input_label="Scene012 / measured depth", caption="Colorized ground truth: 0.3–5.0 m display ramp; invalid depth is black. Scoring uses metric values.")
    provenance = json.loads((ASSETS / "provenance.json").read_text())
    provenance["workflow_figures"] = [
        {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
         "generator": "tools/docs/build_figures.py", "description": "Vector workflow with sourced RPX imagery; labels are contracts, not model predictions"}
        for path in sorted(ASSETS.glob("*.svg"))]
    (ASSETS / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
