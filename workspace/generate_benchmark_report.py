#!/usr/bin/env python3
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


BASE = Path("/media/naren/New Volume/rpx/paper-metrics/paper-metrics-export-20260906-225228")
OUTPUT = Path("/media/naren/New Volume/rpx/RPX_BENCHMARK_RESULTS.md")

MODEL_NAMES = {
    "vggt-omega": "VGGT-Ω",
    "da3": "DA3-GIANT",
    "cut3r": "CUT3R-512-DPT",
    "mast3r": "MASt3R-ViTL-512",
    "must3r": "MUSt3R-512",
    "dust3r": "DUSt3R-ViTL-512",
    "reloc3r": "Reloc3r-512",
    "pi3x": "Pi3X",
    "fast3r": "Fast3R-ViT-Large-512",
    "monst3r": "MonST3R-ViTL-512",
}


def read_tsv(name):
    with (BASE / name).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def mean(values):
    values = [v for v in (number(x) for x in values) if v is not None]
    return statistics.fmean(values) if values else None


def fmt(value, digits=3, scale=1):
    value = number(value)
    if value is None:
        return "—"
    return f"{value * scale:.{digits}f}"


def integer(value):
    value = number(value)
    return "—" if value is None else f"{int(round(value)):,}"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def json_at(path):
    with path.open() as handle:
        return json.load(handle)


rcpe = read_tsv("rcpe-main-table.tsv")
phases = read_tsv("rcpe-phase-table.tsv")
tracking_overall = read_tsv("tracking-overall-table.tsv")
tracking_split = read_tsv("tracking-split-table.tsv")
tracking_phase = read_tsv("tracking-phase-table.tsv")
tracking_validation = read_tsv("tracking-validation.tsv")

split_order = {"easy": 0, "medium": 1, "hard": 2}
model_order = {key: i for i, key in enumerate(MODEL_NAMES)}
rcpe.sort(key=lambda r: (model_order.get(r["model_key"], 99), split_order[r["split"]]))

complete_rcpe = [r for r in rcpe if r["status"] == "COMPLETE"]
complete_tracking_models = {
    "cutie", "dam4sam", "edgetam", "mits",
    "sam2", "sam2-plus", "sam2long", "xmem",
}

sections = []
sections.append("""# RPX Benchmark Results Handover

This document consolidates the final downloaded RCPE and object-tracking benchmark artifacts. Values are reproduced from the exported result JSON/Parquet files; missing measurements are shown as `—`, never as zero.

## Validation and scope

- Archive SHA-256: `8eb91a139e4dfb2b7d2d60db0b819e1f7c5f5733c829e797254b6293f86da879` (verified locally).
- RCPE: **30/30 complete cells** — 10 models × Easy/Medium/Hard.
- RCPE prediction counts: Easy **6,676**, Medium **6,715**, Hard **6,879** unique predictions per model.
- Tracking MOS, frame budget 250: **8 complete models**, each with 300 cells, 100 scenes, and 75,000 evaluated frames.
- Tracking Ego, frame budget 250: **8 complete models**, each with 100 cells, 100 scenes, and 23,121 frames.
- A partial MOS MOTIP result (35/300 cells; 12/100 scenes) exists but is excluded from all headline tables and comparisons.

## Metric interpretation

- RCPE rotation, translation-angle, pose error, and drift: **lower is better**.
- RCPE AUC@5/10/20: **higher is better**. These values are in `[0,1]`.
- Metric translation in metres is only reported where the model/run supplies metric scale. Its absence is not treated as failure.
- Tracking HOTA, DetA, AssA, IDF1, and MOTA: **higher is better**. Tables display them as percentages (`raw value × 100`).
- Tracking ID switches and latency: **lower is better**. Throughput: **higher is better**.
- RCPE phase results compare phase 0 with phase 2 independently. Direct cross-phase relative-pose pairs are invalid because the two captures use unrelated T265 local world frames.
""")

# RCPE macro summary
groups = defaultdict(list)
for row in complete_rcpe:
    groups[row["model_key"]].append(row)

macro = []
for model, rows in groups.items():
    macro.append({
        "model": model,
        "rot": mean(r["rotation_error_deg_mean"] for r in rows),
        "trans": mean(r["translation_angular_deg"] for r in rows),
        "pose": mean(r["pose_error_max_deg"] for r in rows),
        "a5": mean(r["auc_5deg"] for r in rows),
        "a10": mean(r["auc_10deg"] for r in rows),
        "a20": mean(r["auc_20deg"] for r in rows),
        "drift": mean(r["mean_drift_rotation_deg"] for r in rows),
    })
macro.sort(key=lambda r: r["pose"] if r["pose"] is not None else math.inf)

sections.append("## RCPE headline summary (macro-average across difficulty splits)\n\n" + markdown_table(
    ["Model", "Rot. err. ° ↓", "Trans. angle ° ↓", "Pose err. ° ↓", "AUC@5 ↑", "AUC@10 ↑", "AUC@20 ↑", "Rot. drift ° ↓"],
    [[MODEL_NAMES.get(r["model"], r["model"]), fmt(r["rot"]), fmt(r["trans"]), fmt(r["pose"]), fmt(r["a5"], 4), fmt(r["a10"], 4), fmt(r["a20"], 4), fmt(r["drift"])] for r in macro]
))

# RCPE full split table
sections.append("## RCPE full split-wise results\n\n" + markdown_table(
    ["Model", "Split", "Pairs", "Rot. mean ° ↓", "Rot. median ° ↓", "Trans. angle ° ↓", "Trans. m ↓", "Pose ° ↓", "AUC@5 ↑", "AUC@10 ↑", "AUC@20 ↑", "Rot. drift ° ↓"],
    [[MODEL_NAMES.get(r["model_key"], r["model"]), r["split"].title(), integer(r["n_pairs"]), fmt(r["rotation_error_deg_mean"]), fmt(r["rotation_error_deg_median"]), fmt(r["translation_angular_deg"]), fmt(r["translation_error_m_mean"], 4), fmt(r["pose_error_max_deg"]), fmt(r["auc_5deg"], 4), fmt(r["auc_10deg"], 4), fmt(r["auc_20deg"], 4), fmt(r["mean_drift_rotation_deg"])] for r in complete_rcpe]
))

# RCPE efficiency from raw result files; report medians across the three runs.
efficiency = []
for model in MODEL_NAMES:
    records = []
    for split in ("easy", "medium", "hard"):
        path = BASE / "raw" / "rcpe" / model / split / "result.json"
        if not path.exists():
            continue
        doc = json_at(path)
        records.append((doc.get("aggregated", {}), doc.get("compute_cost", {})))
    def med(which, key):
        vals = [number(pair[which].get(key)) for pair in records]
        vals = [v for v in vals if v is not None]
        return statistics.median(vals) if vals else None
    efficiency.append([
        MODEL_NAMES[model], fmt(med(1, "params_m"), 3), fmt(med(1, "memory_traffic_gb"), 3),
        fmt(med(1, "latency_ms_per_sample"), 2), fmt(med(0, "latency_ms"), 2),
        fmt(med(1, "peak_memory_mb"), 1),
    ])

sections.append("""## RCPE efficiency and resource measurements

Values are medians across Easy/Medium/Hard runs. All measurements used batch size 1 at 640×480 on an NVIDIA RTX 6000 Ada. FLOPs/MACs were not populated by these runs and are therefore omitted rather than reported as zero.

""" + markdown_table(
    ["Model", "Params (M)", "Memory traffic (GB)", "Profile latency (ms/sample) ↓", "Observed latency (ms) ↓", "Peak memory (MB) ↓"], efficiency
))

# RCPE phase-conditioned deltas.
phase_map = defaultdict(dict)
for row in phases:
    phase_map[(row["model"], row["split"])][str(row["phase"])] = row
phase_rows = []
for model in MODEL_NAMES:
    for split in ("easy", "medium", "hard"):
        p0 = phase_map[(model, split)].get("0", {})
        p2 = phase_map[(model, split)].get("2", {})
        def delta(key):
            a, b = number(p0.get(key)), number(p2.get(key))
            return None if a is None or b is None else b - a
        phase_rows.append([
            MODEL_NAMES[model], split.title(), integer(p0.get("n_pairs")), integer(p2.get("n_pairs")),
            fmt(p0.get("rotation_error_deg")), fmt(p2.get("rotation_error_deg")), fmt(delta("rotation_error_deg")),
            fmt(p0.get("translation_angular_deg")), fmt(p2.get("translation_angular_deg")), fmt(delta("translation_angular_deg")),
            fmt(p0.get("pose_error_max_deg")), fmt(p2.get("pose_error_max_deg")), fmt(delta("pose_error_max_deg")),
            fmt(p0.get("auc_20deg"), 4), fmt(p2.get("auc_20deg"), 4), fmt(delta("auc_20deg"), 4),
        ])

sections.append("""## RCPE phase-conditioned results

Delta is `phase 2 − phase 0`; a negative error delta is an improvement, while a positive AUC delta is an improvement.

""" + markdown_table(
    ["Model", "Split", "P0 N", "P2 N", "P0 rot", "P2 rot", "Δ rot", "P0 trans", "P2 trans", "Δ trans", "P0 pose", "P2 pose", "Δ pose", "P0 AUC20", "P2 AUC20", "Δ AUC20"], phase_rows
))

# Phi/JEDI directly from raw JSON (exported convenience table did not flatten nested objects).
phi_rows = []
for model in MODEL_NAMES:
    for split in ("easy", "medium", "hard"):
        path = BASE / "raw" / "rcpe" / model / split / "phi_jedi.json"
        doc = json_at(path)
        phi = doc.get("phi", {}) or {}
        jedi = doc.get("jedi", {}) or {}
        per = jedi.get("per_phase", {}) or {}
        clutter, clean = number(per.get("clutter")), number(per.get("clean"))
        phi_rows.append([
            MODEL_NAMES[model], split.title(), fmt(phi.get("phi_wilks"), 6),
            fmt(phi.get("p_value"), 6), doc.get("phi_interpretation", "—"),
            fmt(jedi.get("jedi"), 6), fmt(clutter, 6), fmt(clean, 6),
            fmt(None if clutter is None or clean is None else clean - clutter, 6),
            integer(phi.get("n_eff")), integer(doc.get("n_dropped")),
        ])

sections.append("""## RCPE Phi/JEDI analysis

Phi summarizes the multivariate phase effect over AUC@5/10/20. JEDI is reported overall and for clutter/clean phase labels. These are descriptive per-cell analyses; multiplicity correction should be specified before making confirmatory significance claims.

""" + markdown_table(
    ["Model", "Split", "Phi (Wilks)", "p", "Effect", "JEDI", "JEDI clutter", "JEDI clean", "Δ JEDI", "n eff.", "Dropped"], phi_rows
))

# Tracking helpers.
def tracking_row(r, include_split=False):
    row = [r["model"]]
    if include_split:
        row.append(r["split"].title())
    row.extend([
        integer(r["cells"]), integer(r["unique_scenes"]), integer(r["total_frames"]),
        fmt(r["hota_mean"], 2, 100), fmt(r["deta_mean"], 2, 100),
        fmt(r["assa_mean"], 2, 100), fmt(r["idf1_mean"], 2, 100),
        fmt(r["mota_mean"], 2, 100), integer(r["idsw_total"]),
        fmt(r["latency_median_ms"], 2), fmt(r["throughput_median_fps"], 2),
        fmt(number(r["peak_gpu_memory_allocated_mb_max"]) / 1024 if number(r["peak_gpu_memory_allocated_mb_max"]) is not None else None, 2),
        fmt(number(r["parameter_count_max"]) / 1e6 if number(r["parameter_count_max"]) is not None else None, 2),
    ])
    return row

tracking_headers = ["Model", "Cells", "Scenes", "Frames", "HOTA % ↑", "DetA % ↑", "AssA % ↑", "IDF1 % ↑", "MOTA % ↑", "IDSW ↓", "Latency ms ↓", "FPS ↑", "Peak VRAM GiB ↓", "Params M"]

for protocol, title in (("mos", "Tracking MOS — frame budget 250"), ("ego", "Tracking Ego — frame budget 250")):
    rows = [r for r in tracking_overall if r["protocol"] == protocol and r["model"] in complete_tracking_models and number(r["cells"]) == (300 if protocol == "mos" else 100)]
    rows.sort(key=lambda r: number(r["hota_mean"]), reverse=True)
    sections.append(f"## {title}: overall\n\n" + markdown_table(tracking_headers, [tracking_row(r) for r in rows]))

    split_rows = [r for r in tracking_split if r["protocol"] == protocol and r["model"] in complete_tracking_models]
    split_rows.sort(key=lambda r: (r["model"], split_order.get(r["split"], 99)))
    sections.append(f"## {title}: difficulty-wise\n\n" + markdown_table(
        ["Model", "Split"] + tracking_headers[1:],
        [tracking_row(r, include_split=True) for r in split_rows],
    ))

# MOS phase summary, pooled over difficulty tiers for each model/phase.
mos_phase_groups = defaultdict(list)
for r in tracking_phase:
    if r["protocol"] == "mos" and r["model"] in complete_tracking_models:
        mos_phase_groups[(r["model"], r["phase"])].append(r)

mos_phase_rows = []
for (model, phase), rows in sorted(mos_phase_groups.items(), key=lambda x: (x[0][0], x[0][1])):
    weights = [number(r["cells"]) or 0 for r in rows]
    def weighted(key):
        pairs = [(number(r[key]), w) for r, w in zip(rows, weights)]
        pairs = [(v, w) for v, w in pairs if v is not None and w > 0]
        return sum(v*w for v, w in pairs) / sum(w for _, w in pairs) if pairs else None
    mos_phase_rows.append([
        model, phase, integer(sum(weights)),
        fmt(weighted("hota_mean"), 2, 100), fmt(weighted("deta_mean"), 2, 100),
        fmt(weighted("assa_mean"), 2, 100), fmt(weighted("idf1_mean"), 2, 100),
        fmt(weighted("mota_mean"), 2, 100), integer(sum(number(r["idsw_total"]) or 0 for r in rows)),
    ])

sections.append("## Tracking MOS phase-wise results (pooled across difficulty)\n\n" + markdown_table(
    ["Model", "Phase", "Cells", "HOTA % ↑", "DetA % ↑", "AssA % ↑", "IDF1 % ↑", "MOTA % ↑", "IDSW ↓"], mos_phase_rows
))

# Concise derived observations.
best_pose = min(macro, key=lambda r: r["pose"])
best_auc = max(macro, key=lambda r: r["a20"])
best_rot = min(macro, key=lambda r: r["rot"])
observations = [
    f"- Lowest RCPE macro rotation error: **{MODEL_NAMES[best_rot['model']]}** ({fmt(best_rot['rot'])}°).",
    f"- Lowest RCPE macro pose error: **{MODEL_NAMES[best_pose['model']]}** ({fmt(best_pose['pose'])}°).",
    f"- Highest RCPE macro AUC@20: **{MODEL_NAMES[best_auc['model']]}** ({fmt(best_auc['a20'], 4)}).",
]
for protocol, label in (("mos", "MOS-250"), ("ego", "Ego-250")):
    rows = [r for r in tracking_overall if r["protocol"] == protocol and r["model"] in complete_tracking_models and number(r["cells"]) == (300 if protocol == "mos" else 100)]
    best = max(rows, key=lambda r: number(r["hota_mean"]))
    fast = min(rows, key=lambda r: number(r["latency_median_ms"]))
    observations.append(f"- Highest {label} HOTA: **{best['model']}** ({fmt(best['hota_mean'], 2, 100)}%); fastest median inference: **{fast['model']}** ({fmt(fast['latency_median_ms'], 2)} ms).")

sections.insert(2, "## Headline observations\n\n" + "\n".join(observations))

sections.append("""## Reporting cautions

1. Do not include partial MOTIP MOS numbers in model rankings; only 35 of the required 300 cells are available.
2. RCPE metric translation in metres is unavailable for up-to-scale methods. Compare translation-angle and pose error across all models; use metre error only within the metric-scale subset.
3. Difficulty tiers contain different scene compositions, so Easy → Medium → Hard is not guaranteed to be numerically monotonic for every metric.
4. Latency and memory figures are hardware- and software-stack-specific. Report the RTX 6000 Ada, batch size 1, resolution, CUDA/PyTorch, and precision configuration alongside them.
5. HOTA-family means in this document are macro averages over benchmark cells. IDSW is summed.
6. The complete machine-readable exports remain in `paper-metrics/paper-metrics-export-20260906-225228/`, including the exhaustive `*-all-aggregate-scalars.tsv` tables and normalized raw JSON/Parquet artifacts.
""")

OUTPUT.write_text("\n\n".join(sections).rstrip() + "\n")
print(OUTPUT)
