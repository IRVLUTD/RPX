#!/usr/bin/env python3
"""Build the final-compatible 36,000-question RPX VQA benchmark plan.

MOS: 300 (scene, phase) cells, each with exactly 30 frames that are jointly
centered-eligible across all four streams (normal general, normal spatial,
in-context general, in-context spatial). Every frame gets a normal-general
row and a normal-spatial row; a deterministic half (15/15) additionally gets
either an in-context-general or an in-context-spatial third question.
300 * 30 * 3 = 27,000, all immediately runnable.

Ego: 100 scenes, each with exactly 30 centered-for-normal-general frames (a
subset also centered-eligible for in-context general, which ego does have).
Every frame gets a normal-general row (3,000 total); a deterministic 15/30
frames per scene additionally get an in-context-general row (1,500 total) --
currently runnable Ego = 4,500. Ego has no depth map (no normal spatial
exists at all) and no in-context spatial data exists yet, so a normal-spatial
slot is RESERVED (not filled) on all 30 frames, and the other 15 in-context
slots are RESERVED for a future in-context-spatial config -- 3,000 + 1,500 =
4,500 pending. These reservations are plain descriptors, never fabricated
inference rows: they carry no sample_id, no question, no answer.

27,000 + 4,500 = 31,500 available; + 4,500 pending = 36,000 planned total.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from rpx_benchmark.vqa.contract import VQASample
from rpx_benchmark.vqa.sampling import is_centered, rank

SEED = 20260908

NORMAL_GENERAL_TYPES = (
    "attr_single_color",
    "attr_single_material",
    "attr_single_function",
    "attr_composition",
    "attr_odd_one_out",
)
NORMAL_SPATIAL_TYPES = ("spatial_lr_extreme", "depth_closest", "spatial_farthest")
INCONTEXT_GENERAL_TYPES = (
    "inctx_attr_single_color",
    "inctx_attr_single_material",
    "inctx_attr_single_function",
    "inctx_attr_composition",
    "inctx_attr_odd_one_out",
)
INCONTEXT_SPATIAL_TYPES = ("inctx_spatial_farthest",)

FRAMES_PER_MOS_CELL = 30
FRAMES_PER_EGO_SCENE = 30
HALF_SPLIT = 15
MOS_CELLS = 300
EGO_SCENES = 100

EXPECTED_MOS = MOS_CELLS * FRAMES_PER_MOS_CELL * 3
EXPECTED_EGO_AVAILABLE = EGO_SCENES * (FRAMES_PER_EGO_SCENE + HALF_SPLIT)
EXPECTED_EGO_PENDING = EGO_SCENES * (FRAMES_PER_EGO_SCENE + HALF_SPLIT)
EXPECTED_AVAILABLE = EXPECTED_MOS + EXPECTED_EGO_AVAILABLE
EXPECTED_PENDING = EXPECTED_EGO_PENDING
EXPECTED_TOTAL = EXPECTED_AVAILABLE + EXPECTED_PENDING
assert EXPECTED_MOS == 27000
assert EXPECTED_EGO_AVAILABLE == 4500
assert EXPECTED_PENDING == 4500
assert EXPECTED_AVAILABLE == 31500
assert EXPECTED_TOTAL == 36000


def _iter_rows(path: Path, columns: list[str] | None = None):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        yield from batch.to_pylist()


def _row_phase(row: dict) -> int | None:
    raw = row.get("phase")
    if raw is None or (isinstance(raw, float) and raw != raw):  # NaN check without importing math
        return None
    return int(raw)


def _index_by_scene_phase_frame_type(rows, types):
    index: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["type"] not in types or not is_centered(row):
            continue
        key = (row["scene_id"], _row_phase(row), row["frame"])
        index[key][row["type"]].append(row)
    return index


def load_streams(parquet_dir: Path):
    attribute_rows = list(
        _iter_rows(
            parquet_dir / "attribute.parquet",
            columns=["scene_id", "kind", "phase", "frame", "type", "question", "answer_bbox", "img_w", "img_h", "answer"],
        )
    )
    spatial_rows = list(
        _iter_rows(
            parquet_dir / "spatial_bbox.parquet",
            columns=["scene_id", "kind", "phase", "frame", "type", "question", "answer_bbox", "img_w", "img_h", "answer"],
        )
    )
    mos_general = [row for row in attribute_rows if row["kind"] == "mos"]
    ego_general = [row for row in attribute_rows if row["kind"] == "ego"]
    mos_spatial = [row for row in spatial_rows if row["kind"] == "mos"]

    ic_mos_general = list(_iter_rows(parquet_dir / "incontext_mos_attribute_bbox.parquet"))
    ic_ego_general = list(_iter_rows(parquet_dir / "incontext_ego_attribute_bbox.parquet"))
    ic_mos_spatial = list(_iter_rows(parquet_dir / "incontext_mos_spatial_bbox.parquet"))

    return {
        "ng": _index_by_scene_phase_frame_type(mos_general, NORMAL_GENERAL_TYPES),
        "ns": _index_by_scene_phase_frame_type(mos_spatial, NORMAL_SPATIAL_TYPES),
        "ig": _index_by_scene_phase_frame_type(ic_mos_general, INCONTEXT_GENERAL_TYPES),
        "is_": _index_by_scene_phase_frame_type(ic_mos_spatial, INCONTEXT_SPATIAL_TYPES),
        "ego_ng": _index_by_scene_phase_frame_type(ego_general, NORMAL_GENERAL_TYPES),
        "ego_ig": _index_by_scene_phase_frame_type(ic_ego_general, INCONTEXT_GENERAL_TYPES),
    }


def _balanced_assign(frames, index, types, seed, salt):
    """One row per frame, greedily steering toward an even type distribution:
    each frame picks its least-used-so-far type among the types it actually
    has a centered candidate for (ties broken by a seeded hash)."""
    counts = Counter({t: 0 for t in types})
    rows = []
    for frame in frames:
        available = [t for t in types if index[frame].get(t)]
        if not available:
            raise ValueError(f"no candidate for frame {frame} among {types}")
        available.sort(key=lambda t: (counts[t], rank(seed, f"{salt}:{frame}:{t}")))
        chosen_type = available[0]
        candidates = sorted(index[frame][chosen_type], key=lambda r: rank(seed, str(r.get("sample_id") or r["question"])))
        rows.append(candidates[0])
        counts[chosen_type] += 1
    return rows, counts


def _ranked_frames(frame_set, seed, salt, count):
    ordered = sorted(frame_set, key=lambda frame: rank(seed, f"{salt}:{frame}"))
    return ordered[:count]


def _to_plan_row(row: dict, cell: str, slot: str) -> dict:
    sample = VQASample.from_dict(row)
    plan_row = sample.to_dict()
    plan_row["cell"] = cell
    plan_row["slot"] = slot
    return plan_row


def build_mos(streams: dict, seed: int):
    scenes_phases = sorted({key[:2] for key in streams["ng"]} & {key[:2] for key in streams["ns"]})
    available: list[dict] = []
    report_cells = []
    for scene, phase in scenes_phases:
        cell = f"mos:{scene}:{phase}"
        frame_key = lambda frame: (scene, phase, frame)  # noqa: E731
        joint = {
            frame
            for (s, p, frame) in streams["ng"]
            if s == scene
            and p == phase
            and frame_key(frame) in streams["ns"]
            and frame_key(frame) in streams["ig"]
            and frame_key(frame) in streams["is_"]
        }
        if len(joint) < FRAMES_PER_MOS_CELL:
            raise SystemExit(f"{cell}: only {len(joint)} jointly eligible frames, need {FRAMES_PER_MOS_CELL}")
        frames = _ranked_frames(joint, seed, f"mosframes:{cell}", FRAMES_PER_MOS_CELL)
        general_rows, general_counts = _balanced_assign(
            frames, {f: streams["ng"][frame_key(f)] for f in frames}, NORMAL_GENERAL_TYPES, seed, f"ng:{cell}"
        )
        spatial_rows, spatial_counts = _balanced_assign(
            frames, {f: streams["ns"][frame_key(f)] for f in frames}, NORMAL_SPATIAL_TYPES, seed, f"ns:{cell}"
        )
        split_order = sorted(frames, key=lambda frame: rank(seed, f"split:{cell}:{frame}"))
        ic_general_frames = set(split_order[:HALF_SPLIT])
        ic_spatial_frames = set(split_order[HALF_SPLIT:])
        ic_general_rows, ic_general_counts = _balanced_assign(
            sorted(ic_general_frames, key=lambda f: rank(seed, f"ig:{cell}:{f}")),
            {f: streams["ig"][frame_key(f)] for f in ic_general_frames},
            INCONTEXT_GENERAL_TYPES,
            seed,
            f"ig:{cell}",
        )
        ic_spatial_rows, ic_spatial_counts = _balanced_assign(
            sorted(ic_spatial_frames, key=lambda f: rank(seed, f"is:{cell}:{f}")),
            {f: streams["is_"][frame_key(f)] for f in ic_spatial_frames},
            INCONTEXT_SPATIAL_TYPES,
            seed,
            f"is:{cell}",
        )
        for row in general_rows:
            available.append(_to_plan_row(row, cell, "normal_general"))
        for row in spatial_rows:
            available.append(_to_plan_row(row, cell, "normal_spatial"))
        for row in ic_general_rows:
            available.append(_to_plan_row(row, cell, "third_incontext_general"))
        for row in ic_spatial_rows:
            available.append(_to_plan_row(row, cell, "third_incontext_spatial"))
        report_cells.append(
            {
                "cell": cell,
                "joint_eligible_frames": len(joint),
                "frames_selected": len(frames),
                "normal_general_balance": dict(general_counts),
                "normal_spatial_balance": dict(spatial_counts),
                "incontext_general_balance": dict(ic_general_counts),
                "incontext_spatial_balance": dict(ic_spatial_counts),
            }
        )
    return available, report_cells


def build_ego(streams: dict, seed: int):
    scenes = sorted({key[0] for key in streams["ego_ng"]})
    available: list[dict] = []
    pending: list[dict] = []
    report_cells = []
    for scene in scenes:
        cell = f"ego:{scene}"
        joint = {
            frame
            for (s, _p, frame) in streams["ego_ng"]
            if s == scene and (scene, None, frame) in streams["ego_ig"]
        }
        if len(joint) < FRAMES_PER_EGO_SCENE:
            raise SystemExit(f"{cell}: only {len(joint)} joint (ng & ig) frames, need {FRAMES_PER_EGO_SCENE}")
        frames = _ranked_frames(joint, seed, f"egoframes:{cell}", FRAMES_PER_EGO_SCENE)
        frame_key = lambda frame: (scene, None, frame)  # noqa: E731
        general_rows, general_counts = _balanced_assign(
            frames, {f: streams["ego_ng"][frame_key(f)] for f in frames}, NORMAL_GENERAL_TYPES, seed, f"egng:{cell}"
        )
        split_order = sorted(frames, key=lambda frame: rank(seed, f"egsplit:{cell}:{frame}"))
        ic_frames = set(split_order[:HALF_SPLIT])
        pending_ic_frames = set(split_order[HALF_SPLIT:])
        ic_rows, ic_counts = _balanced_assign(
            sorted(ic_frames, key=lambda f: rank(seed, f"egig:{cell}:{f}")),
            {f: streams["ego_ig"][frame_key(f)] for f in ic_frames},
            INCONTEXT_GENERAL_TYPES,
            seed,
            f"egig:{cell}",
        )
        for row in general_rows:
            available.append(_to_plan_row(row, cell, "ego_general"))
        for row in ic_rows:
            available.append(_to_plan_row(row, cell, "ego_incontext_general"))
        for frame in frames:
            pending.append(
                {
                    "cell": cell,
                    "scene_id": scene,
                    "kind": "ego",
                    "phase": None,
                    "frame": frame,
                    "slot": "ego_normal_spatial",
                    "reserved_types": list(NORMAL_SPATIAL_TYPES),
                    "reason": "ego has no depth map; normal spatial ground truth does not exist for ego",
                }
            )
        for frame in pending_ic_frames:
            pending.append(
                {
                    "cell": cell,
                    "scene_id": scene,
                    "kind": "ego",
                    "phase": None,
                    "frame": frame,
                    "slot": "ego_incontext_spatial",
                    "reserved_types": list(INCONTEXT_SPATIAL_TYPES),
                    "reason": "ego in-context spatial ground truth has not been generated yet",
                }
            )
        report_cells.append(
            {
                "cell": cell,
                "joint_ng_ig_frames": len(joint),
                "frames_selected": len(frames),
                "normal_general_balance": dict(general_counts),
                "incontext_general_balance": dict(ic_counts),
                "pending_normal_spatial": len(frames),
                "pending_incontext_spatial": len(pending_ic_frames),
            }
        )
    return available, pending, report_cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    streams = load_streams(args.parquet_dir)
    mos_available, mos_report = build_mos(streams, args.seed)
    ego_available, ego_pending, ego_report = build_ego(streams, args.seed)
    available = mos_available + ego_available

    if len(mos_available) != EXPECTED_MOS:
        raise SystemExit(f"MOS available count {len(mos_available)} != {EXPECTED_MOS}")
    if len(ego_available) != EXPECTED_EGO_AVAILABLE:
        raise SystemExit(f"Ego available count {len(ego_available)} != {EXPECTED_EGO_AVAILABLE}")
    if len(ego_pending) != EXPECTED_PENDING:
        raise SystemExit(f"Ego pending count {len(ego_pending)} != {EXPECTED_PENDING}")
    ids = [row["sample_id"] for row in available]
    if len(ids) != len(set(ids)):
        dupes = [sample_id for sample_id, count in Counter(ids).items() if count > 1]
        raise SystemExit(f"duplicate sample_id in benchmark plan: {dupes[:10]}")
    if len(available) != EXPECTED_AVAILABLE:
        raise SystemExit(f"available count {len(available)} != {EXPECTED_AVAILABLE}")

    available_path = args.out_dir / "benchmark_available_31500.jsonl"
    with available_path.open("w", encoding="utf-8") as handle:
        for row in available:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    pending_path = args.out_dir / "benchmark_pending_4500.json"
    pending_path.write_text(json.dumps(ego_pending, indent=2, sort_keys=True), encoding="utf-8")

    by_slot = Counter(row["slot"] for row in available)
    by_type = Counter(row["question_type"] for row in available)
    plan = {
        "schema_version": "rpx-vqa-benchmark-plan-1.0",
        "seed": args.seed,
        "total_planned": EXPECTED_TOTAL,
        "available": {
            "count": len(available),
            "file": available_path.name,
            "mos": len(mos_available),
            "ego": len(ego_available),
            "by_slot": dict(by_slot),
            "by_question_type": dict(sorted(by_type.items())),
        },
        "pending": {
            "count": len(ego_pending),
            "file": pending_path.name,
            "by_slot": dict(Counter(row["slot"] for row in ego_pending)),
            "reasons": sorted({row["reason"] for row in ego_pending}),
        },
        "mos_cells": MOS_CELLS,
        "ego_scenes": EGO_SCENES,
        "frames_per_mos_cell": FRAMES_PER_MOS_CELL,
        "frames_per_ego_scene": FRAMES_PER_EGO_SCENE,
    }
    plan_path = args.out_dir / "benchmark_plan_36000.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    report = {
        "seed": args.seed,
        "mos_cell_count": len(mos_report),
        "ego_scene_count": len(ego_report),
        "mos_cells": mos_report,
        "ego_cells": ego_report,
        "available_total": len(available),
        "pending_total": len(ego_pending),
        "distinct_scene_frame_available": len(
            {(row["scene_id"], row["kind"], row["phase"], row["frame"]) for row in available}
        ),
    }
    report_path = args.out_dir / "selection_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    import hashlib as _hashlib

    sha_path = args.out_dir / "SHA256SUMS"
    with sha_path.open("w", encoding="utf-8") as handle:
        for path in (available_path, pending_path, plan_path, report_path):
            digest = _hashlib.sha256(path.read_bytes()).hexdigest()
            handle.write(f"{digest}  {path.name}\n")

    print(f"available={len(available)} (mos={len(mos_available)} ego={len(ego_available)})")
    print(f"pending={len(ego_pending)}")
    print(f"total={len(available) + len(ego_pending)}")
    print(f"wrote: {available_path}, {pending_path}, {plan_path}, {report_path}, {sha_path}")


if __name__ == "__main__":
    main()
