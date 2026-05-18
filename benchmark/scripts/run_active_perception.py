"""Run an active-perception model end-to-end against an RPX split.

Mirrors the structure of ``run_nvs.py`` for the active-perception axis
(Task #11 in the methodology proposal). The runner gives each adapter
K context-view poses + a pool of candidate poses (the remaining
trajectory in the same scene/phase) and scores the adapter's
predicted next-best pose against an oracle.

Three oracle objectives are wired (override via ``--oracle``):

* ``median_translation`` (default) — candidate pose whose camera
  centre is closest to the *median* of all candidate translations.
  A simple geometric oracle that exercises the pipeline end-to-end;
  defensible as a scaffold but **not** the Coverage-NBV / Task-NBV
  the paper will ultimately use.
* ``farthest_from_centroid`` — candidate farthest from the *context*
  centroid. Mostly a circular comparison against the
  ``farthest_point`` baseline (which picks the same pose); useful as
  a sanity check that the baseline gets perfect scores.
* ``random`` — uniform candidate. Noise-floor sanity check.

The Coverage-NBV / Uncertainty-NBV / Task-NBV oracles from the
proposal are deliberately deferred — each needs ray-bundle math or
upstream-model dependencies that don't belong in this scaffold PR.
The runner is structured so adding them is one new branch in
``_compute_oracle``.

Usage
-----
::

    PYTHONPATH=. python scripts/run_active_perception.py \\
        --model farthest_point --split easy --max-samples 50 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Make ./scripts importable so `from active_perception_models import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ─────────────────────────────────────────────────────────────────────────────
# Pose loader — cached, shared shape with run_nvs.py
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=2048)
def _load_pose(path: Path) -> "Any":
    """Read T265 .npz (position + orientation quaternion) → 4×4 SE(3)."""
    data = np.load(path)
    position  = np.asarray(data["position"], dtype=np.float64)
    quat_xyzw = np.asarray(data["orientation"], dtype=np.float64)
    x, y, z, w = quat_xyzw / np.linalg.norm(quat_xyzw)
    rot = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3]  = position
    T.flags.writeable = False
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Oracle definitions
# ─────────────────────────────────────────────────────────────────────────────


def _compute_oracle(
    objective: str,
    context_poses: List[np.ndarray],
    candidate_poses: List[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick the oracle pose `T*` from the candidate pool per objective."""
    if not candidate_poses:
        # Degenerate fixture — should never happen on real data.
        return context_poses[-1]

    cand_centres = np.stack([np.asarray(p)[:3, 3] for p in candidate_poses], axis=0)

    if objective == "median_translation":
        median = np.median(cand_centres, axis=0)
        d = np.linalg.norm(cand_centres - median, axis=1)
        return candidate_poses[int(np.argmin(d))]
    if objective == "farthest_from_centroid":
        ctx_centres = np.stack([np.asarray(p)[:3, 3] for p in context_poses], axis=0)
        centroid = ctx_centres.mean(axis=0)
        d = np.linalg.norm(cand_centres - centroid, axis=1)
        return candidate_poses[int(np.argmax(d))]
    if objective == "random":
        return candidate_poses[int(rng.integers(0, len(candidate_poses)))]

    from rpx_benchmark.exceptions import ConfigError  # noqa: PLC0415

    raise ConfigError(
        f"unknown oracle objective: {objective!r}",
        hint="--oracle must be one of: median_translation, farthest_from_centroid, random",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Candidate-pool construction
# ─────────────────────────────────────────────────────────────────────────────


def _build_candidate_pool(
    sample: Any,
    extracted_root: Path,
    parquet_path: Path,
) -> List[np.ndarray]:
    """All poses in the sample's (scene_id, phase_target) that are NOT in
    the K context views. Loaded via the cached `_load_pose`."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_parquet(parquet_path, columns=["scene_id", "phase", "frame_idx", "has_cam_pose"])
    pool_df = df[
        (df["scene_id"] == sample.scene_id)
        & (df["phase"].astype(int) == int(sample.phase_target))
        & df["has_cam_pose"].fillna(False).astype(bool)
    ]
    context_idxs = set(sample.context_frame_idxs)
    candidate_idxs = sorted(
        int(r["frame_idx"]) for _, r in pool_df.iterrows() if int(r["frame_idx"]) not in context_idxs
    )

    poses: List[np.ndarray] = []
    for idx in candidate_idxs:
        path = extracted_root / f"scenes/{sample.scene_id}/{sample.phase_target}/cam_pose/{idx:05d}.npz"
        try:
            poses.append(_load_pose(path))
        except (FileNotFoundError, OSError, KeyError):
            continue
    return poses


# ─────────────────────────────────────────────────────────────────────────────
# Model resolution
# ─────────────────────────────────────────────────────────────────────────────


def _build_model(name: str, device: str) -> Any:
    from active_perception_models import MODEL_REGISTRY, list_models  # noqa: PLC0415

    if name not in MODEL_REGISTRY:
        from rpx_benchmark.exceptions import AdapterError  # noqa: PLC0415

        raise AdapterError(
            f"unknown active-perception model: {name!r}",
            hint=f"registered models: {', '.join(list_models())}",
        )
    return MODEL_REGISTRY[name](device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Assemble three-axis result
# ─────────────────────────────────────────────────────────────────────────────


def _assemble_result(
    *,
    model_key: str,
    display_name: str,
    split: str,
    oracle: str,
    per_sample: List[Dict[str, Any]],
    latencies_ms: List[float],
    wall_seconds: float,
) -> Dict[str, Any]:
    from rpx_benchmark.active_perception_metrics import evaluate_active_perception  # noqa: PLC0415

    eval_out = evaluate_active_perception(per_sample)
    return {
        "task":         "active_perception",
        "model":        display_name,
        "model_key":    model_key,
        "split":        split,
        "oracle":       oracle,
        "num_samples":  len(per_sample),
        # Axis 1 — Task Performance (pose geodesic + translation L2)
        "aggregated":   eval_out.get("aggregated", {}),
        # Axis 2 — Scene-change robustness (stratified breakdowns)
        "robustness": {
            "by_sample_type":   eval_out.get("by_sample_type", {}),
            "by_difficulty":    eval_out.get("by_difficulty", {}),
            "by_context_count": eval_out.get("by_context_count", {}),
        },
        # Axis 3 — Compute cost
        "compute_cost": {
            "params_m":              0.0,  # baselines have no params
            "flops_g":               None,
            "latency_ms_per_sample": float(np.median(latencies_ms)) if latencies_ms else None,
            "operating_point": {
                "precision": "fp64",
                "params_m":  0.0,
                "flops_g":   None,
            },
        },
        # Timing — total wall + sample count
        "timing": {
            "total_wall_seconds": wall_seconds,
            "n_samples":          len(latencies_ms),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    from rpx_benchmark.cleanup import install_signal_cleanup
    install_signal_cleanup()

    ap = argparse.ArgumentParser(
        description="Run an active-perception model end-to-end against an RPX split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    help="registered active-perception adapter key")
    ap.add_argument("--split", default="easy", choices=("easy", "medium", "hard"))
    ap.add_argument("--device", default="cpu",
                    help="device for the adapter (baselines are CPU-only)")
    ap.add_argument("--oracle", default="median_translation",
                    choices=("median_translation", "farthest_from_centroid", "random"),
                    help="oracle objective for T*. SCAFFOLD oracles only — the "
                    "Coverage-NBV / Task-NBV / Uncertainty-NBV oracles from the "
                    "proposal are not yet implemented.")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--extracted-root", type=Path, default=None)
    ap.add_argument("--parquet-path", type=Path, default=None)
    ap.add_argument("--repo-id", default="IRVLUTD/RPX")
    ap.add_argument("--results-root", type=Path, default=Path("./rpx_results"))
    ap.add_argument("--seed", type=int, default=5_062_026)
    ap.add_argument("--strict-io", action="store_true",
                    help="crash on missing pose files instead of skipping")
    args = ap.parse_args()

    from rpx_benchmark import cli_ux  # noqa: PLC0415

    cli_ux.banner(
        "run_active_perception — Task #11 (next-best-view)",
        "scaffolded — oracles are placeholders, see SHARED_CONTEXT for the "
        "Coverage / Task / Uncertainty NBV roadmap",
    )
    cli_ux.config(vars(args))

    # ── Build pair generator (reuses NVSPairGenerator for the K-context selection)
    cli_ux.section("Sample generator")
    with cli_ux.working("constructing NVSPairGenerator (active-perception reuses it)"):
        from rpx_benchmark.nvs_pairs import NVSPairGenerator  # noqa: PLC0415

        if args.extracted_root is None or args.parquet_path is None:
            from local_manifest import _hf_snapshot_root  # noqa: PLC0415

            snap = _hf_snapshot_root(args.repo_id)
            extracted_root = args.extracted_root or (snap / "extracted")
            parquet_path = args.parquet_path or (snap / "manifest" / "frames_v1.parquet")
        else:
            extracted_root = args.extracted_root
            parquet_path = args.parquet_path

        gen = NVSPairGenerator(
            extracted_root=extracted_root,
            parquet_path=parquet_path,
            split=args.split,
        )
    cli_ux.note(gen.summary())

    # ── Model
    cli_ux.section("Model")
    adapter = _build_model(args.model, args.device)
    from active_perception_models import MODEL_DISPLAY_NAMES  # noqa: PLC0415

    display = MODEL_DISPLAY_NAMES.get(args.model, args.model)
    cli_ux.kv("display name", display)
    cli_ux.kv("oracle", args.oracle)

    # ── Iterate
    cli_ux.section("Predict + score")
    rng = np.random.default_rng(args.seed)
    per_sample: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    skipped: List[Dict[str, Any]] = []
    t_wall_start = time.perf_counter()

    from rpx_benchmark.active_perception_metrics import (
        evaluate_active_perception_sample,  # noqa: PLC0415
    )

    samples_iter = gen.iter_samples()
    with cli_ux.progress("samples", total=args.max_samples) as (p, task):
        for i, sample in enumerate(samples_iter):
            if args.max_samples is not None and i >= args.max_samples:
                break
            try:
                context_poses  = [_load_pose(extracted_root / pth) for pth in sample.context_pose_paths]
                candidate_poses = _build_candidate_pool(sample, extracted_root, parquet_path)
                if not candidate_poses:
                    skipped.append({"sample_id": sample.id, "reason": "empty candidate pool"})
                    p.update(task, advance=1)  # type: ignore[attr-defined]
                    continue

                oracle_pose = _compute_oracle(args.oracle, context_poses, candidate_poses, rng)

                t0 = time.perf_counter()
                T_pred = adapter(
                    context_poses,
                    candidate_poses=candidate_poses,
                )
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                row = evaluate_active_perception_sample(T_pred, oracle_pose)
                row.update({
                    "sample_id":   sample.id,
                    "scene_id":    sample.scene_id,
                    "phase":       sample.phase,
                    "n_context":   sample.n_context,
                    "sample_type": sample.sample_type,
                    "difficulty":  getattr(sample, "difficulty", None),
                })
                per_sample.append(row)
            except (FileNotFoundError, OSError, KeyError, ValueError) as e:
                if args.strict_io:
                    raise
                skipped.append({"sample_id": getattr(sample, "id", "?"),
                                "reason": f"{type(e).__name__}: {str(e)[:80]}"})
            p.update(task, advance=1)  # type: ignore[attr-defined]

    wall_seconds = time.perf_counter() - t_wall_start

    if not per_sample:
        cli_ux.warn("no samples evaluated — check extracted_root + parquet_path")
        return

    # ── Result
    cli_ux.section("Aggregate")
    result = _assemble_result(
        model_key=args.model,
        display_name=display,
        split=args.split,
        oracle=args.oracle,
        per_sample=per_sample,
        latencies_ms=latencies_ms,
        wall_seconds=wall_seconds,
    )
    if skipped:
        result["skipped_samples"] = skipped

    out_dir = args.results_root / display / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "result.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cli_ux.step(f"wrote {json_path}")

    # ── Summary panel
    a = result["aggregated"]
    rows: Dict[str, Any] = {
        "samples":  result["num_samples"],
        "wall":     cli_ux.fmt_duration(wall_seconds),
        "oracle":   args.oracle,
    }
    if "pose_geodesic_deg" in a and a["pose_geodesic_deg"]:
        rows["pose-geo (median)"] = f"{a['pose_geodesic_deg']['median']:.2f}°"
    if "translation_l2_m" in a and a["translation_l2_m"]:
        rows["trans L2 (median)"] = f"{a['translation_l2_m']['median']:.3f} m"
    if latencies_ms:
        lat = float(np.median(latencies_ms))
        rows["latency"] = f"{lat:.3f} ms / sample" if lat < 1.0 else f"{lat:.1f} ms / sample"
    cli_ux.summary(rows, title=f"{display} · {args.split} · oracle={args.oracle}")


if __name__ == "__main__":
    main()
