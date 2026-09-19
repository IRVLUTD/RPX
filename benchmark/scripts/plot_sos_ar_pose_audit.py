#!/usr/bin/env python3
"""Visualize outputs from ``audit_sos_ar_pose.py`` without rerunning detection."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", "None", None) else float("nan")


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _overview(audit_dir: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    summary = json.loads((audit_dir / "summary.json").read_text())
    sequences = _load_csv(audit_dir / "per_sequence.csv")
    frames = [
        row
        for row in _load_csv(audit_dir / "per_frame.csv")
        if row.get("detected") == "True"
    ]

    names = [row["object_id"] for row in sequences]
    coverage = np.asarray([_float(row, "detection_coverage") for row in sequences])
    order = np.argsort(coverage)
    trans = np.asarray([_float(row, "static_board_translation_jitter_m") for row in frames]) * 100
    rot = np.asarray([_float(row, "static_board_rotation_jitter_deg") for row in frames])
    reprojection = np.asarray([_float(row, "reprojection_rmse_px") for row in frames])
    marker_count = np.asarray([_float(row, "marker_count") for row in frames])
    sequence_trans = np.asarray(
        [_float(row, "static_board_translation_jitter.p95") for row in sequences]
    ) * 100
    sequence_rot = np.asarray(
        [_float(row, "static_board_rotation_jitter.p95") for row in sequences]
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.bar(np.arange(len(names)), coverage[order] * 100, color="#4c78a8")
    ax.set_title("Strict AR-board pose coverage by SOS sequence")
    ax.set_ylabel("Accepted frames (%)")
    ax.set_xlabel("70 sequences, sorted by coverage")

    ax = axes[0, 1]
    ax.hist(trans[np.isfinite(trans)], bins=60, color="#59a14f", alpha=0.85)
    p50 = summary["aggregate"]["static_board_translation_jitter"]["median"] * 100
    p95 = summary["aggregate"]["static_board_translation_jitter"]["p95"] * 100
    ax.axvline(p50, color="black", linestyle="--", label=f"median {p50:.1f} cm")
    ax.axvline(p95, color="#e15759", linestyle="--", label=f"P95 {p95:.1f} cm")
    ax.set_title("Static-board translation residual")
    ax.set_xlabel("Distance from sequence central pose (cm)")
    ax.set_ylabel("Accepted frames")
    ax.legend()

    ax = axes[1, 0]
    scatter = ax.scatter(
        reprojection,
        trans,
        c=marker_count,
        s=8,
        alpha=0.35,
        cmap="viridis",
        rasterized=True,
    )
    ax.set_title("PnP fit quality versus composed-pose residual")
    ax.set_xlabel("Marker reprojection RMSE (px)")
    ax.set_ylabel("Translation residual (cm)")
    fig.colorbar(scatter, ax=ax, label="Inlier markers")

    ax = axes[1, 1]
    ax.scatter(sequence_trans, sequence_rot, color="#f28e2b", alpha=0.75)
    score = np.nan_to_num(sequence_trans / np.nanmedian(sequence_trans)) + np.nan_to_num(
        sequence_rot / np.nanmedian(sequence_rot)
    )
    for index in np.argsort(score)[-8:]:
        ax.annotate(names[index], (sequence_trans[index], sequence_rot[index]), fontsize=7)
    ax.set_title("Per-sequence P95 jitter")
    ax.set_xlabel("Translation P95 (cm)")
    ax.set_ylabel("Rotation P95 (deg)")

    fig.suptitle("RPX SOS: ALVAR board versus stored T265 camera poses", fontsize=15)
    fig.savefig(output_dir / "overview.png", dpi=180)
    fig.savefig(output_dir / "overview.pdf")
    plt.close(fig)


def _sequence_pages(audit_dir: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _load_csv(audit_dir / "per_frame.csv"):
        if row.get("detected") == "True":
            groups[row["object_id"]].append(row)

    with PdfPages(output_dir / "per_sequence_diagnostics.pdf") as pdf:
        for object_id in sorted(groups):
            rows = sorted(groups[object_id], key=lambda row: int(row["frame_idx"]))
            frame = np.asarray([int(row["frame_idx"]) for row in rows])
            translation = np.asarray(
                [_float(row, "static_board_translation_jitter_m") for row in rows]
            ) * 100
            rotation = np.asarray(
                [_float(row, "static_board_rotation_jitter_deg") for row in rows]
            )
            xyz = np.asarray(
                [
                    [
                        _float(row, "world_board_x_m"),
                        _float(row, "world_board_y_m"),
                        _float(row, "world_board_z_m"),
                    ]
                    for row in rows
                ]
            )
            reprojection = np.asarray([_float(row, "reprojection_rmse_px") for row in rows])
            outlier = np.asarray([row.get("static_board_robust_outlier") == "True" for row in rows])

            fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
            axes[0, 0].plot(frame, translation, color="#4c78a8", linewidth=1)
            axes[0, 0].scatter(frame[outlier], translation[outlier], color="#e15759", s=10)
            axes[0, 0].set(title="Translation jitter", xlabel="Frame", ylabel="cm")
            axes[0, 1].plot(frame, rotation, color="#f28e2b", linewidth=1)
            axes[0, 1].scatter(frame[outlier], rotation[outlier], color="#e15759", s=10)
            axes[0, 1].set(title="Rotation jitter", xlabel="Frame", ylabel="degrees")
            axes[1, 0].plot(xyz[:, 0], xyz[:, 2], color="#59a14f", linewidth=1)
            axes[1, 0].scatter(xyz[0, 0], xyz[0, 2], color="black", s=20, label="first")
            axes[1, 0].axis("equal")
            axes[1, 0].set(title="Composed world-board position", xlabel="X (m)", ylabel="Z (m)")
            axes[1, 0].legend()
            axes[1, 1].plot(frame, reprojection, color="#af7aa1", linewidth=1)
            axes[1, 1].axhline(2.0, color="#e15759", linestyle="--", label="strict gate")
            axes[1, 1].set(title="Board PnP reprojection", xlabel="Frame", ylabel="RMSE (px)")
            axes[1, 1].legend()
            fig.suptitle(
                f"{object_id}: {len(rows)} accepted frames; "
                "translation median/P95="
                f"{np.median(translation):.1f}/{np.percentile(translation, 95):.1f} cm; "
                f"rotation median/P95={np.median(rotation):.1f}/{np.percentile(rotation, 95):.1f}°"
            )
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = _arguments()
    output_dir = args.output_dir or args.audit_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    _overview(args.audit_dir, output_dir)
    _sequence_pages(args.audit_dir, output_dir)
    print(f"Overview: {output_dir / 'overview.pdf'}")
    print(f"Sequence diagnostics: {output_dir / 'per_sequence_diagnostics.pdf'}")


if __name__ == "__main__":
    main()
