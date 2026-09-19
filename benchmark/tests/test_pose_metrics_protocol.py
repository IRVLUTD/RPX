from __future__ import annotations

import numpy as np

from rpx_benchmark.pose_metrics import (
    build_rcpe_cells,
    canonical_metric_vector,
    cross_phase_auc_delta,
    metric_auc_curve,
    temporal_drift,
)


def _perfect_row(*, scene: str = "scene001", phase: int = 0) -> dict:
    return {
        "scene_id": scene,
        "phase": phase,
        "pair_type": "intra_phase",
        "rotation_error_deg": 0.0,
        "translation_error_m": 0.0,
        "translation_angular_deg": 0.0,
        "pose_error_max_deg": 0.0,
    }


def test_metric_auc_curve_is_bounded_and_rewards_jointly_correct_pose() -> None:
    perfect = metric_auc_curve(np.zeros(4), np.zeros(4))
    failed = metric_auc_curve(np.full(4, 30.0), np.full(4, 0.30))
    assert perfect == 1.0
    assert failed == 0.0


def test_active_vector_skips_metric_auc_for_up_to_scale_adapter() -> None:
    metrics = canonical_metric_vector(
        [_perfect_row()], metric_translation_available=False
    )
    assert np.isclose(metrics["auc_5deg"], 1.0)
    assert np.isclose(metrics["auc_10deg"], 1.0)
    assert np.isclose(metrics["auc_20deg"], 1.0)
    assert "metric_auc" not in metrics


def test_up_to_scale_rcpe_suppresses_metric_translation_outputs() -> None:
    from rpx_benchmark.pose_metrics import evaluate_rcpe

    row = {
        **_perfect_row(),
        "pair_type": "temporal_chain",
        "chain_id": "chain0",
        "chain_position": 0,
        "pred_rotation": np.eye(3).tolist(),
        "pred_translation": [1.0, 0.0, 0.0],
        "gt_rotation": np.eye(3).tolist(),
        "gt_translation": [0.1, 0.0, 0.0],
    }
    result = evaluate_rcpe([row], metric_translation_available=False)

    assert result["aggregated"]["translation_error_m"] is None
    assert "translation_error_m" not in result["per_type"]["temporal_chain"]
    assert result["temporal_drift"]["mean_drift_trans_m"] is None


def test_rcpe_cells_use_only_intra_pairs_and_canonical_phase_labels() -> None:
    rows = [_perfect_row(phase=phase) for phase in (0, 2)]
    rows.append({**_perfect_row(), "pair_type": "cross_phase"})
    cells, analysis = build_rcpe_cells(
        rows,
        model_name="test-model",
        difficulty="easy",
        metric_translation_available=True,
    )
    assert len(cells) == 2
    assert {cell["phase"] for cell in cells} == {
        "clutter", "clean",
    }
    assert all(cell["n_samples"] == 1 for cell in cells)
    assert all(set(row) >= set(("auc_5deg", "auc_10deg", "auc_20deg")) for row in analysis)
    assert all("metric_auc" not in row for row in analysis)


def test_cross_phase_delta_is_auc_loss_at_matched_rotation_bin() -> None:
    rows = [
        {
            "pair_type": "intra_phase",
            "rotation_bin": "easy",
            "pose_error_max_deg": 0.0,
        },
        {
            "pair_type": "cross_phase",
            "rotation_bin": "easy",
            "pose_error_max_deg": 20.0,
        },
    ]
    result = cross_phase_auc_delta(rows)
    assert result["delta_overall"] > 0.0
    assert result["delta_easy_auc@10"] > 0.0


def test_temporal_drift_composes_se3_edges_instead_of_summing_errors() -> None:
    rotation_90 = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    identity = np.eye(3)
    rows = []
    for position in range(2):
        rows.append(
            {
                "pair_type": "temporal_chain",
                "chain_id": "chain0",
                "chain_position": position,
                "rotation_error_deg": 90.0,
                "translation_error_m": 0.0,
                "pred_rotation": rotation_90.tolist(),
                "pred_translation": [1.0, 0.0, 0.0],
                "gt_rotation": identity.tolist(),
                "gt_translation": [1.0, 0.0, 0.0],
            }
        )
    result = temporal_drift(rows)
    chain = result["chains"][0]
    assert np.isclose(chain["total_drift_trans_m"], np.sqrt(2.0))
    assert np.isclose(chain["total_drift_rot_deg"], 180.0)
