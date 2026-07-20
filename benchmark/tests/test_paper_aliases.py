"""The paper-facing aliases must point at the generic canonical objects.

Guards against the two drifting apart (e.g. someone edits the generic
function but the D1-* alias still binds the old one).
"""

from __future__ import annotations

from rpx_benchmark.data import clip_dataset as cd
from rpx_benchmark.tasks import video_depth as vd


def test_clip_dataset_aliases_are_identity():
    assert cd.D1VClip is cd.PhaseClip
    assert cd.D1VClipDataset is cd.PhaseClipDataset


def test_video_depth_aliases_are_identity():
    assert vd.evaluate_d1v_clip is vd.evaluate_video_depth_clip
    assert vd.d1v_cell_row is vd.video_depth_cell_row
    assert vd.D1V_TASK == vd.VIDEO_DEPTH_TASK == "video_depth"
    assert vd.D1F_MANOVA_METRICS == vd.FRAME_DEPTH_MANOVA_METRICS
    assert vd.D1V_MANOVA_METRICS == vd.VIDEO_DEPTH_MANOVA_METRICS


def test_metric_tuple_K_values():
    # Locked robotics-first set (paper appendix Table 15):
    #   Image Depth K=5 = (absrel, rmse, delta1, silog, fscore_5cm).
    #   Video Depth K=7 = Image Depth K + (opw, tgm).
    # TAE is excluded because RPX ships RGB-D only (no per-frame poses),
    # so SE(3) reprojection cannot run. OPW uses RGB optical flow and TGM
    # uses GT depth gradients — both pose-free.
    assert vd.FRAME_DEPTH_MANOVA_METRICS == (
        "absrel", "rmse", "delta1", "silog", "fscore_5cm",
    )
    assert vd.VIDEO_DEPTH_MANOVA_METRICS == (
        "absrel", "rmse", "delta1", "silog", "fscore_5cm", "opw", "tgm",
    )
    # Video Depth K-vector extends Image Depth's with exactly two temporal keys.
    assert set(vd.VIDEO_DEPTH_MANOVA_METRICS) - set(vd.FRAME_DEPTH_MANOVA_METRICS) == {"opw", "tgm"}
