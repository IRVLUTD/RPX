"""Unit tests for non-depth metric calculators.

One test per task family, exercising perfect / zero / edge cases.
Shape-mismatch tests are in :mod:`test_metrics_registry`.
"""

from __future__ import annotations

import numpy as np

from rpx_benchmark.api import (
    DetectionGroundTruth,
    DetectionPrediction,
    KeypointCorrespondenceGroundTruth,
    KeypointCorrespondencePrediction,
    NovelViewSynthesisGroundTruth,
    NovelViewSynthesisPrediction,
    RelativePoseGroundTruth,
    RelativePosePrediction,
    SegmentationGroundTruth,
    SegmentationPrediction,
    SparseDepthGroundTruth,
    SparseDepthPrediction,
    TaskType,
    Tracklet,
    TrackletGroundTruth,
    TrackletPrediction,
    VisualGroundingGroundTruth,
    VisualGroundingPrediction,
)
from rpx_benchmark.metrics import compute_metrics


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def test_detection_perfect_match():
    boxes = np.array([[10, 10, 20, 20], [50, 50, 70, 70]], dtype=np.float32)
    labels = ["cup", "bowl"]
    pred = DetectionPrediction(boxes=boxes, scores=[1.0, 1.0], labels=labels)
    gt = DetectionGroundTruth(boxes=boxes, labels=labels)
    out = compute_metrics(TaskType.OBJECT_DETECTION, pred, gt)
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0
    assert out["f1"] == 1.0


def test_detection_missing_gt_yields_zero_recall():
    gt_boxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float32)
    pred = DetectionPrediction(
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=[],
        labels=[],
    )
    gt = DetectionGroundTruth(boxes=gt_boxes, labels=["a", "b"])
    out = compute_metrics(TaskType.OBJECT_DETECTION, pred, gt)
    assert out["recall"] == 0.0


def test_detection_both_empty_returns_unity():
    pred = DetectionPrediction(
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=[],
        labels=[],
    )
    gt = DetectionGroundTruth(boxes=np.zeros((0, 4), dtype=np.float32), labels=[])
    out = compute_metrics(TaskType.OBJECT_DETECTION, pred, gt)
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

def test_segmentation_perfect_mask_miou_one():
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[2:6, 2:6] = 1
    pred = SegmentationPrediction(mask=mask.copy())
    gt = SegmentationGroundTruth(mask=mask.copy())
    out = compute_metrics(TaskType.OBJECT_SEGMENTATION, pred, gt)
    assert out["miou"] == 1.0


def test_segmentation_partial_overlap():
    pred_mask = np.zeros((10, 10), dtype=np.int32)
    gt_mask = np.zeros((10, 10), dtype=np.int32)
    pred_mask[0:6, 0:6] = 1
    gt_mask[3:9, 3:9] = 1
    pred = SegmentationPrediction(mask=pred_mask)
    gt = SegmentationGroundTruth(mask=gt_mask)
    out = compute_metrics(TaskType.OBJECT_SEGMENTATION, pred, gt)
    # Background (0) + class 1 both contribute; 0 < miou < 1
    assert 0.0 < out["miou"] < 1.0


# --------------------------------------------------------------------------- #
# Relative pose
# --------------------------------------------------------------------------- #

def test_relative_pose_identity_is_zero_error():
    eye = np.eye(3, dtype=np.float64)
    zero_t = np.zeros(3, dtype=np.float64)
    pred = RelativePosePrediction(rotation=eye, translation=zero_t)
    gt = RelativePoseGroundTruth(rotation=eye, translation=zero_t)
    out = compute_metrics(TaskType.RELATIVE_CAMERA_POSE, pred, gt)
    assert out["rotation_error_deg"] == 0.0
    assert out["translation_error_m"] == 0.0


def test_relative_pose_known_translation():
    eye = np.eye(3, dtype=np.float64)
    pred_t = np.array([0.0, 0.0, 0.0])
    gt_t = np.array([0.3, 0.4, 0.0])  # L2 = 0.5
    pred = RelativePosePrediction(rotation=eye, translation=pred_t)
    gt = RelativePoseGroundTruth(rotation=eye, translation=gt_t)
    out = compute_metrics(TaskType.RELATIVE_CAMERA_POSE, pred, gt)
    assert out["rotation_error_deg"] == 0.0
    assert abs(out["translation_error_m"] - 0.5) < 1e-6


def test_relative_pose_rotation_error_is_90_deg_for_z_rotation():
    gt_rot = np.eye(3, dtype=np.float64)
    pred_rot = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)  # 90° about Z
    pred = RelativePosePrediction(rotation=pred_rot, translation=np.zeros(3))
    gt = RelativePoseGroundTruth(rotation=gt_rot, translation=np.zeros(3))
    out = compute_metrics(TaskType.RELATIVE_CAMERA_POSE, pred, gt)
    assert abs(out["rotation_error_deg"] - 90.0) < 1e-4


# --------------------------------------------------------------------------- #
# Visual grounding
# --------------------------------------------------------------------------- #

def test_grounding_perfect_single_box_hit():
    box = np.array([[10, 10, 20, 20]], dtype=np.float32)
    pred = VisualGroundingPrediction(boxes=box, scores=np.array([1.0]), labels=["x"])
    gt = VisualGroundingGroundTruth(text="the x", boxes=box)
    out = compute_metrics(TaskType.VISUAL_GROUNDING, pred, gt)
    assert out["grounding_iou"] == 1.0
    assert out["grounding_acc"] == 1.0


def test_grounding_miss_yields_zero_acc():
    pred_box = np.array([[0, 0, 5, 5]], dtype=np.float32)
    gt_box = np.array([[50, 50, 60, 60]], dtype=np.float32)
    pred = VisualGroundingPrediction(
        boxes=pred_box, scores=np.array([1.0]), labels=["x"],
    )
    gt = VisualGroundingGroundTruth(text="the x", boxes=gt_box)
    out = compute_metrics(TaskType.VISUAL_GROUNDING, pred, gt)
    assert out["grounding_acc"] == 0.0


# --------------------------------------------------------------------------- #
# Sparse depth
# --------------------------------------------------------------------------- #

def test_sparse_depth_perfect_points():
    coords = np.array([[5, 5], [10, 10]], dtype=np.float32)
    depths = np.array([1.5, 2.0], dtype=np.float32)
    pred = SparseDepthPrediction(coordinates=coords.copy(), depths=depths.copy())
    gt = SparseDepthGroundTruth(coordinates=coords, depths=depths)
    out = compute_metrics(TaskType.SPARSE_DEPTH, pred, gt)
    assert out["sparse_absrel"] == 0.0
    assert out["sparse_rmse"] == 0.0


def test_sparse_depth_empty_gt_returns_zero():
    pred = SparseDepthPrediction(
        coordinates=np.zeros((0, 2), dtype=np.float32),
        depths=np.zeros((0,), dtype=np.float32),
    )
    gt = SparseDepthGroundTruth(
        coordinates=np.zeros((0, 2), dtype=np.float32),
        depths=np.zeros((0,), dtype=np.float32),
    )
    out = compute_metrics(TaskType.SPARSE_DEPTH, pred, gt)
    assert out["sparse_absrel"] == 0.0


# --------------------------------------------------------------------------- #
# Novel view synthesis
# --------------------------------------------------------------------------- #

def test_nvs_identical_frames_have_high_psnr():
    rgb = (np.random.default_rng(0).uniform(0, 255, size=(8, 8, 3)).astype(np.uint8))
    pred = NovelViewSynthesisPrediction(rgb=rgb.copy())
    gt = NovelViewSynthesisGroundTruth(rgb=rgb.copy())
    out = compute_metrics(TaskType.NOVEL_VIEW_SYNTHESIS, pred, gt)
    assert out["psnr"] >= 99.0
    assert out["ssim"] > 0.99


def test_nvs_different_frames_lower_psnr():
    rgb_gt = np.full((8, 8, 3), 100, dtype=np.uint8)
    rgb_pred = np.full((8, 8, 3), 200, dtype=np.uint8)  # constant offset
    pred = NovelViewSynthesisPrediction(rgb=rgb_pred)
    gt = NovelViewSynthesisGroundTruth(rgb=rgb_gt)
    out = compute_metrics(TaskType.NOVEL_VIEW_SYNTHESIS, pred, gt)
    assert out["psnr"] < 30.0


# --------------------------------------------------------------------------- #
# Keypoint matching
# --------------------------------------------------------------------------- #

def test_keypoint_perfect_matches_accuracy_one():
    points = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
    pred = KeypointCorrespondencePrediction(
        points0=points.copy(), points1=points.copy(),
    )
    gt = KeypointCorrespondenceGroundTruth(
        points0=points, points1=points,
        visibility=np.ones(3, dtype=bool),
    )
    out = compute_metrics(TaskType.KEYPOINT_MATCHING, pred, gt)
    assert out["keypoint_acc"] == 1.0
    assert out["mean_match_error"] == 0.0


def test_keypoint_all_miss_accuracy_zero():
    p0 = np.array([[10, 10]], dtype=np.float32)
    p1_gt = np.array([[20, 20]], dtype=np.float32)
    p1_pred = np.array([[100, 100]], dtype=np.float32)  # way off
    pred = KeypointCorrespondencePrediction(points0=p0, points1=p1_pred)
    gt = KeypointCorrespondenceGroundTruth(
        points0=p0, points1=p1_gt, visibility=np.array([True]),
    )
    out = compute_metrics(TaskType.KEYPOINT_MATCHING, pred, gt)
    assert out["keypoint_acc"] == 0.0


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #

def test_tracking_perfect_single_tracklet():
    boxes = np.array(
        [[[10, 10, 20, 20], [12, 12, 22, 22]]],
        dtype=np.float32,
    )  # 1 track x 2 frames
    tr = Tracklet(track_id="t1", boxes=boxes[0])
    pred = TrackletPrediction(tracks=[tr])
    gt = TrackletGroundTruth(tracks=[tr])
    out = compute_metrics(TaskType.OBJECT_TRACKING, pred, gt)
    assert out["mota"] == 1.0
    assert out["idf1"] == 1.0


def test_tracking_all_missed_has_low_mota():
    pred_boxes = np.array([[0, 0, 5, 5], [0, 0, 5, 5]], dtype=np.float32)
    gt_boxes = np.array([[50, 50, 70, 70], [50, 50, 70, 70]], dtype=np.float32)
    pred = TrackletPrediction(tracks=[Tracklet("p1", pred_boxes)])
    gt = TrackletGroundTruth(tracks=[Tracklet("g1", gt_boxes)])
    out = compute_metrics(TaskType.OBJECT_TRACKING, pred, gt)
    # Zero IoU means every prediction is FP and every gt is FN.
    assert out["mota"] < 0
    assert out["idf1"] == 0.0
