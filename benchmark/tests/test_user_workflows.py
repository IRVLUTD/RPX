"""Public user workflows: real local manifests, callables, scoring and artifacts."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

import rpx_benchmark as rpx
from rpx_benchmark.tasks.registry import get_task_spec

CONFIGS = {
    rpx.TaskType.MONOCULAR_DEPTH: rpx.MonocularDepthRunConfig,
    rpx.TaskType.VIDEO_DEPTH: rpx.VideoDepthRunConfig,
    rpx.TaskType.OBJECT_SEGMENTATION: rpx.SegmentationRunConfig,
    rpx.TaskType.OBJECT_DETECTION: rpx.ObjectDetectionRunConfig,
    rpx.TaskType.OPEN_VOCAB_DETECTION: rpx.ObjectDetectionRunConfig,
    rpx.TaskType.OBJECT_TRACKING: rpx.ObjectTrackingRunConfig,
    rpx.TaskType.RELATIVE_CAMERA_POSE: rpx.RelativePoseRunConfig,
    rpx.TaskType.VISUAL_GROUNDING: rpx.VisualGroundingRunConfig,
    rpx.TaskType.SPARSE_DEPTH: rpx.SparseDepthRunConfig,
    rpx.TaskType.KEYPOINT_MATCHING: rpx.KeypointMatchingRunConfig,
}


def local_case(root, task):
    rgb = np.full((16, 20, 3), 128, np.uint8)
    mask = np.zeros((16, 20), np.uint8)
    mask[2:10, 3:12] = 1
    Image.fromarray(rgb).save(root / "rgb.png")
    Image.fromarray(mask).save(root / "mask.png")
    Image.fromarray(np.full((16, 20), 2000, np.uint16)).save(root / "depth.png")
    np.savez(root / "pose.npz", position=np.zeros(3), orientation=np.array([0., 0., 0., 1.]))
    boxes = np.array([[3, 2, 12, 10]], np.float32)
    points = np.array([[4, 4], [8, 8]], np.float32)
    sample = dict(id="scene001_0_00000", scene="scene001", scene_id="scene001",
                  phase="clutter", difficulty="easy", rgb="rgb.png")
    if task == rpx.TaskType.MONOCULAR_DEPTH:
        sample["depth"] = "depth.png"
        model = rpx.make_numpy_depth_model(lambda rgb: np.full(rgb.shape[:2], 2., np.float32))
    elif task == rpx.TaskType.VIDEO_DEPTH:
        sample.update(phase=0, frame_filenames=["rgb.png"] * 3,
                      depth_filenames=["depth.png"] * 3)
        model = rpx.make_numpy_video_depth_model(lambda rgb: np.broadcast_to(np.linspace(1., 3., rgb.shape[2]), rgb.shape[:3]).astype(np.float32))
    elif task == rpx.TaskType.OBJECT_SEGMENTATION:
        sample["mask"] = "mask.png"
        model = rpx.make_numpy_mask_model(lambda rgb: mask.copy())
    elif task in (rpx.TaskType.OBJECT_DETECTION, rpx.TaskType.OPEN_VOCAB_DETECTION):
        (root / "boxes.json").write_text(json.dumps([{"bbox": boxes[0].tolist(), "label": "cup"}]))
        sample["boxes"] = "boxes.json"
        model = rpx.make_numpy_detection_model(
            lambda rgb: dict(boxes=boxes, scores=np.ones(1), labels=["cup"]), task=task)
    elif task == rpx.TaskType.OBJECT_TRACKING:
        sample["mask"] = "mask.png"
        model = rpx.make_numpy_tracking_model(lambda rgb: [{"track_id": "1", "boxes": boxes}])
    elif task == rpx.TaskType.VISUAL_GROUNDING:
        sample.update(text="the cup", boxes=boxes.tolist())
        model = rpx.make_numpy_grounding_model(lambda rgb, text: dict(boxes=boxes, scores=np.ones(1)))
    elif task == rpx.TaskType.RELATIVE_CAMERA_POSE:
        sample.update(rgb_b="rgb.png", pose_a="pose.npz", pose_b="pose.npz")
        model = rpx.make_numpy_pose_model(lambda a, b: dict(rotation=np.eye(3), translation=np.zeros(3)))
    elif task == rpx.TaskType.SPARSE_DEPTH:
        sample.update(coordinates=points.tolist(), depths=[2., 2.])
        model = rpx.make_numpy_sparse_depth_model(lambda rgb, coords: np.full(len(coords), 2.))
    else:
        sample.update(rgb_b="rgb.png", points0=points.tolist(), points1=points.tolist(), visibility=[True, True])
        model = rpx.make_numpy_keypoint_model(lambda a, b: (points, points))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(dict(task=task.value, root=str(root), samples=[sample])))
    return model, manifest


@pytest.mark.parametrize("task", list(rpx.TaskType))
def test_public_task_callable_to_report(task, tmp_path):
    """Every advertised task works through its public config and runner."""
    model, manifest = local_case(tmp_path, task)
    kwargs = {"compute_fscore": False} if task == rpx.TaskType.VIDEO_DEPTH else {}
    cfg = CONFIGS[task](model=model, split="easy", manifest_path=str(manifest),
                       output_dir=str(tmp_path / "results"), device="cpu", skip_flops=True, **kwargs)
    result, _, paths = get_task_spec(task).run(cfg)
    assert result.num_samples == 1
    report = json.loads(paths["json"].read_text())
    assert report["task"] == task.value
    assert get_task_spec(task).primary_metric in result.aggregated
    assert all(path.exists() for key, path in paths.items() if key != "out_dir")
    cells = rpx.read_cells(paths["cells"])
    assert len(cells) == 1
    assert set(rpx.FIXED_COLUMNS) <= cells[0].keys()
    assert cells[0]["scene_id"] == "scene001"


def test_kdtree_depth_geometry_matches_brute_force():
    """The accelerated metric must preserve the original L1/L2 definitions."""
    from rpx_benchmark.metrics.depth_robotics import _chamfer_l1, _fscore
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(29, 3)), rng.normal(size=(37, 3))
    differences = a[:, None, :] - b[None, :, :]
    l1 = np.abs(differences).sum(axis=-1)
    assert _chamfer_l1(a, b) == pytest.approx(l1.min(axis=1).mean() + l1.min(axis=0).mean())
    l2 = np.linalg.norm(differences, axis=-1)
    for threshold in (0.01, 0.5, 2.0):
        precision, recall = (l2.min(axis=1) < threshold).mean(), (l2.min(axis=0) < threshold).mean()
        expected = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        assert _fscore(a, b, threshold) == pytest.approx(expected)
