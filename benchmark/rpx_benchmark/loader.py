"""Data loading utilities for RPX benchmark.

Reads a JSON *manifest* describing a slice of the RPX dataset (usually
produced by :func:`rpx_benchmark.hub.download_split`) and yields
:class:`~rpx_benchmark.api.Sample` batches the :class:`BenchmarkRunner`
can feed to a model.

Manifest format
---------------

::

    {
      "task":   "monocular_depth",
      "root":   "/absolute/path/or/snapshot/root",
      "samples": [
        {
          "id":         "scene_001_clutter_00000",
          "scene":      "scene_001",
          "phase":      "clutter",
          "difficulty": "hard",
          "rgb":   "scenes/scene_001/0/rgb/00000.png",
          "depth": "scenes/scene_001/0/depth/00000.png",
          "mask":  "scenes/scene_001/0/mask/00000.png",
          "pose":  "scenes/scene_001/0/pose/00000.npz"
        },
        ...
      ]
    }

All sample paths are resolved relative to ``root`` unless they are
absolute. ``depth`` files are 16-bit PNGs in millimetres (as saved by
``save_device_data.py``); the loader converts to float32 metres on
read. ``pose`` files are ``.npz`` with ``position`` and ``orientation``
(T265 convention, ``[x, y, z, w]`` quaternion).

Errors
------

All load-time failures are raised as
:class:`~rpx_benchmark.exceptions.ManifestError` so user code can
catch them specifically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
from PIL import Image

from .api import (
    Difficulty,
    DepthGroundTruth,
    DetectionGroundTruth,
    KeypointCorrespondenceGroundTruth,
    NovelViewSynthesisGroundTruth,
    Phase,
    RelativePoseGroundTruth,
    Sample,
    SegmentationGroundTruth,
    SparseDepthGroundTruth,
    TaskType,
    Tracklet,
    TrackletGroundTruth,
    VisualGroundingGroundTruth,
)
from .exceptions import ManifestError
from .logging_utils import get_logger

log = get_logger(__name__)

# Phase index → Phase enum (matches capture session folder 0/1/2)
_PHASE_INDEX: Dict[int, Phase] = {
    0: Phase.CLUTTER,
    1: Phase.INTERACTION,
    2: Phase.CLEAN,
}


@dataclass
class RPXDataset:
    """Iterates over RPX samples for a specific task.

    Manifest format (JSON)::

        {
          "task": "object_segmentation",
          "root": "/path/to/data",
          "samples": [
            {
              "id": "scene_001_clutter_00000",
              "scene": "scene_001",
              "phase": "clutter",
              "difficulty": "hard",
              "rgb":   "scene_001/0/rgb/00000.png",
              "depth": "scene_001/0/depth/00000.png",
              "mask":  "scene_001/0/mask/00000.png",
              "pose":  "scene_001/0/pose/00000.npz",
              ...
            }
          ]
        }

    All paths are relative to ``root``.
    ``depth`` files are 16-bit PNG in millimetres (as saved by save_device_data.py).
    ``pose`` files are NPZ with keys ``position`` ([x,y,z] metres) and
    ``orientation`` ([x,y,z,w] quaternion) from the T265 tracker.
    """

    samples: List[Dict[str, Any]]
    task: TaskType
    root: Path
    batch_size: int = 1

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        batch_size: int = 1,
        validate: bool = False,
    ) -> "RPXDataset":
        """Load a manifest JSON file from disk and return a dataset.

        Parameters
        ----------
        manifest_path : str or Path
            Path to the manifest JSON. Produced either by
            :func:`rpx_benchmark.hub.download_split` or by a custom
            upload script.
        batch_size : int
            Number of samples per iteration. Default 1.
        validate : bool, optional
            When ``True``, validate the manifest against the strict
            Pydantic schema in :mod:`rpx_benchmark.schemas` before
            constructing the dataset. Requires the ``schemas`` extra
            (``pip install 'rpx-benchmark[schemas]'``). Default
            ``False`` preserves historical, tolerant behaviour.

        Returns
        -------
        RPXDataset

        Raises
        ------
        ManifestError
            If the manifest file is missing, not valid JSON, is missing
            required top-level fields, or (when ``validate=True``)
            fails strict schema validation.
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise ManifestError(
                f"Manifest file not found: {manifest_path}",
                hint="Did the HuggingFace download fail? Try rerunning with "
                     "--cache-dir pointing at a writable location.",
            )
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            raise ManifestError(
                f"Manifest at {manifest_path} is not valid JSON: {e}",
            ) from e
        return cls.from_dict(
            manifest,
            batch_size=batch_size,
            default_root=manifest_path.parent,
            validate=validate,
        )

    @classmethod
    def from_dict(
        cls,
        manifest: Dict[str, Any],
        batch_size: int = 1,
        default_root: str | Path | None = None,
        validate: bool = False,
    ) -> "RPXDataset":
        """Build a dataset from an already-parsed manifest dict.

        Raises
        ------
        ManifestError
            If ``task`` is missing or unknown, or if ``samples`` is
            missing. When ``validate=True``, any Pydantic validation
            error is also surfaced as :class:`ManifestError`.
        """
        if validate:
            # Opt-in strict validation via rpx_benchmark.schemas. Kept
            # lazy so users who don't install the ``schemas`` extra
            # don't hit pydantic import at module load time.
            from .schemas import validate_manifest  # noqa: PLC0415 — lazy import

            model = validate_manifest(manifest)
            manifest = model.model_dump(exclude_none=True)

        if "task" not in manifest:
            raise ManifestError(
                "Manifest is missing required field 'task'.",
                hint="Task must be one of: " +
                     ", ".join(t.value for t in TaskType),
            )
        try:
            task = TaskType(manifest["task"])
        except ValueError as e:
            raise ManifestError(
                f"Manifest task {manifest['task']!r} is not a known TaskType.",
                hint="Expected one of: " +
                     ", ".join(t.value for t in TaskType),
            ) from e

        if "samples" not in manifest:
            raise ManifestError(
                "Manifest is missing required field 'samples'.",
            )

        root = Path(manifest.get("root") or default_root or ".")
        samples = manifest["samples"]
        if not isinstance(samples, list):
            raise ManifestError(
                f"Manifest 'samples' must be a list, got {type(samples).__name__}",
            )
        log.debug("loaded manifest: task=%s root=%s samples=%d",
                  task.value, root, len(samples))
        return cls(samples=samples, task=task, root=root, batch_size=batch_size)

    @classmethod
    def from_hf(
        cls,
        hf_dataset: Any,
        task: TaskType | str | None = None,
        batch_size: int = 1,
    ) -> Any:
        """Wrap a ``datasets.Dataset`` as an iterable RPX source.

        Returns a :class:`rpx_benchmark.data.hf_bridge.RPXHFBridge`
        which is interchangeable with :class:`RPXDataset` at the
        runner's consumer interface (``__len__`` + ``__iter__`` over
        ``list[Sample]``).

        Parameters
        ----------
        hf_dataset : datasets.Dataset
            Dataset whose columns match
            :func:`rpx_benchmark.data.features.features_for_task`.
        task : TaskType or str, optional
            Task this dataset serves. If omitted, we try to infer it
            from ``hf_dataset.info.description`` (where
            :func:`rpx_benchmark.data.load_hf` stamps it).
        batch_size : int, default 1

        Returns
        -------
        RPXHFBridge

        Raises
        ------
        ManifestError
            If ``task`` is neither provided nor discoverable on the
            dataset metadata.
        """
        from .data.hf_bridge import RPXHFBridge  # noqa: PLC0415 — lazy

        if task is None:
            info = getattr(hf_dataset, "info", None)
            config = getattr(info, "config_name", None) if info is not None else None
            if config is None:
                raise ManifestError(
                    "Could not infer task from the HF dataset; pass "
                    "task=TaskType.* explicitly.",
                )
            task = TaskType(config)
        if isinstance(task, str):
            task = TaskType(task)
        return RPXHFBridge(hf_dataset=hf_dataset, task=task, batch_size=batch_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[List[Sample]]:
        batch: List[Sample] = []
        for entry in self.samples:
            batch.append(self._load_sample(entry))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _load_sample(self, entry: Dict[str, Any]) -> Sample:
        rgb = self._load_rgb(entry["rgb"])

        # Phase — from string label or integer session index
        phase: Phase | None = None
        if "phase" in entry:
            v = entry["phase"]
            phase = Phase(v) if isinstance(v, str) else _PHASE_INDEX.get(int(v))

        # Difficulty
        difficulty: Difficulty | None = None
        if "difficulty" in entry:
            difficulty = Difficulty(entry["difficulty"])

        # Camera pose (T265 NPZ → 4×4 SE(3) float64)
        camera_pose: np.ndarray | None = None
        if "pose" in entry:
            camera_pose = self._load_pose(entry["pose"])

        # Fisheye images — stored in metadata if present
        metadata: Dict[str, Any] = entry.get("metadata") or {}
        if "fisheye_left" in entry:
            metadata["fisheye_left"] = self._load_gray(entry["fisheye_left"])
        if "fisheye_right" in entry:
            metadata["fisheye_right"] = self._load_gray(entry["fisheye_right"])
        # Second RGB frame for pair tasks (relative pose, keypoint matching)
        if "rgb_b" in entry:
            metadata["rgb_b"] = self._load_rgb(entry["rgb_b"])

        gt = self._load_ground_truth(entry)

        return Sample(
            id=entry["id"],
            rgb=rgb,
            ground_truth=gt,
            metadata=metadata or None,
            phase=phase,
            difficulty=difficulty,
            camera_pose=camera_pose,
        )

    def _load_ground_truth(self, entry: Dict[str, Any]) -> Any:
        task = self.task

        if task == TaskType.MONOCULAR_DEPTH:
            return DepthGroundTruth(depth_map=self._load_depth(entry["depth"]))

        if task in (TaskType.OBJECT_DETECTION, TaskType.OPEN_VOCAB_DETECTION):
            boxes, labels = self._load_boxes(entry["boxes"])
            return DetectionGroundTruth(boxes=boxes, labels=labels)

        if task == TaskType.OBJECT_SEGMENTATION:
            return SegmentationGroundTruth(mask=self._load_mask(entry["mask"]))

        if task == TaskType.OBJECT_TRACKING:
            return self._load_tracklets(entry)

        if task == TaskType.RELATIVE_CAMERA_POSE:
            return self._load_relative_pose(entry)

        if task == TaskType.VISUAL_GROUNDING:
            return self._load_visual_grounding(entry)

        if task == TaskType.SPARSE_DEPTH:
            return self._load_sparse_depth(entry)

        if task == TaskType.NOVEL_VIEW_SYNTHESIS:
            return self._load_nvs(entry)

        if task == TaskType.KEYPOINT_MATCHING:
            return self._load_keypoints(entry)

        raise ManifestError(
            f"Unsupported task {task} — loader does not know how to parse "
            "its ground-truth entries.",
        )

    # ------------------------------------------------------------------ #
    # Ground-truth loaders
    # ------------------------------------------------------------------ #

    def _load_tracklets(self, entry: Dict[str, Any]) -> TrackletGroundTruth:
        path = self._resolve(entry["tracks"])
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        tracks = []
        for item in data:
            boxes = np.array(item["boxes"], dtype=np.float32)
            scores = np.array(item["scores"], dtype=np.float32) if "scores" in item else None
            tracks.append(Tracklet(track_id=str(item["track_id"]), boxes=boxes, scores=scores))
        return TrackletGroundTruth(tracks=tracks)

    def _load_relative_pose(self, entry: Dict[str, Any]) -> RelativePoseGroundTruth:
        # Load the two poses and compute relative transform
        pose_a = self._load_pose(entry["pose_a"])   # 4×4 SE(3)
        pose_b = self._load_pose(entry["pose_b"])
        T_rel = np.linalg.inv(pose_a) @ pose_b
        return RelativePoseGroundTruth(
            rotation=T_rel[:3, :3],
            translation=T_rel[:3, 3],
        )

    def _load_visual_grounding(self, entry: Dict[str, Any]) -> VisualGroundingGroundTruth:
        text = entry["text"]
        boxes_raw = entry.get("boxes")
        labels = entry.get("labels")
        if boxes_raw is None:
            boxes = np.zeros((0, 4), dtype=np.float32)
        elif isinstance(boxes_raw, str):
            gt_boxes, _ = self._load_boxes(boxes_raw)
            boxes = gt_boxes
        else:
            boxes = np.array(boxes_raw, dtype=np.float32)
        return VisualGroundingGroundTruth(text=text, boxes=boxes, labels=labels)

    def _load_sparse_depth(self, entry: Dict[str, Any]) -> SparseDepthGroundTruth:
        coordinates = self._load_array_or_inline(entry["coordinates"]).astype(np.float32)
        depths = self._load_array_or_inline(entry["depths"]).astype(np.float32)
        return SparseDepthGroundTruth(coordinates=coordinates, depths=depths)

    def _load_nvs(self, entry: Dict[str, Any]) -> NovelViewSynthesisGroundTruth:
        rgb = self._load_rgb(entry["target_rgb"])
        camera_pose: Any = entry.get("target_pose") or entry.get("camera_pose")
        if isinstance(camera_pose, str):
            camera_pose = self._load_pose(camera_pose)
        return NovelViewSynthesisGroundTruth(rgb=rgb, camera_pose=camera_pose)

    def _load_keypoints(self, entry: Dict[str, Any]) -> KeypointCorrespondenceGroundTruth:
        points0 = self._load_array_or_inline(entry["points0"]).astype(np.float32)
        points1 = self._load_array_or_inline(entry["points1"]).astype(np.float32)
        visibility = None
        if "visibility" in entry:
            visibility = self._load_array_or_inline(entry["visibility"]).astype(bool)
        return KeypointCorrespondenceGroundTruth(
            points0=points0, points1=points1, visibility=visibility
        )

    # ------------------------------------------------------------------ #
    # Low-level file loaders
    # ------------------------------------------------------------------ #

    def _resolve(self, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        return path if path.is_absolute() else self.root / path

    def _load_rgb(self, relative_path: str) -> np.ndarray:
        """Load 640×480 RGB image → uint8 H×W×3."""
        path = self._resolve(relative_path)
        with Image.open(path) as im:
            return np.array(im.convert("RGB"), dtype=np.uint8)

    def _load_depth(self, relative_path: str) -> np.ndarray:
        """Load 16-bit PNG depth (millimetres) → float32 H×W in metres."""
        path = self._resolve(relative_path)
        with Image.open(path) as im:
            depth_mm = np.array(im, dtype=np.float32)
        if depth_mm.ndim != 2:
            raise ManifestError(
                f"Depth file at {path} is not 2-D: got shape {depth_mm.shape}",
                hint="RPX depth maps are single-channel 16-bit PNGs in millimetres.",
            )
        # Convert mm → metres; zero pixels = invalid (no return)
        depth_m = depth_mm / 1000.0
        depth_m[depth_mm == 0] = 0.0  # keep 0 as "invalid" sentinel
        return depth_m

    def _load_pose(self, relative_path: str) -> np.ndarray:
        """Load T265 NPZ pose → 4×4 SE(3) float64 (camera-to-world).

        NPZ keys:
          position:    [x, y, z] metres
          orientation: [x, y, z, w] quaternion (T265 convention)
        """
        path = self._resolve(relative_path)
        data = np.load(path)
        position = data["position"].astype(np.float64)        # (3,)
        quat_xyzw = data["orientation"].astype(np.float64)   # (4,) x,y,z,w

        R = _quat_xyzw_to_rotmat(quat_xyzw)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = position
        return T

    def _load_mask(self, relative_path: str) -> np.ndarray:
        """Load segmentation mask PNG → int32 H×W (pixel values = instance IDs)."""
        path = self._resolve(relative_path)
        with Image.open(path) as im:
            mask = np.array(im, dtype=np.int32)
        if mask.ndim != 2:
            raise ManifestError(
                f"Mask file at {path} is not 2-D: got shape {mask.shape}",
                hint="RPX instance masks are single-channel PNGs whose "
                     "pixel values are instance IDs.",
            )
        return mask

    def _load_gray(self, relative_path: str) -> np.ndarray:
        """Load grayscale image (fisheye) → uint8 H×W."""
        path = self._resolve(relative_path)
        with Image.open(path) as im:
            return np.array(im.convert("L"), dtype=np.uint8)

    def _load_boxes(self, relative_path: str):
        path = self._resolve(relative_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        boxes = np.array([item["bbox"] for item in data], dtype=np.float32)
        labels = [item["label"] for item in data]
        return boxes, labels

    def _load_array_or_inline(self, value: Any) -> np.ndarray:
        if isinstance(value, str):
            path = self._resolve(value)
            if path.suffix.lower() == ".npy":
                return np.load(path)
            with path.open("r", encoding="utf-8") as f:
                return np.array(json.load(f))
        return np.array(value)


# ------------------------------------------------------------------ #
# Quaternion helper
# ------------------------------------------------------------------ #

def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] (T265 convention) to 3×3 rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)
