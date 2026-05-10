"""Generic batched-dispatch BenchmarkModel base for multi-modal tasks.

The stock :class:`rpx_benchmark.adapters.base.BenchmarkableModel` iterates
samples one at a time inside ``predict``: even with ``--batch-size 8``,
the underlying adapter still sees one sample per call. ``BatchedDepthBenchmarkModel``
fixes that for monocular depth specifically.

This module generalises the pattern so every other task (segmentation,
relative pose, NVS, sparse depth, keypoint matching, ...) can opt into
true-batched dispatch without each one having to re-implement the
batch-extraction loop, the per-sample save logic, and the prediction-
wrapping ceremony.

Three task-specific hooks
-------------------------
Subclasses override:

* :meth:`extract_inputs(sample) -> Any` — pull whatever the model needs
  off the sample (rgb, rgb_b, fisheye_left, intrinsics, …). The return
  value is opaque to the base class; the adapter receives a list of
  these.
* :meth:`call_adapter(inputs_list) -> outputs_list` — invoke the
  adapter on the full batch in one call. Default: ``self._adapter(inputs_list)``.
* :meth:`wrap_output(model_output, sample) -> Prediction` — wrap one
  model output into the task's Prediction dataclass (e.g.,
  ``DepthPrediction``, ``SegmentationPrediction``,
  ``RelativePosePrediction``).

Optional:

* :meth:`maybe_save(sample, model_output)` — persist per-frame outputs
  for offline analytics (the depth wrapper writes
  ``<save_dir>/<scene>/<phase>/<frame>.npz``; segmentation could write
  PNG masks; tracking writes JSON tracklets).

Sample.metadata is the multi-modal side-channel
-----------------------------------------------
The loader's ``_load_sample`` puts non-RGB modalities into
``sample.metadata`` (``rgb_b``, ``fisheye_left``, ``fisheye_right``,
…). Subclasses pull from there:

::

    class BatchedRelativePoseBenchmarkModel(BatchedTaskBenchmarkModel):
        task = TaskType.RELATIVE_CAMERA_POSE
        def extract_inputs(self, sample):
            rgb_a = np.asarray(sample.rgb, dtype=np.uint8)
            rgb_b = np.asarray(sample.metadata["rgb_b"], dtype=np.uint8)
            return {"rgb_a": rgb_a, "rgb_b": rgb_b}

The contract is intentionally loose: ``extract_inputs`` returns
whatever shape the adapter expects (numpy array, dict, tuple, ...),
and the adapter dispatches the batch as a list of those.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from ..api import Sample, TaskType

__all__ = [
    "BatchedTaskBenchmarkModel",
    "BatchedSegmentationBenchmarkModel",
    "BatchedRelativePoseBenchmarkModel",
]


class BatchedTaskBenchmarkModel:
    """Abstract base — true-batched-dispatch wrapper for any RPX task.

    Subclasses set the class-level ``task`` (a :class:`TaskType`) and
    override ``extract_inputs`` + ``wrap_output``. The harness contract
    matches ``BenchmarkableModel`` (``setup`` + ``predict(batch)``).

    Parameters
    ----------
    adapter
        The model callable. Receives a list of whatever ``extract_inputs``
        returns (one per sample). Returns a list of model outputs (one per
        sample, same order).
    name
        Display name surfaced in result.json + Box paths.
    save_dir
        If set, ``maybe_save`` writes per-frame artefacts. Default: no save.
    """

    #: Task this wrapper produces predictions for. Subclasses set this.
    task: TaskType = TaskType.MONOCULAR_DEPTH  # placeholder — must be overridden

    def __init__(
        self,
        adapter: Any,
        *,
        name: str,
        save_dir: Optional[str | Path] = None,
    ) -> None:
        self._adapter = adapter
        self.name = name
        self._save_dir = Path(save_dir) if save_dir else None
        # Profiler walker reaches the underlying nn.Module via this attr.
        self.model = adapter

    def setup(self) -> None:
        """No-op — the adapter loads its own weights at construction."""
        return None

    # ────────────────────────  hooks  ────────────────────────

    def extract_inputs(self, sample: Sample) -> Any:
        """Subclass: return whatever the adapter expects per sample.

        Examples:
        - Depth: ``np.asarray(sample.rgb, dtype=np.uint8)``
        - Pose:  ``{"rgb_a": ..., "rgb_b": sample.metadata["rgb_b"]}``
        - Stereo depth: ``{"left": metadata["fisheye_left"],
                            "right": metadata["fisheye_right"]}``
        """
        raise NotImplementedError

    def call_adapter(self, inputs: List[Any]) -> List[Any]:
        """Subclass override point: how the adapter is invoked on a list.

        Default: ``self._adapter(inputs)`` — works when the adapter
        accepts a list directly (HF pipelines, our depth adapters).
        Override for adapters with a different batched contract.
        """
        out = self._adapter(inputs)
        if not isinstance(out, (list, tuple)):
            out = [out]
        return list(out)

    def wrap_output(self, model_output: Any, sample: Sample) -> Any:
        """Subclass: wrap one model output as the task's Prediction dataclass."""
        raise NotImplementedError

    def maybe_save(self, sample: Sample, model_output: Any) -> None:
        """Optional override — persist per-frame artefacts for analytics."""
        return None

    # ────────────────────────  predict()  ────────────────────────

    def predict(self, batch: Sequence[Sample]) -> List[Any]:
        inputs = [self.extract_inputs(s) for s in batch]
        outputs = self.call_adapter(inputs)
        if len(outputs) != len(batch):
            from ..exceptions import AdapterError

            raise AdapterError(
                f"adapter returned {len(outputs)} outputs for a batch of {len(batch)}",
                hint="Check the adapter's batched contract: it must return "
                "one output per input sample.",
            )
        preds: List[Any] = []
        for sample, out in zip(batch, outputs, strict=False):
            preds.append(self.wrap_output(out, sample))
            self.maybe_save(sample, out)
        return preds


# ────────────────────────  Reference subclasses  ────────────────────────


class BatchedSegmentationBenchmarkModel(BatchedTaskBenchmarkModel):
    """Batched wrapper for object-segmentation models.

    Adapter contract: ``adapter(list_of_rgb_uint8_HxWx3) -> list_of_int_mask_HxW``.
    Output mask values are integer instance IDs (0 = background). If the
    model returns a different shape than the input RGB, we nearest-neighbour
    resize back to preserve integer IDs.
    """

    task = TaskType.OBJECT_SEGMENTATION

    def extract_inputs(self, sample: Sample) -> np.ndarray:
        return np.asarray(sample.rgb, dtype=np.uint8)

    def wrap_output(self, model_output: Any, sample: Sample) -> Any:
        from PIL import Image

        from ..api import SegmentationPrediction

        mask = np.asarray(model_output)
        target_hw = np.asarray(sample.rgb).shape[:2]
        if mask.shape != target_hw:
            # Nearest-neighbour preserves integer IDs.
            img = Image.fromarray(mask.astype(np.int32), mode="I")
            img = img.resize((target_hw[1], target_hw[0]), Image.NEAREST)
            mask = np.asarray(img, dtype=np.int32)
        return SegmentationPrediction(mask=mask.astype(np.int32))

    def maybe_save(self, sample: Sample, model_output: Any) -> None:
        if self._save_dir is None:
            return
        from PIL import Image

        meta = getattr(sample, "metadata", None) or {}
        scene = meta.get("scene_id")
        phase = meta.get("phase_idx")
        frame = meta.get("frame")
        if not (scene and frame is not None and phase is not None):
            try:
                scene, phase, frame = str(sample.id).split("__", 2)
            except ValueError:
                scene, phase, frame = "unknown", "0", str(sample.id)
        out = self._save_dir / str(scene) / str(phase) / f"{frame}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        # 16-bit PNG preserves up to 65 K instance ids; sufficient for SAM2.
        Image.fromarray(np.asarray(model_output, dtype=np.uint16), mode="I;16").save(out)


class BatchedRelativePoseBenchmarkModel(BatchedTaskBenchmarkModel):
    """Batched wrapper for paired-frame relative-pose models.

    Adapter contract: ``adapter(list_of_dicts) -> list_of_pose_dicts``,
    where each input dict is ``{"rgb_a": HxWx3 uint8, "rgb_b": HxWx3 uint8}``
    and each output is ``{"rotation": 3x3, "translation": 3}`` (or a tuple).
    """

    task = TaskType.RELATIVE_CAMERA_POSE

    def extract_inputs(self, sample: Sample) -> dict:
        meta = getattr(sample, "metadata", None) or {}
        rgb_a = np.asarray(sample.rgb, dtype=np.uint8)
        rgb_b = meta.get("rgb_b")
        if rgb_b is None:
            from ..exceptions import AdapterError

            raise AdapterError(
                "Relative-pose adapter requires `rgb_b` in sample.metadata; "
                "check the manifest writes `rgb_b` so the loader stashes it.",
            )
        return {"rgb_a": rgb_a, "rgb_b": np.asarray(rgb_b, dtype=np.uint8)}

    def wrap_output(self, model_output: Any, sample: Sample) -> Any:
        from ..api import RelativePosePrediction

        if isinstance(model_output, dict):
            rot = model_output["rotation"]
            trans = model_output["translation"]
        else:
            rot, trans = model_output
        return RelativePosePrediction(
            rotation=np.asarray(rot, dtype=np.float64),
            translation=np.asarray(trans, dtype=np.float64),
        )

    #: CSV columns. 4 ID + 9 rotation + 3 translation = 16 columns.
    _LOG_COLUMNS: tuple[str, ...] = (
        "scene_id",
        "phase",
        "frame_a",
        "frame_b",
        "R00",
        "R01",
        "R02",
        "R10",
        "R11",
        "R12",
        "R20",
        "R21",
        "R22",
        "tx",
        "ty",
        "tz",
    )

    def maybe_save(self, sample: Sample, model_output: Any) -> None:
        """Append one row per pair to ``<save_dir>/predictions.csv``.

        Rationale: paired-pose runs produce ~3 K rows per (model, split);
        a single CSV is friendlier for Box and analytics than 3 K
        per-pair ``.npz`` files. Columns: scene_id, phase, frame_a,
        frame_b, the 9 rotation entries (row-major) and 3 translation
        components. Header is written exactly once per file (on the
        first append to an empty / non-existent log). Idempotent on
        ``(scene, phase, frame_a, frame_b)`` within a single process so
        the runner's deployment-readiness double pass (warmup + measure)
        doesn't duplicate rows.
        """
        if self._save_dir is None:
            return
        import csv

        meta = getattr(sample, "metadata", None) or {}
        scene = meta.get("scene_id") or "unknown"
        phase = meta.get("phase_idx") if meta.get("phase_idx") is not None else "0"
        frame_a = meta.get("frame") or str(sample.id)
        frame_b = meta.get("frame_b") or "?"

        seen = self.__dict__.setdefault("_pose_csv_seen", set())
        key = (scene, str(phase), frame_a, frame_b)
        if key in seen:
            return
        seen.add(key)

        if isinstance(model_output, dict):
            rot = np.asarray(model_output["rotation"], dtype=np.float64)
            trans = np.asarray(model_output["translation"], dtype=np.float64)
        else:
            rot, trans = model_output
            rot = np.asarray(rot, dtype=np.float64)
            trans = np.asarray(trans, dtype=np.float64)

        # Tolerate rotation given as a flat 9-vector or a quaternion.
        rot = rot.reshape(-1)
        if rot.size == 4:
            # Quaternion (x, y, z, w) → 3×3 rotation. We don't depend on
            # scipy here; build it inline from the unit-quaternion formula.
            x, y, z, w = rot.tolist()
            rot = np.asarray(
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
                dtype=np.float64,
            )
        elif rot.size != 9:
            from ..exceptions import AdapterError

            raise AdapterError(
                f"unexpected rotation shape: got {rot.size} entries; "
                "expected 9 (3×3 matrix) or 4 (xyzw quaternion).",
            )
        trans = trans.reshape(-1)
        if trans.size != 3:
            from ..exceptions import AdapterError

            raise AdapterError(
                f"unexpected translation shape: got {trans.size} entries; expected 3.",
            )

        log_path = self._save_dir / "predictions.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists() or log_path.stat().st_size == 0
        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(self._LOG_COLUMNS)
            writer.writerow(
                [
                    scene,
                    phase,
                    frame_a,
                    frame_b,
                    *(f"{v:.10g}" for v in rot.tolist()),
                    *(f"{v:.10g}" for v in trans.tolist()),
                ]
            )
