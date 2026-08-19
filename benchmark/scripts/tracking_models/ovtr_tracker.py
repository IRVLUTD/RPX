"""Official OVTR open-vocabulary multi-object tracker adapter for RPX D3."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

OVTR_MODEL_ID = "jinyanglii/OVTR"
OVTR_SOURCE_REVISION = "500e72c19bf5f7f8717546911a5639fdc26bfee5"
OVTR_MODEL_REVISION = "gdrive-10GKAIBxAseTiXnJXV1MnxnJBTmOHVFh5"
OVTR_CHECKPOINT = "ovtr_5_frame.pth"
OVTR_SOURCE_DIR = "/opt/rpx-models/ovtr/ovtr"

_ASSETS = {
    OVTR_CHECKPOINT: (
        "10GKAIBxAseTiXnJXV1MnxnJBTmOHVFh5",
        239_020_217,
        "7b184a0f149259047ef3f03263cf9178fc9c5e051d08882f065aa52118266d56",
    ),
    "iou_neg5_ens.pth": (
        "1OYvyCQ_y65oq6SDJQKrVm3syzvXStL-0",
        2_464_504,
        "37544c96ff650cd4be4b794c763d33c9f8c7fee86ad8fde85225a3521b3a8d37",
    ),
    "clip_image_embedding_all.pt": (
        "1j5l-BPv-f43fb953hmIijxQ4gSUduWWe",
        2_464_491,
        "67b62ec6ced166c7b7e9480837e6c85b177a974815ac691270c7fbedea7da2b5",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_asset(path: Path, size: int, digest: str) -> bool:
    return path.is_file() and path.stat().st_size == size and _sha256(path) == digest


def _download_assets(cache_root: Path) -> dict[str, Path]:
    import gdown

    asset_root = cache_root / "ovtr" / OVTR_SOURCE_REVISION
    asset_root.mkdir(parents=True, exist_ok=True)
    resolved = {}
    for filename, (file_id, size, digest) in _ASSETS.items():
        destination = asset_root / filename
        if not _valid_asset(destination, size, digest):
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.unlink(missing_ok=True)
            downloaded = gdown.download(id=file_id, output=str(temporary), quiet=False)
            if downloaded is None or not _valid_asset(temporary, size, digest):
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"OVTR asset verification failed: {filename}")
            temporary.replace(destination)
        resolved[filename] = destination
    return resolved


class OVTRTracker:
    """Run the official full OVTR model with its published LVIS vocabulary."""

    model_name = "ovtr"
    model_id = OVTR_MODEL_ID
    source_revision = OVTR_SOURCE_REVISION
    model_revision = OVTR_MODEL_REVISION
    checkpoint_filename = OVTR_CHECKPOINT
    source_directory = OVTR_SOURCE_DIR
    adapter_label = "OVTR"
    prompt_type = "detector"
    tracking_mode = "native-open-vocabulary-mot"
    vocabulary_name = "LVIS-v1-1203-DetPro-CLIP-ensemble"
    vocabulary_size = 1203

    score_threshold = 0.3
    area_threshold = 1.0
    maximum_quantity = 50

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The production OVTR adapter requires CUDA.")

        source_dir = Path(os.environ.get("OVTR_SOURCE_DIR", self.source_directory)).resolve()
        config_path = source_dir / "config" / "ovtr_5_frame_train_val.py"
        if not config_path.is_file():
            raise RuntimeError(f"OVTR config is missing: {config_path}")

        from models import build_model
        from util.list_LVIS import CLASSES
        from util.slconfig import SLConfig
        from util.tool import load_model

        if len(CLASSES) != self.vocabulary_size:
            raise RuntimeError(f"Expected 1203 official LVIS classes, found {len(CLASSES)}.")
        self.class_names = tuple(CLASSES)

        assets = _download_assets(Path(os.environ.get("HF_HOME", "/cache/huggingface")))
        config = SLConfig.fromfile(str(config_path))
        config.Clip_text_embeddings = str(assets["iou_neg5_ens.pth"])
        config.Clip_image_embeddings = str(assets["clip_image_embedding_all.pt"])
        args = SimpleNamespace(
            device="cuda",
            track_query_iteration="CIP",
            sampler_lengths=[2, 3, 4, 5],
            cls_loss_coef=2.0,
            bbox_loss_coef=5.0,
            giou_loss_coef=2.0,
            align_loss_coef=2.0,
            aux_loss=True,
            random_drop=0.1,
            calculate_negative_samples=True,
            with_box_refine=True,
            two_stage=True,
            max_len=100,
            fp_ratio=0.3,
            merger_dropout=0.0,
            update_query_pos=False,
            set_cost_class=3.0,
            set_cost_bbox=5.0,
            set_cost_giou=2.0,
            score_thresh=self.score_threshold,
            filter_score_thresh=self.score_threshold,
            miss_tolerance=5,
        )
        model, _ = build_model(args, config)
        self.model = load_model(model, str(assets[OVTR_CHECKPOINT])).cuda().eval()
        self.model.track_base.score_thresh = self.score_threshold
        self.model.track_base.filter_score_thresh = self.score_threshold
        self.model.track_base.miss_tolerance = 5
        self.model.track_base.maximum_quantity = self.maximum_quantity
        self.model.transformer.decoder.isol_ratio = 5
        self.model.ious_thresh = 0.3
        self.checkpoint_path = str(assets[OVTR_CHECKPOINT].resolve())
        self.checkpoint_sha256 = _ASSETS[OVTR_CHECKPOINT][2]
        self.parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        self._frame_metadata: list[dict[str, Any]] = []

    @staticmethod
    def _image_tensor(path: Path) -> torch.Tensor:
        from torchvision.transforms import functional

        image = Image.open(path).convert("RGB")
        width, height = image.size
        scale = 800.0 / min(height, width)
        if max(height, width) * scale > 1333:
            scale = 1333.0 / max(height, width)
        resized = functional.resize(
            image,
            [int(height * scale), int(width * scale)],
            antialias=True,
        )
        tensor = functional.to_tensor(resized)
        return functional.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ).cuda()

    @staticmethod
    def _rasterize_boxes(
        boxes_xyxy: np.ndarray,
        track_ids: np.ndarray,
        scores: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        height, width = shape
        output = np.zeros(shape, dtype=np.int32)
        for index in np.argsort(scores):
            x1, y1, x2, y2 = boxes_xyxy[index]
            left = max(0, min(width, int(np.floor(x1))))
            top = max(0, min(height, int(np.floor(y1))))
            right = max(0, min(width, int(np.ceil(x2))))
            bottom = max(0, min(height, int(np.ceil(y2))))
            if right > left and bottom > top:
                output[top:bottom, left:right] = int(track_ids[index]) + 1
        return output

    def prediction_metadata(self) -> dict[str, Any]:
        return {
            "classification": "official_OVTR_CLIP_text_image_alignment",
            "vocabulary": self.vocabulary_name,
            "vocabulary_size": self.vocabulary_size,
            "frames": self._frame_metadata,
        }

    def track(
        self,
        video_dir: Path,
        first_frame_mask: np.ndarray,
        frame_count: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        # OVTR is detector-driven; RPX ground-truth initialization is deliberately ignored.
        shape = tuple(int(value) for value in np.asarray(first_frame_mask).shape)
        if len(shape) != 2:
            raise ValueError("first_frame_mask must be a 2-D instance mask.")
        frames = sorted(video_dir.glob("*.jpg"))
        if len(frames) != frame_count:
            raise RuntimeError(f"OVTR received {len(frames)} frames; expected {frame_count}.")

        track_instances = None
        predictions = []
        latencies_ms = []
        self._frame_metadata = []
        with torch.inference_mode():
            for frame_index, frame_path in enumerate(frames):
                image = self._image_tensor(frame_path)
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = self.model.inference_single_image(
                    {"imgs": [image]},
                    track_instances,
                    frame_id=frame_index,
                    ori_img_size=[shape[0], shape[1], 3],
                )
                torch.cuda.synchronize()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
                track_instances = result["track_instances"]
                visible = track_instances.to(torch.device("cpu"))
                keep = (
                    (visible.scores > self.score_threshold)
                    & (visible.disappear_time == 0)
                    & (visible.cls_idxes >= 0)
                )
                visible = visible[keep]
                if len(visible):
                    areas = (visible.boxes[:, 2] - visible.boxes[:, 0]) * (
                        visible.boxes[:, 3] - visible.boxes[:, 1]
                    )
                    visible = visible[areas > self.area_threshold]

                boxes = visible.boxes.detach().float().numpy()
                ids = visible.obj_idxes.detach().numpy()
                scores = visible.scores.detach().float().numpy()
                labels = visible.cls_idxes.detach().numpy()
                predictions.append(self._rasterize_boxes(boxes, ids, scores, shape))
                self._frame_metadata.append(
                    {
                        "frame_index": frame_index,
                        "frame": frame_path.stem,
                        "tracks": [
                            {
                                "track_id": int(track_id) + 1,
                                "class_id": int(label),
                                "class_name": self.class_names[int(label)],
                                "score": float(score),
                                "bbox_xyxy": [float(value) for value in box],
                            }
                            for box, track_id, score, label in zip(
                                boxes, ids, scores, labels, strict=True
                            )
                        ],
                    }
                )
                track_instances.remove("boxes")
                track_instances.remove("labels")
        return predictions, latencies_ms
