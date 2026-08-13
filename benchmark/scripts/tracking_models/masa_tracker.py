"""Official detector-driven MASA-Detic multi-object adapter for RPX D3."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import requests
import torch

MASA_MODEL_ID = "dereksiyuanli/masa"
MASA_MODEL_REVISION = "25ed372c47f2c46cf36fd446d1b657b656bc7ea9"
MASA_CHECKPOINT = "detic_masa.pth"
MASA_CHECKPOINT_BYTES = 726_691_715
MASA_CHECKPOINT_SHA256 = (
    "10c19938af1b70c8bea1ca4a49139198abf2c2cc77c42e8372bfeb0a4e461879"
)
MASA_SOURCE_REVISION = "c5472b9c7615f35abdf1188cb1a0c5408fe50d66"
MASA_SOURCE_DIR = "/opt/rpx-models/masa"
MASA_CONFIG = (
    "configs/masa-detic/open_vocabulary_mot_test/"
    "masa_detic_swinb_open_vocabulary_test.py"
)
MASA_SCORE_THRESHOLD = 0.2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_checkpoint(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == MASA_CHECKPOINT_BYTES
        and _sha256(path) == MASA_CHECKPOINT_SHA256
    )


def _download_checkpoint(cache_root: Path) -> Path:
    destination = cache_root / "masa" / MASA_MODEL_REVISION / MASA_CHECKPOINT
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _valid_checkpoint(destination):
        return destination

    destination.unlink(missing_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = (
        f"https://huggingface.co/{MASA_MODEL_ID}/resolve/"
        f"{MASA_MODEL_REVISION}/{MASA_CHECKPOINT}"
    )
    with requests.get(url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if not _valid_checkpoint(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("MASA-Detic checkpoint failed pinned size/SHA-256 validation.")
    temporary.replace(destination)
    return destination


class MASATracker:
    """Run official unified MASA-Detic without an RPX initialization prompt."""

    model_name = "masa"
    model_id = MASA_MODEL_ID
    model_revision = MASA_MODEL_REVISION
    checkpoint_filename = MASA_CHECKPOINT
    source_revision = MASA_SOURCE_REVISION
    source_directory = MASA_SOURCE_DIR
    config_name = MASA_CONFIG
    adapter_label = "MASA-Detic"
    prompt_type = "detector"
    tracking_mode = "detector-association-mot"
    score_threshold = MASA_SCORE_THRESHOLD

    @classmethod
    def _load_config(cls, source_dir: Path):
        # Importing MASA registers its models and the bundled MMDetection Detic
        # project. The source checkout is deliberately retained in the image.
        import masa  # noqa: F401
        from mmengine.config import Config

        config_path = source_dir / cls.config_name
        if not config_path.is_file():
            raise RuntimeError(
                f"MASA-Detic config is missing: {config_path}. "
                "Use the pinned cumulative image."
            )
        config = Config.fromfile(config_path)

        # The released unified checkpoint contains both Detic and MASA weights.
        # Avoid the config's working-directory-relative auxiliary checkpoint.
        config.model.detector.init_cfg = None

        # Detic constructs its classifier before the unified checkpoint is
        # loaded. A correctly shaped temporary tensor avoids depending on the
        # training-only LVIS metadata file; load_checkpoint replaces it with
        # the released classifier weights.
        for bbox_head in config.model.detector.roi_head.bbox_head:
            bbox_head.cls_predictor_cfg.zs_weight_path = "rand"
            bbox_head.use_fed_loss = False

        # The benchmark config has a dataloader pipeline but no demo inference
        # pipeline. This is the official MASA demo transform for Detic/GDINO.
        config.inference_pipeline = [
            {
                "type": "TransformBroadcaster",
                "transforms": [
                    {"type": "Resize", "scale": (1333, 800), "keep_ratio": True}
                ],
            },
            {"type": "PackTrackInputs"},
        ]
        return config

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise RuntimeError("The production MASA adapter requires device='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to the MASA adapter.")

        source_dir = Path(os.environ.get("MASA_SOURCE_DIR", self.source_directory)).resolve()
        checkpoint = _download_checkpoint(
            Path(os.environ.get("HF_HOME", "/cache/huggingface"))
        )
        config = self._load_config(source_dir)
        from masa.apis import build_test_pipeline, init_masa

        self.model = init_masa(config, str(checkpoint), device="cuda:0")
        self.test_pipeline = build_test_pipeline(self.model.cfg)
        self.device = device
        self.checkpoint_path = str(checkpoint.resolve())
        self.checkpoint_sha256 = MASA_CHECKPOINT_SHA256
        self.parameter_count = int(
            sum(parameter.numel() for parameter in self.model.parameters())
        )

    @staticmethod
    def _rasterize_boxes(
        boxes_xyxy: np.ndarray,
        track_ids: np.ndarray,
        scores: np.ndarray,
        shape: tuple[int, int],
        score_threshold: float = MASA_SCORE_THRESHOLD,
    ) -> np.ndarray:
        height, width = shape
        output = np.zeros(shape, dtype=np.int32)
        # Higher-confidence tracks overwrite lower-confidence overlaps.
        for index in np.argsort(scores):
            if float(scores[index]) <= score_threshold:
                continue
            x1, y1, x2, y2 = boxes_xyxy[index]
            left = max(0, min(width, int(np.floor(x1))))
            top = max(0, min(height, int(np.floor(y1))))
            right = max(0, min(width, int(np.ceil(x2))))
            bottom = max(0, min(height, int(np.ceil(y2))))
            if right > left and bottom > top:
                output[top:bottom, left:right] = int(track_ids[index]) + 1
        return output

    def track(
        self,
        video_dir: Path,
        first_frame_mask: np.ndarray,
        frame_count: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        # Detector-driven MASA deliberately ignores the RPX first-frame mask,
        # including its dimensions; output geometry comes from the RGB frames.
        del first_frame_mask
        frames = sorted(video_dir.glob("*.jpg"))
        if len(frames) != frame_count:
            raise RuntimeError(f"MASA received {len(frames)} frames; expected {frame_count}.")

        import cv2
        from demo.utils import filter_and_update_tracks
        from masa.apis import inference_masa

        instances = []
        latencies_ms = []
        output_shape: tuple[int, int] | None = None
        for frame_id, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"MASA could not read RGB frame: {frame_path}")
            frame_shape = (int(frame.shape[0]), int(frame.shape[1]))
            if output_shape is None:
                output_shape = frame_shape
            elif frame_shape != output_shape:
                raise RuntimeError(
                    f"MASA frame shape changed from {output_shape} to {frame_shape}."
                )
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = inference_masa(
                self.model,
                frame,
                frame_id=frame_id,
                video_len=frame_count,
                test_pipeline=self.test_pipeline,
            )
            torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            instances.append(result.to("cpu"))

        # Match the released demo's track smoothing, giant-box removal and
        # per-track confidence averaging before producing benchmark boxes.
        if output_shape is None:
            raise RuntimeError("MASA received no readable RGB frames.")
        instances = filter_and_update_tracks(
            instances, (output_shape[1], output_shape[0])
        )
        predictions = []
        for result in instances:
            tracks = result[0].pred_track_instances
            predictions.append(
                self._rasterize_boxes(
                    tracks.bboxes.detach().float().cpu().numpy(),
                    tracks.instances_id.detach().cpu().numpy(),
                    tracks.scores.detach().float().cpu().numpy(),
                    output_shape,
                )
            )

        torch.cuda.empty_cache()
        return predictions, latencies_ms
