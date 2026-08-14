"""Official MITS multi-object box-initialized VOS adapter for RPX D3."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MITS_MODEL_ID = "yoxu515/MITS"
MITS_SOURCE_REVISION = "462ebee2c995818998d5f96ab6b615dd86c42688"
MITS_MODEL_REVISION = "gdrive-1Db9DxXc-gyRkxhKs0AMXJ2RH6DyHWCfq"
MITS_CHECKPOINT = "mits.pth"
MITS_CHECKPOINT_GDRIVE_ID = "1Db9DxXc-gyRkxhKs0AMXJ2RH6DyHWCfq"
MITS_CHECKPOINT_BYTES = 385_560_551
MITS_CHECKPOINT_SHA256 = (
    "9291f533b9e47b22803acedc941761fdffbcd023ceffe5c611d540e3509d3f22"
)
MITS_LONG_TERM_MEMORY_GAP = 30
MITS_SHORT_TERM_MEMORY_GAP = 10
MITS_LONG_TERM_MEMORY_MAX = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_checkpoint(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == MITS_CHECKPOINT_BYTES
        and _sha256(path) == MITS_CHECKPOINT_SHA256
    )


def _download_checkpoint(cache_root: Path) -> Path:
    destination = cache_root / "mits" / MITS_MODEL_REVISION / MITS_CHECKPOINT
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _valid_checkpoint(destination):
        return destination

    import gdown

    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.part")
    temporary.unlink(missing_ok=True)
    try:
        result = gdown.download(
            id=MITS_CHECKPOINT_GDRIVE_ID,
            output=str(temporary),
            quiet=False,
        )
        if result is None or not _valid_checkpoint(temporary):
            raise RuntimeError(
                "MITS checkpoint failed its pinned size/SHA-256 integrity check."
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class MITSTracker:
    """Run official MITS using tight first-frame boxes for every RPX object."""

    model_name = "mits"
    model_id = MITS_MODEL_ID
    model_revision = MITS_MODEL_REVISION
    source_revision = MITS_SOURCE_REVISION
    checkpoint_filename = MITS_CHECKPOINT
    adapter_label = "MITS"
    prompt_type = "box"
    tracking_mode = "multi-object-box-initialized-vos"

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise RuntimeError("The production MITS adapter requires device='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to the MITS adapter.")

        from configs.mits import EngineConfig
        from networks.models import build_vos_model
        from utils.checkpoint import load_network

        cfg = EngineConfig("rpx")
        cfg.TEST_BOXT = True
        cfg.TEST_BOX_HEAD = True
        cfg.TEST_LONG_TERM_MEM_GAP = MITS_LONG_TERM_MEMORY_GAP
        cfg.TEST_SHORT_TERM_MEM_GAP = MITS_SHORT_TERM_MEMORY_GAP
        cfg.TEST_LONG_TERM_MEM_MAX = MITS_LONG_TERM_MEMORY_MAX
        cfg.TEST_MAX_SHORT_EDGE = None
        cfg.TEST_MAX_LONG_EDGE = 1040
        cfg.TEST_MULTISCALE = [1.0]
        cfg.TEST_FLIP = False
        cfg.TEST_INPLACE_FLIP = False

        checkpoint = _download_checkpoint(
            Path(os.environ.get("HF_HOME", "/cache/huggingface"))
        )
        model = build_vos_model(cfg.MODEL_VOS, cfg)
        model, removed = load_network(model, str(checkpoint), 0)
        if removed:
            raise RuntimeError(f"MITS checkpoint contains unexpected parameters: {removed}")

        self.model = model.eval()
        self.cfg = cfg
        self.device = device
        self.checkpoint_path = str(checkpoint.resolve())
        self.checkpoint_sha256 = _sha256(checkpoint)
        self.parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    @staticmethod
    def _box_prompt(mask: np.ndarray, object_ids: list[int]) -> tuple[np.ndarray, dict[int, int]]:
        """Rasterize tight boxes with contiguous MITS IDs, preserving small objects."""
        local_by_original = {object_id: index + 1 for index, object_id in enumerate(object_ids)}
        output = np.zeros(mask.shape, dtype=np.int32)
        rectangles: list[tuple[int, int, int, int, int, int]] = []
        for object_id in object_ids:
            ys, xs = np.where(mask == object_id)
            if not len(xs):
                raise ValueError(f"first_frame_mask has no pixels for object {object_id}.")
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            rectangles.append((area, local_by_original[object_id], x0, y0, x1, y1))
        for _, local_id, x0, y0, x1, y1 in sorted(rectangles, reverse=True):
            output[y0 : y1 + 1, x0 : x1 + 1] = local_id
        return output, local_by_original

    def _transform_image(self, path: Path) -> torch.Tensor:
        from dataloaders.video_transforms import MultiRestrictSize, MultiToTensor

        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        resize = MultiRestrictSize(
            self.cfg.TEST_MAX_SHORT_EDGE,
            self.cfg.TEST_MAX_LONG_EDGE,
            False,
            False,
            [1.0],
            self.cfg.MODEL_ALIGN_CORNERS,
        )
        sample = MultiToTensor()(resize({"current_img": image})[0])
        return sample["current_img"].unsqueeze(0).cuda(non_blocking=True)

    def track(
        self,
        video_dir: Path,
        first_frame_mask: np.ndarray,
        frame_count: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        initial = np.asarray(first_frame_mask)
        if initial.ndim != 2 or not np.issubdtype(initial.dtype, np.integer):
            raise ValueError("first_frame_mask must be a 2-D integer instance mask.")
        object_ids = [int(value) for value in np.unique(initial) if value > 0]
        if not object_ids:
            raise ValueError("first_frame_mask contains no positive object IDs.")

        frames = sorted(video_dir.glob("*.jpg"))
        if len(frames) != frame_count:
            raise RuntimeError(f"MITS received {len(frames)} frames; expected {frame_count}.")

        from networks.engines import build_engine

        box_prompt, local_by_original = self._box_prompt(initial, object_ids)
        original_by_local = {local: original for original, local in local_by_original.items()}
        engine = build_engine(
            self.cfg.MODEL_ENGINE,
            phase="eval",
            aot_model=self.model,
            gpu_id=0,
            long_term_mem_gap=MITS_LONG_TERM_MEMORY_GAP,
            short_term_mem_skip=MITS_SHORT_TERM_MEMORY_GAP,
        )
        engine.eval()

        predictions: list[np.ndarray] = [initial.astype(np.int32, copy=True)]
        latencies_ms: list[float] = [0.0] * frame_count
        height, width = initial.shape

        with torch.inference_mode():
            reference = self._transform_image(frames[0])
            prompt = torch.from_numpy(box_prompt).cuda().view(1, 1, height, width).float()
            prompt = F.interpolate(prompt, size=reference.shape[-2:], mode="nearest")
            engine.add_reference_frame(reference, prompt, obj_nums=[len(object_ids)])

            for index, frame_path in enumerate(frames[1:], start=1):
                image = self._transform_image(frame_path)
                torch.cuda.synchronize()
                started = time.perf_counter()
                engine.match_propogate_one_frame(image)
                if len(object_ids) <= self.cfg.MODEL_MAX_OBJ_NUM:
                    engine.decode_current_boxes(img=image)
                logits = engine.decode_current_logits()
                logits = F.interpolate(
                    logits,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=self.cfg.MODEL_ALIGN_CORNERS,
                )
                local_prediction = torch.argmax(logits, dim=1, keepdim=True).float()
                memory_mask = F.interpolate(
                    local_prediction,
                    size=engine.input_size_2d,
                    mode="nearest",
                )
                engine.update_memory(memory_mask)
                torch.cuda.synchronize()
                latencies_ms[index] = (time.perf_counter() - started) * 1000.0

                local = local_prediction[0, 0].cpu().numpy().astype(np.int32, copy=False)
                output = np.zeros((height, width), dtype=np.int32)
                for local_id, original_id in original_by_local.items():
                    output[local == local_id] = original_id
                predictions.append(output)

        del engine
        torch.cuda.empty_cache()
        return predictions, latencies_ms
