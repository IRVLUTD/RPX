"""Shared, lazy GenCeption runtime for RPX adapters.

The official release is JAX based and fixes inference to 81 frames at
480x832.  RPX adapters retain the caller's spatial and temporal geometry by
resizing decoded predictions back to the supplied input grid.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

UPSTREAM_REPOSITORY = "https://github.com/google-deepmind/representations4d"
UPSTREAM_REVISION = "a47e55120cc2b00027c19c9f1937831e77541056"
CHECKPOINT_BASE_URL = "https://storage.googleapis.com/representations4d/checkpoints/genception"
EMBEDDING_BASE_URL = (
    "https://storage.googleapis.com/representations4d/assets/genception_prompt_embeddings"
)
TEXT_ENCODER_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
TEXT_ENCODER_REVISION = "38ec498cb3208fb688890f8cc7e94ede2cbd7f68"
VARIANTS = frozenset({"1.3b", "14b"})
MODEL_FRAMES = 81
OFFICIAL_ASSET_SIZES = {
    "genception_1.3b_transformer.npz": 2_847_810_118,
    "genception_1.3b_config.json": 803,
    "genception_14b_transformer.npz": 28_683_066_174,
    "genception_14b_config.json": 804,
    "vae/config.json": 724,
    "vae/diffusion_pytorch_model.safetensors": 507_591_892,
    "depth.npy": 3_702_912,
}


def _validate_video(video: np.ndarray) -> np.ndarray:
    array = np.asarray(video)
    if array.ndim != 4 or array.shape[-1] != 3 or array.shape[0] < 1:
        raise ValueError(f"expected T x H x W x 3 video, got {array.shape}")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
            raise ValueError("video must contain finite numeric RGB values")
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _resize_2d_frames(values: np.ndarray, target_hw: tuple[int, int], nearest: bool) -> np.ndarray:
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    mode = None if nearest else "F"
    frames = []
    for frame in values:
        image = Image.fromarray(frame) if mode is None else Image.fromarray(frame.astype(np.float32), mode)
        frames.append(np.asarray(image.resize((target_hw[1], target_hw[0]), resample)))
    return np.stack(frames)


def _temporal_indices(source_frames: int, target_frames: int) -> np.ndarray:
    if source_frames < 1 or target_frames < 1:
        raise ValueError("source_frames and target_frames must be positive")
    return np.rint(np.linspace(0, source_frames - 1, target_frames)).astype(np.int64)


def restore_prediction_grid(
    prediction: np.ndarray,
    *,
    target_thw: tuple[int, int, int],
    nearest: bool = False,
) -> np.ndarray:
    """Restore a T x H x W prediction to the original RPX clip grid."""
    values = np.asarray(prediction)
    if values.ndim != 3:
        raise ValueError(f"expected T x H x W prediction, got {values.shape}")
    target_t, target_h, target_w = target_thw
    values = values[_temporal_indices(values.shape[0], target_t)]
    if values.shape[1:] != (target_h, target_w):
        values = _resize_2d_frames(values, (target_h, target_w), nearest)
    return values


def restore_genception_grid(
    prediction: np.ndarray,
    *,
    input_thw: tuple[int, int, int],
    nearest: bool = False,
) -> np.ndarray:
    """Undo the official preprocessor's pad-or-uniform-sample policy."""
    values = np.asarray(prediction)
    input_t = input_thw[0]
    if input_t <= MODEL_FRAMES and values.shape[0] == MODEL_FRAMES:
        # Upstream pads short clips by repeating their final frame, so the
        # first input_t predictions correspond one-to-one with real frames.
        values = values[:input_t]
    return restore_prediction_grid(values, target_thw=input_thw, nearest=nearest)


def segmentation_video_to_masks(video: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Decode GenCeption's RGB RefVOS output into boolean foreground masks."""
    values = np.asarray(video)
    if values.ndim == 5 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"expected T x H x W x 3 RefVOS output, got {values.shape}")
    scaled = values.astype(np.float32)
    if scaled.size and scaled.max() > 1.0:
        scaled /= 255.0
    # RefVOS supervision is monochrome in RGB ambient space.  Taking the
    # channel maximum is robust to small VAE chroma reconstruction errors.
    return np.max(scaled, axis=-1) >= float(threshold)


def segmentation_video_to_scores(video: np.ndarray) -> np.ndarray:
    """Decode the RefVOS reconstruction into a [0, 1] foreground score."""
    values = np.asarray(video)
    if values.ndim == 5 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"expected T x H x W x 3 RefVOS output, got {values.shape}")
    scaled = values.astype(np.float32)
    if scaled.size and scaled.max() > 1.0:
        scaled /= 255.0
    return np.clip(np.max(scaled, axis=-1), 0.0, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetManifest:
    variant: str
    checkpoint_dir: Path
    embeddings_dir: Path

    @property
    def depth_embedding(self) -> Path:
        return self.embeddings_dir / "depth.npy"

    def validate(self) -> None:
        required = {
            self.checkpoint_dir / f"genception_{self.variant}_transformer.npz": (
                f"genception_{self.variant}_transformer.npz"
            ),
            self.checkpoint_dir / f"genception_{self.variant}_config.json": (
                f"genception_{self.variant}_config.json"
            ),
            self.checkpoint_dir / "vae/config.json": "vae/config.json",
            self.checkpoint_dir / "vae/diffusion_pytorch_model.safetensors": (
                "vae/diffusion_pytorch_model.safetensors"
            ),
            self.depth_embedding: "depth.npy",
        }
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing GenCeption assets: " + ", ".join(missing))
        invalid = [
            f"{path} ({path.stat().st_size} bytes)"
            for path, key in required.items()
            if path.stat().st_size != OFFICIAL_ASSET_SIZES[key]
        ]
        if invalid:
            raise RuntimeError("invalid GenCeption asset sizes: " + ", ".join(invalid))


def download_assets(root: str | Path, variant: str) -> AssetManifest:
    """Download one official variant, the shared VAE, and depth embedding."""
    variant = variant.lower()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    root = Path(root).expanduser().resolve()
    checkpoint = root / f"genception_{variant}"
    shared_vae = root / "shared/vae"
    embeddings = root / "prompt_embeddings"
    files = {
        checkpoint / f"genception_{variant}_transformer.npz": (
            f"{CHECKPOINT_BASE_URL}/genception_{variant}_transformer.npz"
        ),
        checkpoint / f"genception_{variant}_config.json": (
            f"{CHECKPOINT_BASE_URL}/genception_{variant}_config.json"
        ),
        shared_vae / "config.json": f"{CHECKPOINT_BASE_URL}/vae/config.json",
        shared_vae / "diffusion_pytorch_model.safetensors": (
            f"{CHECKPOINT_BASE_URL}/vae/diffusion_pytorch_model.safetensors"
        ),
        embeddings / "depth.npy": f"{EMBEDDING_BASE_URL}/depth.npy",
    }
    for destination, url in files.items():
        relative_name = f"vae/{destination.name}" if destination.parent.name == "vae" else destination.name
        expected_size = OFFICIAL_ASSET_SIZES[relative_name]
        if destination.is_file() and destination.stat().st_size == expected_size:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size != expected_size:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"incomplete GenCeption asset {destination.name}: expected "
                f"{expected_size} bytes"
            )
        os.replace(temporary, destination)
    checkpoint.mkdir(parents=True, exist_ok=True)
    vae_link = checkpoint / "vae"
    if not vae_link.exists():
        vae_link.symlink_to(Path("../shared/vae"), target_is_directory=True)
    manifest = AssetManifest(variant, checkpoint, embeddings)
    manifest.validate()
    checksums = {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    }
    (root / f"genception_{variant}_sha256.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


class WanPromptEncoder:
    """Generate the exact 226 x 4096 UMT5 embeddings expected by GenCeption."""

    def __init__(self, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoTokenizer, UMT5EncoderModel
        except ImportError as exc:
            raise ImportError(
                "arbitrary GenCeption referring expressions require transformers and torch"
            ) from exc
        self._torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            TEXT_ENCODER_ID,
            subfolder="tokenizer",
            revision=TEXT_ENCODER_REVISION,
        )
        self.model = UMT5EncoderModel.from_pretrained(
            TEXT_ENCODER_ID,
            subfolder="text_encoder",
            revision=TEXT_ENCODER_REVISION,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device).eval()

    def __call__(self, prompt: str) -> np.ndarray:
        encoded = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=226,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = encoded.input_ids.to(self.device)
        mask = encoded.attention_mask.to(self.device)
        with self._torch.inference_mode():
            hidden = self.model(ids, mask).last_hidden_state[0]
        length = int(mask[0].sum().item())
        output = self._torch.zeros((226, hidden.shape[-1]), dtype=hidden.dtype)
        output[:length] = hidden[:length].cpu()
        return output.float().numpy()[None]


class GenCeptionRuntime:
    """One loaded official pipeline shared by RPX task adapters."""

    def __init__(
        self,
        variant: str,
        checkpoint_dir: str | Path,
        embeddings_dir: str | Path,
        *,
        pipeline: Any | None = None,
        prompt_encoder: Callable[[str], np.ndarray] | None = None,
        segmentation_threshold: float = 0.5,
    ) -> None:
        variant = variant.lower()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        self.variant = variant
        self.assets = AssetManifest(
            variant, Path(checkpoint_dir).expanduser(), Path(embeddings_dir).expanduser()
        )
        self.pipeline = pipeline
        self.prompt_encoder = prompt_encoder
        self.segmentation_threshold = float(segmentation_threshold)
        self._prompt_cache: dict[str, np.ndarray] = {}

    def setup(self) -> None:
        if self.pipeline is not None:
            return
        self.assets.validate()
        try:
            import jax.numpy as jnp
            from representations4d.genception.pipeline import GenCeptionPipeline
        except ImportError as exc:
            raise ImportError(
                f"install the official GenCeption source at {UPSTREAM_REVISION} with "
                "`pip install -e '.[genception]'`"
            ) from exc
        self.pipeline = GenCeptionPipeline.from_pretrained(
            str(self.assets.checkpoint_dir), model_size=self.variant, dtype=jnp.bfloat16
        )

    def _embedding(self, expression: str | None) -> np.ndarray:
        if expression is None:
            return np.load(self.assets.depth_embedding)
        if self.prompt_encoder is None:
            self.prompt_encoder = WanPromptEncoder(
                os.environ.get("RPX_GENCEPTION_TEXT_DEVICE", "cpu")
            )
        if expression not in self._prompt_cache:
            embedding = np.asarray(self.prompt_encoder(expression), dtype=np.float32)
            if embedding.shape != (1, 226, 4096):
                raise ValueError(
                    "GenCeption prompt encoder must return shape (1, 226, 4096); "
                    f"got {embedding.shape}"
                )
            self._prompt_cache[expression] = embedding
        return self._prompt_cache[expression]

    def predict_depth(self, video: np.ndarray) -> np.ndarray:
        self.setup()
        source = _validate_video(video)
        result = self.pipeline(
            prompt_embeds=self._embedding(None), input_video=source
        )
        decoded = np.asarray(self.pipeline.decode_depth(result["video"]), dtype=np.float32)
        return restore_genception_grid(decoded, input_thw=source.shape[:3]).astype(np.float32)

    def predict_masks(self, video: np.ndarray, expression: str) -> np.ndarray:
        return self.predict_mask_scores(video, expression) >= self.segmentation_threshold

    def predict_mask_scores(self, video: np.ndarray, expression: str) -> np.ndarray:
        self.setup()
        source = _validate_video(video)
        result = self.pipeline(
            prompt_embeds=self._embedding(expression), input_video=source
        )
        scores = segmentation_video_to_scores(result["video"])
        return restore_genception_grid(
            scores, input_thw=source.shape[:3], nearest=False
        ).astype(np.float32)


def runtime_from_environment(variant: str, **kwargs: Any) -> GenCeptionRuntime:
    root = Path(os.environ.get("RPX_GENCEPTION_ROOT", "~/.cache/rpx/genception")).expanduser()
    return GenCeptionRuntime(
        variant,
        checkpoint_dir=root / f"genception_{variant}",
        embeddings_dir=root / "prompt_embeddings",
        **kwargs,
    )


def load_rgb_video(paths: Sequence[str | Path]) -> np.ndarray:
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return np.stack(frames)
