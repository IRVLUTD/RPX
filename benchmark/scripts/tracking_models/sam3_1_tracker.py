"""Native SAM 3.1 text-prompted video tracking adapter for RPX."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tracking_text_runtime import TextPrompt

SAM31_MODEL_ID = "facebook/sam3.1"
SAM31_MODEL_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
SAM31_SOURCE_REVISION = "6dbb02bd38288df755dfa1378000a861e65b84f6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SAM31Tracker:
    model_name = "sam3.1"
    model_id = SAM31_MODEL_ID
    model_revision = SAM31_MODEL_REVISION
    source_revision = SAM31_SOURCE_REVISION
    checkpoint_filename = "sam3.1_multiplex.pt"
    adapter_label = "SAM 3.1"
    prompt_type = "text"
    tracking_mode = "native-scene-vocabulary-text-prompted-video-segmentation"
    vocabulary_name = "RPX-primary-color-plus-canonical-name-v1"
    vocabulary_size = 70

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SAM 3.1 requires CUDA.")
        from sam3.model_builder import build_sam3_predictor

        checkpoint = Path(
            hf_hub_download(
                repo_id=SAM31_MODEL_ID,
                revision=SAM31_MODEL_REVISION,
                filename=self.checkpoint_filename,
            )
        )
        self.predictor = build_sam3_predictor(
            version="sam3.1",
            checkpoint_path=str(checkpoint),
            compile=False,
            warm_up=False,
            use_fa3=False,
            async_loading_frames=True,
        )
        self._ignored_init_state_arguments = self._patch_init_state_compatibility()
        self.checkpoint_path = str(checkpoint.resolve())
        self.checkpoint_sha256 = _sha256(checkpoint)
        model = getattr(self.predictor, "model", None)
        self.parameter_count = (
            int(sum(parameter.numel() for parameter in model.parameters()))
            if model is not None
            else 0
        )
        self._metadata: dict[str, Any] = {}

    def _patch_init_state_compatibility(self) -> tuple[str, ...]:
        """Filter base-predictor kwargs unsupported by the multiplex model.

        At the pinned upstream revision, ``Sam3BasePredictor.start_session``
        always forwards ``offload_state_to_cpu``.  SAM 3.1's multiplex
        ``init_state`` does not accept that keyword, despite using the same
        base predictor.  Filter only arguments absent from the concrete model
        signature; supported arguments and model behavior are unchanged.
        """

        original = self.predictor.model.init_state
        signature = inspect.signature(original)
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return ()
        supported = set(signature.parameters)
        base_arguments = {
            "resource_path",
            "offload_video_to_cpu",
            "offload_state_to_cpu",
            "async_loading_frames",
            "video_loader_type",
        }
        ignored = tuple(sorted(base_arguments - supported))

        def compatible_init_state(*args: Any, **kwargs: Any) -> Any:
            filtered = {key: value for key, value in kwargs.items() if key in supported}
            return original(*args, **filtered)

        self.predictor.model.init_state = compatible_init_state
        return ignored

    @staticmethod
    def _merge(
        target: np.ndarray,
        confidence: np.ndarray,
        masks: np.ndarray,
        scores: np.ndarray,
        ids: np.ndarray,
    ) -> None:
        for mask, score, object_id in zip(masks, scores, ids, strict=True):
            binary = np.asarray(mask, dtype=bool).squeeze()
            if binary.shape != target.shape:
                raise RuntimeError(
                    f"SAM 3.1 returned mask shape {binary.shape}; expected {target.shape}."
                )
            replace = binary & (float(score) > confidence)
            target[replace] = int(object_id)
            confidence[replace] = float(score)

    def prediction_metadata(self) -> dict[str, Any]:
        return self._metadata

    def track(
        self,
        video_dir: Path,
        frame_shape: tuple[int, int],
        frame_count: int,
        text_prompts: Sequence[TextPrompt],
    ) -> tuple[list[np.ndarray], list[float]]:
        predictions = [np.zeros(frame_shape, dtype=np.int32) for _ in range(frame_count)]
        confidences = [np.full(frame_shape, -np.inf, dtype=np.float32) for _ in range(frame_count)]
        latencies = [0.0] * frame_count
        prompt_records: list[dict[str, Any]] = []
        next_track_id = 1
        for prompt in text_prompts:
            session_id = None
            internal_to_output: dict[int, int] = {}
            record: dict[str, Any] = {
                "prompt": prompt.prompt_text,
                "source_mask_index": prompt.mask_index,
                "predicted_track_ids": [],
            }
            try:
                response = self.predictor.handle_request(
                    {
                        "type": "start_session",
                        "resource_path": str(video_dir),
                        "offload_video_to_cpu": True,
                        "offload_state_to_cpu": False,
                    }
                )
                session_id = response["session_id"]
                torch.cuda.synchronize()
                started = time.perf_counter()
                self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": prompt.prompt_text,
                    }
                )
                for response in self.predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                        "max_frame_num_to_track": frame_count,
                    }
                ):
                    torch.cuda.synchronize()
                    frame_index = int(response["frame_index"])
                    output = response["outputs"]
                    internal_ids = np.asarray(output["out_obj_ids"], dtype=np.int64)
                    for internal_id in internal_ids:
                        if int(internal_id) not in internal_to_output:
                            internal_to_output[int(internal_id)] = next_track_id
                            record["predicted_track_ids"].append(next_track_id)
                            next_track_id += 1
                    output_ids = np.asarray(
                        [internal_to_output[int(value)] for value in internal_ids],
                        dtype=np.int32,
                    )
                    scores = np.asarray(
                        output.get("out_probs", np.ones(len(output_ids))), dtype=np.float32
                    )
                    self._merge(
                        predictions[frame_index],
                        confidences[frame_index],
                        np.asarray(output["out_binary_masks"]),
                        scores,
                        output_ids,
                    )
                    now = time.perf_counter()
                    latencies[frame_index] += (now - started) * 1000
                    started = now
            finally:
                if session_id is not None:
                    with contextlib.suppress(Exception):
                        self.predictor.handle_request(
                            {"type": "close_session", "session_id": session_id}
                        )
            prompt_records.append(record)
        self._metadata = {
            "initialization": "native SAM 3.1 semantic video prompts",
            "model": {
                "repo": SAM31_MODEL_ID,
                "revision": SAM31_MODEL_REVISION,
                "checkpoint_sha256": self.checkpoint_sha256,
            },
            "compatibility": {
                "filtered_unsupported_init_state_arguments": list(
                    self._ignored_init_state_arguments
                )
            },
            "prompts": prompt_records,
        }
        return predictions, latencies
