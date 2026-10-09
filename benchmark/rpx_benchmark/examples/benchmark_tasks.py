"""Run six paper task examples from an installed wheel.

``--smoke`` creates tiny synthetic fixtures and runs the real public evaluators.
It needs no dataset, weights, GPU, or network. For a real run, provide a
task-specific manifest and ``--model module:function``. T5/T6 use the canonical
VQA JSONL contract, including the verified reference crop for T6.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

import rpx_benchmark as rpx

from ..exceptions import ConfigError
from ..tasks._pipeline import PipelineResult, TaskRunConfig
from ..vqa.contract import VQASample, image_locator, load_manifest, write_manifest
from ..vqa.evaluate import evaluate_vqa
from ..vqa.hub_rgb import image_cache_name, reference_cache_name
from .profile_callable import load_callable

TASKS: dict[
    str,
    tuple[
        Callable[..., rpx.BenchmarkableModel], type[TaskRunConfig], Callable[[Any], PipelineResult]
    ],
] = {
    "T1": (rpx.make_numpy_depth_model, rpx.MonocularDepthRunConfig, rpx.run_monocular_depth),
    "T2": (rpx.make_numpy_video_depth_model, rpx.VideoDepthRunConfig, rpx.run_video_depth),
    "T3": (rpx.make_numpy_tracking_model, rpx.ObjectTrackingRunConfig, rpx.run_object_tracking),
    "T4": (rpx.make_numpy_pose_model, rpx.RelativePoseRunConfig, rpx.run_relative_pose),
}


def demo_depth(rgb):
    """Constant integration fixture; not pretrained inference."""
    return np.full(rgb.shape[:-1], 2.1, np.float32)


def demo_video_depth(rgb):
    """Non-degenerate depth ramp for the video validator's synthetic case."""
    return np.broadcast_to(np.linspace(1.9, 2.1, rgb.shape[2]), rgb.shape[:3]).astype(np.float32)


def demo_tracking(rgb):
    """One fixed track for synthetic data; real trackers need persistent state."""
    return [{"track_id": "1", "boxes": np.array([[3, 2, 12, 10]], np.float32)}]


def demo_pose(rgb_a, rgb_b):
    """Fixed 10-cm translation fixture; not a camera pose estimator."""
    return {"rotation": np.eye(3), "translation": np.array([0.1, 0.0, 0.0])}


def demo_vqa(image_paths, prompt, max_new_tokens, output_kind):
    """Fixed synthetic localization; never use this as a benchmark model."""
    return '{"label":"cup","bbox":[0,0,1000,1000]}'


def create_smoke(root: Path, task: str) -> tuple[Path, Callable[..., Any]]:
    """Create clearly synthetic local inputs with no released annotation edits."""
    root.mkdir(parents=True, exist_ok=True)
    rgb = np.full((16, 20, 3), 128, np.uint8)
    Image.fromarray(rgb).save(root / "rgb.png")
    if task in {"T5", "T6"}:
        image = image_locator({"scene_id": "scene012", "kind": "mos", "phase": 1, "frame": "00113"})
        row = dict(
            sample_id=f"synthetic-{task}",
            scene_id="scene012",
            kind="mos",
            phase=1,
            frame="00113",
            img_w=20,
            img_h=16,
            image=image,
            question_type="attr_single_color",
            question="Locate the gray cup.",
            answer="cup",
            answer_bbox=[0, 0, 19, 15],
        )
        if task == "T6":
            buffer = io.BytesIO()
            Image.new("RGB", (4, 4), "gray").save(buffer, format="PNG")
            crop = buffer.getvalue()
            row.update(
                question_type="inctx_attr_single_color",
                reference_image={**image, "shard": "synthetic/reference/rgb.tar"},
                reference_crop_bbox=[0, 0, 3, 3],
                reference_crop_sha256=hashlib.sha256(crop).hexdigest(),
                source_sample_id="synthetic-T5",
            )
        vqa_sample = VQASample.from_dict(row)
        cache = root / "images"
        cache.mkdir()
        # Populate the normal cache with SYNTHETIC pixels, only under this
        # isolated fixture root. Real runs fetch the manifest's actual images.
        Image.fromarray(rgb).save(cache / image_cache_name(vqa_sample))
        if task == "T6":
            (cache / reference_cache_name(vqa_sample)).write_bytes(crop)
        manifest = root / "manifest.jsonl"
        write_manifest([vqa_sample], manifest)
        return manifest, demo_vqa
    Image.fromarray(np.full((16, 20), 2000, np.uint16)).save(root / "depth.png")
    mask = np.zeros((16, 20), np.uint8)
    mask[2:10, 3:12] = 1
    Image.fromarray(mask).save(root / "mask.png")
    np.savez(root / "pose.npz", position=np.zeros(3), orientation=np.array([0.0, 0.0, 0.0, 1.0]))
    sample: dict[str, Any] = dict(
        id=f"synthetic-{task}",
        scene="scene012",
        scene_id="scene012",
        phase="interaction",
        difficulty="easy",
        rgb="rgb.png",
    )
    task_names = {
        "T1": "monocular_depth",
        "T2": "video_depth",
        "T3": "object_tracking",
        "T4": "relative_camera_pose",
    }
    fn: Callable[..., Any]
    if task == "T1":
        sample["depth"] = "depth.png"
        fn = demo_depth
    elif task == "T2":
        sample.update(phase=1, frame_filenames=["rgb.png"] * 3, depth_filenames=["depth.png"] * 3)
        fn = demo_video_depth
    elif task == "T3":
        sample["mask"] = "mask.png"
        fn = demo_tracking
    else:
        np.savez(
            root / "pose_b.npz",
            position=np.array([0.1, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        sample.update(rgb_b="rgb.png", pose_a="pose.npz", pose_b="pose_b.npz")
        fn = demo_pose
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"task": task_names[task], "root": str(root.resolve()), "samples": [sample]}, indent=2
        )
        + "\n"
    )
    return manifest, fn


def run_task(
    task: str,
    manifest: Path,
    fn: Callable[..., Any],
    output: Path,
    *,
    device: str = "cpu",
    split: str = "hard",
    max_samples: int | None = None,
    image_cache: Path | None = None,
    model_name: str = "custom",
    model_revision: str = "unspecified",
    depth_output_kind: str = "metric",
) -> dict[str, Any]:
    """Run one public task protocol with a supplied callable and manifest."""
    if output.exists() and any(output.iterdir()):
        raise ConfigError(f"Use a new, empty output directory: {output}")
    if task in {"T5", "T6"}:
        samples = load_manifest(manifest)
        if any(sample.is_in_context != (task == "T6") for sample in samples):
            raise ConfigError(f"{task} manifest uses the wrong one-image/two-image protocol")
        if max_samples is not None:
            raise ConfigError("For T5/T6 select a subset manifest first; all its rows are scored")
        return evaluate_vqa(
            manifest,
            fn,
            model_name=model_name,
            model_revision=model_revision,
            output_dir=output,
            image_cache=image_cache or output / "images",
        )
    wrap, config, runner = TASKS[task]
    kwargs = {"depth_output_kind": depth_output_kind} if task in {"T1", "T2"} else {}
    cfg = config(
        model=wrap(fn, name=model_name, **kwargs),
        manifest_path=str(manifest),
        output_dir=str(output),
        device=device,
        split=split,
        require_cuda=device.startswith("cuda"),
        skip_flops=True,
        max_samples=max_samples,
    )
    result, _, paths = runner(cfg)
    config_path = output / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "task": task,
                "model": model_name,
                "model_revision": model_revision,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "device": device,
                "split": split,
                "max_samples": max_samples,
                "depth_output_kind": depth_output_kind if task in {"T1", "T2"} else None,
                "protocol": "public_callable_task_runner",
                "skip_flops": True,
            },
            indent=2,
        )
        + "\n"
    )
    paths["run_config"] = config_path
    return {
        "task": task,
        "num_samples": result.num_samples,
        "aggregated": result.aggregated,
        "paths": {key: str(path) for key, path in paths.items()},
    }


TASK_LABELS = {
    "T1": "Image depth",
    "T2": "Video depth",
    "T3": "Object tracking",
    "T4": "Relative camera pose",
    "T5": "VQA grounding",
    "T6": "In-context VQA",
}
# Headline metrics shown in the console summary (all metrics are in result files).
HEADLINE = {
    "T1": (("absrel", "AbsRel"), ("delta1", "δ1")),
    "T2": (("absrel", "AbsRel"), ("tgm", "TGM")),
    "T3": (("mota", "MOTA"), ("idsw", "IDSW")),
    "T4": (("rotation_error_deg", "Rot°"), ("translation_angular_deg", "Transl°")),
    "T5": (("bbox_accuracy_at_0_5", "Acc@0.5"), ("bbox_mean_giou", "GIoU")),
    "T6": (("bbox_accuracy_at_0_5", "Acc@0.5"), ("bbox_mean_giou", "GIoU")),
}


def summary_line(task: str, report: dict[str, Any], out_dir: Path) -> str:
    """One readable console line: task, sample count, headline metrics, output folder."""
    metrics = report.get("aggregated") or report.get("metrics") or {}
    shown = []
    for key, label in HEADLINE[task]:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            shown.append(f"{label} {value:.3f}")
    samples = report.get("num_samples", "?")
    return f"  {task}  {TASK_LABELS[task]:<21}{samples:>5} samples   {' · '.join(shown):<34} → {out_dir}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "T5", "T6", "all"], required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", help="module:function; the model is loaded by your module")
    parser.add_argument("--model-name", default="custom")
    parser.add_argument("--model-revision", default="unspecified")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--depth-output-kind", choices=["metric", "relative"], default="metric")
    parser.add_argument("--image-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--json", action="store_true", help="print the full reports as JSON lines instead of a summary"
    )
    args = parser.parse_args()
    if args.smoke:
        if args.model or args.manifest or args.image_cache or args.device != "cpu":
            parser.error(
                "--smoke uses isolated synthetic fixtures and CPU; omit model/manifest/cache"
            )
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Use a new, empty smoke output directory")
        task_list = list(TASKS) + ["T5", "T6"] if args.task == "all" else [args.task]
        if not args.json:
            print("RPX smoke check: synthetic inputs, CPU, real evaluators")
        for task in task_list:
            root = args.output / task
            manifest, fn = create_smoke(root / "synthetic_inputs", task)
            report = run_task(
                task,
                manifest,
                fn,
                root / "results",
                split="easy",
                image_cache=root / "synthetic_inputs/images",
                model_name="synthetic-integration-fixture",
                model_revision="demo-v1",
            )
            if args.json:
                print(json.dumps({"task": task, "synthetic": True, "report": report}, default=str))
            else:
                print(summary_line(task, report, root / "results"))
        if not args.json:
            print(
                f"Done: {len(task_list)} task(s) ran end to end; full results under {args.output}/.\n"
                "Next, score your own model: https://irvlutd.github.io/RPX/toolkit-docs/models/"
            )
    else:
        if args.task == "all" or not args.model or not args.manifest:
            parser.error("Real runs require one --task, --manifest and --model")
        report = run_task(
            args.task,
            args.manifest,
            load_callable(args.model),
            args.output,
            device=args.device,
            split=args.split,
            max_samples=args.max_samples,
            image_cache=args.image_cache,
            model_name=args.model_name,
            model_revision=args.model_revision,
            depth_output_kind=args.depth_output_kind,
        )
        if args.json:
            print(json.dumps(report, default=str))
        else:
            print(summary_line(args.task, report, args.output))


if __name__ == "__main__":
    main()
