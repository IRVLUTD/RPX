"""Create one pinned, CUDA-enabled environment for a canonical depth model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

TORCH_VERSION = "2.10.0+cu128"
TORCHVISION_VERSION = "0.25.0+cu128"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

UPSTREAMS = {
    "da3": (
        "https://github.com/ByteDance-Seed/depth-anything-3.git",
        "41736238f5bced4debf3f2a12375d2466874866d",
    ),
    "lotus2": (
        "https://github.com/EnVision-Research/Lotus-2.git",
        "2d5e4522f7213611184fd31992d0fac17ec36035",
    ),
    "metric3d": (
        "https://github.com/YvanYin/Metric3D.git",
        "eb5b6fac0dc155e4e52f576e304fbf11655ff339",
    ),
    "moge2": (
        "https://github.com/microsoft/MoGe.git",
        "07444410f1e33f402353b99d6ccd26bd31e469e8",
    ),
    "unidepth2": (
        "https://github.com/lpiccinelli-eth/UniDepth.git",
        "8d8cfe4c7ee15297099983607febf0d4f32eb3d6",
    ),
    "metadepth": (
        "https://github.com/facebookresearch/metadepth.git",
        "810b77c3e56712813a0de42a130cfbe6f5b19b90",
    ),
    "chrono": (
        "https://github.com/jiahao-shao1/ChronoDepth.git",
        "2580e891eb33a5c3a95eab27b745f64ac28b3514",
    ),
    # v1.0.1 is the last official Python-3.11-compatible release.  Current
    # main requires Python >=3.13 and is a different deployment surface.
    "depthcrafter": (
        "https://github.com/Tencent/DepthCrafter.git",
        "fc83d365f2b781ab05aeb94b13f7e97417df7d97",
    ),
    "monst3r": (
        "https://github.com/Junyi42/monst3r.git",
        "574cc77ad278bad582f470e5382624e01f8769a7",
    ),
    "rolling": (
        "https://github.com/prs-eth/rollingdepth.git",
        "c233765fb7adf9682442a9fbef3eb07212bb70fd",
    ),
    "vggt": (
        "https://github.com/facebookresearch/vggt.git",
        "a288dd0f14786c93483e45524328726ab7b1b4ce",
    ),
    "video_da": (
        "https://github.com/DepthAnything/Video-Depth-Anything.git",
        "4f5ae23172ba60fd7bc11ef671cca678842c7072",
    ),
    "vigeo": (
        "https://github.com/aigc3d/ViGeo.git",
        "c2cf70302be459ca7be758185fa71b89949ee4b5",
    ),
}

MODEL_FAMILY = {
    "da-v2-large": "transformers-image",
    "depth-pro": "transformers-image",
    "hyden": "metadepth",
    "da3-metric-l": "da3",
    "depthlm": "depthlm",
    "lotus-2": "lotus2",
    "metric3d-v2": "metric3d",
    "moge-2-vit-l": "moge2",
    "unidepth-v2": "unidepth2",
    "chrono-depth": "chrono",
    "da3-video": "da3",
    "depth-crafter": "depthcrafter",
    "monst3r": "monst3r",
    "rolling-depth": "rolling",
    "vggt-omega": "vggt",
    "video-da": "video_da",
    "vigeo": "vigeo",
}

VIDEO_MODELS = {
    "chrono-depth",
    "da3-video",
    "depth-crafter",
    "monst3r",
    "rolling-depth",
    "vggt-omega",
    "video-da",
    "vigeo",
}

FAMILY_PACKAGES = {
    "transformers-image": [
        "transformers==5.5.3",
        "timm==1.0.26",
        "accelerate==1.14.0",
        "safetensors",
    ],
    "depthlm": [
        "transformers==5.5.3",
        "accelerate==1.14.0",
        "sentencepiece",
        "safetensors",
    ],
    "da3": [
        "numpy<2",
        "einops",
        "opencv-python",
        "omegaconf",
        "trimesh",
        "e3nn",
        "imageio",
        "safetensors",
        "xformers",
    ],
    "lotus2": [
        "numpy==1.26.4",
        "matplotlib==3.10.0",
        "diffusers==0.32.2",
        # Lotus-2 was published while Transformers 4.x was current.  The
        # unbounded PEFT dependency now resolves to Transformers 5, which
        # removed FLAX_WEIGHTS_NAME required by Diffusers 0.32.2.
        "transformers==4.46.3",
        "accelerate==1.14.0",
        "peft==0.14.0",
        "protobuf==5.29.0",
        "sentencepiece==0.2.0",
        "opencv-python==4.11.0.86",
        "omegaconf==2.3.0",
        "safetensors",
    ],
    "metric3d": [
        "numpy<2",
        "opencv-python",
        "timm==1.0.26",
        "mmengine",
        "plyfile",
        "iopath",
        "imagecorruptions",
        "tensorboardX",
    ],
    "moge2": [],
    "unidepth2": [
        "numpy<2.3",
        "einops>=0.7.0",
        "matplotlib",
        "opencv-python",
        "scipy",
        "timm==1.0.26",
        "trimesh",
        "tabulate",
        "termcolor",
        "wandb",
        "xformers",
    ],
    "metadepth": [],
    "chrono": [
        "numpy==1.26.4",
        "diffusers==0.29.1",
        "transformers==4.43.3",
        "accelerate==0.28.0",
        "easydict==1.13",
        "einops==0.8.0",
        "mediapy==1.2.2",
        "opencv-python",
        "xformers",
    ],
    "depthcrafter": [
        "numpy==1.26.4",
        "diffusers==0.29.1",
        "transformers==4.41.2",
        "accelerate==0.30.1",
        "mediapy==1.2.0",
        "fire==0.6.0",
        "decord==0.6.0",
        "xformers",
    ],
    "monst3r": [
        "numpy<2",
        "roma",
        "matplotlib",
        "tqdm",
        "opencv-python",
        "scipy",
        "einops",
        "gdown",
        "trimesh",
        "pyglet<2",
        "evo",
    ],
    "rolling": [
        "numpy<2",
        "transformers>=4.32.1,<5",
        "einops",
        "matplotlib",
        "av>=13.1.0",
        "omegaconf",
        "xformers",
    ],
    "vggt": ["numpy<2", "einops", "safetensors", "opencv-python"],
    "video_da": [
        "numpy<2",
        "opencv-python",
        "matplotlib",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.4.7",
        "decord",
        "einops",
        "easydict",
        "tqdm",
        "OpenEXR",
        "xformers",
    ],
    "vigeo": ["einops>=0.7"],
}

FAMILY_IMPORTS = {
    "transformers-image": ["transformers"],
    "depthlm": ["transformers"],
    "da3": ["depth_anything_3"],
    "lotus2": ["infer", "pipeline"],
    "metric3d": [],  # loaded by the adapter from the pinned torch.hub ref
    "moge2": ["moge"],
    # Import the adapter's actual entry point. ``import unidepth`` alone does
    # not traverse the model/visualization imports and can miss dependencies.
    "unidepth2": ["unidepth.models"],
    "metadepth": ["metadepth.mogev2"],
    "chrono": ["chronodepth"],
    "depthcrafter": ["depthcrafter.depth_crafter_ppl"],
    "monst3r": ["dust3r"],
    "rolling": ["rollingdepth"],
    "vggt": ["vggt"],
    "video_da": ["video_depth_anything.video_depth"],
    "vigeo": ["vigeo"],
}


def _run(command: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def _clone_pinned(name: str, source_root: Path, *, dry_run: bool) -> Path:
    url, revision = UPSTREAMS[name]
    target = source_root / name
    if not target.exists():
        _run(["git", "clone", "--filter=blob:none", url, str(target)], dry_run=dry_run)
    _run(["git", "fetch", "origin", revision], cwd=target, dry_run=dry_run)
    _run(["git", "checkout", "--detach", revision], cwd=target, dry_run=dry_run)
    return target


def _add_source_path(python: str, source: Path) -> None:
    site_packages = subprocess.run(
        [python, "-c", "import site; print(site.getsitepackages()[0])"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    (Path(site_packages) / f"rpx_{source.name}_source.pth").write_text(
        str(source) + os.linesep,
        encoding="utf-8",
    )


def _storage_preflight(env_root: Path, minimum_free_gb: float) -> float:
    probe = env_root.expanduser().resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1024**3
    if free_gb < minimum_free_gb:
        raise SystemExit(
            f"only {free_gb:.1f} GiB is free for {env_root}; "
            f"environment setup requires at least {minimum_free_gb:.1f} GiB"
        )
    if free_gb < 150:
        print(
            f"WARNING: {free_gb:.1f} GiB free. This is enough for one family, "
            "but reserve at least 150 GiB for the complete 17-model server cache.",
            flush=True,
        )
    return free_gb


def _configure_install_storage(
    env_root: Path,
    *,
    temp_dir: Path | None,
    minimum_free_gb: float,
) -> tuple[Path, Path]:
    """Keep pip's cache and temporary wheels off small system filesystems."""
    root = env_root.expanduser().resolve()
    temporary = (temp_dir or root / ".tmp").expanduser().resolve()
    cache = Path(os.environ.get("PIP_CACHE_DIR", root / ".pip-cache")).expanduser().resolve()

    for path in {root, temporary, cache}:
        _storage_preflight(path, minimum_free_gb)
        path.mkdir(parents=True, exist_ok=True)

    # pip may retain every CUDA wheel in its temporary download directory until
    # dependency resolution completes.  /tmp is commonly a 2-4 GiB tmpfs, so
    # direct all temporary files to the explicitly provisioned setup volume.
    os.environ["TMPDIR"] = str(temporary)
    os.environ["TEMP"] = str(temporary)
    os.environ["TMP"] = str(temporary)
    os.environ["PIP_CACHE_DIR"] = str(cache)
    return temporary, cache


def _last_json_line(output: str) -> object:
    """Parse our final JSON marker while tolerating upstream stdout banners."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("verification command produced no JSON marker")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"verification command ended without valid JSON: {lines[-1]!r}"
        ) from exc


def _verification_stdout(completed: subprocess.CompletedProcess, label: str) -> str:
    if completed.returncode:
        details = "\n".join(
            part.strip()
            for part in (completed.stdout or "", completed.stderr or "")
            if part.strip()
        )
        raise RuntimeError(
            f"{label} failed with status {completed.returncode}"
            + (f":\n{details}" if details else "")
        )
    return completed.stdout or ""


def _torch_install_commands(
    python: str,
    wheelhouse: Path,
    *,
    index: str,
    torch_version: str,
    torchvision_version: str,
) -> tuple[list[str], list[str], list[str]]:
    requirements = [
        f"torch=={torch_version}",
        f"torchvision=={torchvision_version}",
    ]
    offline_download = [
        python,
        "-m",
        "pip",
        "download",
        "--no-index",
        "--find-links",
        str(wheelhouse),
        "--dest",
        str(wheelhouse),
        *requirements,
    ]
    online_download = [
        python,
        "-m",
        "pip",
        "download",
        "--dest",
        str(wheelhouse),
        "--index-url",
        index,
        *requirements,
    ]
    install = [
        python,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheelhouse),
        *requirements,
    ]
    return offline_download, online_download, install


def _ensure_torch_wheelhouse(
    offline_download: list[str],
    online_download: list[str],
    *,
    dry_run: bool,
) -> None:
    if dry_run:
        _run(online_download, dry_run=True)
        return
    print("+", " ".join(offline_download), flush=True)
    completed = subprocess.run(offline_download, check=False)
    if completed.returncode:
        print("Wheelhouse incomplete; downloading missing pinned wheels.", flush=True)
        _run(online_download)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_FAMILY))
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--python", default="python3.11")
    parser.add_argument("--torch-version", default=TORCH_VERSION)
    parser.add_argument("--torchvision-version", default=TORCHVISION_VERSION)
    parser.add_argument("--torch-index", default=TORCH_INDEX)
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="pip temporary directory (default: <env-root>/.tmp)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-other-python",
        action="store_true",
        help="permit a non-3.11 interpreter for a local diagnostic",
    )
    args = parser.parse_args()
    if args.min_free_gb < 0:
        parser.error("--min-free-gb must be non-negative")
    # Model environments involve multi-gigabyte CUDA wheels. Slow university
    # or VPN links routinely exceed pip's short default read timeout.
    os.environ.setdefault("PIP_DEFAULT_TIMEOUT", "120")
    os.environ.setdefault("PIP_RETRIES", "10")
    if not args.dry_run:
        temporary, cache = _configure_install_storage(
            args.env_root,
            temp_dir=args.temp_dir,
            minimum_free_gb=args.min_free_gb,
        )
        print(f"Install temporary directory: {temporary}", flush=True)
        print(f"pip cache: {cache}", flush=True)
    else:
        os.environ.setdefault(
            "PIP_CACHE_DIR",
            str(args.env_root.expanduser().resolve() / ".pip-cache"),
        )

    repo_root = Path(__file__).resolve().parents[2]
    family = MODEL_FAMILY[args.model]
    env_dir = args.env_root.expanduser().resolve() / family
    source_root = args.env_root.expanduser().resolve() / "sources"
    wheelhouse = args.env_root.expanduser().resolve() / ".wheelhouse"
    if not args.dry_run:
        args.env_root.mkdir(parents=True, exist_ok=True)
        source_root.mkdir(parents=True, exist_ok=True)
        wheelhouse.mkdir(parents=True, exist_ok=True)

    version = subprocess.run(
        [
            args.python,
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if version != "3.11" and not args.allow_other_python:
        parser.error(f"Python 3.11 is required; {args.python} reports {version}")

    if not (env_dir / "bin" / "python").exists():
        _run([args.python, "-m", "venv", str(env_dir)], dry_run=args.dry_run)
    python = str(env_dir / "bin" / "python")
    _run([python, "-m", "pip", "install", "--upgrade", "pip==25.3"], dry_run=args.dry_run)
    offline_torch, download_torch, install_torch = _torch_install_commands(
        python,
        wheelhouse,
        index=args.torch_index,
        torch_version=args.torch_version,
        torchvision_version=args.torchvision_version,
    )
    # A wheelhouse makes the multi-family matrix download the multi-gigabyte
    # CUDA stack once. Each family still receives its own isolated install.
    _ensure_torch_wheelhouse(
        offline_torch,
        download_torch,
        dry_run=args.dry_run,
    )
    _run(install_torch, dry_run=args.dry_run)
    _run(
        [python, "-m", "pip", "install", "-e", f"{repo_root / 'benchmark'}[hub]", "psutil"],
        dry_run=args.dry_run,
    )
    packages = FAMILY_PACKAGES[family]
    if packages:
        _run([python, "-m", "pip", "install", *packages], dry_run=args.dry_run)

    upstream_path = None
    if family in UPSTREAMS:
        upstream_path = _clone_pinned(family, source_root, dry_run=args.dry_run)
        if family in {"da3", "unidepth2"}:
            _run(
                [python, "-m", "pip", "install", "--no-deps", "-e", str(upstream_path)],
                dry_run=args.dry_run,
            )
        elif family == "moge2":
            _run(
                [python, "-m", "pip", "install", "-e", str(upstream_path)],
                dry_run=args.dry_run,
            )
        elif family in {"vggt", "vigeo"}:
            _run(
                [python, "-m", "pip", "install", "--no-deps", "-e", str(upstream_path)],
                dry_run=args.dry_run,
            )
        elif family == "rolling":
            _run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "-e",
                    str(upstream_path / "diffusers"),
                ],
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                _add_source_path(python, upstream_path)
        elif family == "monst3r":
            _run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=upstream_path,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                _add_source_path(python, upstream_path)
        elif family in {"lotus2", "chrono", "depthcrafter", "video_da"}:
            if not args.dry_run:
                _add_source_path(python, upstream_path)
        elif family == "metadepth" and not args.dry_run:
            # MetaDepth uses package-relative imports (metadepth.mogev2,
            # metadepth.da2), so Python needs the checkout's parent.
            _add_source_path(python, upstream_path.parent)

    if not args.dry_run:
        import_completed = subprocess.run(
            [
                python,
                "-c",
                (
                    "import importlib, json; "
                    f"mods={FAMILY_IMPORTS[family]!r}; "
                    "[importlib.import_module(name) for name in mods]; "
                    "print(json.dumps(mods))"
                ),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        import_output = _verification_stdout(import_completed, "upstream import check")
        import_check = _last_json_line(import_output)
        check_completed = subprocess.run(
            [
                python,
                "-c",
                (
                    "import json, torch; "
                    "assert torch.cuda.is_available(); "
                    "print(json.dumps({'torch': torch.__version__, "
                    "'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}))"
                ),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        check_output = _verification_stdout(check_completed, "CUDA visibility check")
        check = _last_json_line(check_output)
        freeze = subprocess.run(
            [python, "-m", "pip", "freeze"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        (env_dir / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
        payload = {
            "model": args.model,
            "family": family,
            "python": python,
            "torch_request": {
                "torch": args.torch_version,
                "torchvision": args.torchvision_version,
                "index": args.torch_index,
            },
            "torch": check,
            "import_check": import_check,
            "upstream": (
                {"path": str(upstream_path), "revision": UPSTREAMS[family][1]}
                if upstream_path is not None
                else None
            ),
        }
        (env_dir / "rpx-environment.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    print(f"{'Would create' if args.dry_run else 'Ready'}: {python}")
    print(
        "Run: "
        f"{python} {repo_root / 'benchmark/scripts/run_depth_smoke_gate.py'} "
        f"--task {'video' if args.model in VIDEO_MODELS else 'image'} "
        f"--model {args.model} --gate micro"
    )


if __name__ == "__main__":
    main()
