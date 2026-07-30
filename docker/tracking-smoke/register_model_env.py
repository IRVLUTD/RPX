#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _module_location(module_name: str) -> str:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        return str(Path(module_file).resolve())

    namespace_paths = sorted(
        str(Path(value).resolve())
        for value in getattr(module, "__path__", ())
    )
    if namespace_paths:
        return os.pathsep.join(namespace_paths)

    raise RuntimeError(
        f"imported module {module_name!r} has neither __file__ nor __path__"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint-repo", default="")
    parser.add_argument("--checkpoint-revision", default="")
    parser.add_argument("--import", dest="imports", action="append", default=[])
    parser.add_argument("--python-path", action="append", default=[])
    args = parser.parse_args()

    source = Path(args.source_dir).resolve()
    if not source.is_dir():
        raise SystemExit(f"missing source directory: {source}")

    site_packages = Path(
        subprocess.check_output(
            [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
        ).strip()
    )
    pth = site_packages / f"rpx_{args.model.replace('-', '_').replace('.', '_')}.pth"
    paths = [str(source), *(str((source / p).resolve()) for p in args.python_path)]
    pth.write_text("\n".join(paths) + "\n")
    for value in reversed(paths):
        sys.path.insert(0, value)

    imported = {}
    for module in args.imports:
        imported[module] = _module_location(module)

    freeze_path = Path(sys.prefix) / "rpx-pip-freeze.txt"
    freeze_path.write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True)
    )

    manifest = {
        "container": True,
        "model": args.model,
        "prompt": args.prompt,
        "python": sys.executable,
        "rpx_git_sha": os.environ["RPX_GIT_SHA"],
        "source_dir": str(source),
        "source_revision": args.source_revision,
        "checkpoint_repo": args.checkpoint_repo or None,
        "checkpoint_revision": args.checkpoint_revision or None,
        "imports": imported,
        "torch": importlib.import_module("torch").__version__,
        "cuda_build": importlib.import_module("torch").version.cuda,
        "weights_baked_into_image": False,
        "pip_freeze": str(freeze_path),
    }
    output = Path(sys.prefix) / "rpx-environment.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
