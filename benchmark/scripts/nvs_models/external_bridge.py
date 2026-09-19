"""Subprocess boundary for NVS projects with isolated CUDA environments.

Each cumulative image installs an executable accepting ``--input`` and
``--output`` NPZ paths. This avoids importing mutually incompatible upstream
packages into the evaluator process.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from shutil import which
from typing import Any

import numpy as np

from rpx_benchmark.exceptions import AdapterError

from .specs import NVSModelSpec


class ExternalBridgeNVS:
    native_precision = "fp32"
    torch_module = None

    def __init__(self, spec: NVSModelSpec, *, device: str = "cuda", **_: Any) -> None:
        self.spec = spec
        self.name = spec.key
        self.device = device
        env_name = f"RPX_NVS_{spec.key.upper()}_BRIDGE".replace("-", "_")
        configured = os.environ.get(env_name)
        self.command = shlex.split(configured) if configured else [
            f"/usr/local/bin/rpx-nvs-{spec.key}-bridge"
        ]

    def __call__(
        self,
        context_rgbs: list[np.ndarray],
        context_depths: list[np.ndarray],
        context_poses: list[np.ndarray],
        target_pose: np.ndarray,
    ) -> dict[str, Any]:
        if not context_rgbs:
            raise ValueError(f"{self.spec.display_name} requires context RGB frames")
        if self.spec.uses_sensor_depth and len(context_depths) != len(context_rgbs):
            raise ValueError(
                f"{self.spec.display_name} requires one RPX depth map per RGB frame"
            )
        executable = self.command[0]
        if not (Path(executable).is_file() or which(executable)):
            env_name = f"RPX_NVS_{self.spec.key.upper()}_BRIDGE".replace("-", "_")
            raise AdapterError(
                f"{self.spec.display_name} bridge is not installed.",
                hint=(f"Build its cumulative NVS image or set {env_name}. "
                      f"Upstream: {self.spec.upstream}"),
            )

        with tempfile.TemporaryDirectory(prefix=f"rpx-{self.spec.key}-") as tmp:
            root = Path(tmp)
            input_path = root / "input.npz"
            output_path = root / "output.npz"
            np.savez_compressed(
                input_path,
                context_rgbs=np.stack(context_rgbs),
                context_depths=(np.stack(context_depths) if context_depths
                                else np.empty((0,), dtype=np.float32)),
                context_poses=np.stack(context_poses),
                target_pose=np.asarray(target_pose),
                device=np.asarray(self.device),
            )
            completed = subprocess.run(
                [*self.command, "--input", str(input_path), "--output", str(output_path)],
                check=False, text=True, capture_output=True,
            )
            if completed.returncode != 0:
                message = (completed.stderr or completed.stdout).strip()[-2000:]
                raise AdapterError(
                    f"{self.spec.display_name} bridge failed with exit status "
                    f"{completed.returncode}: {message}"
                )
            if not output_path.is_file():
                raise AdapterError(
                    f"{self.spec.display_name} bridge did not create {output_path}"
                )
            with np.load(output_path) as output:
                if "rgb" not in output:
                    raise AdapterError(
                        f"{self.spec.display_name} bridge output is missing 'rgb'"
                    )
                rgb = np.asarray(output["rgb"], dtype=np.uint8)
                depth = (np.asarray(output["depth"], dtype=np.float32)
                         if "depth" in output and output["depth"].size else None)
            expected_hw = context_rgbs[0].shape[:2]
            if rgb.shape != (*expected_hw, 3):
                raise AdapterError(
                    f"{self.spec.display_name} returned RGB shape {rgb.shape}; "
                    f"expected {(*expected_hw, 3)}"
                )
            if depth is not None and depth.shape != expected_hw:
                raise AdapterError(
                    f"{self.spec.display_name} returned depth shape {depth.shape}; "
                    f"expected {expected_hw}"
                )
            return {"rgb": rgb, "depth": depth}


def build_external(spec: NVSModelSpec, *, device: str = "cuda", **kwargs: Any) -> Any:
    return ExternalBridgeNVS(spec, device=device, **kwargs)
