"""Shared scaffolding for pose adapters — reduces duplication across the zoo."""

from __future__ import annotations

import numpy as np


def validate_pair(pair: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pull rgb_a / rgb_b out of an extract_inputs dict, validate shape."""
    if "rgb_a" not in pair or "rgb_b" not in pair:
        from rpx_benchmark.exceptions import AdapterError

        raise AdapterError(
            f"pose adapter expects {{'rgb_a','rgb_b'}}, got {sorted(pair.keys())}",
            hint="BatchedRelativePoseBenchmarkModel.extract_inputs returns the "
            "right shape; check your custom invoker.",
        )
    rgb_a = np.asarray(pair["rgb_a"], dtype=np.uint8)
    rgb_b = np.asarray(pair["rgb_b"], dtype=np.uint8)
    for tag, r in (("rgb_a", rgb_a), ("rgb_b", rgb_b)):
        if r.ndim != 3 or r.shape[2] != 3:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"pose adapter {tag} expects H×W×3 uint8, got shape {r.shape}",
            )
    return rgb_a, rgb_b


def coerce_pose_output(rot: np.ndarray, trans: np.ndarray) -> dict:
    """Normalise (rotation, translation) → ``{rotation: 3x3, translation: 3}``.

    Tolerates rotation as 3×3 matrix or quaternion (xyzw).
    """
    rot = np.asarray(rot, dtype=np.float64).reshape(-1)
    if rot.size == 4:
        x, y, z, w = rot.tolist()
        rot = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
    elif rot.size == 9:
        rot = rot.reshape(3, 3)
    else:
        from rpx_benchmark.exceptions import AdapterError

        raise AdapterError(
            f"unexpected rotation shape: {rot.size} entries; expected 9 (3×3) or 4 (xyzw).",
        )
    trans = np.asarray(trans, dtype=np.float64).reshape(-1)
    if trans.size != 3:
        from rpx_benchmark.exceptions import AdapterError

        raise AdapterError(
            f"unexpected translation shape: {trans.size} entries; expected 3.",
        )
    return {"rotation": rot, "translation": trans}


def identity_pose() -> dict:
    """Fallback pose for adapters that fail on a particular pair (e.g. no
    matches found by a feature matcher). Identity rotation, zero translation."""
    return {
        "rotation": np.eye(3, dtype=np.float64),
        "translation": np.zeros(3, dtype=np.float64),
    }
