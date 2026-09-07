"""Deterministic reference-crop construction for the in-context VQA tasks'
Image 1: one canonical RGB crop per published SOS object (see sos_catalog.py
for why the pool is the 70 officially published objects, not the wider local
catalog).

Published SOS layout, verified directly against the HF repo (do not assume
extensions):

    objects/<object_id>/0/rgb.tar             -> rgb/00000.webp .. 00499.webp
    objects/<object_id>/0/labels/masks/v1.tar -> sam2/masks/00000.png .. 00499.png

RGB members are .webp; mask members are .png, single-instance binary
(0=background, 1=object) -- unlike scene masks, which are multi-instance
indexed. Every object has exactly one phase folder, "0" (checked across all
70 objects' file listings).

Only a *stride-sampled* subset of each object's 500 frames is ever decoded
(default stride=10 -> 50 candidates), read directly out of the tar via
TarFile.extractfile() without extracting to disk, to bound both download
follow-on CPU and disk use. Crops are cached per object_id -- a second call
against an unchanged input is a no-op that returns the cached manifest row.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

CROP_PADDING_FRAC = 0.12  # midpoint of the requested 10-15% range
CANDIDATE_STRIDE = 10     # every 10th frame of 500 -> 50 candidates per object
# Reference crops need a stronger floor than the general MIN_MASK_AREA_PX=100
# (that floor is for *scene* objects that may be legitimately small/distant;
# a reference crop is a deliberately chosen "best" frame and should never be
# a near-degenerate sliver). Measured (pilot/audit_reference_quality.py,
# every 5th frame x all 70 published objects, 7000 samples):
#   p1=0.00401 p5=0.00700 p10=0.00958 p25=0.01495 p50=0.02498
#   p75=0.04061 p90=0.06649 p95=0.08589 p99=0.12836  min=0.00157 max=0.17568
# Floor set at p5 (reject only the worst 5% tail -- an object's whole SOS
# sequence sometimes has it far from camera in every stride-sampled frame,
# so an aggressive floor risks NoAcceptableFrameError on real objects for no
# benefit). Ceiling set well above the observed max (0.176) -- defensive
# only, guards against a corrupt/aberrant mask filling the whole frame,
# never expected to bind at this dataset's actual scale.
MIN_REF_MASK_AREA_FRAC = 0.007
MIN_REF_MASK_AREA_PX = 500       # secondary hard floor, still 5x the general MIN_MASK_AREA_PX=100
MAX_REF_MASK_AREA_FRAC = 0.30    # defensive ceiling, ~1.7x the observed real max
MIN_EDGE_MARGIN_PX = 4           # reject masks touching the frame border (clipped object)


@dataclass(frozen=True)
class ReferenceCrop:
    object_id: str
    rgb_shard: str
    rgb_member: str
    mask_shard: str
    mask_member: str
    frame_id: str
    mask_bbox: list          # [x0,y0,x1,y1] in the SOS frame's own coords, unpadded
    crop_bbox: list          # [x0,y0,x1,y1] in the SOS frame's own coords, padded+clamped
    crop_w: int
    crop_h: int
    mask_area_px: int
    mask_area_frac: float
    edge_distance_px: int
    center_dist_norm: float
    sharpness: float
    crop_path: str
    crop_sha256: str


def _score_candidate(mask: np.ndarray, rgb: np.ndarray, W: int, H: int):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    area = int(mask.sum())
    area_frac = area / (W * H)
    edge_dist = min(x0, y0, W - 1 - x1, H - 1 - y1)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    center_dist = float(np.hypot(cx - W / 2.0, cy - H / 2.0) / np.hypot(W / 2.0, H / 2.0))
    crop = rgb[y0:y1 + 1, x0:x1 + 1]
    gray = np.asarray(Image.fromarray(crop).convert("L"), dtype=np.float64)
    # Deterministic Laplacian-variance sharpness -- standard blur proxy, no
    # randomness, cv2-free (avoids a heavier dependency for one metric).
    lap = (-4 * gray
           + np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
           + np.roll(gray, 1, 1) + np.roll(gray, -1, 1))
    sharpness = float(lap.var())
    return {
        "bbox": [x0, y0, x1, y1], "area": area, "area_frac": area_frac,
        "edge_dist": edge_dist, "center_dist": center_dist, "sharpness": sharpness,
    }


def _passes_quality_gate(score: dict) -> bool:
    if score["area"] < MIN_REF_MASK_AREA_PX:
        return False
    if score["area_frac"] < MIN_REF_MASK_AREA_FRAC or score["area_frac"] > MAX_REF_MASK_AREA_FRAC:
        return False
    if score["edge_dist"] < MIN_EDGE_MARGIN_PX:
        return False
    return True


def _rank_key(frame_id: str, score: dict):
    # Deterministic, no RNG: prefer further from the frame edge (padding
    # won't clip), then sharper, then larger, then lowest frame id as a
    # final tiebreak so ties are 100% reproducible.
    return (-score["edge_dist"], -score["sharpness"], -score["area"], frame_id)


def select_best_frame(candidates: dict[str, dict]) -> Optional[str]:
    """candidates: frame_id -> score dict (from _score_candidate). Returns
    the winning frame_id, or None if nothing passed the quality gate."""
    survivors = {fid: s for fid, s in candidates.items() if _passes_quality_gate(s)}
    if not survivors:
        return None
    return min(survivors, key=lambda fid: _rank_key(fid, survivors[fid]))


def _pad_and_clamp(bbox, W, H, pad_frac=CROP_PADDING_FRAC):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0 + 1, y1 - y0 + 1
    px, py = int(round(w * pad_frac)), int(round(h * pad_frac))
    return [max(0, x0 - px), max(0, y0 - py), min(W - 1, x1 + px), min(H - 1, y1 + py)]


def build_reference_crop(object_id: str, catalog_obj, out_dir: Path, revision: str = "main",
                          repo_id: str = "IRVLUTD/RPX", stride: int = CANDIDATE_STRIDE,
                          manifest: Optional[dict] = None) -> ReferenceCrop:
    """Downloads objects/<object_id>/0/{rgb.tar, labels/masks/v1.tar} once,
    scores a stride-sampled subset of frames, keeps only the winning crop on
    disk, deletes the tars (both the cache symlink and its backing blob --
    same pattern as pilot/run_full_dataset_gt.py's pull(), for the same
    reason: HF's local cache is content-addressed, and the symlink alone is
    tiny while the blob it points to is not). Idempotent: if `manifest`
    (object_id -> ReferenceCrop-shaped dict, e.g. loaded from
    reference_crops_v1.parquet) already has this object_id, returns that
    cached record untouched rather than re-downloading."""
    if manifest and object_id in manifest:
        return ReferenceCrop(**manifest[object_id])

    from huggingface_hub import hf_hub_download

    rgb_shard = f"{catalog_obj.sos_data_path}0/rgb.tar"
    mask_shard = f"{catalog_obj.sos_data_path}0/labels/masks/v1.tar"
    rgb_tar_path = hf_hub_download(repo_id, repo_type="dataset", filename=rgb_shard, revision=revision)
    mask_tar_path = hf_hub_download(repo_id, repo_type="dataset", filename=mask_shard, revision=revision)

    scores = {}
    frame_members = {}
    with tarfile.open(rgb_tar_path) as rgb_tf, tarfile.open(mask_tar_path) as mask_tf:
        rgb_names = sorted(n for n in rgb_tf.getnames() if n.lower().endswith((".webp", ".png", ".jpg", ".jpeg")))
        mask_names = {os.path.splitext(os.path.basename(n))[0]: n for n in mask_tf.getnames()
                      if n.lower().endswith(".png")}
        candidates = rgb_names[::stride]
        rgb_arrays = {}
        for rgb_member in candidates:
            frame_id = os.path.splitext(os.path.basename(rgb_member))[0]
            mask_member = mask_names.get(frame_id)
            if mask_member is None:
                continue  # missing counterpart -- skip this candidate, not fatal
            rgb_bytes = rgb_tf.extractfile(rgb_member).read()
            mask_bytes = mask_tf.extractfile(mask_member).read()
            try:
                rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
                mask_img = Image.open(io.BytesIO(mask_bytes))
            except Exception:
                continue  # corrupt member -- skip, not fatal
            rgb_arr = np.asarray(rgb_img)
            mask_arr = np.asarray(mask_img)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]
            if rgb_arr.shape[:2] != mask_arr.shape[:2]:
                continue  # dimension mismatch -- reject this candidate, not fatal
            H, W = mask_arr.shape[:2]
            score = _score_candidate(mask_arr > 0, rgb_arr, W, H)
            if score is None:
                continue  # empty mask -- reject
            score["frame_id"], score["rgb_member"], score["mask_member"] = frame_id, rgb_member, mask_member
            score["W"], score["H"] = W, H
            scores[frame_id] = score
            frame_members[frame_id] = (rgb_member, mask_member, rgb_arr)
            rgb_arrays[frame_id] = rgb_arr

        best = select_best_frame(scores)
        if best is None:
            _delete_hf_blob(rgb_tar_path)
            _delete_hf_blob(mask_tar_path)
            raise NoAcceptableFrameError(
                f"{object_id}: no candidate frame passed the reference-crop quality gate "
                f"({len(scores)} scored, 0 survivors)")

        s = scores[best]
        rgb_member, mask_member, rgb_arr = frame_members[best]
        crop_bbox = _pad_and_clamp(s["bbox"], s["W"], s["H"])
        x0, y0, x1, y1 = crop_bbox
        crop_arr = rgb_arr[y0:y1 + 1, x0:x1 + 1]

    _delete_hf_blob(rgb_tar_path)
    _delete_hf_blob(mask_tar_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    crop_path = out_dir / f"{object_id}.png"
    buf = io.BytesIO()
    Image.fromarray(crop_arr).save(buf, format="PNG", compress_level=6)
    crop_bytes = buf.getvalue()
    crop_path.write_bytes(crop_bytes)
    sha256 = hashlib.sha256(crop_bytes).hexdigest()

    return ReferenceCrop(
        object_id=object_id, rgb_shard=rgb_shard, rgb_member=rgb_member,
        mask_shard=mask_shard, mask_member=mask_member, frame_id=best,
        mask_bbox=s["bbox"], crop_bbox=crop_bbox,
        crop_w=crop_arr.shape[1], crop_h=crop_arr.shape[0],
        mask_area_px=s["area"], mask_area_frac=s["area_frac"],
        edge_distance_px=s["edge_dist"], center_dist_norm=s["center_dist"],
        sharpness=s["sharpness"], crop_path=str(crop_path), crop_sha256=sha256,
    )


class NoAcceptableFrameError(RuntimeError):
    pass


def _delete_hf_blob(cached_path: str):
    """cached_path is a symlink into HF's content-addressed blobs/ store --
    deleting just the symlink leaves the real (large) blob behind. Same fix
    as pilot/run_full_dataset_gt.py's pull()."""
    real = os.path.realpath(cached_path)
    if os.path.exists(cached_path):
        os.remove(cached_path)
    if os.path.exists(real) and real != cached_path:
        os.remove(real)


def load_reference_manifest(path) -> dict[str, dict]:
    """object_id -> plain dict with the SAME shapes save_reference_manifest
    wrote from (mask_bbox/crop_bbox decoded back from their JSON-string
    Parquet encoding into real lists) -- so `ReferenceCrop(**row)` always
    works on the result, whether it came fresh from build_reference_crop()
    or round-tripped through this loader."""
    path = Path(path)
    if not path.exists():
        return {}
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    out = {}
    for row in table.to_pylist():
        row = dict(row)
        row["mask_bbox"] = json.loads(row["mask_bbox"]) if isinstance(row["mask_bbox"], str) else row["mask_bbox"]
        row["crop_bbox"] = json.loads(row["crop_bbox"]) if isinstance(row["crop_bbox"], str) else row["crop_bbox"]
        out[row["object_id"]] = row
    return out


def save_reference_manifest(records: list[ReferenceCrop], path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = [asdict(r) for r in records]
    for r in rows:
        r["mask_bbox"] = json.dumps(r["mask_bbox"])
        r["crop_bbox"] = json.dumps(r["crop_bbox"])
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
