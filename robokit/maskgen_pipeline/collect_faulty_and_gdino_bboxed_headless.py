#!/usr/bin/env python3
"""
collect_faulty_and_gdino_bboxed_headless.py

Collect the most-recent faulty frames from ALL scenes under a base directory,
write a unified manifest, and then run GroundingDINO (GDINO) on ALL collected
faulty frames.

UPDATES:
  - CRITICAL FIX: Converts cxcywh -> xyxy before NMS (Fixes "no merging" bug).
  - Debugging: Prints counts before/after NMS to verify merges are happening.
  - Tqdm Write: Ensures logs aren't swallowed by progress bars.

Python: 3.9+
"""

import argparse
import shutil
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torchvision.ops.boxes as box_ops
import torchvision.ops as ops  # Needed for box_convert

from robokit.perception import GroundingDINOObjectPredictor
from robokit.utils import annotate
from .config import logger
from .bbox_utils import (
    apply_nms,
    sort_boxes_by_area,
)


def z5(n: int) -> str:
    return f"{n:05d}"


def extract_scene_id(scene_name: str) -> Optional[int]:
    s = scene_name.strip()
    m = re.search(r"scene[_\-]?(\d+)", s, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    if s.isdigit():
        return int(s)
    m2 = re.search(r"(\d+)", s)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    return None


def normalize_obj_name(name: str) -> str:
    s = str(name).strip()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def load_scene_object_map(scene_objects_json: Optional[Path]) -> Dict[int, List[str]]:
    if scene_objects_json is None:
        return {}
    if not scene_objects_json.exists():
        raise FileNotFoundError(f"--scene_objects_json not found: {scene_objects_json}")

    data = json.loads(scene_objects_json.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {scene_objects_json}, got: {type(data)}")

    out: Dict[int, List[str]] = {}
    for entry in data:
        if not isinstance(entry, dict) or "scene_id" not in entry:
            continue
        try:
            sid_int = int(entry["scene_id"])
        except Exception:
            continue
        objs = entry.get("objects", [])
        names: List[str] = []
        if isinstance(objs, list):
            for o in objs:
                if isinstance(o, dict) and "name" in o:
                    names.append(str(o["name"]))
        out[sid_int] = names
    return out


def build_prompt_from_names(names: List[str], sep: str = ". ") -> Tuple[str, List[str]]:
    norm: List[str] = []
    seen = set()
    for n in names:
        nn = normalize_obj_name(n)
        if nn and nn not in seen:
            seen.add(nn)
            norm.append(nn)
    if not norm:
        return "", []
    prompt = sep.join(norm).strip()
    if not prompt.endswith("."):
        prompt += "."
    return prompt, norm


def find_highest_iter_faulty(sam2_dir: Path) -> Tuple[int, Optional[Path]]:
    if not sam2_dir.exists():
        return -1, None
    candidates: List[Tuple[int, Path]] = []
    for p in sam2_dir.glob("iter*_faulty.txt"):
        m = re.match(r"iter(\d+)_faulty\.txt$", p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return -1, None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][0], candidates[0][1]


def read_frame_list(txt_path: Path) -> List[int]:
    frames: List[int] = []
    if not txt_path.exists():
        return frames
    for line in txt_path.read_text().splitlines():
        s = line.strip()
        if s:
            try:
                frames.append(int(s))
            except ValueError:
                logger.warning(f"Bad frame index '{s}' in {txt_path}")
    return sorted(set(frames))


def group_contiguous(nums: List[int]) -> List[List[int]]:
    if not nums:
        return []
    runs: List[List[int]] = []
    cur: List[int] = [nums[0]]
    for a, b in zip(nums, nums[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    return runs


def collect_faulty_frames(
    base_dir: Path,
    out_root: Path,
    scene_id_to_names: Dict[int, List[str]],
    default_prompt: str,
    prompt_sep: str,
    objects_only_prompt: bool,
) -> Dict[str, Dict]:
    out_frames = out_root / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)

    manifest_lines: List[str] = []
    scene_meta: Dict[str, Dict] = {}

    total_collected = 0
    scene_counts: Dict[str, int] = {}
    scene_part_counts: Dict[str, int] = {}
    missing_pngs_total = 0

    for scene_dir in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        scene_name = scene_dir.name
        sid = extract_scene_id(scene_name)

        if objects_only_prompt:
            scene_prompt = "objects"
            obj_names_norm: List[str] = []
        else:
            obj_names_raw = scene_id_to_names.get(sid, []) if sid is not None else []
            scene_prompt, obj_names_norm = build_prompt_from_names(obj_names_raw, sep=prompt_sep)
            if not scene_prompt:
                scene_prompt = "" 
                obj_names_norm = []

        part_dirs = [p for p in scene_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        part_dirs.sort(key=lambda x: int(x.name))

        for part_dir in part_dirs:
            part_name = part_dir.name
            sam2_dir = part_dir / "sam2"
            iter_num, faulty_txt = find_highest_iter_faulty(sam2_dir)
            if iter_num < 0 or faulty_txt is None:
                continue

            faulty = read_frame_list(faulty_txt)
            if not faulty:
                continue

            rgb_dir = part_dir / "rgb"
            if not rgb_dir.exists():
                logger.warning(f"Missing rgb dir: {rgb_dir}")
                continue

            # copy_dst = out_frames / scene_name / part_name / "rgb"
            # copy_dst.mkdir(parents=True, exist_ok=True)
            copy_dst = out_root / scene_name / "frames" / part_name / "rgb"
            copy_dst.mkdir(parents=True, exist_ok=True)


            copied: List[int] = []
            missing = 0

            for f in faulty:
                src = rgb_dir / f"{z5(f)}.png"
                if not src.exists():
                    missing += 1
                    logger.warning(f"Missing RGB: {src}")
                    continue

                dst = copy_dst / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

                manifest_lines.append(f"{scene_name}_{part_name}_{z5(f)}")
                copied.append(f)

            missing_pngs_total += int(missing)
            if not copied:
                continue

            copied = sorted(set(copied))
            n = len(copied)
            total_collected += n
            scene_counts[scene_name] = scene_counts.get(scene_name, 0) + n
            sp_key = f"{scene_name}_{part_name}"
            scene_part_counts[sp_key] = scene_part_counts.get(sp_key, 0) + n

            key = f"{scene_name}_{part_name}"
            scene_meta[key] = {
                "scene": scene_name,
                "part": part_name,
                "iter": int(iter_num),
                "faulty": copied,
                "num_faulty": int(n),
                "missing_faulty_pngs": int(missing),
                "sequences": group_contiguous(copied),
                "scene_id_inferred": sid,
                "scene_prompt": scene_prompt,
                "scene_object_names": obj_names_norm,
                "objects_only_prompt": bool(objects_only_prompt),
            }

    (out_root / "latest_faulty_all.txt").write_text(
        "\n".join(manifest_lines) + ("\n" if manifest_lines else "")
    )
    (out_root / "sequences.json").write_text(json.dumps(scene_meta, indent=2))

    counts = {
        "total_collected_faulty_frames": int(total_collected),
        "num_scenes_with_faulty": int(len(scene_counts)),
        "num_scene_parts_with_faulty": int(len(scene_part_counts)),
        "missing_faulty_pngs_total": int(missing_pngs_total),
        "per_scene_counts": dict(sorted(scene_counts.items(), key=lambda kv: kv[0])),
        "per_scene_part_counts": dict(sorted(scene_part_counts.items(), key=lambda kv: kv[0])),
        "objects_only_prompt": bool(objects_only_prompt),
    }
    (out_root / "counts.json").write_text(json.dumps(counts, indent=2))
    return scene_meta


def build_flat_jobs(scene_meta: Dict[str, Dict]) -> List[Tuple[str, str, int]]:
    jobs: List[Tuple[str, str, int]] = []
    for key in sorted(scene_meta.keys()):
        meta = scene_meta[key]
        scene = meta["scene"]
        part = meta["part"]
        for f in meta["faulty"]:
            jobs.append((scene, part, int(f)))
    return jobs


def init_gdino(device: str) -> GroundingDINOObjectPredictor:
    try:
        return GroundingDINOObjectPredictor(device=device)
    except TypeError:
        return GroundingDINOObjectPredictor()


@torch.no_grad()
def run_gdino_on_all_faulty(
    out_root: Path,
    scene_meta: Dict[str, Dict],
    resume: bool,
    force: bool,
    shuffle: bool,
    seed: Optional[int],
    default_prompt: str,
    iou_thresh: float,
    min_conf: float,
    save_overlays: bool,
    device: str,
    objects_only_prompt: bool,
):
    # bboxes_dir = out_root / "bboxes"
    # overlays_dir = out_root / "overlays"
    # bboxes_dir.mkdir(parents=True, exist_ok=True)
    # overlays_dir.mkdir(parents=True, exist_ok=True)

    # copied_root = out_root / "frames"

    jobs = build_flat_jobs(scene_meta)
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(jobs)

    total = len(jobs)
    if total == 0:
        logger.info("No faulty frames to process.")
        return

    gdino = init_gdino(device=device)
    torch.set_grad_enabled(False)

    scene_prompt_map: Dict[str, str] = {}
    scene_objnames_map: Dict[str, List[str]] = {}
    scene_sid_map: Dict[str, Optional[int]] = {}
    
    for _, meta in scene_meta.items():
        s = meta.get("scene")
        if s and s not in scene_prompt_map:
            scene_prompt_map[s] = meta.get("scene_prompt", "")
            scene_objnames_map[s] = meta.get("scene_object_names", [])
            scene_sid_map[s] = meta.get("scene_id_inferred", None)

    logger.info(f"Starting inference. Union+NMS Mode. IoU Thresh: {iou_thresh}")

    with tqdm(total=total, desc="GDINO Union") as pbar:
        for idx, (scene_name, part_name, f) in enumerate(jobs, start=1):
            frame_id = z5(f)
            key = f"{scene_name}_{part_name}_{frame_id}"
            
            scene_dir = out_root / scene_name
            
            # --- Add part_name to create the 0, 1, 2 subfolders ---
            bboxes_dir = scene_dir / "bboxes" / part_name
            bboxes_dir.mkdir(parents=True, exist_ok=True)
            
            if save_overlays:
                overlays_dir = scene_dir / "overlays" / part_name
                overlays_dir.mkdir(parents=True, exist_ok=True)
                overlay_path = overlays_dir / f"{key}.png"
            else:
                overlay_path = None
            
            # Set the new destination path for the .npz file
            npz_path = bboxes_dir / f"{key}.npz"

            # Read the image from the matching frames/part folder
            img_path = scene_dir / "frames" / part_name / "rgb" / f"{frame_id}.png"
            if not img_path.exists():
                logger.warning(f"[missing] {idx}/{total} {key} frame {frame_id} @ {img_path}")
                pbar.update(1)
                continue

            if npz_path.exists() and not force:
                pbar.update(1)
                continue

            # Determine Specific Prompt
            specific_prompt = scene_prompt_map.get(scene_name, "")
            obj_names_norm = scene_objnames_map.get(scene_name, [])
            sid = scene_sid_map.get(scene_name, None)

            run_specific = True
            if objects_only_prompt:
                run_specific = False
            elif not specific_prompt:
                run_specific = False 

            try:
                img_pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to open image {img_path}: {e}")
                pbar.update(1)
                continue

            w, h = img_pil.size

            # Lists to collect results from BOTH runs (TENSORS)
            all_boxes = []
            all_phrases = []
            all_confs = []

            # === RUN 1: Generic "objects" ===
            try:
                res_gen = gdino.predict(img_pil, "objects")
                bg, pg, cg = res_gen
                
                bg = torch.as_tensor(bg).to(device)
                cg = torch.as_tensor(cg).to(device)
                
                mask_g = cg >= min_conf
                all_boxes.append(bg[mask_g])
                all_confs.append(cg[mask_g])
                pg = [p for p, ok in zip(pg, mask_g.tolist()) if ok]
                all_phrases.extend(pg)
                count_gen = len(pg)
            except Exception as e:
                logger.warning(f"GDINO 'objects' failed on {key}: {e}")
                count_gen = 0

            # === RUN 2: Specific Prompt ===
            count_spec = 0
            if run_specific:
                try:
                    res_spec = gdino.predict(img_pil, specific_prompt)
                    bs, ps, cs = res_spec
                    
                    bs = torch.as_tensor(bs).to(device)
                    cs = torch.as_tensor(cs).to(device)
                    
                    mask_s = cs >= min_conf
                    all_boxes.append(bs[mask_s])
                    all_confs.append(cs[mask_s])
                    ps = [p for p, ok in zip(ps, mask_s.tolist()) if ok]
                    all_phrases.extend(ps)
                    count_spec = len(ps)
                except Exception as e:
                    logger.warning(f"GDINO specific prompt failed on {key}: {e}")

            # === MERGE / UNION + NMS ===
            if len(all_boxes) > 0:
                # Concatenate raw boxes (which are cx, cy, w, h normalized)
                concat_boxes_cxcywh = torch.cat(all_boxes, dim=0) 
                concat_confs = torch.cat(all_confs, dim=0)
                
                total_boxes = concat_boxes_cxcywh.size(0)

                if total_boxes > 0:
                    # CRITICAL FIX: Convert cxcywh -> xyxy before NMS
                    # GroundingDINO predicts cx, cy, w, h. NMS needs x1, y1, x2, y2.
                    concat_boxes_xyxy = ops.box_convert(concat_boxes_cxcywh, in_fmt='cxcywh', out_fmt='xyxy')

                    # --- DEBUG SECTION ---
                    tqdm.write(f"\n[DEBUG] {key} | Generic: {count_gen} | Specific: {count_spec} | Total Raw: {total_boxes}")
                    
                    # Perform NMS on xyxy boxes
                    keep_idx = box_ops.nms(concat_boxes_xyxy, concat_confs, iou_threshold=iou_thresh)
                    
                    num_kept = keep_idx.size(0)
                    tqdm.write(f"      -> NMS Result: Kept {num_kept} (Dropped {total_boxes - num_kept})")

                    # We want to save the original cxcywh format for downstream tools (gdino.bbox_to_scaled_xyxy)
                    # So we use the indices to slice the ORIGINAL cxcywh tensor
                    boxes_final = concat_boxes_cxcywh[keep_idx]
                    confs_final = concat_confs[keep_idx]
                    keep_idx_list = keep_idx.tolist()
                    phrases_final = [all_phrases[i] for i in keep_idx_list]
                else:
                    boxes_final = torch.empty((0, 4), device=device)
                    confs_final = torch.empty((0,), device=device)
                    phrases_final = []
            else:
                boxes_final = torch.empty((0, 4), device=device)
                confs_final = torch.empty((0,), device=device)
                phrases_final = []

            # Convert to pixel space (Requires Tensor input)
            if boxes_final.size(0) > 0:
                # Note: bbox_to_scaled_xyxy expects CXCYWH normalized input
                image_bboxes_t = gdino.bbox_to_scaled_xyxy(boxes_final.cpu(), w, h)
                
                image_bboxes_np = image_bboxes_t.numpy()
                confs_np = confs_final.cpu().numpy()
                
                image_bboxes, phrases_final, confs_final = sort_boxes_by_area(image_bboxes_np, phrases_final, confs_np)
                
                bboxes_arr = np.array(image_bboxes, dtype=np.float32)
                phrases_arr = np.array(phrases_final, dtype=object)
                confs_arr = np.array(confs_final, dtype=np.float32)
            else:
                bboxes_arr = np.zeros((0, 4), dtype=np.float32)
                phrases_arr = np.array([], dtype=object)
                confs_arr = np.zeros((0,), dtype=np.float32)

            final_prompt_used = "UNION(objects + specific)" if run_specific else "objects"

            np.savez(
                npz_path,
                bboxes=bboxes_arr,
                phrases=phrases_arr,
                confs=confs_arr,
                scene=scene_name,
                part=part_name,
                frame=int(f),
                w=int(w),
                h=int(h),
                prompt=final_prompt_used,
                scene_id=sid if sid is not None else -1,
                scene_object_names=np.array(obj_names_norm, dtype=object),
                iou_thresh=float(iou_thresh),
                min_conf=float(min_conf),
                device=device,
                objects_only_prompt=bool(objects_only_prompt),
                merge_mode="UNION_NMS"
            )

            if save_overlays:
                try:
                    overlay = annotate(img_pil, bboxes_arr, confs_arr, phrases_arr.tolist())
                    overlay.save(overlay_path)
                except Exception as e:
                    logger.warning(f"Failed to save overlay for {key}: {e}")

            pbar.update(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--out_name", type=str, default="faulty_collection_gdino")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--text_prompt", type=str, default="objects",
                    help="Fallback generic prompt if no specific prompt is found.")
    ap.add_argument("--scene_objects_json", type=str, default=None,
                    help="FewSOL scenes JSON mapping scene_id -> objects list.")
    ap.add_argument("--prompt_sep", type=str, default=". ",
                    help="Separator for object names in prompt.")

    ap.add_argument("--objects_only_prompt", action="store_true",
                    help="Disable dual-prompt merge and ONLY run 'objects'.")

    ap.add_argument("--iou_thresh", type=float, default=0.8,
                    help="IoU threshold for NMS merging. <0.5 merges loosely, >0.5 merges strictly.")
    ap.add_argument("--min_conf", type=float, default=0.25,
                    help="Minimum confidence to keep a detection. Default: 0.25")
    ap.add_argument("--save_overlays", action="store_true")
    ap.add_argument("--device", type=str, default=None)

    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_root = base_dir / args.out_name
    out_root.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    scene_id_to_names: Dict[int, List[str]] = {}
    if (not args.objects_only_prompt) and args.scene_objects_json:
        scene_id_to_names = load_scene_object_map(Path(args.scene_objects_json))

    scene_meta = collect_faulty_frames(
        base_dir=base_dir,
        out_root=out_root,
        scene_id_to_names=scene_id_to_names,
        default_prompt=args.text_prompt,
        prompt_sep=args.prompt_sep,
        objects_only_prompt=bool(args.objects_only_prompt),
    )
    if not scene_meta:
        print("[INFO] No scenes/parts with faulty frames found.")
        return

    run_gdino_on_all_faulty(
        out_root=out_root,
        scene_meta=scene_meta,
        resume=args.resume,
        force=args.force,
        shuffle=args.shuffle,
        seed=args.seed,
        default_prompt=args.text_prompt,
        iou_thresh=args.iou_thresh,
        min_conf=args.min_conf,
        save_overlays=args.save_overlays,
        device=device,
        objects_only_prompt=bool(args.objects_only_prompt),
    )

    print("\n[DONE]")
    print(f"- Manifest:  {out_root/'latest_faulty_all.txt'}")
    print(f"- Sequences: {out_root/'sequences.json'}")
    print(f"- Counts:    {out_root/'counts.json'}")
    # print(f"- Bboxes:    {out_root/'bboxes'}")
    # print(f"- Frames:    {out_root/'frames'}")
    # if args.save_overlays:
    #     print(f"- Overlays:  {out_root/'overlays'}")
    print(f"- Data (Frames, Bboxes, Overlays) is now organized inside individual scene folders.")


if __name__ == "__main__":
    main()