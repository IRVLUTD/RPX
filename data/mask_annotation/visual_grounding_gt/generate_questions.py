import random
import glob
from typing import Dict, Any
import argparse, os, json, math, random, re, glob
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image

# ---------------------------- Frame Selection ----------------------------

def select_random_frames(scene_root, num_frames=15):
    """
    Select 15 random frames from the scene if no specific frames are provided.
    """
    # Get all possible frames from the masks folder
    mask_folder = os.path.join(scene_root, 'sam2', 'masks')
    fids = sorted([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(mask_folder, '*.png'))])
    
    # Randomly select up to `num_frames` frames
    selected_frames = random.sample(fids, min(num_frames, len(fids)))
    
    return selected_frames

# ---------------------------- Frame Data Loading ----------------------------

def imread_gray(path):
    im = Image.open(path)
    if im.mode not in ('L', 'I;16', 'I'):
        im = im.convert('L')
    arr = np.array(im)
    return arr

def imread_mask(path):
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int32)

def load_mapping(path) -> Dict[int, Dict[str, Any]]:
    with open(path, 'r') as f:
        data = json.load(f)
    mp = {}
    for k, v in data.items():
        try:
            key = int(k)
        except:
            continue
        if isinstance(v, dict):
            mp[key] = v
        else:
            mp[key] = {"oid": str(v), "name": str(v)}
    return mp

def compute_instances(mask) -> Dict[int, Dict[str, Any]]:
    """
    Compute the instances from a mask (extracting properties like centroid and area).
    """
    instances = {}
    ids = [i for i in np.unique(mask).tolist() if i != 0]
    for mid in ids:
        m = (mask == mid)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        area = int(m.sum())
        cx = float(xs.mean())
        cy = float(ys.mean())
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        instances[mid] = {
            "mask_id": mid,
            "area": area,
            "centroid": [cx, cy],
            "bbox": [xmin, ymin, xmax, ymax],
            "mask": m,  # keep for local use; removed before save
        }
    return instances

def load_frame(scene_root, fid):
    """
    Load all necessary data (rgb, mask, depth, mapping) for a given frame.
    """
    rgb_path = os.path.join(scene_root, 'rgb', f'{fid}.png')
    mask_path = os.path.join(scene_root, 'sam2', 'masks', f'{fid}.png')
    depth_path = os.path.join(scene_root, 'depth', f'{fid}.png')
    map_path = os.path.join(scene_root, 'sam2', 'mask_to_object.json')

    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)
    mask = imread_mask(mask_path)
    H, W = mask.shape
    depth = None
    if os.path.exists(depth_path):
        depth = imread_gray(depth_path).astype(np.float32)
        if depth.max() > 5000:
            depth = depth / 1000.0  # Normalize if needed (to meters)
    
    # Load the mask to object mapping
    mp = load_mapping(map_path)

    instances = compute_instances(mask)
    for mid, inst in list(instances.items()):
        if mid not in mp: 
            instances.pop(mid)
            continue
        meta = mp[mid]
        inst["name"] = meta.get("name", meta.get("oid", str(mid)))
        inst["oid"] = str(meta.get("oid", str(mid)))
        instances[mid] = inst  # Update the instance with its metadata

    return {
        "rgb_path": rgb_path,
        "mask_path": mask_path,
        "depth_path": depth_path if os.path.exists(depth_path) else None,
        "instances": instances,
        "depth": depth
    }

# ---------------------- Main Question Generator -----------------------------

class Generator:
    def __init__(self, scene_root, objects_dir=None):
        self.scene = scene_root
        self.objects_dir = objects_dir
        self.buckets = {"attr": 6, "2d": 6, "3d": 8, "occ": 3, "count": 5, "ref": 4, "temp": 4}
        self.margin_px = 0.02
        self.margin_depth = 0.02

def generate_for_frame(self, fids, out_dir):
    total = 0
    print(f"Starting to process {len(fids)} frames...")  # Added log for frame count
    for fid in fids:  # Loop through the randomly selected frames
        print(f"Processing frame {fid}...")  # Log each frame
        ctx = load_frame(self.scene, fid)  # Pass valid `fid` to load_frame()
        
        if not ctx["instances"]:
            print(f"No instances found for frame {fid}. Skipping.")  # Added check for empty frames
            continue
        
        insts = list(ctx["instances"].values())
        if len(insts) < 2:
            print(f"Not enough instances in frame {fid}. Skipping.")  # Additional check
            continue

        print(f"Loaded {len(insts)} instances in frame {fid}.")  # Log number of instances
        quotas = dict(self.buckets)
        answer_budget = {inst["oid"]: 0 for inst in insts}
        used_templates = set()
        packed = []

        def try_add_batch(batch, bucket_key):
            nonlocal packed, quotas, used_templates, answer_budget
            for q in batch:
                if quotas[bucket_key] <= 0: break
                if q.answer_oid is not None and answer_budget.get(q.answer_oid, 0) >= 5:
                    continue
                key = (bucket_key, q.type, tuple(q.predicates))
                if key in used_templates: 
                    continue
                packed.append(q)
                used_templates.add(key)
                if q.answer_oid is not None:
                    answer_budget[q.answer_oid] = answer_budget.get(q.answer_oid, 0) + 1
                quotas[bucket_key] -= 1
            return

        # Process the current frame and generate output
        out_path, n = self.generate_for_frame([fid], out_dir)  # Pass list of frames
        print(f"[OK] {fid}: wrote {n} Qs -> {out_path}")
        total += n
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="path to scene root")
    ap.add_argument("--out", required=True, help="output dir for JSONL")
    ap.add_argument("--objects_dir", default="", help="FewSOL all_object_folders path with questionnaire.txt files")
    args = ap.parse_args()

    # Initialize the Generator class
    gen = Generator(args.scene, objects_dir=args.objects_dir)

    # Get 15 random frames (or fewer if not available)
    fids = select_random_frames(args.scene, 15)

    # Generate questions for the selected frames
    total = gen.generate_for_frame(fids, args.out)  # Pass the list of frames (fids)
    
    print(f"Done. {total} questions total.")

if __name__ == "__main__":
    main()
