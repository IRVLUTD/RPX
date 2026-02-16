#!/usr/bin/env python3
"""
propagate_bboxes.py

Batch processes sequences of frames using Lucas-Kanade Optical Flow.
It finds manually verified anchors in `bboxes_verified` and propagates 
those boxes to adjacent unverified frames.

Crucially, it saves these guesses into a staging folder called `bboxes_propagated`
so it does not overwrite or pollute your actual verified data.
"""

import argparse
import numpy as np
import cv2
from pathlib import Path

def get_points_in_bbox(bbox, gray_img, max_corners=100):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray_img.shape[1]-1, x2), min(gray_img.shape[0]-1, y2)

    if x2 <= x1 or y2 <= y1: return None

    mask = np.zeros_like(gray_img)
    mask[y1:y2, x1:x2] = 255

    pts = cv2.goodFeaturesToTrack(gray_img, maxCorners=max_corners, qualityLevel=0.01, minDistance=5, mask=mask)

    if pts is None or len(pts) < 4:
        xs = np.linspace(x1, x2, num=5)
        ys = np.linspace(y1, y2, num=5)
        xv, yv = np.meshgrid(xs, ys)
        pts = np.vstack([xv.ravel(), yv.ravel()]).T
        pts = np.float32(pts).reshape(-1, 1, 2)

    return pts

def apply_optical_flow(img_prev, img_next, prev_bboxes):
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    new_bboxes = []
    
    for bbox in prev_bboxes:
        pts_prev = get_points_in_bbox(bbox, img_prev)
        if pts_prev is None:
            new_bboxes.append(bbox) 
            continue
            
        pts_next, status, err = cv2.calcOpticalFlowPyrLK(img_prev, img_next, pts_prev, None, **lk_params)
        valid_prev, valid_next = pts_prev[status == 1], pts_next[status == 1]
        
        if len(valid_prev) < 2:
            new_bboxes.append(bbox)
            continue
            
        dx, dy = np.median(valid_next[:, 0] - valid_prev[:, 0]), np.median(valid_next[:, 1] - valid_prev[:, 1])
        x1, y1, x2, y2 = bbox
        
        new_bboxes.append([
            max(0, x1 + dx), max(0, y1 + dy),
            min(img_next.shape[1]-1, x2 + dx), min(img_next.shape[0]-1, y2 + dy)
        ])
    return new_bboxes

def load_npz_bboxes(npz_path):
    try:
        data = np.load(npz_path, allow_pickle=True)
        return data['bboxes'].tolist() if 'bboxes' in data else []
    except:
        return []

def save_propagated_bboxes(bboxes, orig_npz_path, frame_id, part_name, scene_dir):
    """SAVES TO bboxes_propagated STAGING FOLDER"""
    prop_dir = scene_dir / "bboxes_propagated" / part_name
    prop_dir.mkdir(parents=True, exist_ok=True)
    
    new_npz_path = prop_dir / (orig_npz_path.name if orig_npz_path else f"{frame_id}.npz")
    
    try:
        data = dict(np.load(orig_npz_path, allow_pickle=True)) if orig_npz_path and orig_npz_path.exists() else {}
    except:
        data = {}

    n_boxes = len(bboxes)
    data['bboxes'] = np.array(bboxes, dtype=np.float32)
    if 'confs' in data: data['confs'] = np.ones((n_boxes,), dtype=np.float32)
    if 'phrases' in data: data['phrases'] = np.array(["optical_flow_prop"] * n_boxes, dtype=object)

    np.savez(new_npz_path, **data)

def main():
    parser = argparse.ArgumentParser(description="Propagate BBoxes using Optical Flow")
    parser.add_argument("--base_dir", type=str, required=True, help="Root output directory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    print(f"Scanning {base_dir} for frames and verified anchors...")

    parts = {}
    for npz_path in base_dir.rglob("*.npz"):
        if "bboxes_propagated" in str(npz_path): continue # Ignore existing propagated files
        
        scene_dir, part_name = npz_path.parent.parent.parent, npz_path.parent.name
        frame_id = npz_path.stem.split('_')[-1]
        part_key = f"{scene_dir.name}/{part_name}"
        
        if part_key not in parts:
            parts[part_key] = {'frames': set(), 'verified': {}, 'original': {}, 'rgb': {}, 'scene_dir': scene_dir}
            
        parts[part_key]['frames'].add(frame_id)
        
        if "bboxes_verified" in str(npz_path): parts[part_key]['verified'][frame_id] = npz_path
        elif "bboxes" in str(npz_path): parts[part_key]['original'][frame_id] = npz_path
            
        img_path = scene_dir / "frames" / part_name / "rgb" / f"{frame_id}.png"
        if not img_path.exists(): img_path = scene_dir / "frames" / part_name / "rgb" / f"{npz_path.stem}.png"
        if img_path.exists(): parts[part_key]['rgb'][frame_id] = img_path

    for part_key, data in parts.items():
        frame_ids, verified_ids = sorted(list(data['frames'])), sorted(list(data['verified'].keys()))
        if not verified_ids or len(verified_ids) == len(frame_ids): continue
            
        scene_dir, part_name = data['scene_dir'], part_key.split('/')[-1]
        print(f"\n--- Processing {part_key} ({len(verified_ids)} verified anchor(s) found) ---")

        for k, v_id in enumerate(verified_ids):
            idx = frame_ids.index(v_id)
            # Calculate standard halfway boundaries
            halfway_bw = 0 if k == 0 else (frame_ids.index(verified_ids[k-1]) + idx) // 2 + 1
            halfway_fw = len(frame_ids) - 1 if k == len(verified_ids) - 1 else (idx + frame_ids.index(verified_ids[k+1])) // 2

            # Clamp the propagation to a MAXIMUM of 5 frames in either direction
            MAX_PROPAGATION = 5
            limit_bw = max(halfway_bw, idx - MAX_PROPAGATION)
            limit_fw = min(halfway_fw, idx + MAX_PROPAGATION)

            # Propagate Forward
            if limit_fw > idx:
                curr_bboxes = load_npz_bboxes(data['verified'][v_id])
                prev_img = cv2.cvtColor(cv2.imread(str(data['rgb'][v_id])), cv2.COLOR_BGR2GRAY)
                for i in range(idx + 1, limit_fw + 1):
                    f_id = frame_ids[i]
                    if f_id in data['verified'] or f_id not in data['rgb']: break
                    curr_img = cv2.cvtColor(cv2.imread(str(data['rgb'][f_id])), cv2.COLOR_BGR2GRAY)
                    curr_bboxes = apply_optical_flow(prev_img, curr_img, curr_bboxes)
                    save_propagated_bboxes(curr_bboxes, data['original'].get(f_id), f_id, part_name, scene_dir)
                    print(f"  [Forward]  Propagated -> {f_id}")
                    prev_img = curr_img

            # Propagate Backward
            if limit_bw < idx:
                curr_bboxes = load_npz_bboxes(data['verified'][v_id])
                next_img = cv2.cvtColor(cv2.imread(str(data['rgb'][v_id])), cv2.COLOR_BGR2GRAY)
                for i in range(idx - 1, limit_bw - 1, -1):
                    f_id = frame_ids[i]
                    if f_id in data['verified'] or f_id not in data['rgb']: break
                    curr_img = cv2.cvtColor(cv2.imread(str(data['rgb'][f_id])), cv2.COLOR_BGR2GRAY)
                    curr_bboxes = apply_optical_flow(next_img, curr_img, curr_bboxes)
                    save_propagated_bboxes(curr_bboxes, data['original'].get(f_id), f_id, part_name, scene_dir)
                    print(f"  [Backward] Propagated -> {f_id}")
                    next_img = curr_img

    print("\n[Finished] Optical Flow propagation complete! Saved to `bboxes_propagated` folders.")

if __name__ == "__main__":
    main()