#!/usr/bin/env python3
"""
review_faulty_bboxes.py

Interactive review tool for GroundingDINO BBoxes.
Features:
  - Resizable OpenCV window, initially large (1600x1000).
  - Persistent distinct colors & IDs for each bounding box.
  - Left/Right arrow keys to skip/traverse frames WITHOUT saving.
  - Can traverse BACKWARDS into previously verified frames to re-edit them.
  - 'Q' (Save & Next) and 'ESC' (Save & Quit) are the ONLY triggers for saving.
  - Bottom panel with clickable buttons to delete specific boxes.
  - Left-Click & Drag on image to draw new boxes.
  - Right-Click inside a box to delete it.
  - Absolute Frame Counter and Clean ID tracking (Scene#/Part/Frame).
  - --no_prop flag to completely hide/skip propagated frames from the queue.
"""

import argparse
import numpy as np
import cv2
import re
from pathlib import Path
from PIL import Image

# --- Globals & State ---
COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), 
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (0, 255, 128), (128, 255, 0), (255, 0, 128)
]

PANEL_BG = (40, 40, 40)
BOTTOM_PANEL_H = 180
BTN_W = 120
BTN_H = 35
BTN_PAD = 10

drawing = False
ix, iy = -1, -1
temp_bbox = None
img_w, img_h = 0, 0

# Store boxes as dicts to maintain ID and Color when others are deleted
boxes_data = []
next_id = 0

def load_npz_bboxes(npz_path):
    if not npz_path.exists():
        return []
    try:
        data = np.load(npz_path, allow_pickle=True)
        if 'bboxes' in data:
            return data['bboxes'].tolist()
        return []
    except Exception as e:
        print(f"[ERROR] Loading {npz_path}: {e}")
        return []

def get_clean_id(npz_path):
    """Extracts just the scene#, part, and frame number (ignoring extra folder text)"""
    scene_name = npz_path.parent.parent.parent.name
    part_name = npz_path.parent.name
    frame_num = npz_path.stem.split('_')[-1]
    
    # regex to match "scene21" or "scene_21" and ignore suffixes
    match = re.search(r'scene_?\d+', scene_name, re.IGNORECASE)
    clean_scene = match.group(0) if match else scene_name
    
    return f"{clean_scene}/{part_name}/{frame_num}"

def save_verified_data(original_npz_path, base_dir):
    """Saves the current bboxes into the 'bboxes_verified' directory."""
    global boxes_data
    
    bboxes = [b['bbox'] for b in boxes_data]
    full_id = get_clean_id(original_npz_path)
    
    part_name = original_npz_path.parent.name
    scene_dir = original_npz_path.parent.parent.parent
    
    verified_dir = scene_dir / "bboxes_verified" / part_name
    verified_dir.mkdir(parents=True, exist_ok=True)
    new_npz_path = verified_dir / original_npz_path.name

    try:
        original_data = dict(np.load(original_npz_path, allow_pickle=True))
    except:
        original_data = {}

    original_data['bboxes'] = np.array(bboxes, dtype=np.float32)
    n_boxes = len(bboxes)
    
    if 'confs' in original_data:
        old_confs = original_data['confs']
        new_confs = np.ones((n_boxes,), dtype=np.float32)
        min_len = min(len(old_confs), n_boxes)
        new_confs[:min_len] = old_confs[:min_len]
        original_data['confs'] = new_confs
        
    if 'phrases' in original_data:
        old_phrases = original_data['phrases']
        new_phrases = np.array(["verified"] * n_boxes, dtype=object)
        min_len = min(len(old_phrases), n_boxes)
        new_phrases[:min_len] = old_phrases[:min_len]
        original_data['phrases'] = new_phrases

    np.savez(new_npz_path, **original_data)

    tracker_file = Path(base_dir) / "verified_frames.txt"
    
    # Read existing verified frames to prevent duplicates
    verified_set = set()
    if tracker_file.exists():
        with open(tracker_file, "r") as f:
            verified_set = set(line.strip() for line in f.readlines() if line.strip())
            
    # Append only if it's new
    if original_npz_path.name not in verified_set:
        with open(tracker_file, "a") as f:
            f.write(f"{original_npz_path.name}\n")
            
    # Print success to terminal
    print(f"[VERIFIED] Saved {full_id} -> {new_npz_path.name}")


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, temp_bbox, boxes_data, img_w, img_h, next_id

    # --- BOTTOM PANEL INTERACTION ---
    if y > img_h:
        if event == cv2.EVENT_LBUTTONDOWN:
            cols = max(1, img_w // (BTN_W + BTN_PAD))
            row = (y - img_h - 40) // (BTN_H + BTN_PAD)
            col = (x - BTN_PAD) // (BTN_W + BTN_PAD)
            
            if row >= 0 and col >= 0 and col < cols:
                idx = row * cols + col
                bx = BTN_PAD + col * (BTN_W + BTN_PAD)
                by = img_h + 40 + row * (BTN_H + BTN_PAD)
                
                if bx <= x <= bx + BTN_W and by <= y <= by + BTN_H:
                    if 0 <= idx < len(boxes_data):
                        del boxes_data[idx]
        return 

    # --- IMAGE INTERACTION ---
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        temp_bbox = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_bbox = [min(ix, x), min(iy, y), max(ix, x), max(iy, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                # Add new box with unique ID and color
                boxes_data.append({
                    'id': next_id,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'color': COLORS[next_id % len(COLORS)]
                })
                next_id += 1
            temp_bbox = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i in range(len(boxes_data) - 1, -1, -1):
            bx1, by1, bx2, by2 = boxes_data[i]['bbox']
            if min(bx1, bx2) <= x <= max(bx1, bx2) and min(by1, by2) <= y <= max(by1, by2):
                del boxes_data[i]
                break

def draw_bottom_panel(canvas, img_w, img_h, full_id):
    canvas[img_h:, :] = PANEL_BG
    
    # Display the clean ID next to the header text
    header_text = f"Verified BBoxes - Click to Remove:      [ {full_id} ]"
    cv2.putText(canvas, header_text, (BTN_PAD, img_h + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cols = max(1, img_w // (BTN_W + BTN_PAD))
    for i, b in enumerate(boxes_data):
        row = i // cols
        col = i % cols
        bx = BTN_PAD + col * (BTN_W + BTN_PAD)
        by = img_h + 40 + row * (BTN_H + BTN_PAD)
        
        color = b['color']
        cv2.rectangle(canvas, (bx, by), (bx + BTN_W, by + BTN_H), color, -1)
        
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        
        cv2.putText(canvas, f"X  Box {b['id']}", (bx + 15, by + 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

def review_frame(img_path, npz_path, base_dir, total_pairs, current_idx):
    global boxes_data, next_id, img_w, img_h, temp_bbox
    
    full_id = get_clean_id(npz_path)
    part_name = npz_path.parent.name
    
    # --- Check for previously verified or propagated data to load instead ---
    scene_dir = npz_path.parent.parent.parent
    verified_npz_path = scene_dir / "bboxes_verified" / part_name / npz_path.name
    propagated_npz_path = scene_dir / "bboxes_propagated" / part_name / npz_path.name
    
    if verified_npz_path.exists():
        load_path = verified_npz_path
        status_text = "[VERIFIED]"
        status_color = (0, 255, 0) # Green
    elif propagated_npz_path.exists():
        load_path = propagated_npz_path
        status_text = "[PROPAGATED]"
        status_color = (0, 255, 255) # Yellow
    else:
        load_path = npz_path
        status_text = "[UNVERIFIED]"
        status_color = (0, 165, 255) # Orange
        
    raw_bboxes = load_npz_bboxes(load_path)
    boxes_data = []
    for i, box in enumerate(raw_bboxes):
        boxes_data.append({
            'id': i,
            'bbox': box,
            'color': COLORS[i % len(COLORS)]
        })
    next_id = len(raw_bboxes)
    temp_bbox = None

    if not img_path.exists():
        print(f"[Skip] Image missing: {img_path}")
        return 'next' 
    
    pil_img = Image.open(img_path).convert("RGB")
    cv_img_orig = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_h, img_w = cv_img_orig.shape[:2]
    
    window_name = "Bounding Box Review Tool"
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 1000)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return 'quit'

        canvas = np.zeros((img_h + BOTTOM_PANEL_H, img_w, 3), dtype=np.uint8)
        display_img = cv_img_orig.copy()
        
        for b in boxes_data:
            color = b['color']
            x1, y1, x2, y2 = map(int, b['bbox'])
            
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            
            # Smaller ID tags
            cv2.rectangle(display_img, (x1, y1 - 15), (x1 + 20, y1), color, -1)
            
            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            
            # Reduced font scale to 0.4 and thickness to 1
            cv2.putText(display_img, str(b['id']), (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            
        if temp_bbox is not None:
            tx1, ty1, tx2, ty2 = map(int, temp_bbox)
            cv2.rectangle(display_img, (tx1, ty1), (tx2, ty2), (0, 255, 0), 1)

        canvas[:img_h, :] = display_img
        
        # Pass the full_id down to draw_bottom_panel
        draw_bottom_panel(canvas, img_w, img_h, full_id)
        
        # Navigation UI Hints
        nav_text = f"Frame: {current_idx}/{total_pairs} {status_text} | Skip: Arrows (A/D). ONLY 'Q' or 'ESC' will SAVE."
        cv2.putText(canvas, nav_text, (BTN_PAD, img_h + BOTTOM_PANEL_H - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKeyEx(20)
        
        # --- EXCLUSIVE SAVE TRIGGERS ---
        if key == ord('q'):
            save_verified_data(npz_path, base_dir)
            return 'next'
            
        elif key == 27: # ESC
            save_verified_data(npz_path, base_dir)
            cv2.destroyAllWindows()
            return 'quit'

        # --- NON-SAVING NAVIGATION ---
        # Left Arrow or 'A' (Skip Backwards)
        elif key in (65361, 2424832, 2, ord('a')):
            return 'prev'
            
        # Right Arrow, 'D', or Space (Skip Forwards)
        elif key in (65363, 2555904, 3, ord('d'), 32): 
            return 'next'

def get_verified_frames(base_dir):
    tracker_file = Path(base_dir) / "verified_frames.txt"
    if not tracker_file.exists():
        return set()
    with open(tracker_file, "r") as f:
        return set(line.strip() for line in f.readlines() if line.strip())

def find_pairs(base_dir):
    pairs = []
    base_dir = Path(base_dir)
    npz_files = sorted(list(base_dir.rglob("*.npz")))
    
    for npz in npz_files:
        if "bboxes" not in str(npz.parent.parent.name) and "bboxes" not in str(npz.parent.name):
            continue
        # We NO LONGER skip verified folders here because we want to keep them in the master list
        # Ignore our verified and staging folders when building the base list
        if "bboxes_verified" in str(npz) or "bboxes_propagated" in str(npz):
            continue

        try:
            part_name = npz.parent.name
            scene_dir = npz.parent.parent.parent 
            frame_id = npz.stem.split('_')[-1] 
            
            img_path = scene_dir / "frames" / part_name / "rgb" / f"{frame_id}.png"
            if not img_path.exists():
                img_path = scene_dir / "frames" / part_name / "rgb" / f"{npz.stem}.png"
            
            if img_path.exists():
                pairs.append((img_path, npz))
        except Exception:
            continue
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Interactive Review Tool")
    parser.add_argument("--base_dir", type=str, required=True, help="Root output directory")
    parser.add_argument("--no_prop", action="store_true", help="Only show strictly unverified frames (skips propagated)")
    args = parser.parse_args()

    print(f"Scanning {args.base_dir}...")
    all_pairs = find_pairs(args.base_dir)
    verified_set = get_verified_frames(args.base_dir)
    
    # --- NO_PROP FILTER ---
    if args.no_prop:
        print("Filtering out propagated frames (--no_prop enabled)...")
        filtered_pairs = []
        for img, npz in all_pairs:
            part_name = npz.parent.name
            scene_dir = npz.parent.parent.parent
            prop_path = scene_dir / "bboxes_propagated" / part_name / npz.name
            
            # If the frame hasn't been propagated, keep it in our queue
            if not prop_path.exists():
                filtered_pairs.append((img, npz))
        all_pairs = filtered_pairs

    if not all_pairs:
        print("No frames found to review! Check your directory structure.")
        return

    total_all_frames = len(all_pairs)

    # Find the first unverified frame to start at
    start_idx = 0
    for i, (img, npz) in enumerate(all_pairs):
        if npz.name not in verified_set:
            start_idx = i
            break
    else:
        # If all are verified, start at the last frame
        print("All frames are already verified! Starting at the last frame.")
        start_idx = total_all_frames - 1

    print(f"Starting review. Resuming at frame {start_idx + 1} out of {total_all_frames} total.")
    
    idx = start_idx
    while 0 <= idx < total_all_frames:
        img_path, npz_path = all_pairs[idx]
        
        # Pass the global frame index (idx + 1)
        action = review_frame(img_path, npz_path, args.base_dir, total_all_frames, idx + 1)
        
        if action == 'next':
            idx += 1
        elif action == 'prev':
            idx = max(0, idx - 1)
        elif action == 'quit':
            break
            
    print("\n[Finished] Session complete.")

if __name__ == "__main__":
    main()