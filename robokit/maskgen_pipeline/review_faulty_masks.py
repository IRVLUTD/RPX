#!/usr/bin/env python3
"""
review_faulty_masks.py

Interactive review tool for SAM2 Contour Masks.
Features:
  - Resizable OpenCV window, initially large (1600x1000).
  - ONLY extracts pure SAM2 colors by reading the unblended `palette` images.
  - Bottom panel with clickable buttons to HIDE/UNHIDE specific masks.
  - Hiding a mask dilates the bounds to completely erase white contour lines.
  - Left/Right arrow keys to skip/traverse frames WITHOUT saving.
  - 'Q' (Save & Next) and 'ESC' (Save & Quit) are the ONLY triggers for saving.
  - Saves your hidden states directly to `verified_masks.txt` so you can resume later.
  - Live session tracking: navigating back to a saved frame shows it as VERIFIED with the correct masks hidden.
"""

import argparse
import numpy as np
import cv2
import requests
from pathlib import Path

# --- Globals & State ---
PANEL_BG = (40, 40, 40)
BOTTOM_PANEL_H = 220
BTN_W = 120
BTN_H = 35
BTN_PAD = 10

img_w, img_h = 0, 0
masks_data = []  
OFFICIAL_PALETTE = set()
verified_dict = {}  # Global dictionary to track verified frames and their hidden masks

def load_bgr_palette():
    """Fetches the official SAM2 palette and converts to BGR for OpenCV."""
    url = "https://raw.githubusercontent.com/IRVLUTD/fewsol-toolkit/refs/heads/main/palette.txt"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        # The URL is RGB, but OpenCV needs BGR
        return set(tuple(map(int, line.strip().split()))[::-1] for line in response.text.strip().split("\n"))
    except Exception as e:
        print(f"[WARN] Failed to fetch official palette: {e}")
        return set()

def save_verified_mask(display_img, verified_dest_path, base_dir, full_id):
    """Saves the visual mask and dynamically updates the tracker file with hidden states."""
    global masks_data, verified_dict
    
    # Save the physical image without the hidden masks
    verified_dest_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(verified_dest_path), display_img)

    # Update global verified tracking state
    hidden_ids = [m['id'] for m in masks_data if m['hidden']]
    verified_dict[full_id] = hidden_ids

    # Write the entire dictionary state back to the text file
    tracker_file = Path(base_dir) / "verified_masks.txt"
    with open(tracker_file, "w") as f:
        for fid in sorted(verified_dict.keys()):
            h_ids = verified_dict[fid]
            if h_ids:
                f.write(f"{fid} | hidden:{','.join(map(str, h_ids))}\n")
            else:
                f.write(f"{fid}\n")
            
    print(f"[VERIFIED] Saved {full_id} -> masks_verified")

def mouse_callback(event, x, y, flags, param):
    """Handles clicks in the bottom panel to toggle mask visibility."""
    global masks_data, img_w, img_h

    if event == cv2.EVENT_LBUTTONDOWN:
        if y > img_h:
            cols = max(1, img_w // (BTN_W + BTN_PAD))
            row = (y - img_h - 40) // (BTN_H + BTN_PAD)
            col = (x - BTN_PAD) // (BTN_W + BTN_PAD)
            
            if row >= 0 and col >= 0 and col < cols:
                idx = row * cols + col
                bx = BTN_PAD + col * (BTN_W + BTN_PAD)
                by = img_h + 40 + row * (BTN_H + BTN_PAD)
                
                if bx <= x <= bx + BTN_W and by <= y <= by + BTN_H:
                    if 0 <= idx < len(masks_data):
                        masks_data[idx]['hidden'] = not masks_data[idx]['hidden']

def draw_bottom_panel(canvas, img_w, img_h, full_id):
    canvas[img_h:, :] = PANEL_BG
    
    header_text = f"Verified Masks - Click to Hide / Unhide:      [ {full_id} ]"
    cv2.putText(canvas, header_text, (BTN_PAD, img_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cols = max(1, img_w // (BTN_W + BTN_PAD))
    for i, m in enumerate(masks_data):
        row = i // cols
        col = i % cols
        bx = BTN_PAD + col * (BTN_W + BTN_PAD)
        by = img_h + 40 + row * (BTN_H + BTN_PAD)
        
        color_bgr = (int(m['color'][0]), int(m['color'][1]), int(m['color'][2]))
        
        # Dim the button if the mask is currently hidden
        display_color = color_bgr if not m['hidden'] else (60, 60, 60)
        cv2.rectangle(canvas, (bx, by), (bx + BTN_W, by + BTN_H), display_color, -1)
        
        # Determine text color for contrast
        brightness = sum(color_bgr) / 3
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        if m['hidden']: 
            text_color = (150, 150, 150)
        
        status_txt = f"Hidden {m['id']}" if m['hidden'] else f"Mask {m['id']}"
        cv2.putText(canvas, status_txt, (bx + 15, by + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

def review_frame(rgb_path, contour_path, palette_path, verified_dest_path, full_id, base_dir, total_pairs, current_idx):
    global masks_data, img_w, img_h, OFFICIAL_PALETTE, verified_dict

    if not OFFICIAL_PALETTE:
        OFFICIAL_PALETTE = load_bgr_palette()

    rgb_img = cv2.imread(str(rgb_path))
    orig_contour_img = cv2.imread(str(contour_path))
    orig_palette_img = cv2.imread(str(palette_path))
    
    if orig_contour_img is None or rgb_img is None or orig_palette_img is None:
        print(f"[Skip] Images missing for: {full_id}")
        return 'next'
        
    img_h, img_w = rgb_img.shape[:2]
    
    if orig_contour_img.shape != rgb_img.shape:
        orig_contour_img = cv2.resize(orig_contour_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    if orig_palette_img.shape != rgb_img.shape:
        orig_palette_img = cv2.resize(orig_palette_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # Find the PURE colors from the unblended palette image
    colors, counts = np.unique(orig_palette_img.reshape(-1, 3), axis=0, return_counts=True)
    
    is_already_verified = full_id in verified_dict
    hidden_list = verified_dict.get(full_id, [])

    masks_data = []
    idx_counter = 0
    for c in colors:
        # Strictly filter out compression artifacts and black background
        if tuple(c) in OFFICIAL_PALETTE and np.sum(c) > 0:
            masks_data.append({
                'id': idx_counter,
                'color': tuple(c),
                'hidden': (idx_counter in hidden_list)
            })
            idx_counter += 1

    status_text = "[VERIFIED]" if is_already_verified else "[UNVERIFIED]"
    status_color = (0, 255, 0) if is_already_verified else (0, 165, 255)

    window_name = "Contour Mask Review Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 1000)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Dilation Kernel used to erase thick contour lines of hidden masks
    kernel = np.ones((5, 5), np.uint8)

    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return 'quit'

        canvas = np.zeros((img_h + BOTTOM_PANEL_H, img_w, 3), dtype=np.uint8)
        
        # Build the live overlay
        display_img = orig_contour_img.copy()
        
        for m in masks_data:
            if m['hidden']:
                # Find exactly where the mask is using the pure palette image
                color_mask = np.all(orig_palette_img == m['color'], axis=-1).astype(np.uint8)
                
                # Dilate the mask boundaries slightly to catch the thick white contour line
                dilated_mask = cv2.dilate(color_mask, kernel, iterations=1)
                
                # Erase it by pasting the original RGB pixels over it
                display_img[dilated_mask == 1] = rgb_img[dilated_mask == 1]

        canvas[:img_h, :] = display_img
        draw_bottom_panel(canvas, img_w, img_h, full_id)
        
        nav_text = f"Frame: {current_idx}/{total_pairs} {status_text} | Skip: Arrows (A/D). ONLY 'Q' or 'ESC' will SAVE."
        cv2.putText(canvas, nav_text, (BTN_PAD, img_h + BOTTOM_PANEL_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKeyEx(20)
        
        if key == ord('q') or key == ord('Q'):
            save_verified_mask(display_img, verified_dest_path, base_dir, full_id)
            return 'next'
            
        elif key == 27: # ESC
            save_verified_mask(display_img, verified_dest_path, base_dir, full_id)
            cv2.destroyAllWindows()
            return 'quit'

        elif key in (65361, 2424832, 2, ord('a'), ord('A')):
            return 'prev'
            
        elif key in (65363, 2555904, 3, ord('d'), ord('D'), 32): 
            return 'next'

def find_pairs(base_dir):
    pairs = []
    base_dir = Path(base_dir)
    all_pngs = list(base_dir.rglob("*.png"))
    
    for contour_path in all_pngs:
        if "contour_gt_masks" in contour_path.parts:
            try:
                parts = contour_path.parts
                idx = parts.index("contour_gt_masks")
                part_name = parts[idx - 1]
                sam2_name = parts[idx - 2]
                
                if sam2_name != "sam2": 
                    continue
                    
                scene_name = parts[idx - 3]
                scene_dir = Path(*parts[:idx - 2])
                
                rgb_path = scene_dir / "frames" / part_name / "rgb" / contour_path.name
                palette_path = scene_dir / "sam2" / part_name / "palette" / contour_path.name
                
                if rgb_path.exists() and palette_path.exists():
                    full_id = f"{scene_name}/{part_name}/{contour_path.stem}"
                    verified_dest = scene_dir / "sam2" / part_name / "masks_verified" / contour_path.name
                    pairs.append((rgb_path, contour_path, palette_path, verified_dest, full_id))
            except Exception:
                continue
    return pairs

def main():
    global verified_dict
    parser = argparse.ArgumentParser(description="Interactive Review Tool for Masks")
    parser.add_argument("--base_dir", type=str, required=True, help="Root directory containing all scenes")
    args = parser.parse_args()

    print(f"Scanning {args.base_dir} for contour masks...")
    all_pairs = find_pairs(args.base_dir)

    if not all_pairs:
        print("No paired RGB and Contour masks found! Check your directory structure.")
        return

    total_pairs = len(all_pairs)
    tracker_file = Path(args.base_dir) / "verified_masks.txt"
    
    # Load previously verified frames and their hidden states into global dict
    verified_dict = {}
    if tracker_file.exists():
        with open(tracker_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if " | hidden:" in line:
                    fid, h_str = line.split(" | hidden:")
                    verified_dict[fid] = [int(x) for x in h_str.split(",")]
                else:
                    verified_dict[line] = []

    start_idx = 0
    for i, (_, _, _, _, full_id) in enumerate(all_pairs):
        if full_id not in verified_dict:
            start_idx = i
            break
    else:
        print("All masks have been reviewed! Starting at the last frame.")
        start_idx = total_pairs - 1

    print(f"Starting review at Frame {start_idx + 1} / {total_pairs}...")

    idx = start_idx
    while 0 <= idx < total_pairs:
        rgb_path, contour_path, palette_path, verified_dest, full_id = all_pairs[idx]
        
        action = review_frame(rgb_path, contour_path, palette_path, verified_dest, full_id, args.base_dir, total_pairs, idx + 1)
        
        if action == 'next':
            idx += 1
        elif action == 'prev':
            idx = max(0, idx - 1)
        elif action == 'quit':
            break

    print("\n[Finished] Mask verification session complete.")

if __name__ == "__main__":
    main()