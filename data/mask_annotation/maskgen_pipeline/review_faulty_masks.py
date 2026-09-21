#!/usr/bin/env python3
"""
review_faulty_masks.py

Interactive review tool for SAM2 Contour Masks.
Features:
  - Natural File Sorting guarantees frames are ALWAYS displayed in sequential order.
  - 'T' (Toggle Contours): Instantly strips white borders to show only the pure, semi-transparent mask.
  - 'S' (Save & Next) and 'Q' (Save & Exit) are the primary triggers for saving.
  - 'X' (Unverify) removes the frame from verified status and reloads it.
  - 'P' (Publication Mode) toggles off the UI panel and borders for pure screenshots.
  - Bottom panel with clickable buttons to HIDE/UNHIDE specific masks.
  - Left/Right arrow keys to skip/traverse frames WITHOUT saving.
  - 'ESC' exits WITHOUT saving changes to the current frame.
"""

import argparse
import numpy as np
import cv2
import requests
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# --- Style & Theme ---
class Style:
    PANEL_BG = (0, 0, 0)          
    BTN_BG = (28, 28, 30)         
    ACCENT = (10, 132, 255)       
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (174, 174, 178)
    BORDER = (38, 38, 40)         
    
    BOTTOM_PANEL_H = 160
    BTN_W = 110
    BTN_H = 32
    BTN_PAD = 12
    CORNER_RADIUS = 6
    
    FONT_PATH = "/usr/share/fonts/truetype/lato/Lato-Regular.ttf"

# --- Globals & State ---
img_w, img_h = 0, 0
masks_data = []  
OFFICIAL_PALETTE = set()
verified_dict = {}  
pub_mode = False
show_contours = True  # Tracks the visual toggle state

def pil_draw_text(draw, text, pos, font_size, color, anchor="la"):
    font = None
    if Path(Style.FONT_PATH).exists():
        try: font = ImageFont.truetype(Style.FONT_PATH, font_size)
        except: pass
    if font is None:
        font = ImageFont.load_default()
    draw.text(pos, text, fill=color, font=font, anchor=anchor)

def draw_rounded_rect(draw, rect, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)

def load_bgr_palette():
    url = "https://raw.githubusercontent.com/anonymous/fewsol-toolkit/refs/heads/main/palette.txt"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return set(tuple(map(int, line.strip().split()))[::-1] for line in response.text.strip().split("\n"))
    except Exception as e:
        print(f"[WARN] Failed to fetch official palette: {e}")
        return set()

def save_verified_mask(orig_contour_img, orig_palette_img, rgb_img_bgr, verified_dest_path, base_dir, full_id):
    """Strictly saves the contour version of the image, even if visual toggle is off."""
    global masks_data, verified_dict
    
    # Rebuild the final save image by erasing hidden masks from the contour original
    save_img = orig_contour_img.copy()
    kernel = np.ones((5, 5), np.uint8)
    
    for m in masks_data:
        if m['hidden']:
            color_mask = np.all(orig_palette_img == m['color'], axis=-1).astype(np.uint8)
            dilated_mask = cv2.dilate(color_mask, kernel, iterations=1)
            save_img[dilated_mask == 1] = rgb_img_bgr[dilated_mask == 1]

    verified_dest_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(verified_dest_path), save_img)

    hidden_ids = [m['id'] for m in masks_data if m['hidden']]
    verified_dict[full_id] = hidden_ids

    tracker_file = Path(base_dir) / "verified_masks.txt"
    with open(tracker_file, "w") as f:
        for fid in sorted(verified_dict.keys()):
            h_ids = verified_dict[fid]
            if h_ids:
                f.write(f"{fid} | hidden:{','.join(map(str, h_ids))}\n")
            else:
                f.write(f"{fid}\n")
            
    print(f"[VERIFIED] Saved {full_id} -> masks_verified")

def unverify_mask(verified_dest_path, base_dir, full_id):
    global verified_dict
    
    if verified_dest_path.exists():
        verified_dest_path.unlink()
        
    if full_id in verified_dict:
        del verified_dict[full_id]
        
    tracker_file = Path(base_dir) / "verified_masks.txt"
    with open(tracker_file, "w") as f:
        for fid in sorted(verified_dict.keys()):
            h_ids = verified_dict[fid]
            if h_ids:
                f.write(f"{fid} | hidden:{','.join(map(str, h_ids))}\n")
            else:
                f.write(f"{fid}\n")
                
    print(f"[UNVERIFIED] Removed {full_id} from verified status.")

def mouse_callback(event, x, y, flags, param):
    global masks_data, img_w, img_h

    if event == cv2.EVENT_LBUTTONDOWN:
        if y > img_h:
            cols = max(1, img_w // (Style.BTN_W + Style.BTN_PAD))
            row = (y - img_h - 45) // (Style.BTN_H + Style.BTN_PAD)
            col = (x - Style.BTN_PAD) // (Style.BTN_W + Style.BTN_PAD)
            
            if row >= 0 and col >= 0 and col < cols:
                idx = row * cols + col
                bx = Style.BTN_PAD + col * (Style.BTN_W + Style.BTN_PAD)
                by = img_h + 45 + row * (Style.BTN_H + Style.BTN_PAD)
                
                if bx <= x <= bx + Style.BTN_W and by <= y <= by + Style.BTN_H:
                    if 0 <= idx < len(masks_data):
                        masks_data[idx]['hidden'] = not masks_data[idx]['hidden']

def draw_bottom_panel(draw, img_w, img_h, full_id):
    draw.rectangle([0, img_h, img_w, img_h + Style.BOTTOM_PANEL_H], fill=Style.PANEL_BG)
    draw.line([0, img_h, img_w, img_h], fill=Style.BORDER, width=1)
    
    pil_draw_text(draw, "Verified Masks (Click to Hide / Unhide)", (Style.BTN_PAD, img_h + 15), 14, Style.TEXT_PRIMARY)
    pil_draw_text(draw, f"[ {full_id} ]", (img_w - Style.BTN_PAD, img_h + 15), 12, Style.TEXT_SECONDARY, anchor="ra")

    cols = max(1, img_w // (Style.BTN_W + Style.BTN_PAD))
    for i, m in enumerate(masks_data):
        row = i // cols
        col = i % cols
        bx = Style.BTN_PAD + col * (Style.BTN_W + Style.BTN_PAD)
        by = img_h + 45 + row * (Style.BTN_H + Style.BTN_PAD)
        
        c_bgr = m['color']
        c_rgb = (c_bgr[2], c_bgr[1], c_bgr[0])
        
        if m['hidden']:
            bg_color = (40, 40, 42) 
            fg_color = (100, 100, 100) 
            text_color = Style.TEXT_SECONDARY
            txt = f"Hidden {m['id']}"
        else:
            bg_color = Style.BTN_BG
            fg_color = c_rgb
            txt = f"Mask {m['id']}"
            text_color = Style.TEXT_PRIMARY

        draw_rounded_rect(draw, [bx, by, bx + Style.BTN_W, by + Style.BTN_H], Style.CORNER_RADIUS, fill=bg_color)
        draw.ellipse([bx + 8, by + 8, bx + Style.BTN_H - 8, by + Style.BTN_H - 8], fill=fg_color)
        pil_draw_text(draw, txt, (bx + Style.BTN_H, by + 16), 13, text_color, anchor="lm")

def review_frame(rgb_path, contour_path, palette_path, verified_dest_path, full_id, base_dir, total_pairs, current_idx, window_name):
    global masks_data, img_w, img_h, OFFICIAL_PALETTE, verified_dict, pub_mode, show_contours

    if not OFFICIAL_PALETTE:
        OFFICIAL_PALETTE = load_bgr_palette()

    rgb_img_bgr = cv2.imread(str(rgb_path))
    orig_contour_img = cv2.imread(str(contour_path))
    orig_palette_img = cv2.imread(str(palette_path))
    
    if orig_contour_img is None or rgb_img_bgr is None or orig_palette_img is None:
        print(f"[Skip] Images missing for: {full_id}")
        return 'next'
        
    img_h, img_w = rgb_img_bgr.shape[:2]
    
    if orig_contour_img.shape != rgb_img_bgr.shape:
        orig_contour_img = cv2.resize(orig_contour_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    if orig_palette_img.shape != rgb_img_bgr.shape:
        orig_palette_img = cv2.resize(orig_palette_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # np.unique(..., axis=0) is a well-known slow path for row-wise
    # uniqueness on large arrays (~2s on a 1920x1080 frame, measured).
    # Encoding each BGR triple into one uint32 lets us use the fast 1-D
    # unique() instead — same result, ~70x faster.
    flat = orig_palette_img.reshape(-1, 3).astype(np.uint32)
    encoded = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    uniq_encoded, counts = np.unique(encoded, return_counts=True)
    colors = np.stack(
        [(uniq_encoded >> 16) & 255, (uniq_encoded >> 8) & 255, uniq_encoded & 255],
        axis=1,
    ).astype(np.uint8)
    
    is_already_verified = full_id in verified_dict
    hidden_list = verified_dict.get(full_id, [])

    masks_data = []
    idx_counter = 0
    for c in colors:
        if tuple(c) in OFFICIAL_PALETTE and np.sum(c) > 0:
            masks_data.append({
                'id': idx_counter,
                'color': tuple(c),
                'hidden': (idx_counter in hidden_list)
            })
            idx_counter += 1

    status_text = "VERIFIED" if is_already_verified else "UNVERIFIED"
    status_color = (52, 199, 89) if is_already_verified else (255, 59, 48)

    kernel = np.ones((5, 5), np.uint8)

    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return 'quit'

        # 1. Composite the Image using OpenCV based on Toggle Mode
        if show_contours:
            # Mode A: Standard view with dark background and white contours
            display_img_bgr = orig_contour_img.copy()
            for m in masks_data:
                if m['hidden']:
                    color_mask = np.all(orig_palette_img == m['color'], axis=-1).astype(np.uint8)
                    dilated_mask = cv2.dilate(color_mask, kernel, iterations=1)
                    display_img_bgr[dilated_mask == 1] = rgb_img_bgr[dilated_mask == 1]
        else:
            # Mode B: Clean view with raw RGB and semi-transparent colored masks
            display_img_bgr = rgb_img_bgr.copy()
            for m in masks_data:
                if not m['hidden']:
                    color_mask = np.all(orig_palette_img == m['color'], axis=-1)
                    # Blend 60% mask color over the background for visibility
                    mask_pixels = display_img_bgr[color_mask].astype(np.float64)
                    color_array = np.array(m['color'], dtype=np.float64)
                    blended = (mask_pixels * 0.4 + color_array * 0.6).astype(np.uint8)
                    display_img_bgr[color_mask] = blended

        # 2. Convert to PIL for High-Quality UI Drawing
        display_img_rgb = cv2.cvtColor(display_img_bgr, cv2.COLOR_BGR2RGB)
        pil_img_orig = Image.fromarray(display_img_rgb)
        
        final_h = img_h if pub_mode else img_h + Style.BOTTOM_PANEL_H
        canvas_pil = Image.new("RGB", (img_w, final_h), Style.PANEL_BG)
        canvas_pil.paste(pil_img_orig, (0, 0))
        draw = ImageDraw.Draw(canvas_pil)

        # 3. Draw UI
        if not pub_mode:
            draw.rectangle([0, 0, img_w-1, final_h-1], outline=status_color, width=4)
            draw_bottom_panel(draw, img_w, img_h, full_id)
            
            nav_text = f"FRAME {current_idx} / {total_pairs} • {status_text}"
            pil_draw_text(draw, nav_text, (Style.BTN_PAD, final_h - 22), 12, status_color)
            
            hint_text = "Arrows: Nav | S: Save | X: Unv | C: Toggle Contours | P: Pub | Q: Exit | ESC"
            pil_draw_text(draw, hint_text, (img_w - Style.BTN_PAD, final_h - 22), 11, Style.TEXT_SECONDARY, anchor="ra")

        # 4. Display
        canvas_cv = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, canvas_cv)
        key = cv2.waitKeyEx(20)
        
        if key == ord('s') or key == ord('S'):
            save_verified_mask(orig_contour_img, orig_palette_img, rgb_img_bgr, verified_dest_path, base_dir, full_id)
            return 'next'
        elif key == ord('q') or key == ord('Q'):
            save_verified_mask(orig_contour_img, orig_palette_img, rgb_img_bgr, verified_dest_path, base_dir, full_id)
            return 'quit'
        elif key in (ord('t'), ord('T'), ord('c'), ord('C')):
            show_contours = not show_contours
        elif key == ord('x') or key == ord('X'):
            unverify_mask(verified_dest_path, base_dir, full_id)
            return 'reload' 
        elif key == 27: # ESC
            return 'quit'
        elif key in (65361, 2424832, 2, ord('a'), ord('A')):
            return 'prev'
        elif key in (65363, 2555904, 3, 32, ord('d'), ord('D')): 
            return 'next'
        elif key == ord('p') or key == ord('P'):
            pub_mode = not pub_mode

def get_dynamic_target_idx(curr_idx, direction, all_pairs, args):
    idx = curr_idx
    global verified_dict
    
    while 0 <= idx < len(all_pairs):
        full_id = all_pairs[idx][4]
        
        if args.no_verified and (full_id in verified_dict):
            idx += direction
            continue
            
        return idx
        
    return -1

def natural_sort_key(s):
    """Sorts strings naturally by treating internal numbers as integers."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def find_pairs(base_dir):
    """Discover (rgb, contour, palette, verified_dest, full_id) tuples under base_dir.

    Two output layouts are supported (both produce the same tuples):

    Layout A — multi-scene archive (legacy)::

        <base>/<scene>/sam2/<phase>/contour_gt_masks/<frame>.png
        <base>/<scene>/frames/<phase>/rgb/<frame>.png
        <base>/<scene>/sam2/<phase>/palette/<frame>.png

    Layout B — direct ``interactive_gsam2`` output (per-phase tree)::

        <base>/<scene>/<phase>/sam2/contour_gt_masks/<frame>.png
        <base>/<scene>/<phase>/rgb/<frame>.png
        <base>/<scene>/<phase>/sam2/palette/<frame>.png

    ``--base_dir`` can be pointed at:
      * the archive root (layout A → multiple scenes),
      * a single scene dir (layout B → multiple phases),
      * a single phase dir (layout B with one phase).
    """
    pairs = []
    base_dir = Path(base_dir)
    all_pngs = list(base_dir.rglob("*.png"))

    for contour_path in all_pngs:
        # Only consider PNGs whose IMMEDIATE parent dir is contour_gt_masks.
        if contour_path.parent.name != "contour_gt_masks":
            continue
        try:
            # contour_gt_masks ←── pp ←── ppp ←── scene_dir
            pp = contour_path.parent.parent
            ppp = pp.parent
            scene_dir = ppp.parent

            if ppp.name == "sam2":
                # Layout A: <scene_dir>/sam2/<phase>/contour_gt_masks/<frame>.png
                phase = pp.name
                rgb_path = scene_dir / "frames" / phase / "rgb" / contour_path.name
                palette_path = scene_dir / "sam2" / phase / "palette" / contour_path.name
                verified_dest = scene_dir / "sam2" / phase / "masks_verified" / contour_path.name
            elif pp.name == "sam2":
                # Layout B: <scene_dir>/<phase>/sam2/contour_gt_masks/<frame>.png
                phase = ppp.name
                rgb_path = scene_dir / phase / "rgb" / contour_path.name
                palette_path = scene_dir / phase / "sam2" / "palette" / contour_path.name
                verified_dest = scene_dir / phase / "sam2" / "masks_verified" / contour_path.name
            else:
                continue

            scene_name = scene_dir.name
            if rgb_path.exists() and palette_path.exists():
                full_id = f"{scene_name}/{phase}/{contour_path.stem}"
                pairs.append((rgb_path, contour_path, palette_path, verified_dest, full_id))
        except Exception:
            continue

    pairs.sort(key=lambda x: natural_sort_key(x[4]))
    return pairs

def main():
    global verified_dict
    parser = argparse.ArgumentParser(description="Interactive Review Tool for Masks")
    # Both flags are accepted — the rest of the pipeline uses --scene_dir,
    # so we alias it here to avoid the cognitive overhead of remembering
    # which script needs which flag name.
    parser.add_argument("--base_dir", "--scene_dir", dest="base_dir",
                        type=str, required=True,
                        help="Scene or phase directory to review (either layout auto-detected)")
    parser.add_argument("--no_verified", action="store_true", help="Hide previously verified frames.")
    args = parser.parse_args()

    print(f"Scanning {args.base_dir} for contour masks...")
    all_pairs = find_pairs(args.base_dir)

    if not all_pairs:
        print("No paired RGB and Contour masks found! Check your directory structure.")
        return

    tracker_file = Path(args.base_dir) / "verified_masks.txt"
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

    total_all_frames = len(all_pairs)

    # --- DETERMINE START INDEX ---
    if args.no_verified:
        start_idx = get_dynamic_target_idx(0, 1, all_pairs, args)
        if start_idx == -1:
            print("All masks under these filters are already verified! No frames left to review.")
            return
    else:
        start_idx = 0
        for i, pair in enumerate(all_pairs):
            if pair[4] not in verified_dict:
                start_idx = i
                break
        else:
            print("All masks are already verified! Starting at the last frame.")
            start_idx = total_all_frames - 1

    print(f"Starting review. Resuming at frame {start_idx + 1} out of {total_all_frames} total.")

    window_name = "RoboKit | Mask Review"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 1000)
    cv2.setMouseCallback(window_name, mouse_callback)

    idx = start_idx
    while 0 <= idx < total_all_frames:
        rgb_path, contour_path, palette_path, verified_dest, full_id = all_pairs[idx]
        
        action = review_frame(rgb_path, contour_path, palette_path, verified_dest, full_id, args.base_dir, total_all_frames, idx + 1, window_name)
        
        if action == 'next':
            next_idx = get_dynamic_target_idx(idx + 1, 1, all_pairs, args)
            if next_idx == -1:
                break
            idx = next_idx
            
        elif action == 'prev':
            prev_idx = get_dynamic_target_idx(idx - 1, -1, all_pairs, args)
            if prev_idx != -1:
                idx = prev_idx
                
        elif action == 'reload':
            pass 
            
        elif action == 'quit':
            break

    print("\n[Finished] Mask verification session complete.")

if __name__ == "__main__":
    main()