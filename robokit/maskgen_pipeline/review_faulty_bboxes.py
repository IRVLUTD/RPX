#!/usr/bin/env python3
"""
review_faulty_bboxes.py

Interactive review tool for GroundingDINO BBoxes.
Features:
  - Resizable OpenCV window, initially large (1600x1000).
  - Persistent distinct colors & IDs for each bounding box.
  - Left/Right arrow keys to skip/traverse frames WITHOUT saving.
  - Can traverse BACKWARDS into previously verified frames to re-edit them.
  - 'S' (Save & Next) and 'Q' (Save & Exit) are the primary triggers for saving.
  - 'ESC' exits WITHOUT saving changes to the current frame.
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
from PIL import Image, ImageDraw, ImageFont

# --- Style & Theme ---
class Style:
    # Industry-standard Dark Theme Palette
    PANEL_BG = (0, 0, 0)          # Pure black
    BTN_BG = (28, 28, 30)         # Dark gray buttons
    ACCENT = (10, 132, 255)       # iOS-style blue
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (174, 174, 178)
    BORDER = (38, 38, 40)         # Subtle border
    
    # Publication Mode Style
    PUB_LINE_WIDTH = 3
    PUB_FONT_SIZE = 18
    
    # Dimensions
    BOTTOM_PANEL_H = 160
    BTN_W = 110
    BTN_H = 32
    BTN_PAD = 12
    CORNER_RADIUS = 6
    
    # Fonts
    FONT_PATH = "/usr/share/fonts/truetype/lato/Lato-Regular.ttf"

# --- Globals & State ---
COLORS = [
    (255, 59, 48),    # Red
    (52, 199, 89),    # Green
    (0, 122, 255),    # Blue
    (255, 159, 10),   # Orange
    (175, 82, 222),   # Purple
    (255, 214, 10),   # Yellow
    (90, 200, 250),   # Sky Blue
    (255, 45, 85),    # Pink
    (162, 132, 94),   # Brown
    (88, 86, 214),    # Indigo
]

drawing = False
ix, iy = -1, -1
temp_bbox = None
img_w, img_h = 0, 0
pub_mode = False

# Store boxes as dicts to maintain ID and Color when others are deleted
boxes_data = []
next_id = 0

def pil_draw_text(draw, text, pos, font_size, color, anchor="la"):
    font = None
    if Path(Style.FONT_PATH).exists():
        try:
            font = ImageFont.truetype(Style.FONT_PATH, font_size)
        except:
            pass
    if font is None:
        font = ImageFont.load_default()
    draw.text(pos, text, fill=color, font=font, anchor=anchor)

def draw_rounded_rect(draw, rect, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)

def load_npz_bboxes(npz_path):
    if not npz_path.exists():
        return {"bboxes": [], "confs": [], "phrases": []}
    try:
        data = np.load(npz_path, allow_pickle=True)
        out = {
            "bboxes": data['bboxes'].tolist() if 'bboxes' in data else [],
            "confs": data['confs'].tolist() if 'confs' in data else [],
            "phrases": data['phrases'].tolist() if 'phrases' in data else []
        }
        return out
    except Exception as e:
        print(f"[ERROR] Loading {npz_path}: {e}")
        return {"bboxes": [], "confs": [], "phrases": []}

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
    confs = [b.get('conf', 1.0) for b in boxes_data]
    phrases = [b.get('phrase', "verified") for b in boxes_data]
    
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
    original_data['confs'] = np.array(confs, dtype=np.float32)
    original_data['phrases'] = np.array(phrases, dtype=object)

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
            cols = max(1, img_w // (Style.BTN_W + Style.BTN_PAD))
            row = (y - img_h - 45) // (Style.BTN_H + Style.BTN_PAD)
            col = (x - Style.BTN_PAD) // (Style.BTN_W + Style.BTN_PAD)
            
            if row >= 0 and col >= 0 and col < cols:
                idx = row * cols + col
                bx = Style.BTN_PAD + col * (Style.BTN_W + Style.BTN_PAD)
                by = img_h + 45 + row * (Style.BTN_H + Style.BTN_PAD)
                
                if bx <= x <= bx + Style.BTN_W and by <= y <= by + Style.BTN_H:
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
                    'color': COLORS[next_id % len(COLORS)],
                    'conf': 1.0,
                    'phrase': "verified"
                })
                next_id += 1
            temp_bbox = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i in range(len(boxes_data) - 1, -1, -1):
            bx1, by1, bx2, by2 = boxes_data[i]['bbox']
            if min(bx1, bx2) <= x <= max(bx1, bx2) and min(by1, by2) <= y <= max(by1, by2):
                del boxes_data[i]
                break

def draw_bottom_panel(draw, img_w, img_h, full_id):
    # Background
    draw.rectangle([0, img_h, img_w, img_h + Style.BOTTOM_PANEL_H], fill=Style.PANEL_BG)
    draw.line([0, img_h, img_w, img_h], fill=Style.BORDER, width=1)
    
    # Header
    header_text = f"Verified BBoxes (Click to Remove)"
    pil_draw_text(draw, header_text, (Style.BTN_PAD, img_h + 15), 14, Style.TEXT_PRIMARY)
    
    # ID on the right
    pil_draw_text(draw, f"[ {full_id} ]", (img_w - Style.BTN_PAD, img_h + 15), 12, Style.TEXT_SECONDARY, anchor="ra")

    cols = max(1, img_w // (Style.BTN_W + Style.BTN_PAD))
    for i, b in enumerate(boxes_data):
        row = i // cols
        col = i % cols
        bx = Style.BTN_PAD + col * (Style.BTN_W + Style.BTN_PAD)
        by = img_h + 45 + row * (Style.BTN_H + Style.BTN_PAD)
        
        # Swapping to PIL colors (RGB to BGR conversion is handled by canvas conversion later)
        color = b['color']
        # Actually our COLORS are RGB-like in intent, but OpenCV uses BGR. 
        # Since we use PIL to draw on canvas which is converted to BGR, let's keep it consistent.
        
        # Draw button
        draw_rounded_rect(draw, [bx, by, bx + Style.BTN_W, by + Style.BTN_H], Style.CORNER_RADIUS, fill=Style.BTN_BG)
        
        # Color indicator circle
        draw.ellipse([bx + 8, by + 8, bx + Style.BTN_H - 8, by + Style.BTN_H - 8], fill=color)
        
        # Text
        pil_draw_text(draw, f"Box {b['id']}", (bx + Style.BTN_H, by + 16), 13, Style.TEXT_PRIMARY, anchor="lm")

def review_frame(img_path, npz_path, base_dir, total_pairs, current_idx, window_name, index_frame_interval=0, local_idx=0):
    global boxes_data, next_id, img_w, img_h, temp_bbox, pub_mode
    
    full_id = get_clean_id(npz_path)
    part_name = npz_path.parent.name
    
    scene_dir = npz_path.parent.parent.parent
    verified_npz_path = scene_dir / "bboxes_verified" / part_name / npz_path.name
    propagated_npz_path = scene_dir / "bboxes_propagated" / part_name / npz_path.name
    
    if verified_npz_path.exists():
        load_path = verified_npz_path
        status_text = "VERIFIED"
        status_color = (52, 199, 89) # Green
    elif propagated_npz_path.exists():
        load_path = propagated_npz_path
        status_text = "PROPAGATED"
        status_color = (255, 214, 10) # Yellow
    else:
        load_path = npz_path
        status_text = "UNVERIFIED"
        status_color = (255, 59, 48) # Red
        
    raw_data = load_npz_bboxes(load_path)
    raw_bboxes = raw_data['bboxes']
    raw_confs = raw_data['confs']
    raw_phrases = raw_data['phrases']
    
    boxes_data = []
    for i, box in enumerate(raw_bboxes):
        conf = raw_confs[i] if i < len(raw_confs) else 1.0
        phrase = raw_phrases[i] if i < len(raw_phrases) else "verified"
        boxes_data.append({
            'id': i,
            'bbox': box,
            'color': COLORS[i % len(COLORS)],
            'conf': conf,
            'phrase': phrase
        })
    next_id = len(raw_bboxes)
    temp_bbox = None

    if not img_path.exists():
        print(f"[Skip] Image missing: {img_path}")
        return 'next' 
    
    pil_img_orig = Image.open(img_path).convert("RGB")
    img_w, img_h = pil_img_orig.size
    
    # Determine if this is a "Thorough" frame
    # logic: local_idx 0, then every i-th frame (e.g. i=5 -> 0, 4, 9, 14...)
    is_thorough = False
    if index_frame_interval > 0:
        if local_idx == 0 or (local_idx + 1) % index_frame_interval == 0:
            is_thorough = True

    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return 'quit'

        # Create PIL Canvas
        final_h = img_h if pub_mode else img_h + Style.BOTTOM_PANEL_H
        canvas_pil = Image.new("RGB", (img_w, final_h), Style.PANEL_BG)
        canvas_pil.paste(pil_img_orig, (0, 0))
        draw = ImageDraw.Draw(canvas_pil)
        
        # Color the global border with the category color
        if not pub_mode:
            draw.rectangle([0, 0, img_w-1, final_h-1], outline=status_color, width=4)
        
        # Draw BBoxes
        for b in boxes_data:
            color = b['color']
            x1, y1, x2, y2 = b['bbox']
            lw = Style.PUB_LINE_WIDTH if pub_mode else 2
            
            # Simple rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            
            # Label
            label = f"{b['id']}: {b.get('phrase', '')}" if b.get('phrase') else str(b['id'])
            font_size = Style.PUB_FONT_SIZE if pub_mode else 14
            
            try:
                font = ImageFont.truetype(Style.FONT_PATH, font_size)
            except:
                font = ImageFont.load_default()
                
            # Use textbbox to get size for Supervision-style label background
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            tw, th = right - left, bottom - top
            
            # Label padding (Industry standards use approx 4-6px)
            pad_h = 6
            pad_v = 2
            
            # Supervision Style: Label flush with top-left, matching background
            tag_x1, tag_y1 = x1, y1 - th - (pad_v * 2) - 1
            # If label goes off image, put it inside
            if tag_y1 < 0:
                tag_y1 = y1
                tag_y2 = y1 + th + (pad_v * 2)
            else:
                tag_y2 = y1
            
            # Draw label background
            draw.rectangle([x1, tag_y1, x1 + tw + (pad_h * 2), tag_y2], fill=color)
            
            # Text contrast (Industry tools use white/black for high contrast)
            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            draw.text((x1 + pad_h, tag_y1 + pad_v), label, fill=text_color, font=font)
            
        if temp_bbox is not None and not pub_mode:
            draw.rectangle(temp_bbox, outline=(52, 199, 89), width=1)

        if not pub_mode:
            draw_bottom_panel(draw, img_w, img_h, full_id)
            
            # Navigation & Status
            nav_text = f"FRAME {current_idx} / {total_pairs} • {status_text}"
            pil_draw_text(draw, nav_text, (Style.BTN_PAD, final_h - 22), 12, status_color)
            
            # Hint
            hint_text = "Arrows: Nav | S: Save | P: Pub | Q: Save+Exit | ESC: Exit"
            pil_draw_text(draw, hint_text, (img_w - Style.BTN_PAD, final_h - 22), 11, Style.TEXT_SECONDARY, anchor="ra")
            
            # Thorough Marker Badge (Bottom Center)
            if is_thorough:
                badge_text = " MANDATORY THOROUGH REVIEW "
                try:
                    font_badge = ImageFont.truetype(Style.FONT_PATH, 24)
                except:
                    font_badge = ImageFont.load_default()
                l, t, r, b_badge = draw.textbbox((0, 0), badge_text, font=font_badge)
                tw, th = r - l, b_badge - t
                
                badge_x = (img_w - tw) // 2
                badge_y = img_h - th - 10
                draw.rectangle([badge_x - 5, badge_y - 5, badge_x + tw + 5, badge_y + th + 5], fill=(255, 214, 10), outline=(0, 0, 0), width=1)
                draw.text((badge_x, badge_y), badge_text, fill=(0, 0, 0), font=font_badge)

        # Convert PIL to CV for display (RGB -> BGR)
        canvas_cv = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, canvas_cv)
        key = cv2.waitKeyEx(20)
        
        if key == ord('s'):
            save_verified_data(npz_path, base_dir)
            return 'next'
        if key == ord('q'):
            save_verified_data(npz_path, base_dir)
            return 'quit'
        elif key == 27: # ESC
            return 'quit'
        elif key in (65361, 2424832, 2):
            return 'prev'
        elif key in (65363, 2555904, 3, 32): 
            return 'next'
        elif key == ord('p'):
            pub_mode = not pub_mode

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
    parser.add_argument("--index_frame", type=int, default=5, help="Interval for thorough annotation (e.g. 5 means 0, 4, 9...)")
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
    
    window_name = "RoboKit | BBox Review"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 1000)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Pre-calculate local indices for thorough review reset
    local_indices = []
    curr_cat = None
    l_idx = 0
    for i, (img, npz) in enumerate(all_pairs):
        # Category key: (scene, part)
        try:
            key = (npz.parent.parent.parent.name, npz.parent.name)
        except:
            key = ("unknown", "unknown")
            
        if key != curr_cat:
            curr_cat = key
            l_idx = 0
        local_indices.append(l_idx)
        l_idx += 1

    idx = start_idx
    while 0 <= idx < total_all_frames:
        img_path, npz_path = all_pairs[idx]
        
        # Pass both global index (for nav) and local index (for thorough logic)
        action = review_frame(img_path, npz_path, args.base_dir, total_all_frames, idx + 1, window_name, 
                             index_frame_interval=args.index_frame, local_idx=local_indices[idx])
        
        if action == 'next':
            idx += 1
        elif action == 'prev':
            idx = max(0, idx - 1)
        elif action == 'quit':
            break
            
    print("\n[Finished] Session complete.")

if __name__ == "__main__":
    main()