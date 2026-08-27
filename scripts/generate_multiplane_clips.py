"""
scripts/generate_multiplane_clips.py

Generates 20-second synthetic multi-plane ultrasound clips for testing temporal smoothing
and transition latency. 

- Alternates between "settled" planes (low wobble) and "transition" out-of-plane 
  movements (high wobble).
- Normalizes all sampled frames to a common canvas size per clip to prevent 
  dimension mismatches during MP4 encoding.
- Emits a matching `_annotations.json` file per clip with exact 'settle' and 'leave' events.
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Allow imports from project root when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_video import generate_ego_motion_clip, save_clip_as_mp4
from src.data.transforms import load_and_prep_grayscale_to_rgb

TRAIN_CSV = Path("data/splits/train.csv")
OUT_DIR = Path("data/processed/synthetic_clips")
FPS = 24
TARGET_FRAMES = 480  # ~20 seconds
NUM_CLIPS = 8
RANDOM_SEED = 42

def pad_image_to(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad image with black borders to reach target dimensions, keeping it centered."""
    h, w = img.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def main():
    random.seed(RANDOM_SEED)
    df = pd.read_csv(TRAIN_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    unique_classes = df["plane_label"].unique().tolist()
    classes = [c for c in unique_classes if "Other" not in c]
    other_classes = [c for c in unique_classes if "Other" in c]
    other_class = other_classes[0] if other_classes else classes[0]
    
    clips_written = 0
    
    for clip_idx in range(1, NUM_CLIPS + 1):
        print(f"Generating clip {clip_idx}/{NUM_CLIPS}...")
        
        # 1. Plan the segments
        segments = []
        total_frames = 0
        
        while total_frames < TARGET_FRAMES:
            # --- Settled Segment ---
            cls = random.choice(classes)
            dur_frames = int(random.uniform(2.0, 5.0) * FPS)
            wobble = random.uniform(0.5, 1.5)
            
            # Clamp to TARGET_FRAMES if we overshoot
            if total_frames + dur_frames > TARGET_FRAMES:
                dur_frames = TARGET_FRAMES - total_frames
                
            segments.append({
                "type": "settled", 
                "class": cls, 
                "frames": dur_frames, 
                "wobble": wobble
            })
            total_frames += dur_frames
            
            if total_frames >= TARGET_FRAMES:
                break
                
            # --- Transition Segment ---
            # High wobble, shorter duration, usually 'Other'
            dur_frames = int(random.uniform(0.5, 1.5) * FPS)
            wobble = random.uniform(3.0, 6.0)
            
            if total_frames + dur_frames > TARGET_FRAMES:
                dur_frames = TARGET_FRAMES - total_frames
                
            segments.append({
                "type": "transition", 
                "class": other_class, 
                "frames": dur_frames, 
                "wobble": wobble
            })
            total_frames += dur_frames
            
        # 2. Sample images and find max dimensions
        for seg in segments:
            cls_df = df[df["plane_label"] == seg["class"]]
            row = cls_df.sample(n=1, random_state=random.randint(0, 100000)).iloc[0]
            try:
                img = load_and_prep_grayscale_to_rgb(row["image_path"])
                seg["image"] = img
            except FileNotFoundError as e:
                print(f"  [WARNING] File not found: {e}. Padding will crash if image is missing.")
                raise e
            
        max_h = max(seg["image"].shape[0] for seg in segments)
        max_w = max(seg["image"].shape[1] for seg in segments)
        
        print(f"  Normalizing {len(segments)} segments to canvas size: {max_w}x{max_h}")
        
        # 3. Generate frames and annotations
        all_frames = []
        annotations = []
        current_frame = 0
        
        for seg in segments:
            padded_img = pad_image_to(seg["image"], max_h, max_w)
            seed = random.randint(0, 100000)
            
            frames = generate_ego_motion_clip(
                padded_img, 
                n_frames=seg["frames"], 
                seed=seed, 
                wobble_scale=seg["wobble"]
            )
            
            if seg["type"] == "settled":
                annotations.append({"frame": current_frame, "event": "settle", "label": seg["class"]})
                # The frame right before the next segment starts is the 'leave' frame
                annotations.append({"frame": current_frame + seg["frames"] - 1, "event": "leave", "label": seg["class"]})
                
            all_frames.extend(frames)
            current_frame += seg["frames"]
            
        # 4. Save MP4 and Annotations
        out_name = f"multiplane_scan_{clip_idx:02d}.mp4"
        out_path = OUT_DIR / out_name
        save_clip_as_mp4(all_frames, str(out_path), fps=FPS)
        
        json_name = f"multiplane_scan_{clip_idx:02d}_annotations.json"
        json_path = OUT_DIR / json_name
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)
            
        print(f"  Saved {out_name} and {json_name}")
        clips_written += 1
        
    print(f"\nDone. {clips_written} multiplane clips written to {OUT_DIR}/")

if __name__ == "__main__":
    main()
