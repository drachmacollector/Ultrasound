import os
import cv2
import glob
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import logging

import sys
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


from src.realtime.model_loader import load_inference_model
from src.data.transforms import prep_frame_grayscale_to_rgb

def create_contact_sheet(video_path, output_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, f"F:{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frames.append(frame)
    cap.release()
    
    if not frames:
        return
    
    h, w, c = frames[0].shape
    grid_size = int(np.ceil(np.sqrt(num_frames)))
    sheet = np.zeros((h * grid_size, w * grid_size, c), dtype=np.uint8)
    
    for i, frame in enumerate(frames):
        row = i // grid_size
        col = i % grid_size
        frame = cv2.resize(frame, (w, h))
        sheet[row*h:(row+1)*h, col*w:(col+1)*w] = frame
        
    cv2.imwrite(output_path, sheet)

def main():
    os.makedirs("data/processed/tier1_tuning", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/processed/manual_review", exist_ok=True)
    
    log_file = open("logs/tuning/measure_baseline_flicker.txt", "w", encoding="utf-8")
    
    print("Loading model...", file=log_file)
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device.type)
    loaded_model = load_inference_model("checkpoints/convnext_tiny/best.pt", device=device)
    model = loaded_model.model
    transform = loaded_model.transform
    
    video_paths = []
    iugc_clips = []

    # Synthetic clips
    synthetic_clips = sorted(glob.glob("data/processed/synthetic_clips/*.mp4"))
    video_paths.extend(synthetic_clips)

    iugc_dir = "data/raw/iugc_video/DatasetV3"
    iugc_present = os.path.exists(iugc_dir)

    # IUGC clips: use the same direct-glob strategy as tune_tier1_smoothing.py
    # (first 10 .avi per split/videos/) so both scripts characterise identical clip sets.
    # The previous CSV-filtered approach (pos/neg columns + file-existence check) silently
    # found only 1 IUGC clip from train and missed 19 clips that the sweep evaluated,
    # causing a silent disagreement about what "the dataset" was.
    if iugc_present:
        for split in ['train', 'val', 'test']:
            vid_dir = os.path.join(iugc_dir, split, "videos")
            if os.path.exists(vid_dir):
                avs = sorted(glob.glob(os.path.join(vid_dir, "*.avi")))[:10]
                for v_path in avs:
                    video_paths.append(v_path)
                    iugc_clips.append({'path': v_path, 'pos': 'n/a'})

                    
    results = []
    total_frames_processed = 0
    total_inference_time = 0.0
    
    for v_path in video_paths:
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            continue
            
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        prev_label = None
        switches = 0
        switch_confidences = []
        stable_confidences = []
        
        frames_read = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            img_rgb = prep_frame_grayscale_to_rgb(frame)
            tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
            
            t0 = time.time()
            with torch.no_grad(), torch.amp.autocast('cuda'):
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            t1 = time.time()
            
            total_inference_time += (t1 - t0)
            total_frames_processed += 1
            frames_read += 1
            
            label = int(np.argmax(probs))
            conf = float(probs[label])
            
            if prev_label is not None and label != prev_label:
                switches += 1
                switch_confidences.append(conf)
            elif prev_label is not None and label == prev_label:
                stable_confidences.append(conf)
                
            prev_label = label
            
        duration = frames_read / fps_video if fps_video > 0 else 0
        switches_per_sec = switches / duration if duration > 0 else 0
        switches_per_min = switches_per_sec * 60
        
        mean_switch_conf = np.mean(switch_confidences) if switch_confidences else 0
        mean_stable_conf = np.mean(stable_confidences) if stable_confidences else 0
        
        clip_name = os.path.basename(v_path)
        log_line = f"[{clip_name}] Switches/sec: {switches_per_sec:.2f}, Mean Switch Conf: {mean_switch_conf:.2f}, Mean Stable Conf: {mean_stable_conf:.2f}"
        print(log_line)
        print(log_line, file=log_file)
        
        results.append({
            'clip_name': clip_name,
            'source_path': v_path,
            'switches_per_sec': switches_per_sec,
            'switches_per_min': switches_per_min,
            'mean_switch_conf': mean_switch_conf,
            'mean_stable_conf': mean_stable_conf,
            'is_iugc': 'iugc_video' in v_path
        })
        
        cap.release()
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values('switches_per_sec', ascending=False)
    df_res.to_csv("data/processed/tier1_tuning/baseline_flicker_report.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(df_res['clip_name'], df_res['switches_per_sec'])
    plt.xticks(rotation=90)
    plt.ylabel('Switches per Second')
    plt.title('Baseline Flicker: Raw Argmax Switches per Second')
    plt.tight_layout()
    plt.savefig("data/processed/tier1_tuning/baseline_flicker_chart.png")
    
    inference_fps = total_frames_processed / total_inference_time if total_inference_time > 0 else 0
    print(f"\nTotal frames processed: {total_frames_processed}")
    print(f"Total inference time: {total_inference_time:.2f} s")
    print(f"Raw inference FPS: {inference_fps:.2f}")
    
    print(f"Total frames processed: {total_frames_processed}", file=log_file)
    print(f"Total inference time: {total_inference_time:.2f} s", file=log_file)
    print(f"Raw inference FPS: {inference_fps:.2f}", file=log_file)
    log_file.close()
    
    if iugc_present and len(iugc_clips) >= 5:
        candidate_clips = iugc_clips[:5]
            
        print("\nSelected 5 clips for manual annotation:")
        template_dict = {}
        for clip in candidate_clips:
            path = clip['path']
            cname = os.path.basename(path)
            print(f"- {cname} (pos metadata: {clip['pos']})")
            
            sheet_path = f"data/processed/manual_review/{cname}_contact_sheet.png"
            create_contact_sheet(path, sheet_path)
            
            template_dict[cname] = {"transitions": []}
            
        with open("data/processed/manual_review/transition_annotations_template.json", "w", encoding="utf-8") as f:
            json.dump(template_dict, f, indent=2)
            
if __name__ == '__main__':
    main()
