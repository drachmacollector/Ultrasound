"""
scripts/evaluate_transition_latency.py

Evaluates the transition latency of the full smoothing pipeline (Tier 1 + Tier 2a) 
using the synthetic multi-plane clips and their annotations.

Latency is measured from the annotated "settle" frame to the first frame where the 
smoothing pipeline outputs the correct target label with `is_stable=True`.
"""
import argparse
import csv
import glob
import json
import logging
import os
import sys
from pathlib import Path
import pandas as pd

import cv2
import numpy as np
import torch
import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.smoothing.tier1 import Tier1Smoother
from src.smoothing.tier2_mode_filter import Tier2ModeFilter
from src.realtime.model_loader import load_inference_model
from src.data.transforms import prep_frame_grayscale_to_rgb
from src.data.dataset import CLASS_TO_IDX

# --- Config Paths ---
CKPT_PATH = Path("checkpoints/convnext_tiny/best.pt")
TIER1_CFG_PATH = Path("configs/smoothing_tier1.yaml")
TIER2_CFG_PATH = Path("configs/smoothing_tier2a.yaml")
CLIPS_DIR = Path("data/processed/synthetic_clips")
LOG_PATH = Path("logs/eval/evaluate_transition_latency.txt")
CSV_PATH = Path("logs/eval/transition_latency_details.csv")
NATALIA_DIR = Path("data/raw/natalia_pbfus1")

def run_inference(video_path: str, model: torch.nn.Module, transform, device: torch.device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 24.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    all_probs = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = prep_frame_grayscale_to_rgb(frame)
        tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            logits = model(tensor)
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        all_probs.append(probs)
    cap.release()
    return all_probs, fps

def run_inference_natalia(exam_dir: Path, df_exam: pd.DataFrame, model: torch.nn.Module, transform, device: torch.device):
    all_probs = []
    fps = 24.0
    for _, row in df_exam.iterrows():
        img_path = exam_dir / row['file_name']
        if not img_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        img_rgb = prep_frame_grayscale_to_rgb(frame)
        tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            logits = model(tensor)
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        all_probs.append(probs)
    return all_probs, fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--natalia", action="store_true", help="Run latency evaluation on NatalIA transitions")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8", mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    log = logging.getLogger(__name__)
    
    log.info("=" * 60)
    log.info("Transition Latency Evaluation (Phase 6)")
    log.info("=" * 60)

    # 1. Load configurations
    with open(TIER1_CFG_PATH, "r", encoding="utf-8") as f:
        t1_cfg = yaml.safe_load(f)
    log.info(f"Loaded Tier 1 config: {t1_cfg}")

    with open(TIER2_CFG_PATH, "r", encoding="utf-8") as f:
        t2_cfg = yaml.safe_load(f)
    log.info(f"Loaded Tier 2 config: {t2_cfg}\n")

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_inference_model(CKPT_PATH, device=device)
    model, transform = loaded.model, loaded.transform

    all_latencies_ms = []
    transition_details = []
    
    if args.natalia:
        log.info("Running NatalIA transition evaluation...")
        df = pd.read_csv(NATALIA_DIR / "resume.csv")
        df = df[~df['value'].isin([2, 3])] # Exclude heart/spine
        valid_values = {0: "Head", 1: "Fetal_abdomen", 4: "Fetal_femur"}
        
        # Group by exam
        for studie, group in df.groupby('studie'):
            group = group.copy()
            def get_idx(fn):
                try: return int(fn.split('_')[1])
                except: return 0
            group['idx'] = group['file_name'].apply(get_idx)
            group = group.sort_values('idx').reset_index(drop=True)
            
            settle_events = []
            last_plane = None
            for idx, row in group.iterrows():
                val = row['value']
                if val in valid_values:
                    if last_plane is not None and last_plane != val:
                        settle_events.append({"frame": idx, "label": valid_values[val]})
                    last_plane = val
            
            if settle_events:
                log.info(f"Processing {studie} ({len(settle_events)} transitions)...")
                exam_dir = NATALIA_DIR / str(studie)
                all_probs, fps = run_inference_natalia(exam_dir, group, model, transform, device)
                clip_name = str(studie)
                # Continue with the same processing logic for these settle events...
                
                if not all_probs:
                    continue
                    
                ms_per_frame = 1000.0 / fps

                # Process each transition event
                for event in settle_events:
                    settle_f = event["frame"]
                    if settle_f >= len(all_probs):
                        continue
                        
                    # Deriving valid target labels directly from the ground-truth annotation event
                    if event["label"] == "Head":
                        valid_target_labels = {CLASS_TO_IDX[c] for c in ["Brain_Trans_cerebellum", "Brain_Trans_thalamic", "Brain_Trans_ventricular"]}
                    else:
                        valid_target_labels = {CLASS_TO_IDX[event["label"]]}
                    
                    # Re-initialize smoothers
                    tier1 = Tier1Smoother(
                        num_classes=8,
                        alpha=t1_cfg["alpha"],
                        switch_threshold=t1_cfg["switch_threshold"],
                        min_dwell_frames=t1_cfg["min_dwell_frames"],
                        hold_floor=t1_cfg.get("hold_floor")
                    )
                    tier2 = Tier2ModeFilter(
                        window_frames=t2_cfg["window_frames"],
                        min_majority_frac=t2_cfg["min_majority_frac"],
                        min_stable_frames=t2_cfg.get("min_stable_frames", 1)
                    )
                    
                    # Fast-forward smoothing state up to the frame BEFORE the settle event
                    for fi in range(settle_f):
                        t1_label, _, _, _ = tier1.step(all_probs[fi])
                        tier2.step(t1_label)
                        
                    # Now measure how long it takes from the settle frame to stabilize on the correct label
                    found_stable = False
                    for offset, fi in enumerate(range(settle_f, len(all_probs))):
                        t1_label, _, _, _ = tier1.step(all_probs[fi])
                        t2_label, is_stable = tier2.step(t1_label)
                        
                        if t2_label in valid_target_labels and is_stable:
                            latency_ms = offset * ms_per_frame
                            all_latencies_ms.append(latency_ms)
                            found_stable = True
                            transition_details.append({
                                "clip": clip_name,
                                "settle_frame": settle_f,
                                "target_class": event["label"],
                                "latency_ms": latency_ms,
                                "stabilized": True
                            })
                            break
                            
                    if not found_stable:
                        # Penalise with remaining duration if it never stabilises
                        remaining = len(all_probs) - settle_f
                        latency_ms = remaining * ms_per_frame
                        all_latencies_ms.append(latency_ms)
                        log.warning(f"  Transition at frame {settle_f} never stabilized on {event['label']}!")
                        transition_details.append({
                            "clip": clip_name,
                            "settle_frame": settle_f,
                            "target_class": event["label"],
                            "latency_ms": latency_ms,
                            "stabilized": False
                        })
    else:
        clip_paths = sorted(glob.glob(str(CLIPS_DIR / "multiplane_scan_*.mp4")))
        if not clip_paths:
            log.error(f"No multiplane clips found in {CLIPS_DIR}")
            return

        for clip_path in clip_paths:
            clip_name = os.path.basename(clip_path)
            json_name = clip_name.replace(".mp4", "_annotations.json")
            json_path = CLIPS_DIR / json_name
            
            if not json_path.exists():
                log.warning(f"Missing annotations for {clip_name}, skipping.")
                continue
                
            with open(json_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
                
            settle_events = [a for a in annotations if a["event"] == "settle"]
            if not settle_events:
                continue
                
            log.info(f"Processing {clip_name} ({len(settle_events)} transitions)...")
            all_probs, fps = run_inference(clip_path, model, transform, device)
        
            if not all_probs:
                continue
                
            ms_per_frame = 1000.0 / fps

            # Process each transition event
            for event in settle_events:
                settle_f = event["frame"]
                if settle_f >= len(all_probs):
                    continue
                    
                # Deriving valid target labels directly from the ground-truth annotation event
                if event["label"] == "Head":
                    valid_target_labels = {CLASS_TO_IDX[c] for c in ["Brain_Trans_cerebellum", "Brain_Trans_thalamic", "Brain_Trans_ventricular"]}
                else:
                    valid_target_labels = {CLASS_TO_IDX[event["label"]]}
                
                # Re-initialize smoothers
                tier1 = Tier1Smoother(
                    num_classes=8,
                    alpha=t1_cfg["alpha"],
                    switch_threshold=t1_cfg["switch_threshold"],
                    min_dwell_frames=t1_cfg["min_dwell_frames"],
                    hold_floor=t1_cfg.get("hold_floor")
                )
                tier2 = Tier2ModeFilter(
                    window_frames=t2_cfg["window_frames"],
                    min_majority_frac=t2_cfg["min_majority_frac"],
                    min_stable_frames=t2_cfg.get("min_stable_frames", 1)
                )
                
                # Fast-forward smoothing state up to the frame BEFORE the settle event
                for fi in range(settle_f):
                    t1_label, _, _, _ = tier1.step(all_probs[fi])
                    tier2.step(t1_label)
                    
                # Now measure how long it takes from the settle frame to stabilize on the correct label
                found_stable = False
                for offset, fi in enumerate(range(settle_f, len(all_probs))):
                    t1_label, _, _, _ = tier1.step(all_probs[fi])
                    t2_label, is_stable = tier2.step(t1_label)
                    
                    if t2_label in valid_target_labels and is_stable:
                        latency_ms = offset * ms_per_frame
                        all_latencies_ms.append(latency_ms)
                        found_stable = True
                        transition_details.append({
                            "clip": clip_name,
                            "settle_frame": settle_f,
                            "target_class": event["label"],
                            "latency_ms": latency_ms,
                            "stabilized": True
                        })
                        break
                        
                if not found_stable:
                    # Penalise with remaining duration if it never stabilises
                    remaining = len(all_probs) - settle_f
                    latency_ms = remaining * ms_per_frame
                    all_latencies_ms.append(latency_ms)
                    log.warning(f"  Transition at frame {settle_f} never stabilized on {event['label']}!")
                    transition_details.append({
                        "clip": clip_name,
                        "settle_frame": settle_f,
                        "target_class": event["label"],
                        "latency_ms": latency_ms,
                        "stabilized": False
                    })

    # Summary Stats
    if all_latencies_ms:
        mean_lat = np.mean(all_latencies_ms)
        p50 = np.median(all_latencies_ms)
        p90 = np.percentile(all_latencies_ms, 90)
        p99 = np.percentile(all_latencies_ms, 99)
        max_lat = np.max(all_latencies_ms)
        
        log.info("\n" + "=" * 60)
        log.info(f"Transition Latency Results (N={len(all_latencies_ms)} real transitions)")
        log.info("=" * 60)
        log.info(f"Mean Latency-to-Stable : {mean_lat:.1f} ms")
        log.info(f"Median (P50)             : {p50:.1f} ms")
        log.info(f"90th Percentile (P90)    : {p90:.1f} ms")
        log.info(f"99th Percentile (P99)    : {p99:.1f} ms")
        log.info(f"Max Latency              : {max_lat:.1f} ms")
        log.info("=" * 60)
        
        # Write CSV
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["clip", "settle_frame", "target_class", "latency_ms", "stabilized"])
            writer.writeheader()
            writer.writerows(transition_details)
        log.info(f"Detailed results saved to {CSV_PATH}")
    else:
        log.info("No transitions measured.")

if __name__ == "__main__":
    main()
