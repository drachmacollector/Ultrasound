import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path so we can import src without -m
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FocalPlanesDataset
from src.eval.evaluate_test import TEST_CSV, get_eval_transform
from src.models.cam_localisation import cam_to_bbox
from src.models.gradcam import run_gradcam
from src.models.backbone import build_model
from src.realtime.app import build_display_frame
from src.realtime.pipeline import PipelineStats
from pytorch_grad_cam.utils.image import show_cam_on_image

log = logging.getLogger(__name__)

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate CAM spot-check images.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--output-dir", type=str, default="outputs/cam_spotcheck", help="Output directory")
    return p

def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found at {ckpt_path}")
        sys.exit(1)
        
    # Load model
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone_name = checkpoint.get("backbone_name", "convnext_tiny.fb_in22k_ft_in1k")
    model = build_model(backbone_name=backbone_name, num_classes=8, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load dataset
    if not Path(TEST_CSV).exists():
        log.error(f"Test CSV not found at {TEST_CSV}")
        sys.exit(1)
        
    df = pd.read_csv(TEST_CSV)
    
    # Get test transform
    transform = get_eval_transform()
    
    # Sample images per class
    sample_counts = {
        "Brain_Trans_cerebellum": 3,
        "Brain_Trans_thalamic": 3,
        "Brain_Trans_ventricular": 3,
        "Fetal_abdomen": 2,
        "Fetal_femur": 2,
        "Fetal_thorax": 2,
        "Maternal_cervix": 2,
        "Other": 3
    }
    
    sampled_indices = []
    for cls, count in sample_counts.items():
        cls_rows = df[df["plane_label"] == cls]
        take = min(count, len(cls_rows))
        sampled_indices.extend(cls_rows.sample(n=take, random_state=42).index.tolist())
        
    dataset = FocalPlanesDataset(csv_path=TEST_CSV, transform=transform)
    
    count = 0
    for idx in sampled_indices:
        count += 1
        row = df.iloc[idx]
        image_path = row["image_path"]
        canon_label = row["plane_label"]
        
        dataset_item = dataset[idx]
        input_tensor, label = dataset_item[0], dataset_item[1]
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        img_path = Path(image_path)
        if not img_path.exists():
            log.warning(f"Image not found: {img_path}")
            continue
            
        orig_img_bgr = cv2.imread(str(img_path))
        if orig_img_bgr is None:
            log.warning(f"Failed to read image: {img_path}")
            continue
        orig_img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
        orig_float = orig_img_rgb.astype(np.float32) / 255.0
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs.float(), dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            
        # Get CAM (grayscale) using original run_gradcam helper (it returns 224x224)
        cam = run_gradcam(model, input_tensor, pred_idx, backbone_name, original_rgb_float=None)
        
        # Resize CAM to original image dimensions
        h_orig, w_orig = orig_img_bgr.shape[:2]
        cam_resized = cv2.resize(cam, (w_orig, h_orig))
        
        # Create overlay (original size)
        overlay_rgb = show_cam_on_image(orig_float, cam_resized, use_rgb=True)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        
        # Compute bbox on the full-resolution resized CAM
        bbox = cam_to_bbox(cam_resized, method="otsu")
        
        # Construct pipeline result for build_display_frame
        result = {
            "label_name": canon_label,
            "confidence": confidence,
            "is_stable": True,
            "overlay": overlay_bgr,
            "frame": orig_img_bgr,
            "timings": {
                "preprocess_ms": 0.0,
                "forward_ms": 0.0,
                "smoothing_ms": 0.0,
                "gradcam_ms": 0.0,
            },
            "cam_bbox": bbox,
            "bboxes": None,
            "bbox_labels": None,
            "bbox_scores": None,
            "tier2_active": False,
        }
        
        stats = PipelineStats()
        # Mock some timestamps for FPS calculation so it doesn't div/0
        stats._cap_ts.extend([0.0, 1.0])
        stats._inf_ts.extend([0.0, 1.0])
        stats._gradcam_calls = 1
        
        display = build_display_frame(
            result=result,
            show_gradcam=True,
            show_hud=True,
            is_paused=False,
            is_webcam=False,
            stats=stats,
            show_cam_bbox=True
        )
            
        # Save output
        out_name = f"{count:02d}_{canon_label}_{img_path.name}"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), display)
        log.info(f"Saved {out_name}")
        
    log.info(f"Generated {count} images in {out_dir}")

if __name__ == "__main__":
    main()
