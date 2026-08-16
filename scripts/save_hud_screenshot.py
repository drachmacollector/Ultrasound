import torch
import cv2
import argparse
from pathlib import Path
from src.realtime.model_loader import load_inference_model
from src.realtime.pipeline import PipelineStats
from src.realtime.app import build_display_frame
from src.data.transforms import prep_frame_grayscale_to_rgb

def save_hud_screenshot(video_path: str, ckpt_path: str, out_path: str):
    print(f"Loading model from {ckpt_path}...")
    lm = load_inference_model(ckpt_path)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        return
        
    print("Running inference...")
    rgb_hw3 = prep_frame_grayscale_to_rgb(frame)
    augmented = lm.transform(image=rgb_hw3)
    tensor = augmented["image"].unsqueeze(0).to(lm.device)
    
    bboxes = None
    bbox_labels = None
    bbox_scores = None
    
    with torch.no_grad():
        if hasattr(lm.model, "retinanet"):
            logits, det_outputs = lm.model(tensor)
            if len(det_outputs) > 0:
                bboxes = det_outputs[0]["boxes"].cpu().numpy()
                bbox_labels = det_outputs[0]["labels"].cpu().numpy()
                bbox_scores = det_outputs[0]["scores"].cpu().numpy()
        else:
            logits = lm.model(tensor)
            
    import numpy as np
    probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    label = int(np.argmax(probs))
    confidence = float(probs[label])
    
    from src.data.dataset import IDX_TO_CLASS
    
    result = {
        "label": label,
        "label_name": IDX_TO_CLASS.get(label, f"class_{label}"),
        "confidence": confidence,
        "smoothed_probs": probs,
        "is_stable": True,
        "overlay": None,
        "frame": frame,
        "timings": {
            "preprocess_ms": 5.0,
            "forward_ms": 20.0,
            "smoothing_ms": 1.0,
            "gradcam_ms": None,
        },
        "capture_ts": 0.0,
        "inference_ts": 0.0,
        "tier2_active": False,
        "bboxes": bboxes,
        "bbox_labels": bbox_labels,
        "bbox_scores": bbox_scores,
    }
    
    stats = PipelineStats()
    # Mock stats
    stats._cap_ts.extend([0.0, 1.0])
    stats._inf_ts.extend([0.0, 1.0])
    
    # We pass is_webcam=True so that we can see the watermark
    display = build_display_frame(
        result=result,
        show_gradcam=False,
        show_hud=True,
        is_paused=False,
        is_webcam=True,
        stats=stats
    )
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, display)
    print(f"Screenshot saved to {out_path}")

if __name__ == "__main__":
    save_hud_screenshot(
        "data/processed/synthetic_clips/Brain_Trans_thalamic_clip01.mp4",
        "checkpoints/multitask/best.pt",
        "docs/phases/phase_07/phase7_multitask_hud_demo.png"
    )
