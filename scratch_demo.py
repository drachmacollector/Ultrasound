import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, 'd:/Ultrasound')
from src.realtime.app import build_display_frame
from src.realtime.pipeline import PipelineStats

# Create a mock frame
cap = cv2.VideoCapture('d:/Ultrasound/data/processed/synthetic_clips/Brain_Trans_thalamic_clip01.mp4')
ret, frame = cap.read()
if not ret:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Dummy Grad-CAM heatmap
heatmap = np.zeros_like(frame)
heatmap[:, :, 2] = 200 # Red heatmap
overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

result = {
    'frame': frame,
    'overlay': overlay,
    'label_name': 'Brain_Trans-thalamic',
    'confidence': 0.95,
    'is_stable': True,
    'tier2_active': False,
    'timings': {'gradcam_ms': 120.5}
}

stats = PipelineStats()

snap = {
    'capture_fps': 30.0,
    'inference_fps': 24.1,
    'preprocess_ms': 4.5,
    'forward_ms': 42.1,
    'smoothing_ms': 0.1,
    'gradcam_ms': 120.5,
    'gradcam_calls': 20,
    'cap_queue_drops': 0,
    'inf_queue_drops': 0
}
stats.snapshot = lambda: snap

display = build_display_frame(result, True, True, False, True, stats)

out_dir = Path('d:/Ultrasound/docs/phases/phase_07')
out_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_dir / 'hud_clean_demo.png'), display)
print('Screenshot saved successfully.')
