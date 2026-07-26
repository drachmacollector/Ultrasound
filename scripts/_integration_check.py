"""
Final integration check: load config + build model + verify dataset loads one batch.
This does NOT run training. Verifies the full pipeline wiring.
"""
import sys
sys.path.insert(0, ".")

import json
import torch
import yaml

from src.data.dataset import CANONICAL_CLASSES, FocalPlanesDataset
from src.data.transforms import get_train_transform, get_eval_transform
from src.models.backbone import build_model
from src.train.train import load_config, build_class_weight_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load config
cfg = load_config("configs/repvgg_a1.yaml")
print(f"Config loaded: backbone={cfg['backbone']}, image_size={cfg['image_size']}")

# Build model
model = build_model(cfg["backbone"], num_classes=8, pretrained=False)
model = model.to(device)
print(f"Model built: {type(model).__name__}")

# Class weights
wt = build_class_weight_tensor(cfg["class_weights_json"], device)
print(f"Class weights: {wt.shape} -> {wt.cpu().tolist()}")
assert len(wt) == 8, "Weight tensor must have exactly 8 entries"

# Verify weights are indexed in CANONICAL_CLASSES order
with open(cfg["class_weights_json"]) as f:
    raw = json.load(f)
expected = [raw[c] for c in CANONICAL_CLASSES]
actual = wt.cpu().tolist()
# Compare with allclose tolerance: json.load gives float64, tensor.tolist() gives float32 precision.
# The ordering matters; the float32 rounding does not affect training.
import torch as _torch
expected_t = _torch.tensor(expected, dtype=_torch.float32)
actual_t = _torch.tensor(actual, dtype=_torch.float32)
assert _torch.allclose(expected_t, actual_t, atol=1e-5), (
    f"Weight ordering mismatch (beyond float32 tolerance): {expected} != {actual}"
)
print("Class weight ordering: CORRECT (indexed by CANONICAL_CLASSES, float32 precision OK)")

# Dataset - just peek at the first item
ds = FocalPlanesDataset(cfg["train_csv"], transform=get_eval_transform(cfg["image_size"]))
print(f"Dataset size: {len(ds)}")
tensor, label = ds[0]
print(f"Sample 0: tensor.shape={tensor.shape}, label={label} ({CANONICAL_CLASSES[label]})")
assert tensor.shape == (3, cfg["image_size"], cfg["image_size"]), f"Shape mismatch: {tensor.shape}"

# Forward pass
model.eval()
with torch.no_grad():
    out = model(tensor.unsqueeze(0).to(device))
print(f"Forward pass output: {out.shape}")
assert out.shape == (1, 8)

print()
print("Integration check PASSED. Training pipeline is fully wired.")
