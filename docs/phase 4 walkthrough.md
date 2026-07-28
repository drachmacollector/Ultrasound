# Phase 4 Implementation Walkthrough

## What Was Built

### New Files

| File | Purpose |
|---|---|
| [src/models/__init__.py](file:///d:/Ultrasound/src/models/__init__.py) | Package init |
| [src/models/backbone.py](file:///d:/Ultrasound/src/models/backbone.py) | `build_model()` + `get_pretrained_cfg()` via timm |
| [src/models/gradcam.py](file:///d:/Ultrasound/src/models/gradcam.py) | Grad-CAM with verified per-backbone target layers |
| [src/models/test_backbone.py](file:///d:/Ultrasound/src/models/test_backbone.py) | Shape unit tests (no data/pretrained needed) |
| [src/train/__init__.py](file:///d:/Ultrasound/src/train/__init__.py) | Package init |
| [src/train/train.py](file:///d:/Ultrasound/src/train/train.py) | Config-driven training loop |
| [configs/repvgg_a1.yaml](file:///d:/Ultrasound/configs/repvgg_a1.yaml) | backbone config |
| [configs/repvgg_a2.yaml](file:///d:/Ultrasound/configs/repvgg_a2.yaml) | backbone config |
| [configs/mobilenetv3_large_100.yaml](file:///d:/Ultrasound/configs/mobilenetv3_large_100.yaml) | backbone config |
| [configs/efficientnet_lite0.yaml](file:///d:/Ultrasound/configs/efficientnet_lite0.yaml) | backbone config |
| [configs/tf_efficientnetv2_s.yaml](file:///d:/Ultrasound/configs/tf_efficientnetv2_s.yaml) | backbone config — **300×300, TF normalization** |
| [scripts/lr_finder.py](file:///d:/Ultrasound/scripts/lr_finder.py) | LR range test (self-contained, no extra dependency) |
| [scripts/smoke_test.py](file:///d:/Ultrasound/scripts/smoke_test.py) | 5-epoch smoke test for all 6 backbones |
| [scripts/inspect_pretrained_cfg.py](file:///d:/Ultrasound/scripts/inspect_pretrained_cfg.py) | Print pretrained_cfg for all backbones |
| [docs/EXPERIMENTS.md](file:///d:/Ultrasound/docs/EXPERIMENTS.md) | Pre-populated with verified cfg data; tables ready |

---

## Verification Results

### ✅ All backbone unit tests (src/models/test_backbone.py)
```
repvgg_a1          → torch.Size([2, 8]) ✓
repvgg_a2          → torch.Size([2, 8]) ✓
mobilenetv3_large_100 → torch.Size([2, 8]) ✓
efficientnet_lite0 → torch.Size([2, 8]) ✓
tf_efficientnetv2_s → torch.Size([1, 8]) ✓  (300×300)
```

### ✅ All module imports clean
```
src.models.backbone: OK
src.models.gradcam: OK
src.train.train: OK
scripts.lr_finder: OK
scripts.smoke_test: OK
```

### ✅ All GradCAM target layers resolve (timm 1.0.28)
```
repvgg_a1       → RepVggBlock   (model.stages[3][-1])
repvgg_a2       → RepVggBlock   (model.stages[3][-1])
mobilenetv3     → ConvBnAct     (model.blocks[6][-1])
efficientnet_lite0 → InvertedResidual (model.blocks[6][-1])
tf_effnetv2_s   → InvertedResidual (model.blocks[5][-1])
```

### ✅ Integration check (config + model + dataset + forward pass)
```
Device: cuda
Config loaded: backbone=repvgg_a1, image_size=224
Model built: ByobNet
Class weights: torch.Size([8]) — CORRECT ordering (CANONICAL_CLASSES-indexed)
Dataset size: 6106
Sample 0: tensor.shape=torch.Size([3, 224, 224]), label=7 (Other)
Forward pass output: torch.Size([1, 8])
Integration check PASSED.
```

---

## Key Design Decisions Made

### 1. EfficientNetV2-S at 300×300 with TF normalization (§0.3)
### Critical §0.3 correction applied
`tf_efficientnetv2_s` uses **300×300 and mean/std=0.5** (TF-pretrained normalization) — verified from the actual `pretrained_cfg`. The YAML is set correctly; feeding 224×224 with ImageNet stats would have artificially tanked its score.

### Pre-Training Fixes Applied (User Feedback)
Before running training, the following 6 concrete issues were successfully addressed:
1. **Deduplicated Transform Pipeline:** Removed duplicate inline augmentations from `train.py` and `lr_finder.py`. Upgraded `src/data/transforms.py` to accept `mean`/`std` overrides, serving as the single source of truth to prevent drift.
2. **LR Comment Consistency:** Added the `# placeholder — replace with lr_finder result` reminder to all 5 configs.
3. **Robust Checkpoint Verification:** Replaced the dead `dummy` tensor in `smoke_test.py::verify_checkpoint` with an actual forward pass at native `image_size`. It now asserts `out.shape == (1, 8)` to strictly catch head mismatches.
4. **Accuracy-First EfficientNetV2-S Tag:** Verified and swapped the bare tag for `tf_efficientnetv2_s.in21k_ft_in1k` in the config, ensuring maximum top-1 performance (ImageNet-21k pretrain → 1k fine-tune). Tag query verified normalization remains identical at `0.5/0.5/0.5`.
5. **Full Reproducibility:** Added `random.seed(seed)` to `train.py` alongside PyTorch and NumPy seeds, ensuring Albumentations' probability gates (`p=...`) are deterministic across runs.
6. **RepVGG Reparameterization Warning:** Added an explicit warning to `src/models/gradcam.py` documenting that Phase 4 Step 6 (reparam) will change the model's graph structure, meaning `_verify_gradcam_layers.py` must be re-run on the reparameterized model.

### 2. Class weights indexed by CANONICAL_CLASSES (§4.1)
`build_class_weight_tensor()` in `train.py` explicitly indexes `raw_weights[cls] for cls in CANONICAL_CLASSES` — never `.values()`. Verified correct in integration check.

### 3. GradCAM target layers verified empirically (§11)
Instead of assuming `.layer4[-1]` (ResNet-specific), `gradcam.py` uses a path-traversal function that was verified against actual `named_modules()` output for each backbone in timm 1.0.28.

### 4. No external `torch-lr-finder` dependency
`lr_finder.py` implements the range test natively using exponential LR multiplier + bias-corrected loss smoothing. Zero extra installs needed.

---

## How to Run Training

### Step 1 — LR range test (per backbone)
```bash
conda run -n fetalplane python scripts/lr_finder.py --config configs/repvgg_a1.yaml
# Opens logs/repvgg_a1/lr_finder.png — find the steepest descent point
# Update the lr: field in configs/repvgg_a1.yaml with the found value
# Repeat for all 6 configs
```

### Step 2 — Smoke test all backbones
```bash
conda run -n fetalplane python scripts/smoke_test.py
# Should print: All smoke tests PASSED.
```

### Step 3 — Full training runs
```bash
# Run one at a time (or sequentially):
conda run -n fetalplane python -m src.train.train --config configs/repvgg_a1.yaml
conda run -n fetalplane python -m src.train.train --config configs/repvgg_a2.yaml
conda run -n fetalplane python -m src.train.train --config configs/mobilenetv3_large_100.yaml
conda run -n fetalplane python -m src.train.train --config configs/efficientnet_lite0.yaml
conda run -n fetalplane python -m src.train.train --config configs/tf_efficientnetv2_s.yaml
conda run -n fetalplane python -m src.train.train --config configs/convnext_tiny.yaml
```

### Step 4 — Monitor with TensorBoard
```bash
conda run -n fetalplane tensorboard --logdir logs/
```
Compare `MacroF1/val` across all backbones. **Backbone with highest val macro-F1 wins.**

### Step 5 — Fill in EXPERIMENTS.md
After runs complete, fill in the backbone comparison table in [docs/EXPERIMENTS.md](file:///d:/Ultrasound/docs/EXPERIMENTS.md).

---

## Flaw Found and Fixed

The initial `gradcam.py` had incorrect target layer paths using `-1` indices for RepVGG stages (timm uses `.stages`, not `.stage4`), and used `-1` for block groups when explicit indices are required. After running `scripts/_inspect_layers.py`, the correct paths were determined and hardcoded:
- RepVGG: `stages[3][-1]`
- MobileNetV3/EfficientNet-Lite0: `blocks[6][-1]`
- EfficientNetV2-S: `blocks[5][-1]`

All paths now verified to resolve to the correct layer types.
