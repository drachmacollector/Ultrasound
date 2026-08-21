# Phase 7 — Goal 1 Walkthrough: Multi-Task Detection + Classification

This document is the authoritative record of all implementation decisions,
architectural choices, bugs found, and fixes applied during the development
of the **detection-informed multi-task head** (Stretch Goal 1 of Phase 7).

It supersedes the earlier placeholder walkthrough and covers the full arc from
initial implementation through all bug-fix cycles.

---

## 1. Architecture Overview

### 1.1 Design Goal

Build a single neural network that simultaneously:
1. **Classifies** the fetal plane (Brain thalamic / cerebellum / etc.)
2. **Detects** the anatomical structure with a bounding box (Head / Abdomen / Femur)

Both tasks share one ConvNeXt backbone, run in one forward pass, so there is zero redundant compute.

### 1.2 Model: `MultiTaskConvNeXt`

**File**: `src/models/multitask_model.py`

```
Input image (B x 3 x 224 x 224)
        |
        v
ConvNeXt-Tiny backbone (timm/convnext_tiny.fb_in22k_ft_in1k)
  |-- stage outputs: {0, 1, 2, 3}  (feature maps at different scales)
  |
  |---> Classification head
  |       |-- Global avg pool on stage-2 features -> Linear(768, 8) -> cls_logits
  |
  +---> FPN (Feature Pyramid Network)
          |-- RetinaNet detection head
                 |-- cls + bbox regression per anchor
```

**Key decision — backbone weights**: We explicitly load `timm/convnext_tiny.fb_in22k_ft_in1k`
(ImageNet-22k pre-trained, then fine-tuned on 1k), **not** vanilla `convnext_tiny`
ImageNet-1k weights. This matters because the Phase 4 production classifier also used the
22k→1k checkpoint (test macro-F1 0.8927). Initialising from the weaker checkpoint would
throw away all of Phase 4's fine-tuning signal.

**Key decision — shared backbone**: The model calls the ConvNeXt backbone **once**, then
routes the feature pyramid both to the classification head (stage 2 only) and to the
FPN+RetinaNet head (all stages). Zero redundant compute.

---

## 2. Data Pipeline

### 2.1 Datasets Used

| Dataset | Role | Notes |
|---|---|---|
| FETAL_PLANES_DB | Classification supervision | Phase 3 train/val split (patient-stratified) |
| HC18 | Detection supervision (Head only) | Independent institution; 999 images |
| UCL | Detection supervision (Head/Abdomen/Femur) | 159/130/135 images |
| FP | EXCLUDED | Re-release of FETAL_PLANES_DB — patient overlap |
| MULTICENTRE | EXCLUDED | Policy violation per phase documentation |

**Why exclude FP and MULTICENTRE?**
FP is a landmark-annotated re-release of ~1,047 FETAL_PLANES_DB patients. Using it for
detection training creates patient overlap with our classification training set, invalidating
all evaluation claims. MULTICENTRE was explicitly designated as held-out cross-device data
across all phase documentation and must never appear in any training split.

### 2.2 Label Masking (`-100`)

HC18 and UCL provide coarse structure labels ("Head", "Abdomen", "Femur") not fine-grained
plane labels like "Brain_Trans_thalamic". We cannot fabricate a fine-grained label from a
coarse annotation.

**Solution**: Map all detection-only annotations to `cls_label = -100`. PyTorch's
`CrossEntropyLoss` ignores samples with `target = -100` by default (`ignore_index=-100`).

- The detection head trains on all HC18/UCL images
- The classification head trains **only** on FETAL_PLANES_DB images
- No coarse label is ever injected into the fine-grained classification distribution

The `-100` masking was propagated through every metric calculation (accuracy, macro-F1) so
that evaluation numbers reflect only the real classification subset.

### 2.3 Bounding Box Derivation

**File**: `scripts/derive_bboxes.py`

HC18/UCL annotation CSVs provide landmark coordinates (OFD/BPD endpoints for head; TAD/APAD
diameters for abdomen; FL endpoints for femur). Bounding boxes are derived by:

1. Computing the enclosing axis-aligned rectangle of all landmark points
2. Expanding by a **20% margin** (relative to structure size) for full anatomical capture
3. Clamping to actual image boundaries using `PIL.Image` dimensions (not assumed constants)

The 20% margin was selected as conservative and interpretable — enough to capture the full
structure edge without cropping into neighbouring anatomy.

### 2.4 Manifest Building

**File**: `scripts/build_multitask_manifest.py`

The manifest combines:
- FETAL_PLANES_DB Phase 3 train/val rows (no bbox)
- HC18/UCL annotated rows (with bbox, `cls_label=-100`)

Final manifests after all fixes:
- `data/splits/multitask_train.csv` — 7,271 rows (1,165 with bbox)
- `data/splits/multitask_val.csv` — 1,281 rows (258 with bbox)

---

## 3. Training Infrastructure

### 3.1 Loss Function

```
L_total = L_cls + w_det * L_det
```

- `L_cls`: `CrossEntropyLoss(weight=class_weights, ignore_index=-100)`
- `L_det`: Sum of RetinaNet's focal classification loss + smooth-L1 box regression loss
- `w_det = 1.0` (configurable via `det_loss_weight` in `configs/multitask.yaml`)

When a batch contains no annotated images with real boxes, `L_det = 0`.

### 3.2 Mixed Precision and Collation

- `torch.amp.autocast('cuda')` + `GradScaler` for FP16 training
- Custom `multitask_collate_fn` stacks images into `(B, 3, H, W)` while keeping detection
  targets as a Python list of dicts (required by RetinaNet's internal transform)

---

## 4. Bugs Found, Root Causes, and Fixes

### Bug 1 — Detection Head Trained on Zero Real Bounding Boxes

**Symptom**: The detection head received no supervision despite HC18/UCL data existing.

**Root cause**: `build_multitask_manifest.py` scanned HC18/UCL CSVs to find images, but the
training script consumed `data/splits/train.csv` — the Phase 3 FETAL_PLANES_DB split. These
two datasets have completely different filename conventions and different institutions. No
`bboxes.json` entries would ever match a FETAL_PLANES_DB filename, so detection loss was
always zero.

**Fix**: Rewrote `build_multitask_manifest.py` to produce dedicated `multitask_train.csv`
and `multitask_val.csv` that union both FETAL_PLANES_DB rows (no bbox) and HC18/UCL rows
(with bbox). Training script updated to consume these.

---

### Bug 2 — Wrong Backbone Weights

**Symptom**: After 1 epoch, macro-F1 was 0.76 with zero recall on `Brain_Trans_ventricular`.

**Root cause**: `MultiTaskConvNeXt` called `timm.create_model('convnext_tiny', pretrained=True)`
— vanilla ImageNet-1k weights. The Phase 4 winning checkpoint used `convnext_tiny.fb_in22k_ft_in1k`.
Starting from the weaker init threw away all of Phase 4's fine-tuning signal.

**Fix**: Changed backbone tag to `timm/convnext_tiny.fb_in22k_ft_in1k`. The `head.fc.*`
keys are intentionally missing from the pretrained model (they are task-specific). This is
expected and logged as `[INFO] Missing keys — expected`.

---

### Bug 3 — `num_det_classes=4` Should Be `3`

**Root cause**: Detection head configured with 4 classes. RetinaNet uses explicit foreground
classes only (no implicit background class). Correct value is 3 (Head, Abdomen, Femur).

**Fix**: Set `num_det_classes=3` in all model instantiation paths and config.

---

### Bug 4 — FP and MULTICENTRE Data in Training Set

**Root cause**: `ANNOTATED_SUBSETS = ["UCL", "HC18", "FP", "MULTICENTRE"]`. FP is a
re-release of a FETAL_PLANES_DB subset — patient overlap with classification training.
MULTICENTRE was designated held-out data in all phase documentation.

**Fix**: `ANNOTATED_SUBSETS = ["UCL", "HC18"]`. Hard assertion added in the leakage verifier.

---

### Bug 5 — Per-Image Hash Split Causes Cross-Structure Patient Leakage

**Symptom 1**: UCL patients could have their Head image in train and Abdomen image in val.

**Symptom 2**: In one test run, Python's `hash()` sent all 1,423 annotated rows to train
(0 to val). This is because `hash()` is randomised by `PYTHONHASHSEED`.

**Investigation**: The `SubjectID`/`Patient` columns exist in CSV headers but are NaN
throughout — the dataset was released with those fields blank. The image filename encodes
the patient as a leading numeric prefix: `002_HC.jpeg`, `002_3HC.jpeg`, `002_AC.jpg`,
`002_FL.jpeg` → all patient `"002"`.

**Fixes**:
1. `_patient_id_from_filename()`: extracts leading digits via `re.match(r'^(\d+)', image_name)`
2. `_is_train_split()`: uses `hashlib.md5` (not `hash()`) for deterministic cross-environment
   reproducibility
3. Split key is the **patient prefix** — all of a patient's structures land in the same split

**Result**: 644 unique patients in train, 162 in val, 0 overlap.

---

### Bug 6 — Vacuous Leakage Checker

**Root cause**: The check filtered `patient_id != -1`. Since every annotated row had
`patient_id = -1` (blank CSV columns), the filter excluded the entire annotated portion.
The assertion `len(overlap) == 0` always passed — it never examined the one place leakage
could actually occur. A passing assertion that cannot see its own failure mode manufactures
false confidence.

**Fixes**:
- **Check 2** (new, runs first): `≥95%` of `has_bbox` rows must have a real `patient_id != -1`
  before the disjointness check proceeds. If the manifest builder fails to populate patient
  IDs, this assertion fails loudly. Current result: 100.0% in both splits.
- **Check 3** scoped to `has_bbox=True` rows only (HC18/UCL annotated subset). FETAL_PLANES_DB
  patients are a separate cohort correctly split in Phase 3 and would produce false positives.

---

### Bug 7 — Evaluation Crash: RetinaNet Anchor Shape Mismatch

**Symptom**: Training epoch completed, then crash at validation:
```
IndexError: too many indices for tensor of dimension 1
  widths = boxes[:, 2] - boxes[:, 0]
```

**Root cause**: In `multitask_model.py` eval mode, we called:
```python
detections = self.retinanet.postprocess_detections(head_outputs, anchors, ...)
```

`postprocess_detections` expects anchors as a **list of per-level lists** (split by FPN
level), but `anchor_generator` returns a flat concatenated tensor per image. In training mode,
`compute_loss` accepts the flat format. In eval mode, the postprocessor tries to index into
individual level anchor tensors treated as 2D boxes, hitting the 1D flat tensor instead.

**Root cause traced by**: Reading `torchvision.RetinaNet.forward()` source, which shows:
```python
num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
A = HWA // HW  # anchors per spatial location
split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]
detections = self.postprocess_detections(split_head_outputs, split_anchors, ...)
```

**Fix**: Replicated the exact same split logic in `multitask_model.py`'s eval branch.

**Verification**: Isolated `model.eval()` forward pass on random `(2, 3, 224, 224)` input
→ `Success torch.Size([2, 8])`.

---

### Bug 8 — Validation Loss NaN

**Root cause**: Validation loop called `criterion(cls_logits, labels)` even for batches
containing only HC18/UCL images (all labels = -100). With `ignore_index=-100`, if all targets
are masked, `CrossEntropyLoss` returns `NaN` (no valid samples to average over).

**Fix**: Added `valid_mask` guard in the validation loop, returning `tensor(0.0)` when the
entire batch is detection-only.

---

### Bug 9 — `FocalLoss` Crash on `-100` Labels (Dormant Landmine)

**Root cause**: `FocalLoss.forward()` called `.gather(dim=1, index=targets.unsqueeze(1))`
without filtering `-100`. Passing `target = -100` into `.gather()` either crashes or silently
accesses out-of-bounds memory. Currently dormant (config uses `CrossEntropyLoss`) but would
crash any focal-loss ablation on the multitask line.

**Fix**: Added `ignore_index=-100` parameter; validity mask applied before any indexing:
```python
valid_mask = targets != self.ignore_index
if not valid_mask.any():
    return logits.sum() * 0.0  # preserve autograd graph

logits  = logits[valid_mask]
targets = targets[valid_mask]
```

---

### Bug 10 — `float(loss)` Holds Autograd Graph (Memory Leak Warning)

**Root cause**: Training loop accumulated losses via `float(loss)` on tensors with
`requires_grad=True`. PyTorch raises a `UserWarning` and may prevent garbage collection of
the computation graph, causing steadily increasing GPU memory across batches.
### 2.4 Manifest Building

**File**: `scripts/build_multitask_manifest.py`

The manifest combines:
- FETAL_PLANES_DB Phase 3 train/val rows (no bbox)
- HC18/UCL annotated rows (with bbox, `cls_label=-100`)

Final manifests after all fixes:
- `data/splits/multitask_train.csv` — 7,271 rows (1,165 with bbox)
- `data/splits/multitask_val.csv` — 1,281 rows (258 with bbox)

---

## 3. Training Infrastructure

### 3.1 Loss Function

```
L_total = L_cls + w_det * L_det
```

- `L_cls`: `CrossEntropyLoss(weight=class_weights, ignore_index=-100)`
- `L_det`: Sum of RetinaNet's focal classification loss + smooth-L1 box regression loss
- `w_det = 1.0` (configurable via `det_loss_weight` in `configs/multitask.yaml`)

When a batch contains no annotated images with real boxes, `L_det = 0`.

### 3.2 Mixed Precision and Collation

- `torch.amp.autocast('cuda')` + `GradScaler` for FP16 training
- Custom `multitask_collate_fn` stacks images into `(B, 3, H, W)` while keeping detection
  targets as a Python list of dicts (required by RetinaNet's internal transform)

---

## 4. Bugs Found, Root Causes, and Fixes

### Bug 1 — Detection Head Trained on Zero Real Bounding Boxes

**Symptom**: The detection head received no supervision despite HC18/UCL data existing.

**Root cause**: `build_multitask_manifest.py` scanned HC18/UCL CSVs to find images, but the
training script consumed `data/splits/train.csv` — the Phase 3 FETAL_PLANES_DB split. These
two datasets have completely different filename conventions and different institutions. No
`bboxes.json` entries would ever match a FETAL_PLANES_DB filename, so detection loss was
always zero.

**Fix**: Rewrote `build_multitask_manifest.py` to produce dedicated `multitask_train.csv`
and `multitask_val.csv` that union both FETAL_PLANES_DB rows (no bbox) and HC18/UCL rows
(with bbox). Training script updated to consume these.

---

### Bug 2 — Wrong Backbone Weights

**Symptom**: After 1 epoch, macro-F1 was 0.76 with zero recall on `Brain_Trans_ventricular`.

**Root cause**: `MultiTaskConvNeXt` called `timm.create_model('convnext_tiny', pretrained=True)`
— vanilla ImageNet-1k weights. The Phase 4 winning checkpoint used `convnext_tiny.fb_in22k_ft_in1k`.
Starting from the weaker init threw away all of Phase 4's fine-tuning signal.

**Fix**: Changed backbone tag to `timm/convnext_tiny.fb_in22k_ft_in1k`. The `head.fc.*`
keys are intentionally missing from the pretrained model (they are task-specific). This is
expected and logged as `[INFO] Missing keys — expected`.

---

### Bug 3 — `num_det_classes=4` Should Be `3`

**Root cause**: Detection head configured with 4 classes. RetinaNet uses explicit foreground
classes only (no implicit background class). Correct value is 3 (Head, Abdomen, Femur).

**Fix**: Set `num_det_classes=3` in all model instantiation paths and config.

---

### Bug 4 — FP and MULTICENTRE Data in Training Set

**Root cause**: `ANNOTATED_SUBSETS = ["UCL", "HC18", "FP", "MULTICENTRE"]`. FP is a
re-release of a FETAL_PLANES_DB subset — patient overlap with classification training.
MULTICENTRE was designated held-out data in all phase documentation.

**Fix**: `ANNOTATED_SUBSETS = ["UCL", "HC18"]`. Hard assertion added in the leakage verifier.

---

### Bug 5 — Per-Image Hash Split Causes Cross-Structure Patient Leakage

**Symptom 1**: UCL patients could have their Head image in train and Abdomen image in val.

**Symptom 2**: In one test run, Python's `hash()` sent all 1,423 annotated rows to train
(0 to val). This is because `hash()` is randomised by `PYTHONHASHSEED`.

**Investigation**: The `SubjectID`/`Patient` columns exist in CSV headers but are NaN
throughout — the dataset was released with those fields blank. The image filename encodes
the patient as a leading numeric prefix: `002_HC.jpeg`, `002_3HC.jpeg`, `002_AC.jpg`,
`002_FL.jpeg` → all patient `"002"`.

**Fixes**:
1. `_patient_id_from_filename()`: extracts leading digits via `re.match(r'^(\d+)', image_name)`
2. `_is_train_split()`: uses `hashlib.md5` (not `hash()`) for deterministic cross-environment
   reproducibility
3. Split key is the **patient prefix** — all of a patient's structures land in the same split

**Result**: 644 unique patients in train, 162 in val, 0 overlap.

---

### Bug 6 — Vacuous Leakage Checker

**Root cause**: The check filtered `patient_id != -1`. Since every annotated row had
`patient_id = -1` (blank CSV columns), the filter excluded the entire annotated portion.
The assertion `len(overlap) == 0` always passed — it never examined the one place leakage
could actually occur. A passing assertion that cannot see its own failure mode manufactures
false confidence.

**Fixes**:
- **Check 2** (new, runs first): `≥95%` of `has_bbox` rows must have a real `patient_id != -1`
  before the disjointness check proceeds. If the manifest builder fails to populate patient
  IDs, this assertion fails loudly. Current result: 100.0% in both splits.
- **Check 3** scoped to `has_bbox=True` rows only (HC18/UCL annotated subset). FETAL_PLANES_DB
  patients are a separate cohort correctly split in Phase 3 and would produce false positives.

---

### Bug 7 — Evaluation Crash: RetinaNet Anchor Shape Mismatch

**Symptom**: Training epoch completed, then crash at validation:
```
IndexError: too many indices for tensor of dimension 1
  widths = boxes[:, 2] - boxes[:, 0]
```

**Root cause**: In `multitask_model.py` eval mode, we called:
```python
detections = self.retinanet.postprocess_detections(head_outputs, anchors, ...)
```

`postprocess_detections` expects anchors as a **list of per-level lists** (split by FPN
level), but `anchor_generator` returns a flat concatenated tensor per image. In training mode,
`compute_loss` accepts the flat format. In eval mode, the postprocessor tries to index into
individual level anchor tensors treated as 2D boxes, hitting the 1D flat tensor instead.

**Root cause traced by**: Reading `torchvision.RetinaNet.forward()` source, which shows:
```python
num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
A = HWA // HW  # anchors per spatial location
split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]
detections = self.postprocess_detections(split_head_outputs, split_anchors, ...)
```

**Fix**: Replicated the exact same split logic in `multitask_model.py`'s eval branch.

**Verification**: Isolated `model.eval()` forward pass on random `(2, 3, 224, 224)` input
→ `Success torch.Size([2, 8])`.

---

### Bug 8 — Validation Loss NaN

**Root cause**: Validation loop called `criterion(cls_logits, labels)` even for batches
containing only HC18/UCL images (all labels = -100). With `ignore_index=-100`, if all targets
are masked, `CrossEntropyLoss` returns `NaN` (no valid samples to average over).

**Fix**: Added `valid_mask` guard in the validation loop, returning `tensor(0.0)` when the
entire batch is detection-only.

---

### Bug 9 — `FocalLoss` Crash on `-100` Labels (Dormant Landmine)

**Root cause**: `FocalLoss.forward()` called `.gather(dim=1, index=targets.unsqueeze(1))`
without filtering `-100`. Passing `target = -100` into `.gather()` either crashes or silently
accesses out-of-bounds memory. Currently dormant (config uses `CrossEntropyLoss`) but would
crash any focal-loss ablation on the multitask line.

**Fix**: Added `ignore_index=-100` parameter; validity mask applied before any indexing:
```python
valid_mask = targets != self.ignore_index
if not valid_mask.any():
    return logits.sum() * 0.0  # preserve autograd graph

logits  = logits[valid_mask]
targets = targets[valid_mask]
```

---

### Bug 10 — `float(loss)` Holds Autograd Graph (Memory Leak Warning)

**Root cause**: Training loop accumulated losses via `float(loss)` on tensors with
`requires_grad=True`. PyTorch raises a `UserWarning` and may prevent garbage collection of
the computation graph, causing steadily increasing GPU memory across batches.

**Fix**: Replaced with `.detach().item()`:
```python
total_loss     += loss.detach().item()     * batch_size
total_cls_loss += cls_loss.detach().item() * batch_size
total_det_loss += det_loss.detach().item() * batch_size
```

---

### Bug 11 — NaN Validation Loss During Checkpointing

**Symptom**: During training, the validation loss was `NaN`, but the checkpoint was still saved because the selection criteria (`val_macro_f1`) didn't check for it.

**Root cause**:
1. `det_loss_weight` was a flat constant with no warm-up. RetinaNet's loss dwarfed the classification loss early on, leading to instability.
2. `CrossEntropyLoss` didn't have an explicit `ignore_index=-100` passed during instantiation.
3. No gradient clipping or NaN/Inf checks on the gradients before the optimizer step.

**Fix**:
- Passed `ignore_index=-100` explicitly to `CrossEntropyLoss`.
- Added gradient clipping (`max_norm=5.0`) and a guard to skip the optimizer step if the gradient norm is not finite.
- Implemented linear warm-up for `det_loss_weight` over `det_loss_warmup_steps`.
- Added a hard crash on `NaN` validation loss during the best checkpoint saving block.

---

### Bug 12 — Cross-Device Evaluation Allowed Contaminated Models

**Symptom**: Running `scripts/evaluate_cross_device.py` against the multitask checkpoint succeeded without any warnings, despite HC18/UCL being part of the training set.

**Root cause**: The warning that multitask models shouldn't be evaluated on the cross-device dataset was only documented in markdown files, not enforced in code.

**Fix**: Added `_assert_not_multitask_contaminated(ckpt_path)` to `scripts/evaluate_cross_device.py` that checks the config's `train_csv` inside the checkpoint and raises a `RuntimeError` if the multitask manifest was used.

---

### Bug 13 — `has_bbox` vs True Annotation Verification

**Symptom**: In `build_multitask_manifest.py`, all rows from HC18/UCL were blindly flagged with `has_bbox=True`, even if `derive_bbox` failed (e.g., due to missing landmarks). These rows contributed no signal but consumed compute.

**Fix**: Renamed the column to `is_annotated_subset` and introduced a genuine `has_valid_bbox` column that is set to `True` only if a bounding box was successfully extracted and saved. Both are printed at manifest build time.

---

### Bug 14 — Hardcoded `min_size` / `max_size` in RetinaNet

**Symptom**: `MultiTaskConvNeXt` hardcoded `224` into RetinaNet's `min_size` and `max_size`.

**Root cause**: While harmless when the global `image_size` config is 224, it becomes a silent landmine if the config changes (e.g., to 300), causing RetinaNet to silently re-resize images and distort box coordinates.

**Fix**: Parametrized `img_size` in the `MultiTaskConvNeXt` constructor and passed it from `cfg.get("image_size")` in the training script.

---

## 5. Verification Results

After all fixes, 1-epoch smoke test completed successfully:

```
Epoch 1 TRAIN | loss=0.7785 (cls=0.5560, det=0.2225) | acc=0.7922 | 26 img/s
               det_pos_rate=0.746 (677/908 batches had real boxes)
Epoch 1 VAL   | acc=0.8847 | macro_f1=0.8421
  --> New best checkpoint saved: checkpoints/multitask/best.pt
```

**Key observations**:
- `det_pos_rate=0.746`: 74.6% of training batches had at least one real bounding box,
  confirming HC18/UCL data flows correctly into the detection head.
- `macro_f1=0.8421` after 1 epoch with the strong 22k init — comparable to the Phase 4
  fine-tuned classifier, confirming the `fb_in22k_ft_in1k` initialisation carries over.

### Leakage Verification Output

```
[OK] Check 1: No FP/MULTICENTRE images in either split.
[OK] Check 2 [Train]: 100.0% of annotated rows have a real patient_id.
[OK] Check 2 [Val]:   100.0% of annotated rows have a real patient_id.
[OK] Check 3: 0 overlapping annotated patients (644 train | 162 val).

All checks passed.
  Train: 7,271 rows (1,165 with bbox)
  Val:   1,281 rows (258 with bbox)
```

---

## 6. Known Limitations and Evaluation Boundaries

> **CRITICAL**: HC18/UCL are no longer held-out data. The multitask checkpoint
> (`checkpoints/multitask/best.pt`) must **never** be evaluated against
> `cross_device_manifest.csv` as a genuine cross-device generalisation test.
> Any such comparison is data snooping.

**HC18 is Head-only**: No Abdomen or Femur CSVs exist for HC18. Abdomen and Femur
detection relies entirely on UCL (130 + 135 images). Future work should source additional
institutions for these structures.

**SubjectID/Patient columns are blank**: Patient grouping for the train/val split relies on
the leading numeric prefix in the image filename. This works for all verified UCL/HC18
images. Zero fallbacks were logged in the final build run.

---

## 7. Files Modified Summary

| File | Change |
|---|---|
| `src/models/multitask_model.py` | Architecture; eval-mode anchor split fix; parameterized img_size |
| `scripts/build_multitask_manifest.py` | Patient-level split via filename prefix + hashlib.md5; FP/MULTICENTRE excluded; -100 cls labels; distinct has_valid_bbox vs is_annotated_subset columns |
| `scripts/verify_no_multitask_leakage.py` | Non-vacuous Check 2 (patient-ID coverage >= 95%); Check 3 scoped to is_annotated_subset rows |
| `scripts/train_multitask.py` | -100 masking in metrics; val NaN guard; .detach().item() loss accumulation; NaN val_loss guard; explicit ignore_index; gradient clipping; loss warm-up |
| `scripts/evaluate_cross_device.py` | Hard code-level guard against multitask checkpoint contamination |
| `scripts/derive_bboxes.py` | Removed dangerous standalone main() block; import-only module |
| `src/train/losses.py` | FocalLoss: ignore_index=-100 mask before .gather() |
| `configs/multitask.yaml` | num_det_classes: 3; det_loss_weight: 1.0; correct backbone tag |
| `docs/EVAL_REPORT.md` | Warning: multitask checkpoint invalidates HC18/UCL as held-out |
| `docs/EXPERIMENTS.md` | Same evaluation boundary constraint |
