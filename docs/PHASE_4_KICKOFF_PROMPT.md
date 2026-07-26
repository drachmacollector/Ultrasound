# Phase 4 Kickoff Prompt — Model Training

Phase 3 is complete and verified (12,400-row manifest, patient-disjoint
train/val/test splits, `class_weights.json`, `cross_device_manifest.csv`,
synthetic clips). This prompt assumes all of that already exists on disk
exactly as described in [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md).
the created files are within /scripts /src/data/

---

## 0. Non-negotiable corrections to the plan doc — read this before anything else

Three things below aren't in [04_MODEL_TRAINING.md](04_MODEL_TRAINING.md) verbatim — they come from
things I found by actually reading your Phase 3 code and by researching
current library/checkpoint state. Don't skip them.

### 0.1 — This is 8-way single-softmax, not the reference repo's 2-stage cascade
`inference.py` (the original student reference) uses a binary Stage-1 model +
7-class Stage-2 model. **This project does not.** Per
[00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md) §4 and confirmed by `src/data/dataset.py`'s own
`CANONICAL_CLASSES`, this is **one 8-way softmax head** over:

```python
CANONICAL_CLASSES = [
    "Brain_Trans_cerebellum",   # 0
    "Brain_Trans_thalamic",     # 1
    "Brain_Trans_ventricular",  # 2
    "Fetal_abdomen",            # 3
    "Fetal_femur",              # 4
    "Fetal_thorax",             # 5
    "Maternal_cervix",          # 6
    "Other",                    # 7
]
```

`num_classes=8` everywhere. Import `CANONICAL_CLASSES`, `CLASS_TO_IDX`,
`IDX_TO_CLASS`, `NUM_CLASSES` from `src/data/dataset.py` — do not redefine
this list a second time anywhere in the training/eval code. That file's own
docstring calls it "the single source of truth"; treat it as such.

### 0.2 — User directive: accuracy is prioritized over latency, explicitly and repeatedly
This project is not latency-starved ([00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md) §3, confirmed
again by the project owner directly). When Step 9 (backbone decision) comes
around: **weight val macro-F1 as the primary decision criterion.**
Training/inference speed on the 4060 is a tie-breaker, or a reason to
document a limitation, but is **not** a reason to pick a faster/smaller
backbone over a more accurate one unless the more accurate one is so slow
that it meaningfully blocks *iteration* (e.g., a single epoch taking hours).
Do not default to RepVGG or MobileNetV3 "because they're the primary
candidate" if EfficientNetV2-S beats them on macro-F1 and is still trainable
in reasonable time on 8GB VRAM — [04_MODEL_TRAINING.md](04_MODEL_TRAINING.md) §1 already says this
explicitly, this note is just reinforcing it since it's the single most
important call in this phase.

### 0.3 — Per-backbone resolution and normalization must be verified, not assumed
Your current `src/data/transforms.py` hardcodes `IMG_SIZE=224` and
ImageNet mean/std for every backbone. **This is not automatically correct for
all four candidates.** Confirmed by checking timm's own model source: the
`tf_efficientnetv2_*` family ships pretrained configs where `input_size` and
`test_input_size` are **not** 224 — some variants use `(3,384,384)` train /
`(3,480,480)` test resolution, with `crop_pct=1.0` and `crop_mode='squash'`.
Feeding 224×224 into a backbone whose pretrained weights expect ~384–480 can
artificially and unfairly tank that backbone's score in your comparison —
which would be a real bug in the ablation, not a genuine finding about the
architecture.

**Required action before any training run:** for each of the four
candidates, run:

```python
import timm
m = timm.create_model("tf_efficientnetv2_s", pretrained=True)
print(m.pretrained_cfg)   # inspect input_size, mean, std, crop_pct
```

and repeat for `repvgg_a1`, `mobilenetv3_large_100`, `efficientnet_lite0`.
Record all four `pretrained_cfg`s in [EXPERIMENTS.md](EXPERIMENTS.md) (see §11 below).
**Decision to make and document, not guess:** either (a) standardize all four
at 224×224 with ImageNet mean/std for a clean apples-to-apples comparison,
explicitly noting this may under-represent EfficientNetV2-S's true ceiling,
or (b) run each backbone at its own native pretrained resolution/normalization
for the comparison, then separately re-confirm the winner still trains fine
at whatever resolution the real-time pipeline (Phase 5) will actually use.
Given the accuracy-first directive in §0.2, **(b) is the more defensible
choice** — pick it unless VRAM or time genuinely doesn't allow it, and say so
either way in the write-up.

### 0.4 — A minor inconsistency in the planning docs, flagged so you don't chase a ghost
[04_MODEL_TRAINING.md](04_MODEL_TRAINING.md) §5.3 says early-stopping patience "~8-10 epochs,
matching the reference repo's Stage 1 approach" — but the actual reference
[README.md](README.md) documents Stage 1's real patience as **4** epochs. These two
docs disagree with each other; it's not something you did wrong. Since the
architecture is different anyway (8-way vs. 2-stage), there's no strong
reason to match either number exactly — use 8-10 as [04_MODEL_TRAINING.md](04_MODEL_TRAINING.md)
recommends (more patience is safer for a from-scratch comparison across
4 unfamiliar backbones with class-weighted loss, which can have noisier
validation curves early on), just don't be confused if you go looking for
where "4" came from and can't find it consistently.

---

## 1. Repository state check (do this first, report back before writing code)

**IMPORTANT** run them after running
 "conda activate fetalplane" as all packages have been installed in the conda environment not the main environment.  

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
ls data/splits/          # train.csv, val.csv, test.csv
ls data/processed/       # manifest.csv, class_weights.json, cross_device_manifest.csv
python -c "import timm; print(timm.__version__)"
python -c "import timm; print([m for m in timm.list_models('repvgg*')])"
python -c "import timm; print([m for m in timm.list_models('*efficientnetv2_s*')])"
python -c "import timm; print([m for m in timm.list_models('mobilenetv3_large*')])"
python -c "import timm; print([m for m in timm.list_models('efficientnet_lite0')])"
```
Confirm every one of the four target identifiers actually resolves in your
installed `timm` version before writing any training code — timm renames or
deprecates model strings between versions, and [04_MODEL_TRAINING.md](04_MODEL_TRAINING.md) itself
warned "verify current exact string" rather than hardcoding blindly.

---

## 2. Build `src/models/backbone.py`

A single generic wrapper, not four separate model files:

```python
import timm
import torch.nn as nn

def build_model(backbone_name: str, num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    """
    backbone_name: one of 'repvgg_a1', 'repvgg_a2', 'mobilenetv3_large_100',
                   'efficientnet_lite0', 'tf_efficientnetv2_s'
    """
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
    return model

def get_pretrained_cfg(backbone_name: str) -> dict:
    """Returns input_size, mean, std, crop_pct etc. for the given backbone —
    use this to build a per-backbone-correct eval transform, per §0.3 above."""
    m = timm.create_model(backbone_name, pretrained=True)
    return m.pretrained_cfg
```

`timm.create_model(..., num_classes=8)` already swaps the final classifier
head for you — no manual `model.fc = nn.Linear(...)` needed the way
`inference.py`'s hand-rolled `build_stage1_model()`/`build_stage2_model()`
did. That hand-rolled pattern was fine for a 2-model static repo; don't
carry it forward into a 4-backbone comparison, it doesn't generalize across
architectures (RepVGG/MobileNetV3/EfficientNet don't all expose a `.fc`
attribute the same way ResNet does).

---

## 3. Config schema — `configs/<backbone_name>.yaml`

One YAML per backbone candidate, e.g. `configs/repvgg_a1.yaml`:

```yaml
backbone: repvgg_a1
num_classes: 8
pretrained: true
image_size: 224          # or per-backbone native size, per §0.3 decision
normalize_mean: [0.485, 0.456, 0.406]
normalize_std: [0.229, 0.224, 0.225]

train_csv: data/splits/train.csv
val_csv: data/splits/val.csv
class_weights_json: data/processed/class_weights.json

batch_size: 32            # tune per backbone/VRAM; RepVGG/MobileNet can likely go higher
num_workers: 4
epochs_smoke_test: 5
epochs_max: 60
early_stopping_patience: 10
early_stopping_metric: val_macro_f1

optimizer: adamw
lr: 3e-4                  # placeholder — replace with LR range test result, see §5
weight_decay: 0.01
scheduler: cosine         # or step
amp: true

log_dir: logs/repvgg_a1
checkpoint_dir: checkpoints/repvgg_a1
seed: 42
```

---

## 4. Build `src/train/train.py`

Key implementation details, in order of "easy to get subtly wrong":

### 4.1 Class weights — must be indexed explicitly, never rely on dict order
`class_weights.json` is a flat dict of `{class_name: weight}`. Do **not**
call `.values()` on it and assume the order matches `CANONICAL_CLASSES`
index order — that's an implicit assumption riding on Python dict insertion
order, which is fragile if the json is ever regenerated by different code.
Index explicitly:

```python
import json, torch
from src.data.dataset import CANONICAL_CLASSES

with open(config["class_weights_json"]) as f:
    raw_weights = json.load(f)

weight_tensor = torch.tensor(
    [raw_weights[cls] for cls in CANONICAL_CLASSES], dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weight_tensor)
```

### 4.2 Data loading
Use `FocalPlanesDataset` and `get_train_transform()` / `get_eval_transform()`
exactly as already implemented in `src/data/dataset.py` and
`src/data/transforms.py` — these are done, tested, and match the class-index
ordering above. Do not rewrite a parallel Dataset class. If §0.3 leads you to
use a non-224 resolution for a specific backbone, parameterize
`get_train_transform(img_size=...)` / `get_eval_transform(img_size=...)` per
config rather than hardcoding — both functions already accept `img_size` as
an argument, so this is a config wiring change only, not a rewrite.

### 4.3 Optimizer / scheduler / AMP
- AdamW, weight decay ~0.01 as a starting point.
- Cosine annealing (`torch.optim.lr_scheduler.CosineAnnealingLR`) or step
  decay — either is fine, document which you picked and why in
  [EXPERIMENTS.md](EXPERIMENTS.md).
- `torch.cuda.amp.autocast()` + `GradScaler` for mixed precision — the 4060
  benefits meaningfully from this, use it for every run, not just the
  "final" one.

### 4.4 Metrics + early stopping
- Track **macro-F1** (`sklearn.metrics.f1_score(..., average='macro')`) as
  the early-stopping and model-selection metric — not accuracy. This
  matches both the reference repo's own convention and
  [04_MODEL_TRAINING.md](04_MODEL_TRAINING.md) §3's explicit reasoning (accuracy lets "Other" and
  majority classes mask minority-class failure).
- Also log per-class F1, precision, recall, and a confusion matrix every N
  epochs (or at minimum: final epoch + best-checkpoint epoch).
- Log to TensorBoard: loss, accuracy, macro-F1, per-class F1, LR, images/sec
  throughput — per [01_ENVIRONMENT_SETUP.md](01_ENVIRONMENT_SETUP.md)'s TensorBoard decision, already
  locked in, don't re-litigate W&B here.

### 4.5 Checkpointing
Save best-by-val-macro-F1 to `checkpoints/<backbone_name>/best.pt`. Save the
full training config (the YAML) alongside it so a later phase can reload
without guessing what hyperparameters produced it.

---

## 5. LR range test (Leslie Smith-style) — run before committing to a fixed LR

Don't guess a learning rate. For each backbone, run a short LR range test
(exponentially increasing LR from ~1e-7 to ~1, single epoch or partial
epoch, plot loss vs. LR, pick the LR where loss is descending fastest just
before it diverges). `torch-lr-finder` is a small, well-known package that
does this if you'd rather not hand-roll it — either is fine, document which
you used.

---

## 6. Smoke test — all 4 backbones, ~5 epochs, before any full run

Confirm for each backbone: pipeline runs end-to-end, loss decreases,
no shape/dtype errors, checkpoint saves and reloads correctly, TensorBoard
logs appear. **Do not start a full run on any backbone until all four pass
this.** This is cheap insurance against burning hours on a backbone that
has, e.g., a normalization mismatch from §0.3 that would only show up as
"suspiciously bad accuracy" after a full run otherwise.

---

## 7. Full training runs, all 4 backbones

Early stopping on val macro-F1, patience 8-10 epochs (§0.4). Save best
checkpoint per backbone. Produce, per backbone:
- Final val macro-F1, per-class F1
- Confusion matrix (save as image, `data/processed/sanity_checks/` or a new
  `checkpoints/<backbone>/` location — your call, just be consistent)
- Training-time-to-convergence and approximate GPU-hours, for the record
  (informational, not the deciding factor per §0.2)

---

## 8. Backbone decision — make it empirically, write it down

Compare all four on val macro-F1 (primary) and per-class F1 for the brain
sub-plane cluster specifically (Trans-ventricular vs. Trans-thalamic is the
literature-documented hardest pair — expect it to remain the weakest
category regardless of backbone, per your own README's Stage 2 numbers
where Trans-ventricular already scored lowest F1 at 0.759).

Per §0.2: **pick the backbone with the best val macro-F1**, full stop,
unless there's a genuinely blocking training-time problem. If
EfficientNetV2-S wins, that's a fine outcome — don't force RepVGG to "win"
because it was labeled "primary candidate" in the original plan; that label
reflected an edge-deployment assumption this project explicitly rejected in
[00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md) §3.

---

## 9. Pretraining-init ablation — concrete candidates found via research

[04_MODEL_TRAINING.md](04_MODEL_TRAINING.md) §2 left this open-ended ("search for a public
fetal-ultrasound or general-ultrasound SSL checkpoint"). Here's what
actually exists publicly as of mid-2026, so you don't burn time
rediscovering this:

| Checkpoint | Encoder type | Usable for your CNN backbones? |
|---|---|---|
| **FUSC** (BioMedIA-MBZUAI/FUSC, SimCLR-pretrained on fetal ultrasound 2nd-trimester scans) | CNN-based, SimCLR pretext | **Yes — this is your best candidate.** Check the repo for a downloadable checkpoint; if the released weights are for a ResNet-family encoder, they may be adaptable to your winning backbone's stem, or at minimum usable as a like-for-like comparison point if your winner happens to be ResNet-compatible. |
| USFM (UltraSound Foundation Model), MOFO, UltraSAM, FetalCLIP | All ViT-based encoders | **No, not without extra work.** These use Vision Transformer backbones — dropping a ViT checkpoint into a RepVGG/MobileNetV3/EfficientNet training loop isn't a weight-swap, it's a different architecture and out of scope for this ablation. Don't chase these unless you're open to adding a 5th, ViT-based candidate as its own separate experiment (arguably interesting as a stretch goal, but not part of the Step 2 ablation as scoped). |

A recent systematic review on fetal-plane classification (arXiv 2601.00990)
found self-supervised US-domain pretraining conferred a median accuracy gain
of ~3.1 percentage points (IQR 1.8–4.9) over ImageNet-pretrained baselines
on 6-class fetal plane tasks — useful as a sanity-check magnitude: if your
ablation shows a much bigger or non-existent gap, that's worth double
checking rather than taking at face value.

**If FUSC's checkpoint isn't practically portable to your winning backbone**
(likely, since FUSC's own encoder choice may not match your winner),
document that explicitly and fall back to ImageNet init only — per
[04_MODEL_TRAINING.md](04_MODEL_TRAINING.md)'s own instruction, this ablation is expected to help,
not required to work.

---

## 10. RepVGG re-parameterization (only if RepVGG wins §8)

If — and only if — RepVGG wins the backbone comparison: implement the
structural reparam step (timm has built-in support via
`model.reparameterize()` on RepVGG models, or use the official RepVGG
reparam utility) that collapses the multi-branch training-time graph into a
single-branch plain conv stack. **Verify numerically** — run the same
held-out batch through both the pre-reparam and post-reparam model and
confirm outputs match within float tolerance (e.g. `torch.allclose(..., atol=1e-4)`)
before trusting the reparam model anywhere downstream.

---

## 11. Grad-CAM module — use the library, current API confirmed

[04_MODEL_TRAINING.md](04_MODEL_TRAINING.md) §7 correctly says: use `pytorch-grad-cam`
(`pip install grad-cam`), don't hand-roll hooks the way `inference.py` did.
Confirmed current API (v1.5.5, `jacobgil/pytorch-grad-cam`):

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layers = [model.layer4[-1]]   # or the equivalent final conv block
                                       # for your winning backbone — check
                                       # the model's named_modules() output,
                                       # this won't literally be "layer4" for
                                       # RepVGG/MobileNetV3/EfficientNet

with GradCAM(model=model, target_layers=target_layers) as cam:
    targets = [ClassifierOutputTarget(predicted_class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

visualization = show_cam_on_image(original_rgb_float_image, grayscale_cam, use_rgb=True)
```

Note: newer versions of this library dropped the `use_cuda=` constructor
argument in favor of just moving the model to the right device yourself —
don't copy older tutorial code that still passes `use_cuda=True`.

**Identifying the right `target_layers` per backbone is not optional busywork** —
each architecture's final spatial conv block has a different attribute path.
Print `model.named_modules()` for each winning-candidate backbone and pick
the last block before global pooling, don't assume `.layer4[-1]` works
universally (it's a ResNet-specific naming convention that `inference.py`
happened to rely on).

Visually spot-check on a validation batch: confirm activation lands on
plausible anatomical regions for at least the well-separated classes (femur,
cervix) before trusting it on the harder brain sub-planes, per
[04_MODEL_TRAINING.md](04_MODEL_TRAINING.md) §7.

---

## 12. Focal loss ablation — conditional, not automatic

Only run this if, after class-weighted CE on the winning backbone, the
confusion matrix still shows Trans-ventricular ↔ Trans-thalamic confusion as
a live problem (which, per the reference repo's own numbers, is likely).
Compare focal loss vs. class-weighted CE specifically on brain sub-plane
F1 — not a full re-run of the 4-backbone comparison, just this one
targeted ablation on the winner.

---

## 13. Write [EXPERIMENTS.md](EXPERIMENTS.md)

Should include, at minimum:
- Per-backbone `pretrained_cfg` findings from §0.3, and which resolution
  policy you chose and why
- Backbone comparison table (val macro-F1, per-class F1, brain sub-plane F1,
  rough training time) — all four candidates, not just the winner
- Final backbone decision + explicit reasoning tied to the accuracy-priority
  directive
- Pretraining-init ablation result (or explicit "FUSC checkpoint not
  portable, skipped" note)
- RepVGG reparam verification result, if applicable
- Focal-loss-vs-CE result, if that ablation was triggered
- Grad-CAM spot-check notes (which classes looked anatomically plausible,
  which didn't)

---

## Deliverables checklist (from [04_MODEL_TRAINING.md](04_MODEL_TRAINING.md), reproduced for tracking)

- [ ] All 4 backbones trained, compared on val macro-F1 and per-class F1
- [ ] Per-backbone `pretrained_cfg` verified (resolution/normalization), policy documented
- [ ] Pretraining-init ablation run and documented (FUSC checked for portability)
- [ ] Final backbone decision made and justified in writing, weighted toward accuracy per user directive
- [ ] Best checkpoint saved
- [ ] Re-parameterization implemented and numerically verified (if RepVGG chosen)
- [ ] Grad-CAM module implemented via `pytorch-grad-cam`, correct target_layers identified per architecture, visually spot-checked
- [ ] Focal loss ablation run only if confusion matrix warrants it
- [ ] [EXPERIMENTS.md](EXPERIMENTS.md) written
