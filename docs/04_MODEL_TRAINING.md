# 04 — Model Training

`[AGENT]`-heavy phase. Human role: review training curves/metrics at checkpoints, make the final backbone selection call, sanity-check confusion matrices.

---

## Step 1 — Backbone candidates (all available via `timm`)

Train and compare all six before committing to one for the final real-time system:

| Backbone | Role in comparison | timm identifier (verify current exact string) |
|---|---|---|
| RepVGG-A1 or A2 | Primary candidate — trains multi-branch, re-parameterizes to single-branch plain conv net for inference (fastest possible inference graph, cleanest ONNX export) | `repvgg_a1`, `repvgg_a2` |
| MobileNetV3-Large | Standard efficient baseline | `mobilenetv3_large_100` |
| EfficientNet-Lite0 | Standard efficient baseline, alternative family | `efficientnet_lite0` |
| EfficientNetV2-S | Accuracy-ceiling reference (per FAUSP-NET's own benchmark, this or Swin-Tiny scored highest but slowest) | `tf_efficientnetv2_s` |
| ConvNeXt-Tiny | Modern pure-CNN baseline added during investigation | `convnext_tiny` |

Since we're not latency-starved (demo/portfolio deployment target), the decision axis is **accuracy vs. training/inference time on the 4060**, not a hard latency ceiling. If EfficientNetV2-S is meaningfully more accurate and still comfortably real-time on this GPU (which FAUSP-NET's own 24ms/frame number on similar-class hardware suggests it should be), there's no reason to force RepVGG as the final choice purely on the earlier "edge deployment" reasoning — that reasoning applied to a different deployment target than the one we settled on. **Make this call empirically once you have the comparison numbers, not a priori.**

## Step 2 — Pretraining initialization ablation

Run at least the primary backbone candidate under two initializations:
1. Standard ImageNet weights (via `timm(pretrained=True)`)
2. **`[MANUAL] search for a public fetal-ultrasound or general-ultrasound self-supervised checkpoint`** (the literature review found references to a DINOv2-based fetal-US foundation model and SimCLR-pretrained fetal-plane checkpoints — search for a publicly released version; if none is actually downloadable, skip this ablation and note it as unavailable rather than blocking on it)

Compare val accuracy/F1 and training-epochs-to-convergence between the two. Domain-specific pretraining is expected to help data efficiency, not required to work — if it doesn't help or isn't available, fall back to ImageNet init without further debate.

## Step 3 — Loss, optimizer, schedule

- Loss: class-weighted cross-entropy as the default; keep focal loss as a documented alternative to try specifically on the brain sub-plane confusion (Trans-ventricular vs. Trans-thalamic — the literature-documented hardest confusion pair) if the confusion matrix shows it's still a problem after class weighting.
- Optimizer: AdamW, standard starting LR ~1e-3 to 3e-4 for a from-scratch head / fine-tuned backbone combo — tune via a short LR range test rather than guessing.
- Scheduler: cosine annealing or step decay with early stopping on validation macro-F1 (not accuracy — macro-F1 avoids the "Other" and majority classes masking poor minority-class performance, matching the reference repo's own choice of macro-averaged metrics for Stage 2).
- Mixed precision (`torch.cuda.amp`) — use it, the 4060 benefits meaningfully and there's no reason not to.

## Step 4 — Augmentation policy

Standard augmentation for ultrasound classification (via `albumentations`):
- Horizontal flip (probability ~0.5) — note: verify this doesn't create anatomically implausible images for laterality-dependent classes (e.g., if any class distinguishes left/right structures — check FETAL_PLANES_DB's actual class definitions before assuming flip is safe for every class, mirroring the caution the intrapartum dataset paper explicitly noted about flip augmentation changing anatomical plausibility)
- Random rotation (±10°), scaling (0.9–1.1×), translation (~10%)
- Brightness/contrast jitter (±0.2)
- Speckle noise (ultrasound-appropriate, not generic Gaussian noise — multiplicative noise models speckle more faithfully)
- Do NOT apply augmentations that would be inconsistent with the synthetic ego-motion clips built in Phase 3 — keep the "small continuous perturbation" character consistent between static-image training augmentation and the video-jitter simulation, so the model's learned invariances line up with what the smoothing layer will later assume.

## Step 5 — Training loop (chronological)

1. Implement `src/train/train.py` — config-driven (reads a YAML from `configs/`), logs to TensorBoard (loss, accuracy, macro-F1, per-class F1, LR, throughput).
2. Run a short (~5 epoch) smoke test on each of the 6 backbone candidates to confirm the pipeline runs end-to-end and loss decreases, before committing to full training runs.
3. Full training run per backbone candidate (expect convergence within a modest number of epochs given dataset size — monitor val macro-F1 plateau rather than fixing epoch count in advance; use early stopping, patience ~8-10 epochs, matching the reference repo's Stage 1 approach).
4. Save best checkpoint per candidate (by val macro-F1) to `checkpoints/<backbone_name>/best.pt`.
5. Run the pretraining-init ablation (Step 2) on whichever backbone wins Step 5.3.

## Step 6 — RepVGG re-parameterization (if RepVGG is the final choice)

If RepVGG wins the comparison, implement the structural re-parameterization step (`timm` has built-in support for this, or use RepVGG's official reparam utility) that collapses the multi-branch training-time architecture into a single-branch plain conv stack for inference. Verify numerically that re-parameterized outputs match the original training-time model outputs (within floating point tolerance) on a held-out batch before trusting it in the real-time pipeline.

## Step 7 — Grad-CAM module

Implement using the `pytorch-grad-cam` library rather than hand-rolling hooks (the reference repo hand-rolled this — fine for a static pipeline, but we want a maintained library for the real-time throttled version). Target the last convolutional block of the chosen backbone. Verify visually on a batch of validation images that activation maps land on plausible anatomical regions for at least the well-separated classes (femur, cervix) before trusting it on the harder brain sub-planes.

## Deliverables checklist

- [ ] All 6 backbones trained, compared on val macro-F1 and per-class F1
- [ ] Pretraining-init ablation run and documented
- [ ] Final backbone decision made and justified in writing (a short [EXPERIMENTS.md](EXPERIMENTS.md) note is enough)
- [ ] Best checkpoint saved
- [ ] Re-parameterization implemented and numerically verified (if RepVGG chosen)
- [ ] Grad-CAM module implemented and visually spot-checked
