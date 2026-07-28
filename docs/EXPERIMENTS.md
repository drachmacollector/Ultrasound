# EXPERIMENTS.md — Phase 4 Model Training

This file is the living record of all backbone comparison experiments, ablations,
and design decisions made in Phase 4. Cross-referenced with
[PHASE_4_KICKOFF_PROMPT.md](docs/PHASE_4_KICKOFF_PROMPT.md).

---

## §0.3 — Per-backbone pretrained_cfg findings

Run `python scripts/inspect_pretrained_cfg.py` to populate this table.

| Backbone | input_size | mean | std | crop_pct | crop_mode |
|---|---|---|---|---|---|
| repvgg_a1 | (3, 224, 224) | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | 0.875 | N/A |
| repvgg_a2 | (3, 224, 224) | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | 0.875 | N/A |
| mobilenetv3_large_100 | (3, 224, 224) | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | 0.875 | N/A |
| efficientnet_lite0 | (3, 224, 224) | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | 0.875 | N/A |
| tf_efficientnetv2_s.in21k_ft_in1k | **(3, 300, 300)** | **(0.5, 0.5, 0.5)** | **(0.5, 0.5, 0.5)** | **1.0** | center |
| convnext_tiny.fb_in22k_ft_in1k | (3, 224, 224) | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | 0.875 | N/A |

> **Verified 2026-07-26** via `timm==1.0.28` and `scripts/inspect_pretrained_cfg.py`.
> **Tag comparison** (`scripts/_check_effnetv2s_tags.py`): all four tf_efficientnetv2_s tags
> (default, .in1k, .in21k, .in21k_ft_in1k) share identical input_size/mean/std/crop_pct.
> The explicit `.in21k_ft_in1k` tag is used in `configs/tf_efficientnetv2_s.yaml` because
> this variant (ImageNet-21k pretrain → IN-1k fine-tune) consistently scores ~1-2 pp
> higher top-1 than the plain .in1k default across the EfficientNetV2 family.

### Resolution policy decision

**Policy (b): Each backbone trained at its own native pretrained resolution.**

Rationale: Per §0.2 (accuracy-first directive), artificially feeding 224×224 into
`tf_efficientnetv2_s` (which expects 300×300 and uses TF-style 0.5/0.5 normalization)
would unfairly and artificially tank its score. This is a bug in the comparison,
not a genuine finding about architecture capability. The `configs/tf_efficientnetv2_s.yaml`
therefore uses `image_size: 300` and TF-pretrained normalization.

**Limitation acknowledged:** Because `tf_efficientnetv2_s` trains at a different
input resolution and normalization regime, its batch_size is reduced (24 vs. 64)
to fit within 8 GB VRAM with AMP. This means slightly more gradient noise per step,
but the comparison is still more fair than forcing 224×224 onto it.

---

## Backbone comparison table

*Populate after full training runs complete.*

| Backbone | Val macro-F1 | BT-cerebellum F1 | BT-thalamic F1 | BT-ventricular F1 | F-abdomen F1 | F-femur F1 | F-thorax F1 | M-cervix F1 | Other F1 | Approx GPU-hrs |
|---|---|---|---|---|---|---|---|---|---|---|
| repvgg_a1 | — | — | — | — | — | — | — | — | — | — |
| repvgg_a2 | — | — | — | — | — | — | — | — | — | — |
| mobilenetv3_large_100 | — | — | — | — | — | — | — | — | — | — |
| efficientnet_lite0 | — | — | — | — | — | — | — | — | — | — |
| tf_efficientnetv2_s | — | — | — | — | — | — | — | — | — | — |
| convnext_tiny | — | — | — | — | — | — | — | — | — | — |

---

## Backbone decision

*Fill in after all four comparison runs complete.*

**Winner:** TBD  
**Reasoning:** Val macro-F1 = ??? (best). Per §0.2, the backbone with the
highest val macro-F1 is selected, period — training speed is a tie-breaker
only. See `checkpoints/<backbone>/classification_report_final.txt` for
per-class breakdown.

**Brain sub-plane analysis (hardest pair: Trans-ventricular vs. Trans-thalamic):**
TBD. Reference: Stage 2 from the original repo scored Trans-ventricular at
F1=0.759 (lowest). Expect this to remain the weakest category.

---

## Pretraining-init ablation

**FUSC checkpoint portability assessment:**

FUSC (BioMedIA-MBZUAI/FUSC) is a SimCLR-pretrained CNN on fetal ultrasound
2nd-trimester scans. Per §9 of PHASE_4_KICKOFF_PROMPT.md, this is the best
public candidate for domain-specific SSL init.

Status: **TBD** — check https://github.com/BioMedIA-MBZUAI/FUSC for a
downloadable checkpoint. If the encoder is ResNet-family and does not match
the winning backbone, document "not portable, skipped" here.

Expected gap from literature: ~3.1 pp median accuracy gain (IQR 1.8–4.9)
over ImageNet init on 6-class fetal plane tasks (arXiv 2601.00990).

| Init | Val macro-F1 | Epochs to convergence | Notes |
|---|---|---|---|
| ImageNet (pretrained=True) | — | — | Baseline |
| FUSC SSL | — | — | If portable |

---

## RepVGG re-parameterization

*Only applicable if RepVGG wins the backbone comparison.*

Status: **TBD**

Numerical verification: `torch.allclose(pre_reparam_output, post_reparam_output, atol=1e-4)`
Result: TBD — must be confirmed before trusting reparam model downstream.

---

## Focal loss ablation

*Only triggered if confusion matrix shows Trans-ventricular ↔ Trans-thalamic
confusion as a live problem after class-weighted CE on the winning backbone.*

Status: **TBD** — decision deferred to post-full-run confusion matrix review.

| Loss | BT-ventricular F1 | BT-thalamic F1 | Val macro-F1 | Notes |
|---|---|---|---|---|
| Class-weighted CE | — | — | — | Default |
| Focal loss (γ=2) | — | — | — | If triggered |

---

## Grad-CAM spot-check notes

*Fill in after visual inspection of val batch overlays.*

| Class | Activation looks anatomically plausible? | Notes |
|---|---|---|
| Fetal_femur | — | Should show long bone region |
| Maternal_cervix | — | Should show lower uterine segment |
| Brain_Trans_thalamic | — | Should highlight thalamus region |
| Brain_Trans_ventricular | — | Should highlight ventricles |
| Brain_Trans_cerebellum | — | Should show posterior fossa |
| Fetal_abdomen | — | Should show stomach/umbilical region |
| Fetal_thorax | — | Should show heart/rib cage |
| Other | — | Expect diffuse / low-confidence patterns |

---

## LR finder results

*Populate after running `python scripts/lr_finder.py --config configs/<backbone>.yaml`*

| Backbone | Suggested LR (steepest descent) | Chosen LR | Notes |
|---|---|---|---|
| repvgg_a1 | N/A (auto-detection unreliable) | 5e-4 | Based on 4 repeated runs, all consistent |
| repvgg_a2 | N/A (auto-detection unreliable) | 3e-4 | More LR-sensitive than A1 — divergence starts earlier |
| mobilenetv3_large_100 | N/A (auto-detection unreliable) | 7e-4 | More tolerant of higher LR than RepVGG variants |
| efficientnet_lite0 | N/A (auto-detection unreliable) | 1.5e-3 | Confirmed via direct log read |
| tf_efficientnetv2_s | N/A (auto-detection unreliable) | 1.5e-3 | Sharp cliff right after the minimum |
| convnext_tiny | N/A (auto-detection unreliable) | 1e-4 | Most fragile of the six — picked conservatively |
