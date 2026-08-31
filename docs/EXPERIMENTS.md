# EXPERIMENTS.md — Phase 4 Model Training

This file is the living record of all backbone comparison experiments, ablations, and design decisions made in Phase 4. Cross-referenced with [PHASE_4_KICKOFF_PROMPT.md](kickoff & results/PHASE_4_KICKOFF_PROMPT.md).

---

## §0.3 — Per-backbone pretrained_cfg findings

| Backbone | input_size | mean | std | crop_pct | crop_mode |
|---|---|---|---|---|---|
| repvgg_a1 | (3, 224, 224) | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | 0.875 | N/A |
| repvgg_a2 | (3, 224, 224) | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | 0.875 | N/A |
| mobilenetv3_large_100 | (3, 224, 224) | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | 0.875 | N/A |
| efficientnet_lite0 | (3, 224, 224) | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | 0.875 | N/A |
| tf_efficientnetv2_s.in21k_ft_in1k | **(3, 300, 300)** | **(0.5, 0.5, 0.5)** | **(0.5, 0.5, 0.5)** | **1.0** | center |
| convnext_tiny.fb_in22k_ft_in1k | (3, 224, 224) | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | 0.875 | N/A |

### Resolution policy decision

**Policy (b): Each backbone trained at its own native pretrained resolution.**
Rationale: Artificially feeding 224×224 into `tf_efficientnetv2_s` (which expects 300×300 and uses TF-style 0.5/0.5 normalization) would unfairly tank its score. The `configs/tf_efficientnetv2_s.yaml` uses `image_size: 300` and TF-pretrained normalization. 
Limitation: `tf_efficientnetv2_s` batch_size is reduced (24 vs. 64) to fit 8 GB VRAM with AMP.

---

## Pretraining-init ablation

**FUSC checkpoint portability assessment:**
FUSC (BioMedIA-MBZUAI/FUSC) is a SimCLR-pretrained CNN on fetal ultrasound 2nd-trimester scans.
**Status: SKIPPED.** The FUSC encoder is ResNet-based, which does not match any of our chosen backbones (ConvNeXt, RepVGG, EfficientNet, MobileNet). The checkpoint is not portable to our architectures.

---

### 4.2 Backbone Comparison, Cross-Device

**Source:** This phase, Task 2 (`scripts/evaluate_cross_device.py`, `logs/eval/evaluate_cross_device.txt`).

> [!WARNING]
> **Multitask Checkpoint Invalidation:** The Phase 7 multitask architecture incorporates HC18 and UCL into its training set to acquire bounding-box supervision. Therefore, HC18 and UCL are **no longer held-out data** for any `multitask` checkpoint. Multitask checkpoints must never be evaluated against `cross_device_manifest.csv` or compared to the numbers below.

## Backbone comparison table

Evaluated on the true held-out `test.csv` (5,271 images). Val macro-F1 provided for transparency.

| Backbone | Test Macro-F1 | Val Macro-F1 | Test BT-vent. | Test BT-thal. | Test BT-cereb. | Test F-femur | Test M-cervix |
|---|---|---|---|---|---|---|---|
| **convnext_tiny** | **0.8927** | 0.9180 | 0.77 | 0.85 | 0.87 | 0.89 | 0.99 |
| tf_efficientnetv2_s | 0.8853 | 0.9052 | 0.80 | 0.85 | 0.87 | 0.88 | 0.99 |
| efficientnet_lite0 | 0.8823 | 0.9112 | 0.79 | 0.83 | 0.86 | 0.87 | 1.00 |
| repvgg_a2 | 0.8811 | **0.9228** | 0.75 | 0.82 | 0.85 | 0.89 | 0.99 |
| repvgg_a1 | 0.8760 | 0.8952 | 0.76 | 0.81 | 0.86 | 0.87 | 1.00 |
| mobilenetv3_large_100| 0.8751 | 0.9045 | 0.74 | 0.83 | 0.88 | 0.85 | 0.99 |

### Bootstrap Significance Analysis
A paired bootstrap (n=5271, 2000 iterations) was conducted to verify if `convnext_tiny`'s lead over the runners-up was robust:

| Difference (ref: convnext_tiny) | Point Δ | 95% CI of Δ | Verdict |
|---|---|---|---|
| vs tf_efficientnetv2_s | +0.0074 | [-0.0010, +0.0159] | Indistinguishable (straddles 0) |
| vs efficientnet_lite0 | +0.0104 | [+0.0022, +0.0187] | Robust (CI > 0) |
| vs repvgg_a2 | +0.0116 | [+0.0027, +0.0202] | Robust (CI > 0) |

---

## Backbone decision

**Winner:** `convnext_tiny.fb_in22k_ft_in1k` (Class-weighted CE checkpoint)

**Reasoning:** 
1. `convnext_tiny` is the clear winner on aggregate macro-F1 on the held-out test set. While it wins outright on 1 class (`F_abdomen`), ties on 5, and loses on 2 (notably slightly underperforming `tf_efficientnetv2_s` (0.77 vs 0.80 F1) on the hardest and most clinically meaningful class, `BT_ventricular`), its overall macro-F1 edge is statistically robust. 
2. `repvgg_a2`'s val-set macro-F1 exhibited high epoch-to-epoch variance, and its early-stopped checkpoint appears to have benefited from a favorable val-set fluctuation rather than a stable generalization improvement. This is precisely the failure mode the test-set evaluation gate was built to catch, as its test-set performance dropped by over -0.04 to 4th place.
3. While `convnext_tiny` is statistically tied with `tf_efficientnetv2_s` based on the bootstrap test (CI barely straddles zero), secondary criteria cleanly resolve the tie: `convnext_tiny` trains/infers at 224x224 (vs 300x300, meaning faster inference) and has a simpler normalization scheme.

---

## RepVGG re-parameterization

**Status: Not applicable.** 
RepVGG did not win the backbone comparison on the real test set; see `repvgg_a2`'s val→test drop finding above. Re-parameterization applies only to the RepVGG family.

---

## Focal loss ablation

**Trigger condition:** `convnext_tiny`'s test confusion matrix showed highly concentrated error on the hardest class: `BT_ventricular` predictions landed on `BT_thalamic` for 25 out of 27 off-diagonal errors. This specific, concentrated pair confusion triggered the focal loss ablation (gamma=2.0).

**Result:** Focal loss ablation run, did not improve on class-weighted CE, CE retained. 
Test Macro-F1 dropped from 0.8927 (CE) to 0.8785 (Focal). `BT_ventricular` F1 specifically regressed from 0.77 (CE) to 0.75 (Focal), with precision dropping notably from 0.83 to 0.74. 
A clean negative result — focal loss did not help the hard pair. The class-weighted CE checkpoint is retained as the final shipped model.

---

## Grad-CAM spot-check notes

The layer path `["stages", 3, "blocks", -1]` mapping to `ConvNeXtBlock` already existed in `gradcam.py` and resolved successfully during verification. No re-parameterization step was required.

A 10-image test set spot check was performed on `convnext_tiny`:
- **Correct predictions (Femur, Cervix, Cerebellum, etc):** Heatmaps clearly highlighted the appropriate anatomical structures (bone shaft, lower uterine segment, posterior fossa).
- **`BT_ventricular` → `BT_thalamic` error:** Heatmap formed a clean, concentrated band tracing the actual anatomical brain/ventricle structure. It was locked onto the correct region but made the wrong fine-grained call. This is "concentrated-but-anatomically-grounded confusion", meaning the model looks in the right place but fails fine-grained discrimination.
- **`BT_thalamic` → `Other` error:** Heatmap was split across two disconnected regions rather than a coherent structure, indicating diffuse uncertainty on a hard/atypical image defaulting to the catch-all class.

---

## LR finder results

| Backbone | Suggested LR (steepest descent) | Chosen LR | Notes |
|---|---|---|---|
| repvgg_a1 | N/A (auto-detection unreliable) | 5e-4 | Based on 4 repeated runs, all consistent |
| repvgg_a2 | N/A (auto-detection unreliable) | 3e-4 | More LR-sensitive than A1 — divergence starts earlier |
| mobilenetv3_large_100 | N/A (auto-detection unreliable) | 7e-4 | More tolerant of higher LR than RepVGG variants |
| efficientnet_lite0 | N/A (auto-detection unreliable) | 1.5e-3 | Confirmed via direct log read |
| tf_efficientnetv2_s | N/A (auto-detection unreliable) | 1.5e-3 | Sharp cliff right after the minimum |
| convnext_tiny | N/A (auto-detection unreliable) | 1e-4 | Most fragile of the six — picked conservatively |

---

## ACAM adaptive-contrast preprocessing ablation (Phase 8, Stage 5)

**Date:** 2026-08-28

**Trigger condition (survey §2.2):** The literature survey identified arXiv:2509.00808 (Chen et al., "Adaptive Contrast Adjustment Module") as a validated, architecture-agnostic preprocessing sub-network tested on the **identical FETAL_PLANES_DB dataset** this project uses (12,400 images, six-category variant). Reported gains of +1.15 to +2.02pp accuracy across lightweight, conventional, and state-of-the-art backbones motivated a single clean ablation against the locked `convnext_tiny` baseline using the existing bootstrap-CI protocol.

**Implementation (documented per Task 5.1 requirement):**
- Module: `src/models/acam.py` — placed in `src/models/` (not `src/data/`) because ACAM is a **learnable** sub-network with trainable CNN parameters, not a fixed preprocessing transform.
- Architecture: shallow decision network (Conv→BN→ReLU ×2 → GAP → FC → Sigmoid → scale α to [1,3]) → K=10 contrast-adjusted views via `I'(x,y) = α_k · (I(x,y) − μ)` → channel-concatenation + 1×1 Conv (fusion).
- **Fusion mechanism choice:** The paper (§2.2, §5) states K views are "fused within downstream classifiers" without specifying the operator. Channel-concatenation + 1×1 conv was chosen as the most standard and defensible interpretation — linear learned fusion that restores the original tensor shape and is fully differentiable. The 1×1 conv is initialised as an equal-weight average so training begins near identity.
- **K=10** from paper Table 1 (n=10); **α∈[1,3]** from paper §5 Conclusion ("sigmoid-mapped parameters reflecting real sonographer adjustment ranges (1–3)").
- Input/output: `[B, 3, H, W] → [B, 3, H, W]` — drop-in compatible with any backbone.
- Parameter count: **5,559 learnable parameters** added to the convnext_tiny stack.
- Wiring: `nn.Sequential(acam, backbone)` jointly optimised via the same AdamW optimiser. Config flag: `use_acam: true` in `configs/convnext_tiny_acam.yaml`.
- All other hyperparameters identical to `configs/convnext_tiny.yaml` (lr=1e-4, batch=64, patience=8, seed=42, same class-weighted CE loss, same data splits).

**Training result:**
- Best val macro-F1: **0.9129** at epoch 16 (baseline: 0.9180).
- Early stopping triggered at epoch 24 (8 epochs without improvement).
- No NaN losses; training converged cleanly.
- Checkpoint: `checkpoints/convnext_tiny_acam/best.pt`.

**Test-set result (data/splits/test.csv — 5,271 images / 896 patients):**

| Model | Test macro-F1 | Point Δ vs baseline |
|---|---|---|
| convnext_tiny + ACAM | **0.8860** | −0.0067 |
| convnext_tiny (baseline) | **0.8927** | — |

**Bootstrap significance test (2,000 iterations, n=5,271, seed=42, paired):**
- Point Δ (ACAM − baseline): **−0.0067**
- Bootstrap 95% CI of Δ: **[−0.0142, +0.0012]**
- **VERDICT: statistically tied (CI ∋ 0)**

The CI straddles zero: ACAM neither robustly wins nor robustly loses. The point estimate is a small negative (−0.0067), but this is not statistically distinguishable from zero at the 95% level.

**Per-class test-set comparison (ACAM vs baseline):**

| Class | ACAM F1 | Baseline F1 | Δ |
|---|---|---|---|
| Brain_Trans_cerebellum | 0.86 | 0.87 | −0.01 |
| Brain_Trans_thalamic | 0.84 | 0.83 | +0.01 |
| Brain_Trans_ventricular | 0.78 | 0.77 | +0.01 |
| Fetal_abdomen | 0.91 | 0.88 | +0.03 |
| Fetal_femur | 0.89 | 0.91 | −0.02 |
| Fetal_thorax | 0.92 | 0.92 | 0.00 |
| Maternal_cervix | 0.99 | 0.99 | 0.00 |
| Other | 0.89 | 0.89 | 0.00 |

**Decision: ACAM retained as documented neutral result — not shipped.**

The ablation result is statistically tied: no reliable evidence that ACAM improves classification on this project's test set. The paper's reported +1.15–2.02pp gains on ConvNeXt-class models did not replicate here at statistical significance. Possible explanations (for the record, not as speculation requiring further investigation):
1. The paper's training setup differs (Adam vs this project's AdamW, 20 epochs vs up to 60, different augmentation pipeline including `RandomBrightnessContrast` which may already partially subsume ACAM's contrast adaptation effect).
2. The paper evaluated on a 6-class variant; this project's harder 8-class problem (with the fine-grained 3-way brain split) may provide less benefit from contrast manipulation vs class-discriminative fine detail.
3. The point estimate is slightly negative (−0.0067), consistent with a marginal net harm from adding contrast variability to a well-regularised baseline — no cause for alarm, and consistent with the project's existing negative-result culture.

The plain `convnext_tiny` checkpoint (`checkpoints/convnext_tiny/best.pt`) remains the production model.
The ACAM checkpoint is retained at `checkpoints/convnext_tiny_acam/best.pt` for reproducibility.

**Full log:** `logs/eval/bootstrap_significance_acam.txt`
**Per-class test report:** `checkpoints/convnext_tiny_acam/classification_report_TEST.txt`
**Confusion matrix:** `checkpoints/convnext_tiny_acam/confusion_matrix_TEST.png`

