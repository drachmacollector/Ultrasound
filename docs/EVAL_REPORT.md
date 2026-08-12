# Phase 6 Evaluation Report: Fetal Standard Plane Real-Time Detection

## 1. In-Distribution Performance

Performance on the held-out FETAL_PLANES_DB test set (5,271 images from 896 patients) for the primary selected backbone (`convnext_tiny.fb_in22k_ft_in1k`, class-weighted CE).

- **Macro-F1:** 0.8927
- **Overall Accuracy:** 0.90

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Brain_Trans_cerebellum | 0.82 | 0.92 | 0.87 | 339 |
| Brain_Trans_thalamic | 0.84 | 0.85 | 0.85 | 765 |
| Brain_Trans_ventricular | 0.83 | 0.73 | 0.77 | 302 |
| Fetal_abdomen | 0.88 | 0.97 | 0.92 | 358 |
| Fetal_femur | 0.87 | 0.92 | 0.89 | 524 |
| Fetal_thorax | 0.94 | 0.93 | 0.94 | 660 |
| Maternal_cervix | 1.00 | 0.99 | 0.99 | 645 |
| Other | 0.93 | 0.88 | 0.90 | 1678 |

### Confusion Matrix

![Confusion Matrix](../checkpoints/convnext_tiny/confusion_matrix_TEST.png)

> [!NOTE]
> `Brain_Trans_ventricular` remains the weakest class (F1=0.77, recall=0.73), consistent with the literature-documented hard pair and with the focal-loss ablation's negative result (`docs/EXPERIMENTS.md`).

---

## 2. Cross-Device Generalization (HC18 + UCL)

**Checkpoint:** `checkpoints/convnext_tiny/best.pt`  
**Manifest:** `data/processed/cross_device_manifest.csv` (HC18 + UCL only — 1,423 images)  
**Script:** `scripts/evaluate_cross_device.py`  
**Full log:** `logs/eval/evaluate_cross_device.txt`  
**Per-image CSV:** `data/processed/eval_cross_device/cross_device_results.csv`

### Scope Limitations (explicit)

> [!IMPORTANT]
> 1. **No "Other" examples exist.** Every image in HC18/UCL is a valid standard plane by construction. This evaluation validates plane-*identity* classification cross-device, **not** the standard-vs-Other decision.
> 2. **Only 3 of 7 classes covered.** HC18/UCL cover `Head`, `Fetal_abdomen`, and `Fetal_femur` only. `Fetal_thorax` and `Maternal_cervix` have **zero cross-device coverage** in any available external dataset — those classes are not validated here and must not be implied as cross-device validated.
> 3. **FP and MULTICENTRE subfolders excluded.** They overlap with FETAL_PLANES_DB training data. Only genuinely independent HC18 (Netherlands, Voluson E8/730) and UCL (UCLH) images are used. See `00_PROJECT_OVERVIEW.md §5a`.

### Collapsed-Label Scoring Rule

`Head` rows are scored **CORRECT** if the model's argmax is *any* of `{Brain_Trans_cerebellum, Brain_Trans_thalamic, Brain_Trans_ventricular}`. `Fetal_abdomen` / `Fetal_femur` rows use exact-match scoring. Brain-subplane-to-brain-subplane confusions therefore do **not** count as misclassifications under this metric.

### Overall Accuracy (HC18 + UCL Combined)

**1,190 / 1,423 correct → 83.6%**

### Per-Subset Accuracy (HC18 vs UCL — reported separately, never blended)

| Subset | Label | Correct | Total | Accuracy |
|--------|-------|---------|-------|----------|
| **HC18** | Head | 813 | 999 | **81.4%** |
| **HC18** | *Subset total* | 813 | 999 | **81.4%** |
| **UCL** | Head | 151 | 159 | **95.0%** |
| **UCL** | Fetal_abdomen | 117 | 130 | **90.0%** |
| **UCL** | Fetal_femur | 109 | 135 | **80.7%** |
| **UCL** | *Subset total* | 377 | 424 | **88.9%** |

> [!NOTE]
> HC18 and UCL are reported separately per `03_DATA_PIPELINE.md §3` — they differ meaningfully in site count, device, and per-anatomy coverage (HC18 = Head only from a single Netherlands site; UCL = Head+Abdomen+Femur from UCLH). Blending them into one number would obscure these structural differences.

### Generalization Gap vs. In-Distribution Baseline

The in-distribution baseline uses **collapsed scoring** on `data/splits/test.csv` to match the cross-device task exactly (e.g., any brain-subplane prediction on a brain-subplane ground truth counts as correct), making this a true apples-to-apples comparison.

| Label | In-Dist Acc (collapsed) | Cross-Device Acc | **Gap** |
|-------|------------------------|-----------------|---------|
| Head | 98.0% | 83.2% | **−14.8 pp** |
| Fetal_abdomen | 97.5% | 90.0% | **−7.5 pp** |
| Fetal_femur | 92.4% | 80.7% | **−11.6 pp** |

All three classes show a meaningful generalization drop, consistent with the UCL/HC18 multi-centre paper's own reported cross-device degradation. The Head gap (−14.8 pp) is the largest, driven primarily by HC18 (81.4%) which differs most in device and site from FETAL_PLANES_DB's Barcelona machines.

### Head Misclassification Breakdown

When a Head image is misclassified (194 total errors across HC18+UCL), the model predicts:

| Predicted Class | Count | Note |
|-----------------|-------|------|
| Other | 180 | Dominant failure mode — domain-shifted head images fall to catch-all |
| Fetal_thorax | 7 | Genuine domain-shift red flag |
| Fetal_abdomen | 5 | Genuine domain-shift red flag |
| Maternal_cervix | 2 | Genuine domain-shift red flag |

The overwhelming majority (180/194 = 92.8%) of Head errors land on "Other" rather than a wrong plane class. This is the *expected* and less alarming failure mode: the model recognizes it cannot confidently classify a cross-device head image and defaults to the catch-all, rather than hallucinating a wrong anatomy. The 14 non-"Other" errors (7.2% of Head errors) are genuine domain-shift red flags but minor in scale.

![Head Misclassification Bar Chart](../data/processed/eval_cross_device/cross_device_confusion.png)

### Gate Check

- **Combined Head accuracy: 83.2%** — well above the 50% gate threshold.
- **Result: PASS ✓** — no implementation bug in the collapsed-scoring logic.

