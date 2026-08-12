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

---

## 3. Realistic-Video Evaluation

**Script:** `scripts/evaluate_realistic_video.py`  
**Full log:** `logs/eval/evaluate_realistic_video.txt`  
**Per-clip CSV:** `data/processed/eval_realistic_video/realistic_video_results.csv`  
**Clips evaluated:** 16 synthetic + 30 IUGC = **46 total**

> [!IMPORTANT]
> Per constraint §0.6: **IUGC clips have no accuracy ground truth** — their `pos`/`neg`/PS/AoP/HSD labels are for a different clinical task (intrapartum monitoring) with zero correspondence to our 8 classes. Frame-level accuracy is **only computed for the 16 synthetic clips**. IUGC clips contribute to stability metrics only.

### §3a — Frame-Level Accuracy: Synthetic Clips (raw vs. smoothed)

| Clip | Ground Truth | Raw Acc | Smoothed Acc | Delta |
|------|-------------|---------|--------------|-------|
| Brain_Trans_cerebellum_clip01.mp4 | Brain_Trans_cerebellum | 1.0000 | 1.0000 | +0.0000 |
| Brain_Trans_cerebellum_clip02.mp4 | Brain_Trans_cerebellum | 1.0000 | 1.0000 | +0.0000 |
| Brain_Trans_thalamic_clip01.mp4 | Brain_Trans_thalamic | 1.0000 | 1.0000 | +0.0000 |
| Brain_Trans_thalamic_clip02.mp4 | Brain_Trans_thalamic | 1.0000 | 1.0000 | +0.0000 |
| Brain_Trans_ventricular_clip01.mp4 | Brain_Trans_ventricular | 1.0000 | 1.0000 | +0.0000 |
| Brain_Trans_ventricular_clip02.mp4 | Brain_Trans_ventricular | 1.0000 | 1.0000 | +0.0000 |
| Fetal_abdomen_clip01.mp4 | Fetal_abdomen | 1.0000 | 1.0000 | +0.0000 |
| Fetal_abdomen_clip02.mp4 | Fetal_abdomen | 1.0000 | 1.0000 | +0.0000 |
| Fetal_femur_clip01.mp4 | Fetal_femur | 1.0000 | 1.0000 | +0.0000 |
| Fetal_femur_clip02.mp4 | Fetal_femur | 1.0000 | 1.0000 | +0.0000 |
| Fetal_thorax_clip01.mp4 | Fetal_thorax | 1.0000 | 1.0000 | +0.0000 |
| Fetal_thorax_clip02.mp4 | Fetal_thorax | 1.0000 | 1.0000 | +0.0000 |
| Maternal_cervix_clip01.mp4 | Maternal_cervix | 1.0000 | 1.0000 | +0.0000 |
| Maternal_cervix_clip02.mp4 | Maternal_cervix | 1.0000 | 1.0000 | +0.0000 |
| Other_clip01.mp4 | Other | 1.0000 | 1.0000 | +0.0000 |
| Other_clip02.mp4 | Other | 1.0000 | 1.0000 | +0.0000 |

**Aggregate: Mean raw accuracy = 100.0% · Mean smoothed accuracy = 100.0% · Delta = 0.0%**

No clip showed smoothed accuracy below raw — the expected direction (smoothed ≥ raw) holds uniformly. This result is consistent with the synthetic clips' construction: `scripts/create_synthetic_clips.py` generates within-class jitter only ("Does NOT synthesize class transitions"), so the raw model is already perfectly confident on every frame and the smoother has no accuracy cost to pay.

> [!NOTE]
> A 100% synthetic accuracy ceiling is expected and not surprising — these clips are derived from the same FETAL_PLANES_DB images the model trained on, with mild ego-motion augmentation. The cross-device (§2) and IUGC stability (§3b) results are the more meaningful real-world performance indicators.

### §3b — Video Stability Metrics: All 46 Clips

| Metric | Value |
|--------|-------|
| **Sum raw switches/min** (all 46 clips) | **788.75** |
| **Sum smoothed switches/min** (all 46 clips) | **72.00** |
| **Switch reduction** | **90.9%** |
| **Mean dwell time (smoothed, all clips)** | **2,430 ms** (~2.4 s) |
| Phase 5 reproduction check | **PASS ✓** (788.75 → 72.00, exact match) |

The Phase 5 tuning numbers reproduce exactly at 788.75 → 72.00 switches/min, confirming no data drift in `data/raw/iugc_video/` and no re-implementation inconsistency.

**Mean dwell time sanity check:** 2,430 ms (~2.4 seconds) is well within the expected range of "seconds for stable clips, not the entire clip length." No clip shows mean dwell = full duration *and* raw switches > 0 in a way that indicates a genuine smoother bug (see gate check below).

**Per-clip stability (selected rows — see CSV for all 46):**

| Clip | Raw SPM | Smoothed SPM | Reduction | Dwell (ms) |
|------|---------|-------------|-----------|------------|
| All 16 synthetic clips | 0.00 | 0.00 | — | 667 |
| `20190909T155747__B_产科_0.avi` | 141.64 | 0.00 | 100% | 2,542 |
| `test_14_51.avi` | 35.56 | 0.00 | 100% | 3,375 |
| `test_16_79.avi` | 35.56 | 0.00 | 100% | 3,375 |
| `202101141947512003470I1.avi` | 216.00 | **72.00** | 66.7% | 417 |
| `202101141947512000310I3.avi` | 144.00 | 0.00 | 100% | 833 |
| `202101141947512003470I2.avi` | 216.00 | 0.00 | 100% | 833 |

**Gate check — 5 clips flagged (raw_spm > 0, sm_spm = 0, dwell ≈ full clip):**

Five IUGC clips trigger the gate condition where raw argmax flickered but the smoother held a single label for the entire clip. Investigating: all five show zero genuine plane-to-plane transitions in the raw signal — the "raw switches" are pure oscillation noise between two near-identical predictions (consistent with Phase 5 Task 5's manual annotation finding of zero genuine transitions across all IUGC candidate clips). The smoother correctly suppresses all noise and holds one label throughout; `dwell = full clip` is the intended outcome, not a stuck-smoother bug. This is reported transparently rather than suppressed.

`202101141947512003470I1.avi` (72 smoothed SPM) is the same "stubborn clip" Phase 5 identified — the one clip where residual switching could not be fully eliminated. This is documented and expected.

**Gate result: PASS ✓** — no evidence of a smoother stuck on its cold-start label through a genuine class transition.

### §3b — Latency-to-Stabilize: Honest Limitation Statement

> True transition-latency validation was not possible in this project because no available real video source contains an annotated genuine plane-to-plane transition — the primary dataset (FETAL_PLANES_DB) never released source video, and manual annotation of the 5 IUGC candidate clips (Phase 5, Task 5) found zero raw model-level transitions to time. The dwell-time metric above (§3b) is the closest available proxy for responsiveness, but it measures steady-state holding behaviour, not transition-tracking speed.

