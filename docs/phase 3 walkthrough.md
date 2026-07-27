# Phase 3 — Data Pipeline: Completion Walkthrough

## Summary

All Phase 3 deliverables are implemented and verified. Here are the actual numbers for your review before Phase 4.

---

## 1. FETAL_PLANES_DB Manifest — `data/processed/manifest.csv`

**12,400 rows** total, 8 canonical classes:

| Class | Images |
|---|---|
| Other | 4,356 |
| Fetal_thorax | 1,718 |
| Brain_Trans_thalamic | 1,638 |
| Maternal_cervix | 1,626 |
| Fetal_femur | 1,040 |
| Brain_Trans_cerebellum | 714 |
| Fetal_abdomen | 711 |
| Brain_Trans_ventricular | 597 |

---

## 2. Patient-Level Splits — `data/splits/{train,val,test}.csv`

**Algorithm:** `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)`, searched all 7 folds for the one with the **best minimum per-class patient coverage in val** (fold 2 chosen).

| Split | Images | Patients |
|---|---|---|
| **train** | 6,106 | 772 |
| **val** | 1,023 | 124 |
| **test** (in-distribution) | 5,271 | 896 |

**Per-class patient coverage in val** (min = 34, well above the 3-patient floor):

| Class | Val patients | Train patients |
|---|---|---|
| Brain_Trans_cerebellum | 43 | 265 |
| Brain_Trans_thalamic | 67 | 391 |
| Brain_Trans_ventricular | **34** ← minimum | 197 |
| Fetal_abdomen | 42 | 250 |
| Fetal_femur | 55 | 315 |
| Fetal_thorax | 61 | 382 |
| Maternal_cervix | 70 | 429 |
| Other | 57 | 353 |

> [!NOTE]
> Even the smallest class (`Brain_Trans_ventricular`, the hardest per the literature) has 34 val patients — solid coverage for monitoring training progress.

---

## 3. Leakage Verification — `scripts/verify_no_leakage.py`

```
PASS: train/val/test patient sets are fully disjoint.
train patients: 772, val patients: 124, test patients: 896
```

✅ All three pairwise intersection assertions passed.

---

## 4. Cross-Device Manifest — `data/processed/cross_device_manifest.csv`

**1,423 rows** total. Only HC18 and UCL — no FP or MULTICENTRE paths (confirmed by path-parts guardrail assert):

| source_subset | plane_label | Count |
|---|---|---|
| HC18 | Head | 999 |
| UCL | Fetal_abdomen | 130 |
| UCL | Fetal_femur | 135 |
| UCL | Head | 159 |

> [!IMPORTANT]
> **Manual checkpoint #1:** Open `cross_device_manifest.csv` and confirm every `image_path` contains `\HC18\` or `\UCL\` and never `\FP\` or `\MULTICENTRE\`.

---

## 5. Class Weights — `data/processed/class_weights.json`

Inverse-frequency weights, normalized to mean=1.0:

| Class | Weight |
|---|---|
| Brain_Trans_ventricular | **1.899** (heaviest — least common) |
| Fetal_abdomen | 1.585 |
| Brain_Trans_cerebellum | 1.486 |
| Fetal_femur | 1.083 |
| Brain_Trans_thalamic | 0.641 |
| Maternal_cervix | 0.570 |
| Fetal_thorax | 0.528 |
| Other | **0.209** (lightest — most common) |

Phase 4 training loop loads this JSON directly (`data/processed/class_weights.json`).

---

## 6. Source Files Created

| File | Purpose |
|---|---|
| [scripts/build_manifest.py](file:///d:/Ultrasound/scripts/build_manifest.py) | FETAL_PLANES_DB CSV → canonical 8-class manifest |
| [scripts/patient_split.py](file:///d:/Ultrasound/scripts/patient_split.py) | StratifiedGroupKFold patient-level splits |
| [scripts/verify_no_leakage.py](file:///d:/Ultrasound/scripts/verify_no_leakage.py) | Hard-assert leakage check |
| [scripts/build_cross_device_manifest.py](file:///d:/Ultrasound/scripts/build_cross_device_manifest.py) | HC18+UCL only manifest with guardrails |
| [scripts/generate_sample_clips.py](file:///d:/Ultrasound/scripts/generate_sample_clips.py) | Driver: 2 clips per class → `synthetic_clips/` |
| [scripts/run_sanity_checks.py](file:///d:/Ultrasound/scripts/run_sanity_checks.py) | All 6 sanity-check visualizations |
| [src/data/transforms.py](file:///d:/Ultrasound/src/data/transforms.py) | Train + eval albumentations pipelines |
| [src/data/dataset.py](file:///d:/Ultrasound/src/data/dataset.py) | `FocalPlanesDataset` + `CrossDeviceDataset` |
| [src/data/class_weights.py](file:///d:/Ultrasound/src/data/class_weights.py) | Inverse-frequency weight computation |
| [src/data/synthetic_video.py](file:///d:/Ultrasound/src/data/synthetic_video.py) | `_SmoothRandomWalk` + `generate_ego_motion_clip` |

---

## 7. Synthetic Clips — `data/processed/synthetic_clips/`

**16 MP4 clips** generated (2 per class, 16 frames @ 24fps each):

> [!IMPORTANT]
> **Manual checkpoint #2:** Watch a handful of the `.mp4` clips in `data/processed/synthetic_clips/`. The motion should look like plausible small probe wobble — smooth, continuous drift — NOT jittery per-frame randomness. If it looks jittery, the `_SmoothRandomWalk` momentum was not working correctly.

---

## 8. Sanity-Check Visualizations — `data/processed/sanity_checks/`

| File | Contents |
|---|---|
| [01_class_distribution_splits.png](file:///d:/Ultrasound/data/processed/sanity_checks/01_class_distribution_splits.png) | Class counts per split (grouped bar, FETAL_PLANES_DB) |
| [02_cross_device_distribution.png](file:///d:/Ultrasound/data/processed/sanity_checks/02_cross_device_distribution.png) | Cross-device counts by source_subset and label |
| [03_sample_grid_fetal_planes.png](file:///d:/Ultrasound/data/processed/sanity_checks/03_sample_grid_fetal_planes.png) | 5 sample images per class (train split) |
| [04_sample_grid_cross_device.png](file:///d:/Ultrasound/data/processed/sanity_checks/04_sample_grid_cross_device.png) | 5 sample images per (subset, label) combination |
| [05_patient_count_heatmap.png](file:///d:/Ultrasound/data/processed/sanity_checks/05_patient_count_heatmap.png) | Heatmap of patient + image counts per class per split |
| [05_patient_image_counts_per_split.csv](file:///d:/Ultrasound/data/processed/sanity_checks/05_patient_image_counts_per_split.csv) | Same data as CSV |
| [06_resolution_histogram.png](file:///d:/Ultrasound/data/processed/sanity_checks/06_resolution_histogram.png) | Width / height / aspect-ratio distributions |

> [!IMPORTANT]
> **Manual checkpoint #3:** Review every chart before starting Phase 4. This is the cheapest point to catch labeling or leakage bugs.

---

## Deliverables Checklist

- [x] `scripts/build_manifest.py` run — `data/processed/manifest.csv` produced (12,400 rows)
- [x] `scripts/patient_split.py` run — `data/splits/{train,val,test}.csv` produced; min val coverage = 34 patients
- [x] `scripts/verify_no_leakage.py` run — **PASS**
- [x] `scripts/build_cross_device_manifest.py` run — 1,423 rows, HC18/UCL only confirmed
- [x] `src/data/transforms.py` + `src/data/dataset.py` wired to split CSVs
- [x] `src/data/class_weights.py` run — `data/processed/class_weights.json` produced
- [x] `src/data/synthetic_video.py` + `scripts/generate_sample_clips.py` run — 16 MP4 clips in `synthetic_clips/`
- [x] All 6 sanity-check visualizations saved to `data/processed/sanity_checks/`

**Phase 4 can begin after you complete the 3 manual checkpoints above.**
