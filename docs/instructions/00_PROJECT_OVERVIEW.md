# Fetal Standard Plane Real-Time Detection — Project Overview

**Read this file first in any new chat.** It's the entry point; the other numbered files go deep on each phase. This project will be built almost entirely via AI coding agent ("vibe coded") — every doc is written so an agent can execute it with minimal ambiguity, and every place a human must act manually is flagged `[MANUAL]`.

---

## 1. What we're building

A **real-time fetal standard-plane detection system**: given a live video stream (webcam standing in for an ultrasound probe, or a pre-recorded video file), continuously classify each moment of the scan into one of 7 fetal standard anatomical planes, or "Other" (non-standard/transitional), with a stable (non-flickering) on-screen label, a confidence score, and a Grad-CAM style visual explanation — functioning as an assistive tool for a sonographer.

**Primary target classes (8-way, matching FETAL_PLANES_DB):**
```
0: Brain — Trans-cerebellum
1: Brain — Trans-thalamic
2: Brain — Trans-ventricular
3: Fetal abdomen
4: Fetal femur
5: Fetal thorax
6: Maternal cervix
7: Other (non-standard / transitional / off-plane)
```

## 2. Why this scope (not the alternatives we considered)

We explicitly evaluated and rejected:
- **A second "intrapartum" (labor-monitoring) track** using the IUGC/Zenodo video datasets. Rejected: different exam entirely (transperineal, labor-ward, PS/FH biometry) — no class or anatomical overlap with mid-pregnancy screening. Doing one system well beats two shallow ones. These datasets are **retained only as a temporal-methodology sandbox** (see [05_TEMPORAL_SMOOTHING_AND_REALTIME.md](05_TEMPORAL_SMOOTHING_AND_REALTIME.md)).
- **True multi-task detection (anatomical structure boxes + plane classification)** as the primary deliverable, à la FAUSP-NET. This is clinically the *better* design (see rationale in section 4) but requires structure-level bounding-box annotations we don't have out of the box. It's kept as **Stretch Goal #1** ([07_STRETCH_GOALS_AND_ROADMAP.md](07_STRETCH_GOALS_AND_ROADMAP.md)), not blocking the primary deliverable.
- **Heavy spatio-temporal architectures (3D-CNN / video transformer clips).** Standard planes are defined by single-frame anatomical content, not motion — the "temporal" need here is *smoothing a per-frame signal*, not recognizing a temporal action. A lightweight post-hoc smoothing layer over per-frame predictions is the right level of complexity; validated against real video (IUGC) before committing to anything heavier.

## 3. Deployment target (locks in a lot of downstream decisions)

**Demo/portfolio pipeline** — local laptop (RTX 4060, 8GB VRAM), webcam or video file input, no real clinical hardware integration, no regulatory/FDA scope. This means:
- We are NOT latency-starved (no need for sub-10ms inference); target is **stable, trustworthy signal within ~100–600ms perceived latency**, not raw fps maximization.
- We can afford heavier backbones than an edge/embedded target would allow (see [04_MODEL_TRAINING.md](04_MODEL_TRAINING.md) for backbone comparison), but should still build the model export/quantization path as a stretch goal to prove it *could* scale down.
- ONNX/TensorRT export, mobile deployment, Jetson targeting = stretch goals only, not required for v1.

## 4. Architecture summary (final decisions from planning phase)

| Component | Decision |
|---|---|
| Backbone | RepVGG-A1/A2 primary candidate; benchmark against MobileNetV3-Large, EfficientNet-Lite0, and EfficientNetV2-S (accuracy ceiling reference) |
| Classification head | Single 8-way softmax (7 planes + Other) — NOT the original repo's two-stage cascade. Simpler, one forward pass, and the smoothing layer handles most of what the cascade was trying to buy us |
| Interpretability | Grad-CAM, throttled to a fraction of frames (not every frame) to preserve real-time throughput |
| Temporal stabilization | Tier-1: confidence-weighted moving average + hysteresis + minimum dwell-time before switching displayed label. Tier-2 (learned temporal module) only if Tier-1 is empirically insufficient |
| Pretraining | Try domain-relevant self-supervised init (fetal-US SSL, if a usable public checkpoint exists) as an ablation against plain ImageNet init |
| Serving | Local Python real-time loop (OpenCV capture → preprocessing thread → inference thread → smoothing → overlay render), NOT the original repo's Flask upload/response pattern |

## 5. Datasets (full detail in [02_DATASETS.md](02_DATASETS.md))

| Dataset | Role | Classes covered | Video? |
|---|---|---|---|
| **FETAL_PLANES_DB** (Burgos-Artizzu et al., 2020) | Primary train/val/test | All 7 planes + Other | No (static frames) |
| **UCL/HC18 cross-device generalization set** — **`HC18` + `UCL` subsets only, `FP` and merged `MULTICENTRE` folders excluded** (see §5a below) | Held-out cross-device generalization test | Head, Abdomen, Femur only (3/7) | No |
| **IUGC / Zenodo intrapartum video** (Zenodo `17655183`, CC BY 4.0) | Temporal-smoothing methodology sandbox ONLY — never trains the plane classifier | N/A (different task) | Yes (real) |
| Synthetic ego-motion clips | Jitter-robustness validation for primary task | Derived from FETAL_PLANES_DB frames | Synthetic |

### 5a. `[CORRECTION — confirmed against source data]` The downloaded "multi-centre" bundle is not a clean external test set

The publicly released cross-device benchmark (arXiv 2512.16710 / *Sci Rep* s41598-026-47854-3, "A multi-centre, multi-device benchmark dataset for landmark-based comprehensive fetal biometry") ships as **four folders**, not one: `FP`, `HC18`, `UCL`, and a merged `MULTICENTRE` folder that is just the union of the other three.

The paper itself states this dataset **"combines three sources: the Fetal Plane (FP) dataset [burgos2020zdataset], the HC18 head dataset..., and an expanded cohort from UCLH."** `burgos2020zdataset` is the same Burgos-Artizzu et al. 2020 paper FETAL_PLANES_DB is built from — confirmed independently by `FP`'s own README, which lists the identical two Barcelona sites (Vall d'Hebron, Sant Joan de Déu), the identical device set (Voluson E6/S8/S10, Aloka), and the identical filename convention (`Patient[ID]_Plane[N]_[M]_of_[K].png`) as FETAL_PLANES_DB.

**In plain terms: `FP` is a landmark-annotated re-release of a subset of FETAL_PLANES_DB itself, not an independent dataset.** Its 1,047 patients very likely sit inside (or heavily overlap with) our own 1,792-patient FETAL_PLANES_DB training pool. Only `HC18` (Netherlands, Voluson E8/730) and `UCL` (UCLH, single site) are genuinely independent of FETAL_PLANES_DB.

**Decision:** Use only the `HC18/` and `UCL/` subfolders (999 + 424 = 1,423 images, 857 genuinely unseen subjects) as the cross-device generalization test set. **Never use `FP/` or `MULTICENTRE/` for this purpose** — doing so would silently leak FETAL_PLANES_DB training data into what's supposed to be a held-out generalization report. This supersedes the more general "watch for UCLH overlap" caution in earlier drafts of [02_DATASETS.md](02_DATASETS.md), which was aimed at the wrong sub-source; [02_DATASETS.md](02_DATASETS.md) and [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md) have been updated accordingly.

**Known consequence of this decision:** because `HC18`/`UCL` are pure landmark-biometry datasets, every image in them is by construction a valid standard plane — there are no "Other"/non-standard examples. This cross-device set can therefore only validate *"given a standard image, is the plane label right,"* not Stage 1's standard-vs-other decision cross-device. Documented as an explicit limitation in [06_EVALUATION_VALIDATION.md](06_EVALUATION_VALIDATION.md).

## 6. Chronological phase list (see [08_MASTER_CHECKLIST.md](08_MASTER_CHECKLIST.md) for the full checklist)

1. Environment setup ([01_ENVIRONMENT_SETUP.md](01_ENVIRONMENT_SETUP.md))
2. Dataset acquisition — **heavy `[MANUAL]` content** ([02_DATASETS.md](02_DATASETS.md))
3. Data pipeline: splitting, verification, preprocessing, synthetic augmentation ([03_DATA_PIPELINE.md](03_DATA_PIPELINE.md))
4. Model training: backbone comparison, primary classifier training ([04_MODEL_TRAINING.md](04_MODEL_TRAINING.md))
5. Temporal smoothing design + tuning, real-time serving pipeline ([05_TEMPORAL_SMOOTHING_AND_REALTIME.md](05_TEMPORAL_SMOOTHING_AND_REALTIME.md))
6. Evaluation & validation, including cross-device generalization ([06_EVALUATION_VALIDATION.md](06_EVALUATION_VALIDATION.md))
7. Stretch goals: multi-task detection, export/quantization, UI polish ([07_STRETCH_GOALS_AND_ROADMAP.md](07_STRETCH_GOALS_AND_ROADMAP.md))

## 7. Reference literature (for the agent's context, not to re-read in full)

- SonoNet (Baumgartner et al., IEEE TMI 2017 / arXiv:1612.05601) — real-time fully-convolutional plane detection + weak localization
- FAUSP-NET (Sci Rep 2026, s41598-026-40590-8) — multi-task detection+classification, 24.1ms/frame on consumer GPU, backbone comparison table
- ScanAhead (Men et al., Med Image Anal 2025) — Swin-Transformer + explicit temporal extractor for video-based fetal head plane prediction
- 3DFETUS (arXiv 2511.10412) — spatial-transformer normalization pattern (not directly used, informs design philosophy)
- **UCL/HC18 multi-centre benchmark (arXiv 2512.16710 / Sci Rep s41598-026-47854-3)** — our cross-device generalization test set. **Confirmed via the paper itself that this bundle's `FP` component is sourced from Burgos-Artizzu et al. 2020 (same as FETAL_PLANES_DB) — only its `HC18` and `UCL` components are used by us; see §5a.**
- **IUGC / Zenodo intrapartum dataset (Bai et al., *Scientific Data*, doi:10.5281/zenodo.17655183)** — confirmed 774 videos / 68,106 frames from 3 institutions (JNU, SYSU, SMU), released CC BY 4.0.
- Original student repo (`inference.py`, `app.py`, [README.md](../README.md)) — reference for what to keep (patient-level split, Grad-CAM concept, two-stage philosophy) vs. discard (Flask serving, per-frame double-forward-pass, no temporal handling)

## 8. Non-goals (explicitly out of scope for v1)

- No clinical validation, no regulatory claims, no real patient data
- No real ultrasound hardware integration
- No mobile app
- No multi-task detection head (stretch goal only)
- No production-grade auth/security/multi-user serving
