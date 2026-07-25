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
- **A second "intrapartum" (labor-monitoring) track** using the IUGC/Zenodo video datasets. Rejected: different exam entirely (transperineal, labor-ward, PS/FH biometry) — no class or anatomical overlap with mid-pregnancy screening. Doing one system well beats two shallow ones. These datasets are **retained only as a temporal-methodology sandbox** (see `05_TEMPORAL_SMOOTHING_AND_REALTIME.md`).
- **True multi-task detection (anatomical structure boxes + plane classification)** as the primary deliverable, à la FAUSP-NET. This is clinically the *better* design (see rationale in section 4) but requires structure-level bounding-box annotations we don't have out of the box. It's kept as **Stretch Goal #1** (`07_STRETCH_GOALS_AND_ROADMAP.md`), not blocking the primary deliverable.
- **Heavy spatio-temporal architectures (3D-CNN / video transformer clips).** Standard planes are defined by single-frame anatomical content, not motion — the "temporal" need here is *smoothing a per-frame signal*, not recognizing a temporal action. A lightweight post-hoc smoothing layer over per-frame predictions is the right level of complexity; validated against real video (IUGC) before committing to anything heavier.

## 3. Deployment target (locks in a lot of downstream decisions)

**Demo/portfolio pipeline** — local laptop (RTX 4060, 8GB VRAM), webcam or video file input, no real clinical hardware integration, no regulatory/FDA scope. This means:
- We are NOT latency-starved (no need for sub-10ms inference); target is **stable, trustworthy signal within ~100–200ms perceived latency**, not raw fps maximization.
- We can afford heavier backbones than an edge/embedded target would allow (see `04_MODEL_TRAINING.md` for backbone comparison), but should still build the model export/quantization path as a stretch goal to prove it *could* scale down.
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

## 5. Datasets (full detail in `02_DATASETS.md`)

| Dataset | Role | Classes covered | Video? |
|---|---|---|---|
| **FETAL_PLANES_DB** | Primary train/val/test | All 7 planes + Other | No (static frames) |
| **UCL/HC18 multi-centre landmark dataset** (arXiv 2512.16710) | Held-out cross-device generalization test | Head, Abdomen, Femur only (3/7) | No |
| **IUGC / Zenodo intrapartum video** | Temporal-smoothing methodology sandbox ONLY — never trains the plane classifier | N/A (different task) | Yes (real) |
| Synthetic ego-motion clips | Jitter-robustness validation for primary task | Derived from FETAL_PLANES_DB frames | Synthetic |

**Known, accepted limitation** (documented, not hidden): plane-to-plane transition smoothing is tuned against real video from a *different* clinical exam (IUGC); within-plane jitter smoothing is validated on synthetic ego-motion of the primary dataset. True temporal validation on the primary task's own video does not exist because BCNatal never released source video for FETAL_PLANES_DB.

## 6. Chronological phase list (see `08_MASTER_CHECKLIST.md` for the full checklist)

1. Environment setup (`01_ENVIRONMENT_SETUP.md`)
2. Dataset acquisition — **heavy `[MANUAL]` content** (`02_DATASETS.md`)
3. Data pipeline: splitting, verification, preprocessing, synthetic augmentation (`03_DATA_PIPELINE.md`)
4. Model training: backbone comparison, primary classifier training (`04_MODEL_TRAINING.md`)
5. Temporal smoothing design + tuning, real-time serving pipeline (`05_TEMPORAL_SMOOTHING_AND_REALTIME.md`)
6. Evaluation & validation, including cross-device generalization (`06_EVALUATION_VALIDATION.md`)
7. Stretch goals: multi-task detection, export/quantization, UI polish (`07_STRETCH_GOALS_AND_ROADMAP.md`)

## 7. Reference literature (for the agent's context, not to re-read in full)

- SonoNet (Baumgartner et al., IEEE TMI 2017 / arXiv:1612.05601) — real-time fully-convolutional plane detection + weak localization
- FAUSP-NET (Sci Rep 2026, s41598-026-40590-8) — multi-task detection+classification, 24.1ms/frame on consumer GPU, backbone comparison table
- ScanAhead (Men et al., Med Image Anal 2025) — Swin-Transformer + explicit temporal extractor for video-based fetal head plane prediction
- 3DFETUS (arXiv 2511.10412) — spatial-transformer normalization pattern (not directly used, informs design philosophy)
- UCL/HC18 multi-centre benchmark (arXiv 2512.16710 / Sci Rep s41598-026-47854-3) — our cross-device generalization test set
- Original student repo (`inference.py`, `app.py`, `README.md`) — reference for what to keep (patient-level split, Grad-CAM concept, two-stage philosophy) vs. discard (Flask serving, per-frame double-forward-pass, no temporal handling)

## 8. Non-goals (explicitly out of scope for v1)

- No clinical validation, no regulatory claims, no real patient data
- No real ultrasound hardware integration
- No mobile app
- No multi-task detection head (stretch goal only)
- No production-grade auth/security/multi-user serving
