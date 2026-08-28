# 08 — Master Chronological Checklist

The single top-to-bottom list. Everything here is detailed in files `00`–`07`; this file exists so you (or a new chat/agent) can track progress at a glance without re-reading everything. Tags: `[M]` = you do this manually, `[A]` = the coding agent does this, `[M+A]` = you make a decision, agent implements it.

---

### Phase 0 — Orientation
- [ ] `[M]` Read [00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md) fully before starting anything
- [ ] `[M]` Confirm you agree with the scope decisions recorded there (single-track mid-pregnancy planes, no second clinical track, no multi-task head in v1)

### Phase 1 — Environment ([01_ENVIRONMENT_SETUP.md](01_ENVIRONMENT_SETUP.md))
- [ ] `[M]` Update NVIDIA driver, verify `nvidia-smi`
- [ ] `[M+A]` Create Python env, install PyTorch w/ CUDA, verify `torch.cuda.is_available()`
- [ ] `[A]` Scaffold repo structure
- [ ] `[M]` Decide TensorBoard vs. W&B (recommendation: TensorBoard)
- [ ] `[A]` Install `requirements.txt`
- [ ] `[M]` git init, first commit of empty scaffold

### Phase 2 — Datasets ([02_DATASETS.md](02_DATASETS.md))
- [ ] `[M]` Download FETAL_PLANES_DB, verify CSV schema, spot-check labels
- [ ] `[M]` Download UCL/HC18 multi-centre dataset (manual browser download — site blocks bots), read its train/test docs, understand overlap risk with FETAL_PLANES_DB
- [ ] `[M]` Download IUGC/Zenodo intrapartum video (optional but recommended for smoothing tuning)
- [ ] `[M]` Download NatalIA PBF-US1 dataset, verify `resume.csv` schema
- [ ] `[M]` Record license notes for all sources
- [ ] `[M]` Confirm final `data/raw/` folder structure matches the checklist in file 02

### Phase 3 — Data Pipeline ([03_DATA_PIPELINE.md](03_DATA_PIPELINE.md))
- [ ] `[A]` Build master manifest from FETAL_PLANES_DB CSV, canonical 8-class mapping
- [ ] `[A]` Patient-level train/val split; leakage-verification script (hard assert, not warning)
- [ ] `[A]` Build cross-device test manifest from UCL data; `[M]` verify no train/test contamination
- [ ] `[A]` Preprocessing transforms (resize, normalize, channel handling)
- [ ] `[M+A]` Class imbalance strategy chosen and implemented
- [ ] `[A]` Synthetic ego-motion clip generator; `[M]` manually watch sample clips for plausibility
- [ ] `[A]` Produce all sanity-check visualizations; `[M]` actually review them before Phase 4

### Phase 4 — Model Training ([04_MODEL_TRAINING.md](04_MODEL_TRAINING.md))
- [ ] `[A]` Implement config-driven training loop with TensorBoard logging
- [ ] `[A]` Smoke-test all 6 backbone candidates (RepVGG, MobileNetV3, EfficientNet-Lite0, EfficientNetV2-S, ConvNeXt-Tiny)
- [ ] `[A]` Full training runs, early stopping on val macro-F1
- [ ] `[A]` Pretraining-init ablation (`[M]` search for a usable public domain-specific checkpoint first)
- [ ] `[M+A]` Final backbone decision made empirically, documented
- [ ] `[A]` RepVGG re-parameterization + numerical verification (if applicable)
- [ ] `[A]` Grad-CAM module implemented; `[M]` visually spot-check activation maps

### Phase 5 — Temporal Smoothing & Real-Time Pipeline ([05_TEMPORAL_SMOOTHING_AND_REALTIME.md](05_TEMPORAL_SMOOTHING_AND_REALTIME.md))
- [ ] `[A]` Implement Tier-1 smoothing (EMA + hysteresis + dwell time)
- [ ] `[A]` Run classifier over IUGC (or synthetic fallback) video, log raw flicker baseline
- [ ] `[M+A]` Sweep and pick smoothing parameters; `[M]` mark manual transition points for lag measurement
- [ ] `[A]` (Only if needed) Tier-2 learned temporal module
- [ ] `[A]` Build capture abstraction (file + webcam)
- [ ] `[A]` Build threaded capture/inference/render pipeline with bounded drop-oldest queues
- [ ] `[A]` Implement Grad-CAM throttling
- [ ] `[A]` Implement latency/FPS instrumentation
- [ ] `[M+A]` Decide and build UI (cv2 window first; Streamlit/Gradio as polish)
- [ ] `[M]` Test end-to-end on a real video file first, then webcam

### Phase 6 — Evaluation & Validation ([06_EVALUATION_VALIDATION.md](06_EVALUATION_VALIDATION.md))
- [ ] `[A]` In-distribution test metrics (per-class + macro-F1, confusion matrix)
- [ ] `[A]` Cross-device generalization metrics, gap explicitly reported
- [ ] `[A]` Realistic-video frame-level accuracy, with/without smoothing
- [x] `[A]` Video stability metrics (switches/min, latency-to-stabilize (mean 489.3 ms), dwell time)
- [ ] `[A]` Run all planned ablations
- [ ] `[M+A]` Write final [EVAL_REPORT.md](../EVAL_REPORT.md) including honest limitations section

### Phase 7 — Stretch Goals ([07_STRETCH_GOALS_AND_ROADMAP.md](07_STRETCH_GOALS_AND_ROADMAP.md)) — only after Phases 0-6 are fully done
- [ ] Detection-informed multi-task head
- [ ] ONNX/TensorRT export + quantization
- [x] Tier-2a majority-vote mode filter (`src/smoothing/tier2_mode_filter.py`)
  - Sweep: dual grid A+B, 42 combos. Selected: `window=9, min_majority_frac=0.7` (338ms added lag).
  - Result: 100% flicker elimination on all 46 eval clips (72→0 switches/min on stubborn clip, 0 spurious on all others).
  - Decision: Tier-2b (learned GRU) NOT built — Tier-2a sufficient. See `docs/phases/phase_07/tier2_results.md`.
  - Caveat (documented in results): validated on a single ~20-frame (~833ms) clip with ~1 switch. Win is real but not validated on sustained real-world oscillation.
  - Pipeline integration: opt-in via `--enable-tier2` / `--tier2-config` flags (off by default).
- [ ] Tier-2b learned temporal head (deferred — Tier-2a sufficient)
- [ ] Web UI polish
- [ ] Proper domain-specific self-supervised pretraining
- [ ] Second clinical track (intrapartum monitoring), if desired

---

## If you're starting a brand-new chat with just this checklist

Paste in, at minimum: this file + [00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md). Add the phase-specific file (`01`-`07`) matching whatever step you've reached. You don't need to paste all nine files into every new chat — just the overview plus whichever phase you're actively working on, and the checklist to show where you are.
