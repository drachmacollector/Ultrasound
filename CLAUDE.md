# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FetScan: real-time fetal ultrasound standard-plane classifier (7 planes + "Other", 8-way softmax) with temporal smoothing and Grad-CAM. Research project: no clinical claims, local GPU inference (RTX 4060, 8 GB VRAM). Trained on FETAL_PLANES_DB; HC18/UCL are the cross-device held-out set; NatalIA is phantom data from non-expert operators.

## Environment

- Conda env is named `fetalplane` (README also mentions `ultrasound_env`; `pyrightconfig.json` and the run commands use `fetalplane`). Python 3.11, CUDA 12.x.
- `requirements-lock.txt` is the pinned CUDA 12.8 torch install; `requirements.txt` is unpinned.
- Always run scripts **from the repo root** — many use `sys.path.insert(0, ".")` and relative paths (`data/splits/...`, `checkpoints/...`).
- `data/` is gitignored (except `.md` and `class_weights.json`).

## Commands

```bash
# Data pipeline (order matters)
python scripts/build_manifest.py
python scripts/patient_split.py              # patient-disjoint train/val/test
python scripts/build_cross_device_manifest.py
python scripts/run_sanity_checks.py

# Train (one YAML per backbone/ablation in configs/; writes checkpoints/<name>/{best.pt,config.yaml} and logs/)
python src/train/train.py --config configs/convnext_tiny.yaml [--epochs-override 2]
python scripts/train_multitask.py --config configs/multitask.yaml

# Evaluate
python scripts/evaluate_cross_device.py --checkpoint checkpoints/convnext_tiny/best.pt
python scripts/evaluate_transition_latency.py

# Demos
streamlit run app_streamlit.py                                    # web UI (primary)
python -m src.realtime.app --source <video.mp4> --loop [--debug]  # desktop OpenCV app; --debug enables Grad-CAM

# Tests (plain scripts under scripts/tests/, not a configured runner)
python -m pytest scripts/tests
python -m pytest scripts/tests/test_tier1.py -k flicker
```

No linter/build config exists.

## Architecture

- **`src/models/`**: `backbone.py` (timm backbones), `acam.py` (optional contrast module, wired as `nn.Sequential(acam, backbone)`), `gradcam.py`, `cam_localisation.py` (saliency-derived approximate boxes), `multitask_model.py` (detection head; stretch goal, only 1-epoch smoke-trained).
- **`src/train/`**: `train.py` is config-driven (YAML in `configs/`); `losses.py` has focal loss variants.
- **`src/data/`**: dataset/transforms/class weights, plus `synthetic_video.py` for building multi-plane test clips.
- **`src/smoothing/`**: `tier1.py` (confidence-weighted moving average + hysteresis + min dwell; params in `configs/smoothing_tier1.yaml`) and `tier2_mode_filter.py` (tier 2a, `configs/smoothing_tier2a.yaml`). Tier-1 state machine is the core anti-flicker mechanism.
- **`src/realtime/`**: `pipeline.py` is a threaded CaptureThread → InferenceThread with `DropOldestQueue` (size 2); `app.py` renders. Constraints: `cv2.imshow`/`waitKey` only on the main thread; the GradCAM instance is created once and reused (throttled); backbone, image size and normalization are **always read from `checkpoint["config"]`**, never hardcoded. Training/inference preprocessing must match exactly (`test_transforms_parity.py`).
- **`app_streamlit.py`**: uploads a clip, runs the same pipeline offline, outputs an annotated H.264 MP4. Default checkpoint `checkpoints/convnext_tiny/best.pt`.
- **`scripts/`**: dataset building, evaluation, tuning and clip-generation scripts (one-off, phase-driven).

## Key facts and gotchas

- Production model is plain `convnext_tiny` (test macro-F1 0.8927). The ACAM and focal variants were ablations that did not reliably win; don't switch the default without evidence.
- **Multitask checkpoints are trained on HC18/UCL**, so they must never be evaluated on `cross_device_manifest.csv` or compared to its numbers.
- Each backbone trains at its own native resolution/normalization (e.g. `tf_efficientnetv2_s` is 300px with 0.5/0.5 normalization, batch 24).
- Bounding boxes in the UI are Grad-CAM-derived approximations, not detector output; label them as such.
- Grad-CAM is off by default (clinical mode).
- Project history, design rationale and results live in `docs/` (`EXPERIMENTS.md`, `EVAL_REPORT.md`, `docs/instructions/00-09_*.md` specs, `docs/phases/`). Consult them before changing methodology.
