# Demo 1 Walkthrough: Upload-a-Video, Watch-it-Get-Labeled

**File:** `docs/phases/demo1/demo1_walkthrough.md`
**Date:** 2026-08-25
**Status:** Implementation complete — Demo 1 is functional.

---

## 1. What This Document Covers

This walkthrough documents the complete architecture, design decisions, implementation
details, and known limitations of **Demo 1**: a Gradio web application that accepts an
uploaded fetal ultrasound video clip and returns a fully-annotated MP4 with per-frame
plane labels, confidence scores, and Grad-CAM saliency overlays.

It is intended as a **permanent reference** for anyone picking up this codebase, and
as the honest disclosure document required by the plan (documenting why bounding boxes
are absent, what the known accuracy gaps are, and what was deliberately deferred).

---

## 2. System Context — What Was Already Done

Before Demo 1 was built, the following components were complete and validated:

| Component | Status | Key Numbers |
|---|---|---|
| Data pipeline | ✅ Done | Patient-disjoint splits, leakage verification, cross-device manifest |
| Model training (6 backbones) | ✅ Done | `convnext_tiny` wins: test macro-F1 **0.8927** |
| Tier-1 smoothing | ✅ Done | 210-combo tuning sweep, EMA α=0.2 + hysteresis + dwell=8 |
| Tier-2a mode filter | ✅ Done | 100% residual flicker elimination across 46 eval clips |
| Threaded real-time pipeline | ✅ Done | 23.6–23.7 fps stable on RTX 4060, 120s no-throttle run |
| Headless validation script | ✅ Done | `scripts/validate_pipeline_headless.py` |
| Desktop app (`cv2.imshow`) | ✅ Done | `src/realtime/app.py` |
| **Web UI (Demo 1)** | ✅ **This document** | `app_gradio.py` + `scripts/render_annotated_video.py` |
| Multitask detection model | 🟡 Smoke-test only | 1 epoch, val F1 0.8421 — NOT used in Demo 1 |
| ONNX / quantization | ❌ Stretch goal | Not started |

---

## 3. Architecture Overview

```
Browser upload
     │
     ▼
app_gradio.py (Gradio Blocks)
     │  gr.Video in → process_video() → gr.Video out
     │
     ▼
scripts/render_annotated_video.py :: render_video()
     │
     ├── load_inference_model(checkpoint)          [src/realtime/model_loader.py]
     │       └── reads config FROM checkpoint — backbone, img_size, norm stats
     │           never hardcoded
     │
     ├── cv2.VideoCapture(input_path)              [source video]
     │
     ├── Tier1Smoother(configs/smoothing_tier1.yaml)   [src/smoothing/tier1.py]
     │       α=0.2, switch_threshold=0.7, min_dwell_frames=8
     │
     ├── Tier2ModeFilter(configs/smoothing_tier2a.yaml)  [src/smoothing/tier2_mode_filter.py]
     │       window_frames=9, min_majority_frac=0.7
     │       (optional — checkbox in UI, default ON)
     │
     ├── GradCAM(model, target_layer)              [pytorch_grad_cam]
     │       Persistent instance — hooks registered ONCE, reused per-frame
     │       Runs every gradcam_every_n frames (slider, default 1)
     │
     └── Per-frame loop:
             prep_frame_grayscale_to_rgb()         [src/data/transforms.py]
             → eval albumentations transform
             → torch.no_grad() + autocast forward pass
             → Tier1Smoother.step()
             → Tier2ModeFilter.step()       (if enabled)
             → GradCAM()                    (if frame_idx % every_n == 0)
             → build_display_frame()        [src/realtime/app.py]
             → imageio H.264 writer
     │
     ▼
Browser inline playback (H.264 MP4)
+ JSON label log (per-frame label / confidence / timestamp)
```

### Key design decisions

**Single-threaded offline loop (not CaptureThread + InferenceThread)**
The live desktop app uses a two-thread architecture (capture + inference) because
it must maintain ≤200ms Grad-CAM cadence against a hardware-paced webcam at 23.7 fps.
For offline rendering there is no hard latency budget — the user waits for the whole
render anyway. The simpler sequential loop is preferred: fewer moving parts, easier
to reason about progress reporting, no queue-drop complexity.

**`build_display_frame()` reused without modification**
The annotated frame renderer in `src/realtime/app.py` was intentionally written
to be display-agnostic — it produces a BGR `np.ndarray` and does not know whether
its output goes to `cv2.imshow`, `cv2.imwrite`, a `VideoWriter`, or imageio.
`render_annotated_video.py` calls it identically to how `validate_pipeline_headless.py`
already called it. Zero renderer code was duplicated.

**H.264 via imageio-ffmpeg, not OpenCV mp4v**
OpenCV's default `mp4v` fourcc produces files that Chrome and Safari frequently
cannot play inline (missing moov atom positioning, unsupported profile). `imageio[ffmpeg]`
ships its own ffmpeg binaries (no system install required) and writes proper H.264
with `pix_fmt yuv420p` — the broadest browser-compatible profile. The writer is
invoked as a streaming writer (one `append_data()` per frame), so memory usage is
bounded regardless of clip length.

**Persistent GradCAM instance, not `run_gradcam()`**
`src/models/gradcam.py::run_gradcam()` creates a new GradCAM context-manager on
every call, re-registering hooks each time (~5–15ms overhead). `InferenceThread`
already solved this by creating one GradCAM instance in `__init__` and keeping its
hooks alive. `render_annotated_video.py` follows the same pattern.

**Config always read from checkpoint**
The checkpoint stores the full training config under `ckpt["config"]`. Backbone name,
image size, normalization mean/std are all read from there via `load_inference_model()`.
Nothing is hardcoded in the render script or the Gradio app.

---

## 4. New Files Created

### `scripts/render_annotated_video.py`

The inference + rendering engine. Importable as a library (`from scripts.render_annotated_video import render_video`) or runnable as a CLI script.

**Public API:**
```python
result = render_video(
    input_path="clip.mp4",
    output_path="annotated.mp4",
    checkpoint="checkpoints/convnext_tiny/best.pt",
    smoothing_config="configs/smoothing_tier1.yaml",
    tier2_config="configs/smoothing_tier2a.yaml",
    enable_tier2=True,
    enable_gradcam=True,
    gradcam_every_n=1,
    show_hud=True,
    progress_cb=lambda done, total: ...,  # optional — used by Gradio gr.Progress
)
# result["output_path"]  : str — absolute path to written MP4
# result["total_frames"] : int
# result["fps_in"]       : float
# result["label_log"]    : list[dict] — per-frame label/confidence/timestamp
# result["model_info"]   : dict — backbone, F1, smoothing params, bbox_note
# result["elapsed_s"]    : float
```

**CLI:**
```bash
python scripts/render_annotated_video.py \
    --input  data/processed/synthetic_clips/Brain_Trans_thalamic_clip01.mp4 \
    --output /tmp/annotated.mp4 \
    --checkpoint checkpoints/convnext_tiny/best.pt \
    --gradcam-every-n 1 \
    --enable-tier2
```

**Approximate render times** (RTX 4060, with Grad-CAM every frame):
- 30-second clip @ 25 fps (750 frames): ~60–90 seconds
- 10-second clip @ 25 fps (250 frames): ~20–30 seconds

Disabling Grad-CAM (`--no-gradcam`) roughly halves render time.

### `app_gradio.py`

The Gradio Blocks web application. Thin wrapper around `render_video()`.

**Key UI elements:**
- `gr.Video(sources=["upload"])` — input
- `gr.Video(autoplay=True)` — annotated output, plays inline
- Grad-CAM toggle + every-N slider
- Tier-2a checkbox (default ON)
- HUD toggle
- Per-frame JSON label log (first 200 frames shown, rest truncated)
- Model info accordion (backbone, F1, known limitations)
- Auto-discovered example clips from `data/processed/synthetic_clips/*.mp4`

**Launch:**
```bash
python app_gradio.py               # http://127.0.0.1:7860
python app_gradio.py --share       # public Gradio tunnel
python app_gradio.py --port 8080   # custom port
python app_gradio.py --no-browser  # headless (CI smoke test)
```

### Modified files

| File | Change |
|---|---|
| `requirements.txt` | Added `gradio>=4.0,<5.0` and `imageio[ffmpeg]` |
| `README.md` | Replaced stale WIP real-time section with Demo 1 quick-start |

---

## 5. Smoothing Parameters (Reference)

### Tier-1 (`configs/smoothing_tier1.yaml`)

Selected by a 210-combination sweep over real video clips (Phase 5 Task 6).

```yaml
alpha: 0.2               # EMA weight on raw softmax probs (lower = smoother)
switch_threshold: 0.7    # smoothed probability must exceed this to trigger switch
min_dwell_frames: 8      # hold current label for ≥8 frames after switch
hold_floor: null         # no probability floor applied
```

**Result:** 788.75 → 72 switches/min (on 46 eval clips with spurious flicker).

### Tier-2a (`configs/smoothing_tier2a.yaml`)

Selected by a separate grid search (Phase 5 Task 7b / Tier-2 tuning).

```yaml
window_frames: 9         # trailing window for majority vote
min_majority_frac: 0.7   # fraction of window that must agree before switching display
```

**Result:** 72 → 0 switches/min (100% residual flicker elimination).
Worst-case added latency: **338ms** at 23.7 fps (9 frames × 42.19ms/frame).

---

## 6. Model Performance (Reference)

All numbers from `EVAL_REPORT.md` and `EXPERIMENTS.md`.

### Per-class F1 (test set, convnext_tiny, no smoothing)

| Class | F1 | Note |
|---|---|---|
| Brain_Trans_cerebellum | 0.8924 | |
| Brain_Trans_thalamic | 0.8816 | |
| Brain_Trans_ventricular | 0.7721 | **Weakest** — documented literature confusion with thalamic |
| Fetal_abdomen | 0.8804 | |
| Fetal_femur | 0.8703 | |
| Fetal_thorax | 0.9429 | |
| Maternal_cervix | 0.9413 | |
| Other | 0.9411 | |
| **Macro average** | **0.8927** | |

### Cross-device generalization

| Dataset | Accuracy (head only) | Notes |
|---|---|---|
| In-distribution test | 98.0% | FETAL_PLANES_DB held-out split |
| HC18/UCL unseen devices | 83.2% | 92.8% of gap = correct "Other" fallback |

---

## 7. Known Limitations and Deferred Work

### 7.1 No Bounding Boxes in Demo 1 ⚠️

**This is documented explicitly in `app_gradio.py` (module docstring), `scripts/render_annotated_video.py` (DESIGN DECISIONS section), and this file.**

The multitask object detection model (`checkpoints/multitask/`) was trained for exactly
**1 epoch** as a wiring smoke test to verify the multitask architecture was correctly
integrated. Its val macro-F1 is **0.8421** — *below* the production classifier's 0.9183.
The detection head would produce garbage bounding boxes.

`build_display_frame()` in `src/realtime/app.py` calls `_draw_bboxes()`, which
**safely no-ops** when `result["bboxes"] is None`. Since `load_inference_model()` loads
the classifier-only checkpoint (no `retinanet.*` keys in state dict), `bboxes` is always
`None` — no guard code is needed in the render loop.

**Path to enabling boxes in a future demo:**
1. Train the multitask model for the full configured epochs (`configs/multitask.yaml`)
2. Verify val macro-F1 exceeds the production classifier (0.9183 threshold)
3. Load the multitask checkpoint in `app_gradio.py` (change `_DEFAULT_CHECKPOINT`)
4. Boxes will appear automatically — `_draw_bboxes()` and the render pipeline are
   already fully wired.

### 7.2 Genuine Transition Latency Not Benchmarked

Every Tier-1/Tier-2a smoothing number (788.75 → 72 → 0 switches/min) was measured
on **spurious within-class oscillation** in stable clips — because no available video
source contained annotated genuine plane-to-plane transitions.

It is currently unknown how many frames/ms the system takes to follow a genuine
sonographer motion from one plane to another. This is the single most important unknown
for a live clinical demo. Before showing the system to a hospital, source 3–5 clips
with deliberate manual transitions and measure transition lag directly.

### 7.3 Cross-Device Generalization Gap

Accuracy drops from 98.0% (in-distribution) to 83.2% on HC18/UCL (unseen devices).
92.8% of that gap is the model correctly falling back to "Other" rather than confidently
misclassifying — the safer failure mode. However, on a hospital machine that differs
meaningfully from the Voluson E6/S8/S10 and Aloka training sources, expect degradation
similar to or potentially worse than this, per SonoNet's well-known finding that
curated-image accuracy does not transfer 1:1 to live scanning.

### 7.4 Render Speed

Offline rendering with Grad-CAM every frame is ~2–3× slower than real-time on an
RTX 4060. For a 30-second clip this is ~60–90 seconds — acceptable for a demo, but
plan accordingly for long recordings.

---

## 8. Verification Checklist

### Headless smoke tests (run before showing to anyone)

```bash
# 1. Import test
python -c "from scripts.render_annotated_video import render_video; print('OK')"

# 2. Render a synthetic clip (no Grad-CAM for speed)
python scripts/render_annotated_video.py \
    --input  data/processed/synthetic_clips/Brain_Trans_thalamic_clip01.mp4 \
    --output /tmp/test_annotated.mp4 \
    --no-gradcam

# 3. Verify output is a valid video
python -c "
import imageio
r = imageio.get_reader('/tmp/test_annotated.mp4')
print(f'Frames: {len(list(r))} — OK')
"

# 4. Gradio launch smoke test (--no-browser suppresses auto tab open)
#    Ctrl+C after seeing "Running on local URL: http://127.0.0.1:7860"
python app_gradio.py --no-browser
```

### Manual browser verification

1. Open `http://127.0.0.1:7860` in Chrome and Safari (both).
2. Upload one of the synthetic clips from `data/processed/synthetic_clips/`.
3. Leave all options at defaults and click **▶ Process Video**.
4. Verify:
   - [ ] Progress bar updates during render
   - [ ] Output video plays **inline** without "unsupported format" error in Chrome
   - [ ] Output video plays **inline** in Safari (H.264/yuv420p compatibility)
   - [ ] Label panel visible (class name, confidence %)
   - [ ] Stability badge (STABLE / SETTLING) visible
   - [ ] Grad-CAM overlay visible (heatmap on ultrasound image)
   - [ ] HUD overlay visible (inference timings)
   - [ ] JSON label log populated in the accordion
   - [ ] No bounding boxes shown (correct — classifier checkpoint)
5. Toggle Grad-CAM OFF, re-run — verify faster render, plain frame overlay.
6. Toggle Tier-2a OFF, re-run — verify output still produced (no crash).

---

## 9. Future Work (Demo 2 and Beyond)

| Item | Blocking issue | Path forward |
|---|---|---|
| Bounding boxes | Detection model needs full training | Train `configs/multitask.yaml` to convergence; set `_DEFAULT_CHECKPOINT` |
| Live webcam streaming | WebSocket + streaming needed | Gradio `gr.Video(sources=["webcam"])` + ONNX model for 30 fps target |
| ONNX export | Not started | `src/models/backbone.py` → `torch.onnx.export()` → quantize |
| Transition latency benchmark | No annotated transition clips | Source 3–5 real clips with deliberate plane changes; measure catch-up lag |
| Cross-device fine-tuning | Accuracy gap ~15% on HC18/UCL | Domain-adaptation or test-time augmentation study |
| Docker deployment | Not started | Single-stage Dockerfile with CUDA base, no conda |

---

*End of Demo 1 Walkthrough. Generated 2026-08-25.*
