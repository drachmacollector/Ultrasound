# Demo 1 Walkthrough: Upload-a-Video, Watch-it-Get-Labeled

**File:** `docs/phases/demo1/demo1_walkthrough.md`
**Last Updated:** 2026-08-26
**Status:** Implementation complete — Demo 1 is functional.

---

## 1. What This Document Covers

This walkthrough documents the complete architecture, design decisions, implementation
details, bug history, and known limitations of **Demo 1**: a Gradio web application
that accepts an uploaded fetal ultrasound video clip and returns a fully-annotated MP4
with per-frame plane labels, confidence scores, and Grad-CAM saliency overlays.

It is intended as a **permanent reference** for anyone picking up this codebase, and
as the honest disclosure document for all deferred or limited functionality.

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
     │  Output written to outputs/demo1/ (Gradio allowed_paths)
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
outputs/demo1/annotated_<name>.mp4   (project-local, in Gradio's allowed_paths)
     │
     ▼
Browser inline playback (H.264/yuv420p — Chrome + Safari compatible)
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

**Output files in `outputs/demo1/`, not `tempfile.gettempdir()`**
See §8.2 (Gradio video serving bug) for why the temp directory caused "Video not
playable / Method not implemented" errors and how it was fixed.

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
    --output outputs/test_annotated.mp4 \
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
- `gr.Video(autoplay=True, format="mp4")` — annotated output, plays inline
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
| `requirements.txt` | Added `gradio>=5.0` and `imageio[ffmpeg]` |
| `README.md` | Replaced stale WIP real-time section with Demo 1 quick-start |
| `.gitignore` | Added `outputs/` (generated annotated videos, not tracked) |

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

**Documented explicitly in `app_gradio.py` (module docstring), `scripts/render_annotated_video.py`, and this file.**

The multitask object detection model (`checkpoints/multitask/`) was trained for exactly
**1 epoch** as a wiring smoke test. Its val macro-F1 is **0.8421** — *below* the
production classifier's 0.9183. The detection head produces garbage bounding boxes.

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
sonographer motion from one plane to another. Before showing the system to a hospital,
source 3–5 clips with deliberate manual transitions and measure transition lag directly.

### 7.3 Cross-Device Generalization Gap

Accuracy drops from 98.0% (in-distribution) to 83.2% on HC18/UCL (unseen devices).
92.8% of that gap is the model correctly falling back to "Other". On a hospital machine
that differs meaningfully from the Voluson E6/S8/S10 and Aloka training sources, expect
similar or worse degradation.

### 7.4 Render Speed

Offline rendering with Grad-CAM every frame is ~2–3× slower than real-time on an
RTX 4060. For a 30-second clip this is ~60–90 seconds — acceptable for a demo, but
plan accordingly for long recordings.

---

## 8. Bug History and Post-Launch Fixes (2026-08-26)

Three bugs were discovered and fixed after the initial Demo 1 implementation.
This section documents each in full so that the issue, root cause, and fix are
permanently recorded.

---

### 8.1 Short Synthetic Clips — N_FRAMES Never Updated

**Symptom:** The example clips in `data/processed/synthetic_clips/` were only
**16 frames long** (0.67 seconds at 24 fps). The clips served as examples in the
Gradio UI and for pipeline smoke testing, but a 0.67-second clip is too short to
demonstrate temporal smoothing behaviour meaningfully.

**Root cause:** `N_FRAMES = 16` had been set during early Phase 3 development
(when clips were just sanity checks). The constant was later updated to `N_FRAMES = 120`
in `scripts/generate_sample_clips.py` and the default argument in
`src/data/synthetic_video.py::generate_ego_motion_clip()` was set to `n_frames=120`,
but the generation script was never re-run. The stale 16-frame clips remained on disk.

**Fix:**
- Re-ran `scripts/generate_sample_clips.py` with `N_FRAMES = 120`, `FPS = 24`.
- All 16 example clips (2 per class × 8 classes) regenerated at **5.0 seconds each**
  (`120 frames @ 24fps`), giving enough duration to observe the STABLE badge settling
  and Tier-2a smoothing working across a clip.
- Also upgraded `src/data/synthetic_video.py::save_clip_as_mp4()` to write H.264
  via `imageio[ffmpeg]` (matching the render engine output) rather than OpenCV mp4v,
  so the example clips themselves play correctly inline in browsers.

**Command to regenerate clips if needed:**
```bash
conda run -n fetalplane python scripts/generate_sample_clips.py
# Writes 16 clips to data/processed/synthetic_clips/ at 120 frames @ 24fps
```

---

### 8.2 "Video not playable / Method not implemented" Error in Gradio

**Symptom:** After uploading a video and waiting for processing to complete, the
browser displayed **"Video not playable"** and the server console showed
**"Method not implemented"**.

**Root cause:** `process_video()` was writing the annotated output MP4 to
`tempfile.gettempdir()` (e.g. `C:\Users\Nakul\AppData\Local\Temp\ultrasound_demo1\`).

On Windows, Gradio 5's file-serving layer uses `allowed_paths` to decide which
directories it will serve files from. The system temp directory is **not** in
`allowed_paths` by default. When Gradio tried to stream the file back to the browser
it issued an internal redirect to a static file handler that raised
`NotImplementedError: Method not implemented` — which manifested as "Video not
playable" in the browser.

**Fix:** Two changes:

1. Output is now written to **`outputs/demo1/`** inside the project root:
   ```python
   output_dir = _ROOT / "outputs" / "demo1"
   ```
   This is a project-local, deterministic path that does not change between runs.

2. `demo.launch()` now passes `allowed_paths=[str(_ROOT / "outputs")]`:
   ```python
   demo.launch(..., allowed_paths=[_outputs_dir])
   ```
   Gradio then explicitly permits serving any file under `outputs/`, resolving
   the "Method not implemented" error completely.

3. `outputs/` was added to `.gitignore` so generated annotated videos are not
   accidentally committed.

4. The output `gr.Video` component now specifies `format="mp4"` explicitly so
   Gradio knows the container format without guessing from the file extension.

---

### 8.3 Photocopier Effect in Synthetic Clips

**Symptom:** The synthetic clips in `data/processed/synthetic_clips/` had a highly
unnatural motion appearance — the entire image appeared to slide/rotate as a single
rigid piece, like a photograph being slid under a camera lens. This is particularly
noticeable on ultrasound images which contain distinct texture layers (specular
highlights, acoustic shadows, grainy speckle) that should not move coherently.

**Root cause:** The original `generate_ego_motion_clip()` in `src/data/synthetic_video.py`
applied **a single `cv2.warpAffine` call per frame to the entire image**:

```python
M = cv2.getRotationMatrix2D((cx, cy), angle=rot, scale=1.0 + zoom)
M[0, 2] += dx
M[1, 2] += dy
warped = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
```

This is correct code but for a fundamentally wrong model. A rigid affine warp of
a flat 2D array means every pixel in the image — background tissue, mid-field anatomy,
near-field specular reflections, scan cone boundary — moves by exactly the same
translation/rotation. This is geometrically equivalent to sliding a piece of paper on
a scanner bed: a single plane of content undergoing a single rigid motion.

Real ultrasound probe motion creates **depth-dependent parallax**: structures at
different tissue depths move by *different amounts* relative to the probe face for the
same physical probe displacement:
- Near-field tissue (subcutaneous fat, uterine wall) moves more relative to the
  image frame than far-field structures (deep anatomy).
- Specular highlights and acoustic shadows are fixed relative to the anatomy being
  insonated, not the probe coordinate frame.
- The scan cone boundary (fan-shaped region) is fixed in the probe frame and should
  not translate at all.

No amount of tuning the random walk parameters can fix this — the problem is
structural, not parametric.

**Fix: Depth-Layered Parallax Compositing**

`src/data/synthetic_video.py` was rewritten to use a **two-layer frequency-domain
decomposition** approach:

**Step 1 — Decompose (done once, before the frame loop):**
```python
blur_sigma = max(3.0, w * 0.03)   # ~3% of image width
far_layer  = cv2.GaussianBlur(img_f32, (ksize, ksize), blur_sigma)
near_layer = img_f32 - far_layer  # high-frequency residual
```

- `far_layer` = strongly blurred version of the image → represents slowly-varying
  deep tissue background (low spatial frequency content).
- `near_layer` = difference between original and blurred → represents fine texture,
  specular highlights, speckle (high spatial frequency content).

**Step 2 — Warp each layer with a different affine (per frame):**
```python
PARALLAX_FAR  = 0.35   # deep tissue moves at 35% of full probe speed
PARALLAX_NEAR = 1.00   # surface texture moves at 100% of probe speed

M_far  = scale_affine(M_full, PARALLAX_FAR)
M_near = M_full.copy()   # full motion

far_warped  = cv2.warpAffine(far_layer,  M_far,  ...)
near_warped = cv2.warpAffine(near_layer, M_near, ...)
```

**Step 3 — Composite:**
```python
composite = np.clip(far_warped + near_warped, 0.0, 255.0)
```

This creates the visual impression that the background anatomy drifts slowly while
near-field texture/highlights shift more briskly — matching the qualitative
appearance of real probe hand tremor on ultrasound video. It is a **2.5D parallax**
technique (well-established in computational photography) applied to a medical
imaging context.

**Why this is sufficient for the stated purpose:**
These clips are used for:
1. **Temporal smoothing validation** — verifying that Tier-1/Tier-2a smoothing
   correctly handles within-class label oscillation. The classifier input is the
   individual frame, and the smoothing validation only needs the classifier to
   produce slightly varying outputs across frames, which the motion achieves.
2. **Pipeline smoke testing** — verifying that the full inference+render pipeline
   can process a video end-to-end without errors.
3. **Demo UI examples** — giving users something to upload when first opening the
   Gradio app.

For purposes 1–3, physically-accurate ultrasound simulation is unnecessary. The
parallax approach is visually convincing enough to not be immediately distracting,
while being computationally trivial (no 3D model, no rendering engine required).

**Parameters (tunable in `src/data/synthetic_video.py`):**
```python
PARALLAX_FAR  = 0.35   # 0 = no background motion, 1 = same as near field
PARALLAX_NEAR = 1.00   # scale of near-field motion relative to the walk value
blur_sigma = w * 0.03  # controls frequency cutoff between far/near layers
```

**What this does NOT do:**
- Does not simulate the scan cone boundary (would require masking the fan region).
- Does not simulate acoustic shadowing changes (would require a physical model).
- Does not simulate class transitions (clips are single-class by design).
- Is not suitable as training data augmentation for the classifier — real
  video frames from FETAL_PLANES_DB are used for training.

---

## 9. Verification Checklist

### Headless smoke tests (run before showing to anyone)

```bash
# 1. Import test
conda run -n fetalplane python -c \
    "from scripts.render_annotated_video import render_video; print('OK')"

# 2. Check clip durations (should all be 120 frames @ 24fps = 5.0s)
conda run -n fetalplane python -c "
import cv2, pathlib
for p in sorted(pathlib.Path('data/processed/synthetic_clips').glob('*.mp4')):
    cap = cv2.VideoCapture(str(p))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f'{p.name}: {n:.0f} frames @ {fps:.0f}fps = {n/fps:.1f}s')
"

# 3. Render a synthetic clip (no Grad-CAM for speed)
conda run -n fetalplane python scripts/render_annotated_video.py \
    --input  data/processed/synthetic_clips/Brain_Trans_thalamic_clip01.mp4 \
    --output outputs/test_annotated.mp4 \
    --no-gradcam

# 4. Gradio launch smoke test
#    Ctrl+C after seeing "Running on local URL: http://127.0.0.1:7860"
conda run -n fetalplane python app_gradio.py --no-browser
```

### Manual browser verification

1. Open `http://127.0.0.1:7860` in Chrome.
2. Upload one of the synthetic clips from `data/processed/synthetic_clips/`.
3. Leave all options at defaults and click **▶ Process Video**.
4. Verify:
   - [ ] Progress bar updates during render
   - [ ] Output video plays **inline** without "Video not playable" error
   - [ ] Label panel visible (class name, confidence %)
   - [ ] Stability badge (STABLE / SETTLING) visible
   - [ ] Grad-CAM overlay visible (heatmap on ultrasound image)
   - [ ] HUD overlay visible (inference timings)
   - [ ] JSON label log populated in the accordion
   - [ ] No bounding boxes shown (correct — classifier checkpoint)
5. Toggle Grad-CAM OFF, re-run — verify faster render, plain frame overlay.
6. Toggle Tier-2a OFF, re-run — verify output still produced (no crash).

---

## 10. Future Work (Demo 2 and Beyond)

| Item | Blocking issue | Path forward |
|---|---|---|
| Bounding boxes | Detection model needs full training | Train `configs/multitask.yaml` to convergence; set `_DEFAULT_CHECKPOINT` |
| Live webcam streaming | WebSocket + streaming needed | Gradio `gr.Video(sources=["webcam"])` + ONNX model for 30 fps target |
| ONNX export | Not started | `src/models/backbone.py` → `torch.onnx.export()` → quantize |
| Transition latency benchmark | No annotated transition clips | Source 3–5 real clips with deliberate plane changes; measure catch-up lag |
| Cross-device fine-tuning | Accuracy gap ~15% on HC18/UCL | Domain-adaptation or test-time augmentation study |
| Docker deployment | Not started | Single-stage Dockerfile with CUDA base, no conda |

---

*End of Demo 1 Walkthrough. Last updated 2026-08-26.*
