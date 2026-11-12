# PHASE_5_KICKOFF_PROMPT.md — Temporal Smoothing & Real-Time Pipeline

Read, in this order, before writing any code: [docs/00_PROJECT_OVERVIEW.md](../instructions/00_PROJECT_OVERVIEW.md), [docs/05_TEMPORAL_SMOOTHING_AND_REALTIME.md](../instructions/05_TEMPORAL_SMOOTHING_AND_REALTIME.md), [docs/09_MASTER_CHECKLIST.md](../instructions/09_MASTER_CHECKLIST.md), [docs/EXPERIMENTS.md](../EXPERIMENTS.md) (for the final backbone decision), and this file in full. This file is more prescriptive than [05_TEMPORAL_SMOOTHING_AND_REALTIME.md](../instructions/05_TEMPORAL_SMOOTHING_AND_REALTIME.md) on purpose — where the two disagree on a concrete detail, follow this file, since it was written after seeing the actual Phase 4 outputs (checkpoint contents, config schema, existing Grad-CAM implementation) that [05\_...md](../instructions/05_TEMPORAL_SMOOTHING_AND_REALTIME.md) was written without.

Phases 0–4 are complete and verified. The winning model is **`convnext_tiny.fb_in22k_ft_in1k`**, trained with class-weighted cross-entropy (not the focal-loss variant), checkpoint at [checkpoints/convnext_tiny/best.pt](../checkpoints/convnext_tiny/best.pt). Test macro-F1 = 0.8927.

---

## 0. Non-negotiable constraints — read before writing anything

1. **Do not modify** `data/raw/`, `data/processed/` (except adding new files under new subfolders you create), `src/data/`, `src/models/`, `src/train/`, `src/eval/`, or anything under `checkpoints/`. Phase 5 code lives in `src/smoothing/`, `src/realtime/`, new files under `scripts/`, new files under `configs/`, and `docs/`.
2. **Never hardcode `image_size`, `backbone`, or normalization stats in the real-time path.** Every one of these must be read from `checkpoint["config"]` at load time, exactly the way `src/eval/evaluate_test.py::evaluate_checkpoint()` already does it. This is not optional — it's the difference between a script that stays correct if the checkpoint ever changes and one with a silent landmine in it.
3. **Reuse existing modules — do not reimplement them:**
   - `src.data.dataset.CANONICAL_CLASSES` / `IDX_TO_CLASS` / `NUM_CLASSES` for label mapping.
   - `src.models.backbone.build_model()` for model construction.
   - `src.data.transforms.get_eval_transform()` for the albumentations pipeline.
   - `src.models.gradcam.run_gradcam()` / `get_target_layer()` for Grad-CAM (already has the correct verified target-layer path for `convnext_tiny.fb_in22k_ft_in1k`: `["stages", 3, "blocks", -1]`).
4. **Grad-CAM must be throttled — never run on every frame.** Per `05_TEMPORAL_SMOOTHING_AND_REALTIME.md` §B2, run it every K processed frames or on a fixed wall-clock cadence (default: every 200ms), and reuse the last computed overlay on skipped frames.
5. **All `cv2.imshow` / `cv2.waitKey` calls must happen on the main thread.** This is a hard cross-platform requirement (macOS enforces this; Windows/Linux are more lenient but you should not rely on that). Capture and inference run on background threads; only rendering happens on main.
6. **Tier-2 (learned temporal module) is explicitly out of scope for this phase unless Tier-1 tuning empirically proves insufficient** (per §A3 of the doc). If your Task 7 tuning sweep can't get label-switches-per-minute and latency-to-stabilize into a reasonable range, **stop and report this to the user before building anything** — do not silently start building a GRU/1D-conv temporal head. This is a substantial scope increase and needs explicit sign-off.
7. **Streamlit/Gradio is explicitly out of scope for this phase.** The user has already made the UI decision (§B4 of the doc): `cv2.imshow` first, full stop, for this phase. Do not scaffold a Streamlit app "just in case." That's a separate, later task.
8. **The two manual checkpoints in `05_TEMPORAL_SMOOTHING_AND_REALTIME.md` are handled as follows:**
   - **§B4 (cv2.imshow vs. Streamlit/Gradio) is already decided** — build cv2.imshow now. Do not ask the user about this again.
   - **§A2 (watching ~5 video clips and marking transition points) is NOT decided and requires the user's direct input.** See Task 6 below — you must physically stop execution and wait for the user's response before proceeding to Task 7. This is a hard gate, not a suggestion.
9. **Log significant metrics and debug output to `logs/`.** Whenever a script outputs significant metrics, detailed sweep results, or valuable data that helps debug or pinpoint an issue, you must explicitly write this output to a suitably named `.txt` file in the `logs/` directory (strictly `encoding="utf-8"`). Small, one-off commands (like the Preflight checks) should be reported directly in the chat window, but bulk data, frame-by-frame outputs, or detailed runtime metrics must go to a log file.

---

## 1. Preflight — repository & data state check (do this first, report back before writing code)

Run and report the output of all of these before touching any code:

```bash
conda activate fetalplane
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then, in a small throwaway script (or interactively), load the checkpoint and print its saved config so you know exactly what you're serving:

```python
import torch
ckpt = torch.load("checkpoints/convnext_tiny/best.pt", map_location="cpu", weights_only=False)
print(ckpt["config"])
print("epoch:", ckpt["epoch"], "val_macro_f1:", ckpt["val_macro_f1"])
```

Confirm this prints `backbone: convnext_tiny.fb_in22k_ft_in1k`, `image_size: 224`, `normalize_mean/std: ImageNet values` — i.e., confirm you're loading the CE checkpoint, not `checkpoints/convnext_tiny_focal/best.pt`.

Then check data availability and report which branch of Task 6 applies:

```bash
ls data/raw/iugc_video/DatasetV3/ 2>/dev/null || echo "IUGC NOT PRESENT"
ls data/processed/synthetic_clips/ | wc -l
```

**Report explicitly:** is `data/raw/iugc_video/DatasetV3/{train,val,test}/videos/` present and populated? This single fact determines whether Task 6's manual checkpoint applies at all (see Task 6). Do not guess — check and state it plainly in your first response back to the user.

Confirm `src/smoothing/` and `src/realtime/` currently contain only `.gitkeep` (expected — this phase populates them).

---

## 2. Task 1 — Preprocessing parity function for live frames (train/serve skew prevention)

`src.data.transforms.load_and_prep_grayscale_to_rgb()` takes a file **path** and does `cv2.imread(path, IMREAD_GRAYSCALE)` → `cv2.cvtColor(..., GRAY2RGB)`. Live video/webcam frames arrive as **in-memory BGR arrays** from `cv2.VideoCapture.read()`, not paths — there is currently no function for this case, and writing the real-time preprocessing "close enough by hand" risks silent train/serve skew (a very common source of real-world model degradation that would otherwise be invisible until you notice predictions look off).

Add, in `src/data/transforms.py` (this file is inside the "don't modify" list above **except** for this one additive, backward-compatible function — do not touch anything else in the file):

```python
def prep_frame_grayscale_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """In-memory equivalent of load_and_prep_grayscale_to_rgb() for live video/webcam
    frames (cv2.VideoCapture.read() output), which arrive as BGR arrays, not file paths.
    Must produce bit-identical output to the path-based function for the same underlying
    image, to guarantee training/serving preprocessing parity.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return img_rgb
```

Write a small verification test: save a frame to disk as PNG, load it both via `cv2.imread(path, IMREAD_GRAYSCALE)` (path-based) and by reading it back as a BGR array and calling `prep_frame_grayscale_to_rgb()` (array-based), and assert `np.array_equal()` on the two outputs. Put this in `src/data/test_transforms_parity.py`. This must pass before you proceed — if it doesn't, something about channel order or colorspace conversion is wrong and everything downstream will be subtly miscalibrated.

---

## 3. Task 2 — Centralized model loading for inference

Create `src/realtime/model_loader.py`:

```python
"""
src/realtime/model_loader.py

Single source of truth for loading a trained checkpoint for real-time inference.
Mirrors src/eval/evaluate_test.py's loading pattern exactly — config comes from
the checkpoint, never from a hardcoded default, per PHASE_5_KICKOFF_PROMPT.md §0.2.
"""
from dataclasses import dataclass
from pathlib import Path
import torch
from src.models.backbone import build_model
from src.data.transforms import get_eval_transform

@dataclass
class LoadedModel:
    model: torch.nn.Module
    backbone_name: str
    device: torch.device
    img_size: int
    normalize_mean: tuple
    normalize_std: tuple
    transform: object  # albumentations.Compose

def load_inference_model(ckpt_path: str, device: torch.device | None = None) -> LoadedModel:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    backbone_name = cfg["backbone"]
    img_size = cfg.get("image_size", 224)
    mean = tuple(cfg.get("normalize_mean", [0.485, 0.456, 0.406]))
    std = tuple(cfg.get("normalize_std", [0.229, 0.224, 0.225]))

    model = build_model(backbone_name, num_classes=8, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    transform = get_eval_transform(img_size=img_size, mean=mean, std=std)
    return LoadedModel(model, backbone_name, device, img_size, mean, std, transform)
```

Every script and pipeline component in this phase must go through this function to obtain the model. No other file should call `build_model()` or `torch.load()` directly for inference purposes.

---

## 4. Task 3 — Tier-1 temporal smoothing module

Create `src/smoothing/tier1.py`. Implement exactly the state machine from `05_TEMPORAL_SMOOTHING_AND_REALTIME.md` §A1 — **note a real ambiguity in that doc you need to resolve, not silently pick one interpretation of**: the prose says "hold at >0.4 but require >0.6 to switch," which implies two thresholds (a hold-floor and a switch-ceiling), but the pseudocode given only implements one (`switch_threshold`) and holds unconditionally on the current label regardless of confidence. **Implement the pseudocode's actual logic** (single `switch_threshold`, unconditional hold on match), but add an optional `hold_floor: float | None = None` parameter that, if set, would additionally require `candidate_confidence > hold_floor` even to keep holding the current label — default it to `None` (disabled) so behavior matches the literal pseudocode unless the tuning phase (Task 7) finds a reason to enable it. Document this design choice with a comment in the file, and mention it explicitly in the final walkthrough.

```python
"""
src/smoothing/tier1.py

Tier-1 temporal smoothing: EMA + hysteresis + minimum dwell time.
Per docs/05_TEMPORAL_SMOOTHING_AND_REALTIME.md §A1.
"""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Tier1Smoother:
    num_classes: int
    alpha: float = 0.3
    switch_threshold: float = 0.6
    min_dwell_frames: int = 5
    hold_floor: float | None = None  # see design note above; None = literal pseudocode behavior

    smoothed_probs: np.ndarray = field(init=False, default=None)
    current_displayed_label: int = field(init=False, default=-1)
    frames_since_last_switch: int = field(init=False, default=0)

    def reset(self):
        self.smoothed_probs = None
        self.current_displayed_label = -1
        self.frames_since_last_switch = 0

    def step(self, raw_probs: np.ndarray) -> tuple[int, float, np.ndarray, bool]:
        """
        Args:
            raw_probs: [num_classes] raw per-frame softmax output.
        Returns:
            (current_displayed_label, candidate_confidence, smoothed_probs, is_stable)
            is_stable = True once frames_since_last_switch >= min_dwell_frames
            (i.e. the displayed label has held long enough to be considered settled,
            not mid-transition). Used by the render loop's "STABLE / SETTLING" indicator.
        """
        if self.smoothed_probs is None:
            self.smoothed_probs = raw_probs.copy()
            self.current_displayed_label = int(np.argmax(self.smoothed_probs))
        else:
            self.smoothed_probs = self.alpha * raw_probs + (1 - self.alpha) * self.smoothed_probs

        candidate_label = int(np.argmax(self.smoothed_probs))
        candidate_confidence = float(self.smoothed_probs[candidate_label])

        if candidate_label == self.current_displayed_label:
            self.frames_since_last_switch += 1  # NOTE: counts UP while holding (see below)
        elif (candidate_confidence > self.switch_threshold
              and self.frames_since_last_switch >= self.min_dwell_frames):
            self.current_displayed_label = candidate_label
            self.frames_since_last_switch = 0
        else:
            self.frames_since_last_switch += 1

        is_stable = self.frames_since_last_switch >= self.min_dwell_frames
        return self.current_displayed_label, candidate_confidence, self.smoothed_probs.copy(), is_stable
```

**Important correction to the doc's own pseudocode:** the doc's step 4 resets `frames_since_last_switch = 0` on hold, which would mean `frames_since_last_switch` never actually accumulates while holding a stable label, making `min_dwell_frames` only ever gate _switches away from a fresh state_, not "how long has the current label been stable." The implementation above instead counts `frames_since_last_switch` **up** while holding, and only resets it to 0 at the moment of an actual switch — this is necessary to make `is_stable` meaningful (otherwise it would be true/false only immediately after each switch and never reflect ongoing stability). Verify this reasoning yourself against a few synthetic sequences before trusting it; if you disagree with this correction after testing, flag it in the walkthrough rather than silently reverting to the doc's literal (and, I believe, buggy) reset behavior.

Write `src/smoothing/test_tier1.py` with at least these three cases, using hand-constructed sequences of softmax vectors:

1. **Stable input** (same class dominant every frame) → smoother should track it immediately and stay `is_stable=True`.
2. **Rapid flicker** (argmax alternates between two classes every frame, neither confidently) → smoother should suppress this and keep displaying one label without switching.
3. **Genuine class change** (first N frames class A confidently, then a clean switch to class B confidently for the remaining frames) → smoother should eventually switch to B and reach `is_stable=True` again within a bounded number of frames, and log/assert what that frame-lag actually was.

---

## 5. Task 4 — Baseline (unsmoothed) flicker measurement

Before tuning anything, measure how bad the problem actually is. Create `scripts/measure_baseline_flicker.py`:

- Load the model via `load_inference_model("checkpoints/convnext_tiny/best.pt")`.
- If `data/raw/iugc_video/DatasetV3/` is present: run frame-by-frame inference (raw argmax, no smoothing) over a representative sample of clips (suggest: 20–30 clips spanning `train/`, `val/`, `test`, mixing `pos=TRUE` and `pos=FALSE` per `train_info.csv`/`val_info.csv`/`test_info.csv`). Use `cv2.VideoCapture` directly (do not build the full threaded pipeline yet for this — it's a straight sequential script).
- Also run it over the 16 existing `data/processed/synthetic_clips/*.mp4` from Phase 3 regardless of IUGC availability (these are always available and give you the within-class jitter number even if IUGC is missing).
- For each clip, log: raw label-switches-per-second, mean/median confidence at switch points, mean/median confidence when stable. Aggregate into `data/processed/tier1_tuning/baseline_flicker_report.csv` and a summary Markdown table.
- While processing the clips, also write the detailed per-clip progress and raw metrics to a log file (e.g., `logs/measure_baseline_flicker.txt` with strictly `encoding="utf-8"`) so the verbose output is preserved for debugging.
- **This also gives you the real achieved inference FPS on this hardware for convnext_tiny at 224×224** — record this number explicitly, you need it in Task 7 to convert `min_dwell_frames` into a target millisecond range (the doc wants ~150–500ms dwell; you cannot pick a frame count sensibly without knowing real fps first).
- Produce one chart (`data/processed/tier1_tuning/baseline_flicker_chart.png`): switches-per-second per clip, sorted, so it's visually obvious how bad unsmoothed flicker is before you show the "after smoothing" comparison later.

---

## 6. Task 5 — MANUAL CHECKPOINT: transition annotation request (conditional — read carefully)

**This step only applies if `data/raw/iugc_video/DatasetV3/` was confirmed present in Task 0 (Preflight).** If it is not present, skip directly to the note at the end of this section — do not attempt to fabricate transition ground truth from synthetic clips, since synthetic ego-motion clips (Phase 3) contain _only within-class jitter, never a genuine plane change_ by design (per `src/data/synthetic_video.py`'s own docstring: "Does NOT synthesize class transitions"), so there is nothing real to annotate in them for this purpose.

**If IUGC is present:**

1. From your Task 4 baseline run, select **5 clips** that are good candidates for containing an eyeball-detectable transition between a "standard/clear" view and a "non-standard/blurry/transitional" view — use each clip's `pos`/`neg` frame-index metadata from its corresponding `*_info.csv` (train/val/test) to prefer clips that have **both** SP and NSP segments represented (a clip that's 100% one or the other has no transition to mark). Document exactly which 5 you picked and why (clip filename + its pos/neg index summary) in your response.

2. For each of the 5 clips, generate a **contact sheet** — a single PNG per clip showing ~16 evenly-spaced sampled frames in a grid, each thumbnail labeled with its frame number — saved to `data/processed/manual_review/<clip_name>_contact_sheet.png`. This won't replace watching the actual video, but gives the user a fast way to narrow down roughly where to scrub to before opening it, so they aren't watching 5 full clips cold.

3. Create `data/processed/manual_review/transition_annotations_template.json` pre-filled with the 5 chosen clip filenames and this exact schema (empty arrays, ready for the user to fill in):

```json
{
  "20190909T155747I1.avi": {
    "transitions": []
  },
  "...": {
    "transitions": []
  }
}
```

Each entry the user adds should look like `{"frame": 42, "event": "settle"}` (the moment the view becomes a clear, stable, standard-looking plane) or `{"frame": 110, "event": "leave"}` (the moment it stops being one) — approximate frame indices are fine, this doesn't need to be frame-perfect.

4. **STOP HERE.** Do not proceed to Task 7. Output a message to the user, clearly, that says approximately:

> I've generated contact sheets and a template at `data/processed/manual_review/transition_annotations_template.json` for these 5 clips: [list them]. Please open the actual video files locally, watch them (the contact sheets can help you narrow down where to scrub to), and fill in approximate frame indices for where each clip settles into vs. leaves a clear standard-plane-like view. Reply here with the completed JSON (or confirm you've edited the file directly and I should read it from disk) before I continue to the tuning sweep in Task 6.

Do not write any code for Task 7 until you have received this input back from the user, or the user explicitly tells you to proceed without it.

**If IUGC is not present:** state this plainly in your response (do not silently skip past it) — e.g. "`data/raw/iugc_video/DatasetV3/` was not found, so the manual transition-annotation checkpoint does not apply. Proceeding to Task 7 tuned against synthetic ego-motion clips only (within-class jitter robustness), per the documented fallback in `05_TEMPORAL_SMOOTHING_AND_REALTIME.md`. This means the tuning validates jitter suppression but not genuine plane-transition latency — this limitation will be stated explicitly in the final walkthrough and should also be carried into the Phase 6 `EVAL_REPORT.md` when you get there." Then proceed straight to Task 7 without stopping.

---

## 7. Task 6 — Tier-1 parameter sweep and final tuning

Create `scripts/tune_tier1_smoothing.py`. Sweep:

- `alpha ∈ {0.15, 0.2, 0.25, 0.3, 0.35, 0.4}`
- `switch_threshold ∈ {0.5, 0.55, 0.6, 0.65, 0.7}`
- `min_dwell_frames` — convert your Task 4 measured real inference FPS into frame counts targeting **~150–500ms** dwell (e.g., if you measured ~60fps, that's roughly 9–18 frames; sweep a few values spanning that range plus a bit outside it for context)

For each parameter combination, run the `Tier1Smoother` over the same clip set as Task 4 and measure:

- Label-switches-per-minute with smoothing on, compared against the Task 4 baseline (report both raw numbers and % reduction).
- **If manual annotations were provided (Task 5):** latency-to-first-stable-label after each annotated `"settle"` event — i.e., frames from the annotated event to the point where `Tier1Smoother` reports the correct label with `is_stable=True` — converted to milliseconds using your measured FPS.
- **If manual annotations were not available:** report only the flicker-suppression numbers on synthetic clips, and state explicitly in the output that transition latency was not validated.

Throughout the sweep, log the per-iteration parameter sets and their corresponding detailed metrics to a file (e.g., `logs/tune_tier1_smoothing_sweep.txt` with strictly `encoding="utf-8"`). Pick the parameter set that minimizes spurious switches while keeping stabilize-lag under ~300–400ms (when measurable). Write the full sweep results table plus your chosen values and reasoning to `docs/PHASE5_SMOOTHING_TUNING.md`, and persist the chosen values to `configs/smoothing_tier1.yaml`:

```yaml
alpha: <chosen>
switch_threshold: <chosen>
min_dwell_frames: <chosen>
hold_floor: null # or a value, if Task 3's optional param proved useful
```

**Gate check:** if no parameter combination gets label-switches-per-minute meaningfully below baseline while keeping stabilize-lag reasonable, stop here and report this to the user explicitly rather than proceeding — this is the trigger condition for considering Tier-2 (§0.6 above), and building Tier-2 is a decision for the user to make, not you.

---

## 8. Task 7 — Video/webcam capture abstraction

Create `src/realtime/capture.py`:

```python
"""
src/realtime/capture.py
Single interface over cv2.VideoCapture for either a webcam index (int) or a
video file path (str/Path), so the rest of the pipeline is source-agnostic.
"""
import cv2

class FrameSource:
    def __init__(self, source: int | str, loop: bool = False):
        self.source = source
        self.loop = loop
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret and self.loop and isinstance(self.source, (str,)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()

    @property
    def is_webcam(self) -> bool:
        return isinstance(self.source, int)
```

`loop=True` is useful for demoing against a short video file repeatedly; `is_webcam` is used later to decide whether to burn in the "mechanics test only" watermark (Task 10).

---

## 9. Task 8 — Threaded pipeline: bounded drop-oldest queues, capture/inference threads

Python's `queue.Queue` doesn't natively support "drop oldest on full" — implement a small wrapper first, `src/realtime/queues.py`:

```python
"""
src/realtime/queues.py
A bounded queue that drops the OLDEST item when full, rather than blocking or
raising — used so inference always processes the most recent frame available,
never a backlog of stale ones, per docs/05_TEMPORAL_SMOOTHING_AND_REALTIME.md §B2.
"""
import queue

class DropOldestQueue:
    def __init__(self, maxsize: int = 2):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            self._q.put_nowait(item)

    def get_nowait_or_none(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None
```

Then `src/realtime/pipeline.py`:

- **`CaptureThread`**: reads frames from `FrameSource` as fast as available, pushes `(frame, capture_timestamp)` to a `DropOldestQueue(maxsize=2)`. Tracks rolling capture FPS.
- **`InferenceThread`**: pulls the latest available frame (non-blocking; if queue empty, brief sleep and retry), and per frame:
  1. `prep_frame_grayscale_to_rgb(frame)` → albumentations eval transform → tensor.
  2. Forward pass under `torch.no_grad()` + `torch.amp.autocast('cuda', enabled=...)`, softmax.
  3. `Tier1Smoother.step()` using the tuned params from `configs/smoothing_tier1.yaml`.
  4. **Grad-CAM, throttled**: instantiate the `pytorch_grad_cam.GradCAM` object **once**, outside the per-frame loop (reuse it across the whole session — do not create/destroy it every throttled call, which just adds unnecessary hook registration overhead on top of the throttling that's already there for compute reasons). Every `gradcam_every_n_frames` processed frames (default from a CLI flag, suggest default of 10) or every 200ms wall-clock (whichever the CLI selects), call `run_gradcam()` targeting the currently displayed (smoothed) label, not necessarily the raw argmax. On skipped frames, reuse the last computed overlay.
  5. Push a result dict — `{label, confidence, smoothed_probs, is_stable, overlay_or_none, frame, timings}` — to a second `DropOldestQueue`.
  6. Track and expose: rolling inference FPS, and per-stage timings (preprocess_ms, forward_ms, gradcam_ms-when-run, smoothing_ms) via a small thread-safe stats object (a simple class with a lock is fine, no need for anything fancier).
  7. **Mandatory Re-measurement of Pipeline FPS**: Because the single-threaded baseline FPS (~24.3 fps) measured in Task 4 does not account for threading concurrency (e.g. data preprocessing happening concurrently with inference), the actual real-time pipeline FPS must be empirically measured here. **HOW**: Add a simple script or flag to run the pipeline in headless mode for a fixed number of frames (or time) over a video file, and log the average inference FPS. **WHY**: This real, batched/threaded FPS number is strictly necessary to correctly calibrate time-based budgets downstream (like the Grad-CAM throttle cadence, which expects 200ms equivalents) and to understand actual system capacity.

Both threads should support a clean `stop()` via a shared `threading.Event`.

---

## 10. Task 9 — Main render loop + CLI entrypoint (cv2.imshow)

Create `src/realtime/app.py`:

- CLI args: `--source` (int for webcam index, or path string for video file), `--checkpoint` (default `checkpoints/convnext_tiny/best.pt`), `--smoothing-config` (default `configs/smoothing_tier1.yaml`), `--gradcam-every-n-frames` (default computed from re-measured FPS to target ~200ms; do NOT blindly use 10, calculate it based on Task 8's actual threaded throughput), `--no-gradcam` (disable entirely), `--loop` (for file sources).
- Runs `CaptureThread` and `InferenceThread` as background threads; the **main thread** runs the render loop: pull latest result (non-blocking — if none yet, just redisplay the previous frame so the window doesn't stall), draw onto the frame:
  - Predicted label (from `IDX_TO_CLASS`) + confidence, large and legible.
  - Stability indicator: e.g. a green "● STABLE" vs. amber "◐ SETTLING" text/dot, driven directly by `Tier1Smoother`'s `is_stable` flag.
  - Grad-CAM overlay blended in when available (toggle with a keypress, see below), reusing the last one on throttled-skip frames as described above.
  - A HUD (toggleable) showing: capture FPS, inference FPS, and the per-stage millisecond breakdown from Task 9's stats object — this is the instrumentation `05_TEMPORAL_SMOOTHING_AND_REALTIME.md` §B3 explicitly asks to build in "from the start, not as an afterthought."
- Keybindings: `q`/`ESC` = quit, `g` = toggle Grad-CAM overlay, `h` = toggle HUD, `space` = pause/resume rendering (inference can keep running or pause too, your call — document which).
- On exit, `release()` the capture and join the threads cleanly.

---

## 11. Task 10 — Validation runs + webcam-as-stand-in caveat

1. First, validate end-to-end against a **video file** (use a synthetic clip from `data/processed/synthetic_clips/`, or an IUGC clip if available) with `--loop`. Confirm: no crashes over a few minutes of looped playback, FPS numbers look sane for the RTX 4060, stability indicator behaves sensibly, Grad-CAM overlay updates at the expected throttled cadence.
2. Then validate against the **webcam** (`--source 0`) purely to prove the capture/inference/render mechanics generalize to a live device source. Per `05_TEMPORAL_SMOOTHING_AND_REALTIME.md` §B5: **burn a persistent, visible caption into the frame whenever `FrameSource.is_webcam` is True** — something like `"[WEBCAM DEMO — pipeline mechanics test only, not real ultrasound]"` — so this caveat is impossible to miss even if someone screenshots the running app later, rather than relying on a README note nobody reads. This directly operationalizes the doc's documentation requirement in the UI itself.
3. Save at least one representative annotated frame from each validation run (file-source and webcam) to `data/processed/manual_review/` (or a new `docs/phase5_screenshots/` folder) as proof-of-execution PNGs for the walkthrough.
4. During these validation runs, dump the rolling performance statistics (capture FPS, inference FPS, queue drops, and per-stage timings) periodically to a log file (e.g., `logs/realtime_validation_run.txt` with strictly `encoding="utf-8"`) to provide a concrete record of the pipeline's stability and performance over time.

---

## 12. Task 11 — Final deliverable: `docs/phase 5 walkthrough.md`

Matching the exact style and thoroughness of the existing `docs/phase 3 walkthrough.md` and `docs/phase 4 walkthrough.md` (numbered sections, tables of files created, actual numbers not vague summaries), produce `docs/phase 5 walkthrough.md` containing, at minimum:

1. **New files created** — a table (file path → purpose), matching the format of Phase 4's walkthrough.
2. **IUGC availability finding** — whether it was present, and which branch of Task 5/6 was taken as a result.
3. **Baseline flicker results** — the actual numbers from Task 4 (switches/sec per clip, before smoothing), plus the achieved inference FPS on the RTX 4060 that this was measured against.
4. **Manual checkpoint outcome** — if applicable: which 5 clips were selected and why, confirmation the user's annotations were received and used, or explicit note if this was skipped and why.
5. **Tier-1 tuning results** — the sweep table from Task 6, the chosen `(alpha, switch_threshold, min_dwell_frames)`, the reasoning, and the before/after flicker-rate comparison (and latency-to-stabilize numbers, if measurable).
6. **The `frames_since_last_switch` correction** made in Task 3 relative to the doc's literal pseudocode — state it plainly as a deliberate deviation and why.
7. **Real-time pipeline architecture** — brief description of the threading model, queue behavior, Grad-CAM throttling cadence chosen and why.
8. **End-to-end validation results** — FPS numbers achieved on file playback and webcam, any issues hit and how resolved, the screenshot/frame-grab proof images referenced with their paths.
9. **Explicit statement of what was NOT built and why** — Tier-2 (unless triggered — see gate in Task 6), Streamlit/Gradio UI (explicitly deferred per user decision), and anything else scoped out.
10. **A deliverables checklist** mirroring the one at the bottom of `05_TEMPORAL_SMOOTHING_AND_REALTIME.md`, each item marked done / skipped-with-reason.

---

## Deliverables checklist for this session

- [ ] Preflight report given (GPU check, checkpoint config printed and confirmed correct, IUGC availability confirmed one way or the other)
- [ ] `prep_frame_grayscale_to_rgb()` added and parity-tested against the path-based training function
- [ ] `src/realtime/model_loader.py` created, used everywhere instead of ad-hoc loading
- [ ] `src/smoothing/tier1.py` implemented with the 3 documented unit tests passing, and the `frames_since_last_switch` design deviation explicitly flagged
- [ ] `scripts/measure_baseline_flicker.py` run, baseline report + chart produced
- [ ] **Manual checkpoint (§Task 5) handled correctly**: either the STOP-and-wait was actually honored and annotations incorporated, or the skip was explicitly and correctly justified — this is the item I most want you to get right, don't paper over it
- [ ] `scripts/tune_tier1_smoothing.py` run, `docs/PHASE5_SMOOTHING_TUNING.md` and `configs/smoothing_tier1.yaml` produced
- [ ] `src/realtime/capture.py`, `src/realtime/queues.py`, `src/realtime/pipeline.py`, `src/realtime/app.py` implemented
- [ ] Grad-CAM throttled and using a single persistent `GradCAM` instance, not re-instantiated per call
- [ ] Validated end-to-end on file playback (`--loop`) and on webcam, webcam caveat caption burned into frame
- [ ] `docs/phase 5 walkthrough.md` written with real numbers, not placeholders

Do not silently skip the manual checkpoint gate. If it applies and you proceed past it without the user's input, that's a failure to follow this prompt, not a shortcut.
