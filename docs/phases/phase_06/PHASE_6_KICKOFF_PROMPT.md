# PHASE_6_KICKOFF_PROMPT.md — Evaluation & Validation

Read, in this order, before writing any code: `docs/instructions/00_PROJECT_OVERVIEW.md`,
`docs/instructions/06_EVALUATION_VALIDATION.md`, `docs/instructions/08_MASTER_CHECKLIST.md`,
`docs/EXPERIMENTS.md` (Phase 4 backbone decision), `docs/phases/phase_05/PHASE5_SMOOTHING_TUNING.md`
and `docs/phases/phase_05/phase 5 walkthrough.md` (Phase 5 results), and this file in full.
This file is more prescriptive than `06_EVALUATION_VALIDATION.md` on purpose — where the two
disagree on a concrete detail, follow this file, since it was written after seeing the actual
Phase 4/5 artifacts (checkpoint contents, class taxonomy, smoothing config, dataset manifests)
that `06_...md` was written without.

Phases 0–5 are complete and verified. The winning model is **`convnext_tiny.fb_in22k_ft_in1k`**
(class-weighted CE, not focal), checkpoint at `checkpoints/convnext_tiny/best.pt`,
test macro-F1 = **0.8927**. The tuned Tier-1 smoothing config is locked at
`configs/smoothing_tier1.yaml` (`alpha=0.20, switch_threshold=0.70, min_dwell_frames=8`).

Phase 6 produces **no new model training, no new smoothing tuning, no pipeline changes**.
It is a measurement and reporting phase: run the existing checkpoint and the existing tuned
smoother against data they haven't been formally scored against yet, and compile everything —
old and new — into one honest, cross-referenced `EVAL_REPORT.md`.

---

## 0. Non-negotiable constraints — read before writing anything

1. **Do not retrain, fine-tune, or modify any checkpoint.** `checkpoints/convnext_tiny/best.pt`
   is frozen. If a number in this phase looks bad, that is a *reportable finding*, not a trigger
   to go back and improve the model. Phase 6 measures; it does not fix.
2. **Do not modify `configs/smoothing_tier1.yaml`, `src/smoothing/`, or `src/realtime/`.**
   The Tier-1 parameters are locked from Phase 5. If this phase's stability metrics look worse
   than Phase 5's, report the discrepancy and investigate the *measurement*, not re-tune the
   smoother.
3. **Never hardcode `image_size`, `backbone`, or normalization stats.** Every evaluation script
   must load the model via `src.realtime.model_loader.load_inference_model()` (the same function
   Phase 5 established as the single source of truth) — never `torch.load()` or `build_model()`
   directly.
4. **Reuse existing modules — do not reimplement them:**
   - `src.data.dataset.CANONICAL_CLASSES` / `CLASS_TO_IDX` / `IDX_TO_CLASS` / `NUM_CLASSES`
   - `src.data.dataset.CrossDeviceDataset` (already implements the FP/MULTICENTRE exclusion
     guardrail — do not write a second manifest reader)
   - `src.data.transforms.get_eval_transform()` / `prep_frame_grayscale_to_rgb()`
   - `src.smoothing.tier1.Tier1Smoother` (load params from `configs/smoothing_tier1.yaml`,
     exactly as `src/realtime/pipeline.py::InferenceThread.__init__` already does)
   - `src.eval.metrics_utils.save_confusion_matrix()`
5. **The cross-device "Head" label is collapsed — implement this exactly, see Task 2 for the
   precise algorithm.** Do not attempt exact-string-match scoring against `CrossDeviceDataset`'s
   `plane_label` column for Head rows; it will always fail even on a perfect model.
6. **Realistic-video *accuracy* (with vs. without smoothing) can only be computed on the 16
   synthetic clips in `data/processed/synthetic_clips/`.** They are the only video source with a
   ground-truth label compatible with the 8-class taxonomy (the label is implicit in the filename
   prefix — e.g. `Brain_Trans_cerebellum_clip01.mp4` — and is constant across every frame in the
   clip, per `synthetic_video.py`'s own docstring: "Does NOT synthesize class transitions").
   **Do not compute or report a frame-level "accuracy" number against any IUGC clip.** IUGC's
   `pos`/`neg`/PS/AoP/HSD labels are for a different clinical task and have zero correspondence
   to your 8 classes — the only valid IUGC-derived metrics are the stability metrics already
   established in Phase 5 (switches/min, spurious-switch count). If you find yourself computing
   "correct/total frames" against an IUGC clip, stop — that number is meaningless and must not
   appear in `EVAL_REPORT.md`.
7. **Latency-to-stabilize does not have real ground truth in this project and Phase 6 must not
   try to manufacture one.** Phase 5's `PHASE5_SMOOTHING_TUNING.md` already found that all 5
   manually-annotated IUGC clips showed zero raw argmax transitions — there was no genuine
   model-level transition to time. Cite that finding directly in `EVAL_REPORT.md`'s limitations
   section; do not re-run `measure_transition_latency()`-style logic expecting a different result.
8. **Log everything meaningful.** Any script that produces metrics, a table, a sweep, or
   per-clip/per-image results must write a `.txt`, `.csv`, or `.json` artifact under
   `logs/eval/` (strictly `encoding="utf-8"` for text) — not just print to stdout. Console output
   for quick commands is fine; anything with more than ~20 rows of data, or anything you'd want
   to reference later while writing `EVAL_REPORT.md`, goes to a file. This mirrors the logging
   discipline established in `PHASE_5_KICKOFF_PROMPT.md §0.9`.
9. **New folders for this phase:** `logs/eval/`, `data/processed/eval_cross_device/`,
   `data/processed/eval_realistic_video/`, `docs/phases/phase_06/`. Do not write eval outputs
   into `checkpoints/`, `data/raw/`, or any Phase 4/5 directory.
10. **`docs/EVAL_REPORT.md` is the single canonical deliverable.** It lives at the repo-doc root
    (matching `06_EVALUATION_VALIDATION.md §6`'s own path), not under `docs/phases/phase_06/`.
    The phase-06 subfolder is for working artifacts (screenshots, intermediate tables) that
    `EVAL_REPORT.md` references.

---

## 1. Preflight — repository & data state check (do this first, report back before writing code)

```bash
conda activate fetalplane
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Confirm the checkpoint you're about to score is the one you think it is:

```python
import torch
ckpt = torch.load("checkpoints/convnext_tiny/best.pt", map_location="cpu", weights_only=False)
print(ckpt["config"]["backbone"])       # expect: convnext_tiny.fb_in22k_ft_in1k
print(ckpt["val_macro_f1"], ckpt["epoch"])
```

Confirm the cross-device manifest and synthetic clips are present and match Phase 3's numbers:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/processed/cross_device_manifest.csv'); print(len(df)); print(df.groupby(['source_subset','plane_label']).size())"
ls data/processed/synthetic_clips/*.mp4 | wc -l   # expect 16
```

Confirm the tuned smoothing config matches what Phase 5 shipped:

```bash
cat configs/smoothing_tier1.yaml
# expect: alpha: 0.2, switch_threshold: 0.7, min_dwell_frames: 8, hold_floor: null
```

Report all of the above explicitly before starting Task 1. If the cross-device manifest row
count doesn't match Phase 3's documented **1,423 rows** (999 HC18 Head + 159 UCL Head +
130 UCL Abdomen + 135 UCL Femur), stop and flag it — do not proceed on a manifest that's drifted.

---

## 2. Task 1 — In-distribution test metrics (consolidation, not re-run)

This was already computed correctly in Phase 4 (`src/eval/evaluate_test.py`,
`docs/test_classification_metrics_all.txt`, per-backbone `classification_report_TEST.txt` and
`confusion_matrix_TEST.png`). **Do not re-run inference.** Task 1 is purely to pull the
`convnext_tiny` (CE, not focal) numbers into a normalized form for `EVAL_REPORT.md`:

- Macro-F1: **0.8927**
- Per-class precision/recall/F1 (pull verbatim from `checkpoints/convnext_tiny/classification_report_TEST.txt`)
- Confusion matrix: reference `checkpoints/convnext_tiny/confusion_matrix_TEST.png` directly —
  do not regenerate it.
- Note explicitly in `EVAL_REPORT.md`: `Brain_Trans_ventricular` remains the weakest class
  (F1=0.77, recall=0.73), consistent with the literature-documented hard pair and with the
  focal-loss ablation's negative result (`docs/EXPERIMENTS.md`).

**Output:** a short "§1 In-Distribution Performance" section in `EVAL_REPORT.md`. No new script,
no new log file needed for this task — it's pure consolidation of existing, already-logged Phase 4
results.

---

## 3. Task 2 — Cross-device generalization (HC18 + UCL only)

Create `scripts/evaluate_cross_device.py`.

**The collapsed-label scoring algorithm — implement exactly as follows:**

```python
"""
scripts/evaluate_cross_device.py

Evaluates checkpoints/convnext_tiny/best.pt on data/processed/cross_device_manifest.csv
(HC18 + UCL only — FP/MULTICENTRE are excluded upstream by build_cross_device_manifest.py
and CrossDeviceDataset's own guardrail assert).

Scoring rule for the collapsed "Head" label:
  A "Head" row is scored CORRECT if the model's argmax is ANY of:
    {Brain_Trans_cerebellum, Brain_Trans_thalamic, Brain_Trans_ventricular}
  "Fetal_abdomen" / "Fetal_femur" rows use exact-match scoring (single canonical class each).

No "Other" class exists in this manifest by construction (every image is a valid standard
plane) -- this evaluation validates plane-IDENTITY classification cross-device, NOT the
standard-vs-Other decision. State this explicitly in the report.

No "Fetal_thorax" / "Maternal_cervix" coverage exists in ANY available external dataset --
state this explicitly too, do not imply those two classes were cross-device validated.
"""
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from src.data.dataset import CrossDeviceDataset, CLASS_TO_IDX, IDX_TO_CLASS
from src.realtime.model_loader import load_inference_model

BRAIN_SUBCLASSES = {"Brain_Trans_cerebellum", "Brain_Trans_thalamic", "Brain_Trans_ventricular"}

def score_row(true_label: str, pred_class_name: str) -> bool:
    if true_label == "Head":
        return pred_class_name in BRAIN_SUBCLASSES
    return pred_class_name == true_label
```

Wire this against a `DataLoader(CrossDeviceDataset(...), batch_size=..., shuffle=False)`, run
inference via `load_inference_model("checkpoints/convnext_tiny/best.pt")`, and compute:

1. **Overall collapsed-label accuracy**, combined HC18+UCL.
2. **Per-`source_subset` accuracy** (HC18 vs UCL), reported **separately, never averaged into
   one blended number** — HC18 and UCL differ meaningfully in site count, device, and per-anatomy
   coverage (HC18 = Head only; UCL = Head+Abdomen+Femur), and `03_DATA_PIPELINE.md §3` explicitly
   warns against blending them.
3. **Per-label accuracy within each subset** (Head / Fetal_abdomen / Fetal_femur where
   applicable — HC18 has no Abdomen/Femur rows, don't compute a metric for a category with zero
   support).
4. **The generalization gap** — compare cross-device Head accuracy against the equivalent
   *collapsed* in-distribution number, not the raw per-subclass in-distribution F1. To do this
   fairly, also compute an in-distribution "collapsed-Head" accuracy from `data/splits/test.csv`
   (a brain-subclass prediction counts correct if the true label is any brain subclass AND the
   argmax is any brain subclass) — this is the correct apples-to-apples baseline, since comparing
   cross-device collapsed accuracy against in-distribution *sub-plane* F1 would be comparing two
   different tasks. Likewise compute in-distribution exact-match accuracy for Fetal_abdomen and
   Fetal_femur alone (not the full 8-class accuracy) as the baseline for those two gaps.
5. Confusion breakdown for Head rows specifically: when a Head image is misclassified, which
   non-brain class does it fall into? (This tells you whether cross-device failures look like
   "confused the brain sub-plane" — fine, expected — vs. "thought a brain image was Femur" —
   a genuine domain-shift red flag.)

**Log to:** `logs/eval/evaluate_cross_device.txt` (full per-row breakdown optional but the
summary tables are mandatory) and `data/processed/eval_cross_device/cross_device_results.csv`
(per-image: image_path, source_subset, true_label, pred_class_name, correct[bool]).
Save a confusion-style bar chart or heatmap to
`data/processed/eval_cross_device/cross_device_confusion.png` via
`src.eval.metrics_utils` patterns (extend if needed, don't hand-roll a second plotting utility).

**Gate check:** if overall Head accuracy is below ~50% (worse than random-among-3), stop and
report before writing the final section — that would suggest a genuine implementation bug in the
collapsed-scoring logic, not just a hard generalization gap, and needs a second pair of eyes
before it goes in the report.

---

## 4. Task 3 — Realistic-video evaluation: accuracy (synthetic only) + stability (all clips)

Create `scripts/evaluate_realistic_video.py`. This task has two genuinely different halves —
keep them structurally separate in the script and in the output, do not merge them into one
table that implies IUGC clips have an accuracy number (see constraint §0.6).

### 4a. Frame-level accuracy, with vs. without Tier-1 smoothing — synthetic clips only

For each of the 16 `data/processed/synthetic_clips/*.mp4`:
1. Parse the ground-truth class from the filename prefix (reuse the same mapping logic as
   `scripts/generate_sample_clips.py` used to name them — the class name up to `_clipNN.mp4`).
2. Run frame-by-frame inference (reuse `prep_frame_grayscale_to_rgb` + `get_eval_transform`,
   exactly as `measure_baseline_flicker.py` and `tune_tier1_smoothing.py` already do).
3. **Raw accuracy:** fraction of frames where `argmax(raw_probs) == ground_truth_class`.
4. **Smoothed accuracy:** fresh `Tier1Smoother` instance per clip (loaded from
   `configs/smoothing_tier1.yaml`, not hardcoded), fraction of frames where the *displayed*
   label equals ground truth.
5. Report per-clip and aggregate (mean across 16 clips) raw vs. smoothed accuracy, plus the
   delta. Expected direction: smoothed ≥ raw, since these are within-class jitter clips by
   construction (no real transition to lag behind) — if smoothed accuracy is *lower* than raw
   on any clip, that's worth flagging explicitly in the report as a candidate red flag (it would
   mean the smoother is dragging a stale wrong label rather than correcting jitter — check
   whether that clip's raw accuracy was already poor, which is the more likely explanation, and
   say so).

### 4b. Video stability metrics — all 46 clips (16 synthetic + IUGC set from Phase 5)

Re-run (not re-tune) the tuned config's stability numbers as an independent Phase 6
confirmation of Phase 5's sweep result, using the identical clip set Task 6 of Phase 5 used
(`data/processed/synthetic_clips/*.mp4` + first 10 `.avi` per split under
`data/raw/iugc_video/DatasetV3/{train,val,test}/videos/`, matching the glob pattern already
established in `tune_tier1_smoothing.py::load_video_paths()`):

- **Label-switches-per-minute**, with smoothing vs. raw argmax baseline (should reproduce
  Phase 5's 788.75 → 72.00/min, 90.9% reduction, within measurement noise — if it doesn't
  reproduce closely, investigate before reporting; a mismatch would indicate either data drift
  in `data/raw/iugc_video/` or an inconsistency in this phase's re-implementation).
- **Mean dwell time on displayed label** (new metric, not computed in Phase 5): average number
  of consecutive frames — converted to milliseconds using this phase's own measured inference
  FPS — that the smoother holds a label before switching. This is the sanity check
  `06_EVALUATION_VALIDATION.md §4` asks for: confirm the system isn't just permanently stuck on
  one label (which would trivially minimize switches while being useless). A reasonable dwell
  time here should be on the order of seconds for stable clips, not "the entire clip length" —
  if every clip's mean dwell time equals its full duration, flag that as a possible sign the
  smoother never leaves its cold-start label, which would be a different (and more serious)
  finding than what Phase 5 reported.
- **Latency-to-stabilize:** do NOT attempt to recompute this from IUGC annotations (see §0.7).
  Instead, write the following into the report verbatim as the honest position: *"True
  transition-latency validation was not possible in this project because no available real
  video source contains an annotated genuine plane-to-plane transition — the primary dataset
  (FETAL_PLANES_DB) never released source video, and manual annotation of the 5 IUGC candidate
  clips (Phase 5, Task 5) found zero raw model-level transitions to time. The dwell-time metric
  above (§4b) is the closest available proxy for responsiveness, but it measures steady-state
  holding behavior, not transition-tracking speed."*

**Log to:** `logs/eval/evaluate_realistic_video.txt` (per-clip lines, same style as
`logs/tuning/measure_baseline_flicker.txt`) and
`data/processed/eval_realistic_video/realistic_video_results.csv` (columns: clip_name,
is_synthetic[bool], ground_truth_class_or_null, raw_accuracy_or_null, smoothed_accuracy_or_null,
raw_switches_per_min, smoothed_switches_per_min, mean_dwell_ms).

---

## 5. Task 4 — Ablations summary (consolidation)

No new experiments required — every ablation `06_EVALUATION_VALIDATION.md §5` asks for already
exists somewhere in Phases 4–5. This task is purely to assemble them into one comparison section
of `EVAL_REPORT.md`, cited from their original sources:

| Ablation | Source | One-line result |
|---|---|---|
| Backbone comparison (6 candidates, in-distribution) | `docs/EXPERIMENTS.md`, `docs/test_classification_metrics_all.txt` | convnext_tiny wins (0.8927), statistically robust vs efficientnet_lite0/repvgg_a2, statistically tied with tf_efficientnetv2_s (bootstrap CI) |
| Backbone comparison, cross-device | **This phase, Task 2** — first time this has been measured for any backbone. State plainly that only the winning backbone (convnext_tiny) was evaluated cross-device, per the project's accuracy-first, single-final-checkpoint deployment target — a full 6-backbone cross-device sweep was never in scope. |
| Pretraining init (ImageNet vs. domain SSL) | `docs/EXPERIMENTS.md` | FUSC checkpoint not portable (ResNet encoder vs. our chosen architectures) — skipped, documented, not a negative result |
| Smoothing on/off | **This phase, Task 3** + `docs/phases/phase_05/PHASE5_SMOOTHING_TUNING.md` | 90.9% switch-rate reduction, zero spurious switches; frame-accuracy delta from Task 3 |
| Class-weighted CE vs. focal loss | `docs/EXPERIMENTS.md` | Focal loss regressed `Brain_Trans_ventricular` F1 (0.77→0.75) and overall macro-F1 (0.8927→0.8785) — clean negative result, CE retained |

Do not re-run any of these. If you notice a genuine gap (e.g., "cross-device was only ever
measured for the winning backbone, not compared across candidates"), name it as a documented
scope limitation in `EVAL_REPORT.md`'s limitations section rather than silently expanding scope
to fill it — that's a Phase 7 stretch-goal-sized undertaking (re-running Task 2's script against
5 more checkpoints), not a Phase 6 task, and should be flagged as a candidate `07_...` addition
if you think it's worth doing.

---

## 6. Task 5 — Write `docs/EVAL_REPORT.md`

Single canonical deliverable. Structure (matching `06_EVALUATION_VALIDATION.md §6`'s
requirements plus the honesty requirements woven through this prompt):

1. **In-distribution performance** (Task 1) — macro-F1, per-class table, confusion matrix image
   embedded, explicit callout on `Brain_Trans_ventricular` as the persistent weak class.
2. **Cross-device generalization** (Task 2) — HC18/UCL numbers reported separately, the gap
   table vs. the correctly-matched in-distribution collapsed baseline, the explicit scope
   statement ("validates plane-identity only, not standard-vs-Other; no Thorax/Cervix coverage
   available from any external source").
3. **Realistic-video evaluation** (Task 3) — the synthetic-clip accuracy table (raw vs.
   smoothed), and the stability metrics table (switches/min, dwell time) covering all 46 clips,
   with the explicit "IUGC has no accuracy ground truth" statement and the latency-to-stabilize
   honest-limitation paragraph verbatim from §4b above.
4. **Ablations** (Task 4) — the consolidated table above, each row citing its source document.
5. **Limitations** (mandatory, per `06_EVALUATION_VALIDATION.md`'s own final deliverables
   checklist) — compile from what's already scattered across Phase 2–5 docs into one section:
   - Video-transition validation gap (no real annotated transition ever existed to test against)
   - Cross-device set covers 3/8 classes, no "Other" examples, gap reported for 2 of those 3
     classes only (HC18 has no Abdomen/Femur)
   - `FP`/`MULTICENTRE` exclusion rationale (same-source data leak risk, confirmed via source paper)
   - No true clinical validation, no real patient data beyond the public research datasets used
   - Cross-device backbone comparison only covers the single winning checkpoint, not all 6
     candidates (name this as a Phase 7 candidate if you think it's worth doing)
6. **Deliverables checklist** — mirror `06_EVALUATION_VALIDATION.md`'s own checklist at the
   bottom, each item marked done/skipped-with-reason, same style as Phase 5's walkthrough.

---

## Deliverables checklist for this session

- [ ] Preflight report given (GPU check, checkpoint config confirmed, manifest/clip counts confirmed against Phase 3/5 documented numbers)
- [ ] Task 1: in-distribution numbers consolidated into `EVAL_REPORT.md` §1, no re-run
- [ ] Task 2: `scripts/evaluate_cross_device.py` implemented with the exact collapsed-Head scoring rule; run; `logs/eval/evaluate_cross_device.txt` + `data/processed/eval_cross_device/*` produced; gap table computed against the *correctly matched* collapsed in-distribution baseline (not raw sub-plane F1)
- [ ] Task 3a: synthetic-clip raw-vs-smoothed frame accuracy computed and logged
- [ ] Task 3b: stability metrics (switches/min reproduction + new mean-dwell-time metric) computed across all 46 clips and logged; latency-to-stabilize section states the honest limitation verbatim, no fabricated number
- [ ] Task 4: ablations table assembled from existing sources, nothing re-run, cross-device-per-backbone gap named as scope limitation if unaddressed
- [ ] Task 5: `docs/EVAL_REPORT.md` written with all 6 sections, embedded images, and the mandatory limitations section
- [ ] No checkpoint, smoothing config, or `src/realtime/`/`src/smoothing/` file was modified anywhere in this phase

If any gate check in Task 2 or Task 3 fires (Head accuracy below chance, dwell time equal to
full clip duration), stop and report before writing `EVAL_REPORT.md` — do not paper over an
implementation bug with a confident-sounding report section.