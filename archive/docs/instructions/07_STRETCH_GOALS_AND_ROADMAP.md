# 07 — Stretch Goals & Future Roadmap

None of this blocks the v1 deliverable (Phases in files 01-06). Pursue only after the primary system is working and evaluated. Ordered roughly by value-to-effort ratio.
Phase 6 is complete (`docs/EVAL_REPORT.md`). All of Phase 0–6 is frozen — nothing below modifies
`checkpoints/convnext_tiny/best.pt`, `configs/smoothing_tier1.yaml`, or any existing `src/`
module in place. New work lives in new files/folders only, following the pattern already
established in Phases 5–6 (`src/realtime/`, `src/smoothing/`, `logs/eval/`, etc.).

**Priority order for this round (reordered from the original doc):**

1. Stretch Goal 1 — Detection-informed multi-task head 
2. **Stretch Goal 2 — Tier-2 temporal smoothing** 
3. **Stretch Goal 3 — ONNX / TensorRT export and quantization** 
4. Stretch Goal 4 — Web UI polish
5. Stretch Goal 5 — Domain-specific self-supervised pretraining
6. Stretch Goal 6 — Second clinical track *(kept in the doc for completeness; **not being pursued**, not detailed below)*
---

## Stretch Goal 1 — Detection-informed multi-task head (highest value, was our preferred design if data allowed)

Per the FAUSP-NET pattern: add a structure-detection head alongside the plane classifier, so the system can say *"Trans-thalamic plane — cavum septum pellucidum ✓, thalami ✓, midline falx not clearly visible"* instead of just a bare confidence score. This is what actually makes the tool clinically actionable rather than a black box.

**What's needed:**
- Structure-level bounding-box annotations. The UCL/HC18 dataset's landmark annotations (BPD/OFD, TAD/APAD, FL points) are a partial starting point for head/abdomen/femur — landmarks aren't boxes, but a box could be derived around each landmark cluster with a reasonable margin as a first pass.
- For thorax/cervix/other brain sub-planes, no existing annotation source was identified during planning — would require either manual annotation of a subset (time-intensive) or finding another public dataset with structure-level labels for these specific planes (worth a fresh search when this stretch goal is picked up, literature moves fast in this space).
- Architecture: shared backbone, two heads — a lightweight detection head (single-scale is probably sufficient given our anatomies are usually large in-frame, unlike FAUSP-NET's small structures like IVC) and the existing classification head, with the plane label optionally derivable from which structures are jointly detected (á la FAUSP-NET) rather than a separate softmax.

## Stretch Goal 2 — Tier-2 learned temporal module
### 2.0 Read this first: the evidence-required gate, and why the plan below is different from the original sketch

The original Phase 5/6 roadmap set an explicit bar for building Tier-2 (`07_STRETCH_GOALS_AND_ROADMAP.md` §Stretch Goal 3, pre-reorder): all three of (1) residual flicker still clinically disruptive, (2) larger `min_dwell_frames` doesn't fix it without unacceptable lag, (3) the problem is genuine classification-level oscillation rather than model quality. Phase 6's Task 3 re-confirmed Phase 5's finding **exactly**: across all 46 clips, Tier-1 achieves a 90.9% switch-rate reduction with **zero spurious switches**, and there is exactly **one** residual clip — `202101141947512003470I1.avi`, holding at 72 switches/min after smoothing, confirmed via exhaustive 210-combination sweep (`docs/phases/phase_05/PHASE5_SMOOTHING_TUNING.md`) to be a **hard structural floor** for the EMA+hysteresis+dwell parameter family, not a tuning gap.

That's evidence criterion (3) — yes. It is *not* strong evidence for (1): one clip out of 46, not a general pattern across real probe motion. Building the originally-sketched full learned-GRU-over-embeddings pipeline (sequence dataset construction, embedding extraction infra, a training loop, a second model artifact to maintain) to fix one clip's residual oscillation is disproportionate engineering weight for the actual problem.

**Decision made here, overriding the original doc's single monolithic Tier-2 design:** build this in two tiers of its own, cheapest-first, and only escalate if the cheap one fails.

- **Tier-2a — majority-vote / mode filter (build and evaluate first).** A trailing-window mode filter is the textbook fix for exactly this failure mode — periodic 2-state oscillation that individually-confident EMA+hysteresis can't fully suppress because each oscillating frame briefly crosses `switch_threshold`. This needs no training data, no new model, no embedding extraction — it's a second, optional smoothing stage that consumes `Tier1Smoother`'s own output stream. This is also literally what `docs/phases/phase_05/PHASE5_SMOOTHING_TUNING.md` itself proposes as "the correct path forward for this residual" — this plan follows that pointer instead of reaching straight for the GRU.
- **Tier-2b — learned temporal head (only if 2a fails the criteria below).** Keep the original GRU/TCN-over-embeddings design as the fallback, detailed in full at the end of this section, so it doesn't need to be redesigned from scratch if it turns out to be necessary.

This is worth pushing back on explicitly if you disagree — but building 2a first costs roughly an afternoon and directly targets the diagnosed mechanism, versus 2b which is a multi-day undertaking (sequence dataset, training loop, a second artifact to version and integrate) that the current evidence doesn't clearly justify yet.

---

### 2.1 Tier-2a — Majority-Vote / Mode Filter

**File:** `src/smoothing/tier2_mode_filter.py`

**Design:** A causal trailing-window mode filter that sits *after* `Tier1Smoother` in the pipeline, not instead of it. Tier-1 already does the expensive work (EMA denoising the raw softmax, hysteresis against low-confidence flicker, dwell-time gating). Tier-2a's only job is to catch the specific residual case Tier-1 structurally can't: a candidate label that keeps crossing `switch_threshold` with real (if brief) confidence, oscillating between two labels faster than `min_dwell_frames` can suppress on every single occurrence.

```python
"""
src/smoothing/tier2_mode_filter.py

Tier-2a: causal majority-vote filter over Tier-1's displayed-label stream.
Stateful, one-call-per-frame, mirrors Tier1Smoother's step() interface so it
composes as a second stage without changing src/realtime/pipeline.py's core
control flow — only the InferenceThread wiring changes (see §2.3).

Not a replacement for Tier1Smoother. Consumes its *output* label stream.
"""
from __future__ import annotations
from collections import deque, Counter
from dataclasses import dataclass, field

@dataclass
class Tier2ModeFilter:
    window_frames: int = 15       # tune empirically, see §2.2 sweep
    min_majority_frac: float = 0.6  # fraction of window a label must hold to be adopted

    _buffer: deque[int] = field(init=False, default_factory=deque)
    _displayed_label: int = field(init=False, default=-1)

    def reset(self) -> None:
        self._buffer.clear()
        self._displayed_label = -1

    def step(self, tier1_label: int) -> int:
        """Feed one Tier-1 displayed label per frame; returns the Tier-2 displayed label."""
        self._buffer.append(tier1_label)
        if len(self._buffer) > self.window_frames:
            self._buffer.popleft()

        if self._displayed_label == -1:
            self._displayed_label = tier1_label
            return self._displayed_label

        counts = Counter(self._buffer)
        top_label, top_count = counts.most_common(1)[0]
        if top_label != self._displayed_label and (top_count / len(self._buffer)) >= self.min_majority_frac:
            self._displayed_label = top_label
        return self._displayed_label
```

**Why this composes safely with Tier-1 rather than fighting it:** Tier-1's own dwell/hysteresis already suppresses the vast majority of noise before Tier-2a ever sees it (confirmed: 45/46 clips reach 0 residual switches from Tier-1 alone). Tier-2a only has real work to do on the one clip where Tier-1's output stream itself still oscillates. On every other clip its input is already constant, so `step()` is a no-op after the first frame — there is no risk of Tier-2a degrading the 90.9%-reduction result already achieved, only of it improving the one remaining outlier. Verify this claim explicitly in the eval script (§2.4) rather than assuming it.

**Added latency cost:** `window_frames=15` at the pipeline's measured steady-state ~23.6 fps (`logs/realtime/realtime_validation_run_20260811_233812.txt`) is ~636ms of *additional* worst-case lag on top of Tier-1's own ~340ms dwell window, before Tier-2a will adopt a genuinely new label. This is a real, non-trivial trade-off — a longer window suppresses more oscillation but makes the system slower to track an actual plane change. Sweep `window_frames` the same way Phase 5 swept Tier-1's parameters (§2.2) rather than picking a number by feel.

### 2.2 Tier-2a parameter sweep

**File:** `scripts/tune_tier2_mode_filter.py`

Reuse `tune_tier1_smoothing.py`'s clip-loading and inference-caching machinery (`load_video_paths()`, the 46-clip precompute pass) rather than reimplementing it — import from that module or factor the shared bits into `src/smoothing/tuning_utils.py` if duplication starts to hurt.

- Fix Tier-1's parameters at the Phase 5 locked values (`configs/smoothing_tier1.yaml` — do not re-sweep Tier-1, it's frozen).
- Sweep: `window_frames ∈ {5, 9, 13, 15, 19, 25}`, `min_majority_frac ∈ {0.5, 0.6, 0.7}`.
- For each combo, run Tier-1 → Tier-2a in series over all 46 clips and record: residual switches/min on `202101141947512003470I1.avi` specifically, total switches/min across all 46 (must not regress above Tier-1-only's 72.00 baseline on any clip), and — critically — verify the "no-op on already-stable clips" claim from §2.1 holds for every combination (i.e., the other 45 clips' switch counts stay at exactly what Tier-1 alone produced).
- Output: `logs/eval/tier2a_sweep.txt`, `data/processed/tier2a_tuning/sweep_results.csv`, chosen values written to `configs/smoothing_tier2a.yaml` (new file — do not add these keys to `smoothing_tier1.yaml`, keep the tiers' configs separate since Tier-2a is optional/toggleable).

**Success criterion for Tier-2a (decide before running the sweep, not after):** the stubborn clip's residual should drop meaningfully below 72/min (ideally toward 0) with zero regression on the other 45 clips, at a `window_frames` value that keeps total added worst-case lag under roughly 500ms combined with Tier-1's own dwell (so under ~150-200ms budget left for Tier-2a itself, meaning `window_frames` likely needs to land closer to 5-9 than 25 — the sweep will tell you exactly where the trade-off breaks).

### 2.3 Pipeline integration

`src/realtime/pipeline.py`'s `InferenceThread` currently calls `Tier1Smoother.step()` once per frame and pushes the result. Add an **optional** second stage:

- New constructor arg `tier2_config_path: str | None = None` (default `None` = Tier-2a disabled, matching current behavior exactly — this must be opt-in, not silently on by default, since it changes system latency characteristics).
- If provided, instantiate `Tier2ModeFilter` from `configs/smoothing_tier2a.yaml` alongside the existing `Tier1Smoother`, and feed Tier-1's `current_displayed_label` output into it each frame; the result dict's `label`/`label_name` fields become Tier-2a's output instead of Tier-1's when enabled.
- `src/realtime/app.py`: add a `--enable-tier2` CLI flag (default off) so `--source <file> --loop` A/B comparisons between Tier-1-only and Tier-1+Tier-2a are trivial to run for the write-up.
- Do **not** modify `Tier1Smoother` itself or its existing unit tests (`src/smoothing/test_tier1.py`) — Tier-2a is strictly additive.

### 2.4 Evaluation

Extend (don't replace) `scripts/evaluate_realistic_video.py`'s pattern into a new `scripts/evaluate_tier2.py` — same 46-clip set, same CSV/log conventions (`logs/eval/evaluate_tier2.txt`, `data/processed/eval_tier2/tier2_results.csv`), reporting Tier-1-only vs. Tier-1+Tier-2a side by side (same per-clip table shape as Phase 6's §3b, with an extra column pair). Explicitly re-run the "spurious switches introduced" check from Phase 5/6 — this is the one number that must stay at zero or the whole exercise has made things worse, not better.

### 2.5 Tier-2b — Learned temporal head (fallback only, build only if 2a fails)

If Tier-2a's sweep can't get the stubborn clip's residual meaningfully below 72/min without regressing other clips or blowing the latency budget, fall back to the originally-scoped learned approach. Keeping the full design here so it doesn't need to be rebuilt from scratch:

**Architecture:**
```
[convnext_tiny backbone, frozen] → penultimate feature vector (768-dim for convnext_tiny,
  via model.forward_features() + global pool, per timm's API) per frame
  → circular buffer of last N=8-16 feature vectors
  → causal 2-layer GRU (hidden_dim=256) OR causal 1D-depthwise-conv stack
  → softmax over 8 classes
```
- Backbone weights frozen (`model.requires_grad_(False)`) — this is a temporal head only, not a fine-tune.
- Must be causal (no look-ahead) to stay real-time compatible.
- Target inference overhead: <5ms GPU per frame (profile against convnext_tiny's own measured ~20ms forward pass — the temporal head must not become the bottleneck).
- Runs **instead of** Tier-1 in the inference thread when enabled (not stacked with Tier-1a/Tier-2a — a learned head absorbs the smoothing role directly).

**Data problem (the actual hard part):** there is still no annotated real plane-to-plane transition anywhere in this project (Phase 5 Task 5's finding, reconfirmed in Phase 6 §5.1). Training data would have to come from synthetic sequence construction — concatenating feature vectors from different within-class clips at a known frame boundary — which validates temporal-head *stability* but not genuine *transition-tracking latency*, the same limitation already documented for Tier-1. State this explicitly again in Tier-2b's write-up if you get here; don't let a synthetic-transition-trained head's dwell numbers get reported as if they measure something Tier-1's numbers don't.

**Criteria to call Tier-2b an improvement (unchanged from the original doc, all must hold):** total residual flicker <72/min on the 46-clip set with zero spurious switches; mean-latency-to-stable ≤400ms; pipeline FPS ≥80% of Tier-1's; no regression on the static-image test macro-F1 (≥0.89, confirming the frozen backbone + new head didn't break anything); subjective stability rated better than Tier-1 by at least two independent observers on real probe-motion video.

### 2.6 Non-negotiable constraints for this goal

1. `checkpoints/convnext_tiny/best.pt` and `configs/smoothing_tier1.yaml` stay frozen and untouched — Tier-2 (either tier) is strictly additive.
2. `src/smoothing/tier1.py` and `src/smoothing/test_tier1.py` are not modified.
3. New smoothing configs live in their own files (`configs/smoothing_tier2a.yaml`, and `configs/smoothing_tier2b.yaml` if you get that far) — never merged into `smoothing_tier1.yaml`.
4. Tier-2 must be opt-in at the CLI/config level, off by default, so existing Phase 5/6 behavior is reproducible without any flag changes.
5. Every sweep/eval writes to `logs/eval/` and a `data/processed/eval_tier2*/` or `data/processed/tier2*_tuning/` folder — same logging discipline as Phases 5–6.

### 2.7 Deliverables checklist

- [ ] `src/smoothing/tier2_mode_filter.py` implemented, unit-tested (stable-input, oscillation-suppression, and no-op-on-already-stable-Tier1-output cases, mirroring `test_tier1.py`'s structure)
- [ ] `scripts/tune_tier2_mode_filter.py` run, `configs/smoothing_tier2a.yaml` produced, sweep logged
- [ ] `src/realtime/pipeline.py` + `app.py` updated with opt-in Tier-2a support, existing Tier-1-only path unchanged when the flag is off
- [ ] `scripts/evaluate_tier2.py` run across all 46 clips, zero spurious-switch regression confirmed
- [ ] Decision documented in writing: did Tier-2a resolve the stubborn clip well enough to stop here, or does the evidence justify building Tier-2b?
- [ ] If Tier-2b is built: embedding extraction script, sequence dataset construction, training loop, all 5 success criteria in §2.5 evaluated and reported, synthetic-transition-data limitation stated explicitly
- [ ] Short write-up (`docs/phases/phase_07/tier2_results.md`) in the same style as `PHASE5_SMOOTHING_TUNING.md`


> **Status: Explicitly deferred — do not implement until the triggering evidence below is observed.**
>
> Per `PHASE_5_KICKOFF_PROMPT.md §0.6` and `05_TEMPORAL_SMOOTHING_AND_REALTIME.md §A3`:
> Tier-2 is a stretch goal, not a default path. Tier-1 tuning has been completed and
> documented. The gate for Tier-2 is empirical, not scheduled.

### Why Tier-2 is not being implemented now

Tier-1 (EMA + hysteresis + minimum dwell) achieved **90.9% overall flicker reduction** across
46 IUGC clips (788.75 → 72.0 switches/min total), with zero spurious switches introduced on
any previously-stable clip. The one residual clip (`202101141947512003470I1.avi`, 72/min)
has been confirmed, via exhaustive search of all 210 parameter combinations, to be a **hard
structural floor** for the EMA+hysteresis+dwell parameter family — not a tuning failure.

Building Tier-2 before the full real-time pipeline (Tasks 7–9) is in production and
evaluated would:
1. add substantial training infrastructure (sequence dataset construction, temporal head
   training loop, integration testing) on top of an unvalidated base system;
2. make it impossible to attribute any residual flicker to the correct cause (model
   quality, preprocessing, smoothing, or rendering latency);
3. risk unnecessary architectural complexity if real-world deployment shows the system
   is already stable enough for its intended use.

The correct engineering order is: ship the complete Tier-1 pipeline → run Task 10
validation → collect real-world evidence → decide on Tier-2.

---

### Evidence required to justify Tier-2

Tier-2 should only be considered if **all three** of the following are observed
after the full pipeline (Tasks 7–10) is deployed and running in real conditions:

1. **Residual flicker remains clinically disruptive** — the display switches more than
   ~1–2 times per second on genuine probe-motion clips during real use (not just on the
   known hard-floor clip `202101141947512003470I1.avi` with its 72/min structural limit).
2. **Larger Tier-1 dwell values do not solve it** — increasing `min_dwell_frames` beyond
   the chosen value causes unacceptable latency to genuine transitions (>500ms observed
   against annotated `settle` events, or >400ms measured in the real-time pipeline's
   `mean_latency_ms` statistic).
3. **The problem is classification-level oscillation, not model quality** — the backbone's
   per-frame softmax output genuinely alternates between two classes on stable anatomy
   (vs. the backbone simply being wrong on this anatomy type, which Tier-2 cannot fix).

If only (3) is true without (1) and (2), the problem is model quality (more training data,
data augmentation, or a different backbone — not a temporal head). If only (1) is true
without (2), try larger dwell first. Tier-2 adds complexity that is only justified when
EMA+dwell are definitively exhausted.

---

### Files, logs, and metrics to review before deciding

Before starting any Tier-2 implementation, the following must be reviewed:

| Source | What to check |
|:--|:--|
| `docs/PHASE5_SMOOTHING_TUNING.md` | Per-clip residual table; confirm the hard-floor clip is the only one remaining. |
| `logs/tune_tier1_smoothing_sweep.txt` | Verify sweep was run on GPU (check "device=cuda" in FPS log line). Confirm all 210 combos were evaluated. |
| `data/processed/tier1_tuning/sweep_results.csv` | Check that no parameter combination achieves `total_residual < 72.0` with `spurious_new == 0`. |
| `logs/realtime_validation_run.txt` (after Task 10) | Inspect actual pipeline flicker rate, queue drops, and latency-to-stabilize under real probe motion. |
| `configs/smoothing_tier1.yaml` | Confirm the deployed dwell_ms is correctly calibrated to GPU FPS (not CPU). |
| `docs/phase 5 walkthrough.md` (after Task 11) | Review the documented validation results and any residual issues flagged. |
| Manual observation of running pipeline | Watch at least 10 minutes of real probe motion; note subjectively whether label stability is acceptable for clinical decision support. |

---

### Proposed Tier-2 implementation (if later decided)

Per `05_TEMPORAL_SMOOTHING_AND_REALTIME.md §A3`, the proposed approach is a **small causal
GRU or 1D-conv over the last N backbone embeddings**, not over raw pixels.

**Architecture:**
```
[Backbone] → penultimate-layer feature vector (1536-dim for convnext_tiny) per frame
           → circular buffer of last N=8–16 feature vectors
           → lightweight temporal head (GRU or causal 1D-conv)
           → softmax over 8 classes (temporal prediction)
```

The temporal head replaces the role of Tier-1 smoothing in the pipeline — it sees the
feature history and learns to predict the stable plane label from temporal context.
The backbone weights are **frozen** during Tier-2 training (do not modify the static
classifier; it is the source of ground truth embeddings for the temporal training set).

**Key design constraints:**
- The GRU/conv must be **causal** (no look-ahead into future frames) to maintain real-time
  compatibility.
- Inference latency of the temporal head must be <5ms (GPU) or <20ms (CPU) per frame to
  not materially affect the pipeline's overall throughput.
- The temporal head runs **instead of** (not in addition to) Tier-1 smoothing in the
  inference thread — the result queue still carries `(label, confidence, is_stable, overlay)`.

---

### Sequence data and labels required

Tier-2 training requires **sequence-level labels**, not frame-level labels:

1. **Plane-transition annotations**: frame indices where the displayed plane genuinely
   changes (the user's `transition_annotations_template.json` format from Task 5/6, but
   at much larger scale — at least 50–100 annotated transition events across diverse clips).
2. **Stable-period labels**: for each clip segment between transitions, a ground-truth
   plane label that the temporal head should learn to output stably.
3. **Embedding extraction**: run the frozen backbone over all labelled clips and save the
   penultimate-layer features as `.npy` arrays — do NOT re-run the full training pipeline
   for this.

**Fallback if manual annotations are unavailable**: synthetic sequences can be constructed
by concatenating feature vectors from different within-class clips with a transition at
a known frame boundary. This validates temporal head stability but not genuine-transition
latency (the same limitation noted for Task 5/6 in the kickoff prompt).

---

### Training and integration procedure

1. **Extract backbone embeddings** for all available clips using `scripts/extract_embeddings.py`
   (to be created) — `model.forward_features(x)` (timm API) yields the penultimate-layer tensor.
2. **Construct sequence dataset**: sliding window of N frames with the transition label at
   the last frame of each window. Balance the dataset by transition type (plane→plane pairs).
3. **Train temporal head only** — freeze backbone via `model.requires_grad_(False)`. Train
   with cross-entropy on the balanced sequence dataset. A 2-layer GRU with hidden_dim=256
   is a reasonable starting point; profile vs. 1D-depthwise-conv blocks if latency is tight.
4. **Checkpoint and integrate**: save the temporal head separately from the backbone checkpoint.
   Load both at inference time in `src/realtime/pipeline.py`; replace the `Tier1Smoother.step()`
   call with the temporal head's forward pass, passing the last N cached embeddings.
5. **Do not modify `src/models/`, `src/train/`, or `src/eval/`** — temporal head lives in
   `src/smoothing/tier2.py` and is loaded only by the real-time pipeline.

---

### Comparison metrics (before vs. after Tier-2)

Measure all of the following on the same 46-clip IUGC set used for Tier-1 tuning, plus
any additional annotated clips:

| Metric | Tool / Source | Tier-1 baseline to beat |
|:--|:--|:--|
| Total switches/min (all clips) | `scripts/tune_tier1_smoothing.py` | 72.00/min |
| Max switches/min (worst clip) | same | 72.00/min (`202101141947512003470I1.avi`) |
| Spurious switches introduced | same, `spurious_new` column | 0.00/min (must remain 0) |
| Mean latency-to-stable (annotated settle events) | `measure_transition_latency()` | ~0ms (cold-start artefact; needs real annotated transitions) |
| Max latency-to-stable | same | same |
| Inference throughput (FPS) | `logs/realtime_validation_run.txt` | GPU FPS of Tier-1 pipeline |
| Per-stage latency breakdown | pipeline stats object | forward_ms + smoothing_ms |
| Temporal head inference overhead | `timeit` benchmark | — (target: <5ms GPU) |

---

### Criteria for concluding Tier-2 improves the system

Tier-2 must satisfy **all** of the following to be considered an improvement:

1. **Total residual flicker < 72/min** on the 46-clip set with `spurious_new == 0`.
2. **Mean latency-to-stable ≤ 400ms** on real annotated transitions (not the cold-start artefact).
3. **Pipeline FPS remains ≥ 80% of Tier-1 FPS** — the temporal head must not become the bottleneck.
4. **No regression on the backbone's per-frame classification accuracy** on the static test set
   (run `scripts/evaluate_test.py` and confirm macro-F1 ≥ 0.89 still holds for the backbone alone).
5. **Subjective stability** — at least two independent observers rate the display as "noticeably
   more stable" than Tier-1 on real probe-motion video, without describing it as "sluggish."

If (1)–(3) pass but (5) fails (system appears frozen/inert), the dwell window is too long
and Tier-2 has made the system slower, not better. If only (1) passes, the improvement is
too small to justify the added complexity. All five criteria must be met.

## Stretch Goal 3 — ONNX / TensorRT Export and Quantization

**Purpose:** prove the shipped model *could* run on constrained/edge hardware, even though the actual deployment target (RTX 4060 laptop demo) never required it. This is a portfolio/credibility artifact, not a functional requirement — treat correctness and honest reporting of the accuracy/latency trade-off as more important than hitting a specific speed number.

### 3.1 Export pipeline

**File:** `scripts/export_onnx.py`, output artifacts under a new top-level `exports/onnx/` folder (not `checkpoints/` — keep Phase 6's convention of not writing derived artifacts into the frozen checkpoint directory).

```python
"""
scripts/export_onnx.py

Exports checkpoints/convnext_tiny/best.pt to exports/onnx/convnext_tiny.onnx.
Loads via src.realtime.model_loader.load_inference_model() — never a second,
hand-rolled model-loading path (same rule Phases 5/6 established).
"""
import torch
from src.realtime.model_loader import load_inference_model

OUT_PATH = "exports/onnx/convnext_tiny.onnx"
OPSET = 17  # verify against installed onnxruntime's supported-opset table before locking this

loaded = load_inference_model("checkpoints/convnext_tiny/best.pt")
model = loaded.model.eval()
dummy = torch.zeros(1, 3, loaded.img_size, loaded.img_size, device=loaded.device)

torch.onnx.export(
    model, dummy, OUT_PATH,
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=OPSET,
)
```

Record `loaded.img_size`, `loaded.normalize_mean`, `loaded.normalize_std` alongside the `.onnx` file (a small sidecar `exports/onnx/convnext_tiny.meta.json`) — the ONNX graph itself has no memory of preprocessing, and every consumer of this export needs those exact values to avoid a silent train/serve mismatch (the same class of bug §2 of `PHASE_5_KICKOFF_PROMPT.md` was explicit about for the live-frame preprocessing path).

### 3.2 Numerical parity verification (mandatory before any benchmarking)

**File:** `scripts/verify_onnx_parity.py`. Run both the PyTorch model and an `onnxruntime` `InferenceSession` over the same held-out batch (reuse `data/splits/test.csv` via `FocalPlanesDataset`, a few hundred images is enough) and assert `np.allclose(pytorch_logits, onnx_logits, atol=1e-3, rtol=1e-3)` per-sample, plus confirm **argmax predictions match on 100% of the sample** (looser logit tolerance is fine; a single flipped argmax is not — that's the number that actually matters for the deployed system). Log the max absolute logit difference and the argmax-agreement rate to `logs/eval/onnx_parity.txt`. Do not proceed to benchmarking or quantization until this passes.

### 3.3 Latency benchmarking

**File:** `scripts/benchmark_onnx.py`. Compare, on the same GPU (RTX 4060) for an apples-to-apples number, and separately on CPU if you want the edge-deployment story to mean something:
- Native PyTorch (`torch.no_grad()`, no AMP — match whatever `src/realtime/pipeline.py` actually uses in production, currently AMP-enabled fp16 on GPU) — reuse `scripts/benchmark_pipeline.py`'s timing pattern.
- `onnxruntime` with `CUDAExecutionProvider` (GPU) and `CPUExecutionProvider` (CPU), both with a proper warm-up loop before timing (first-call overhead in onnxruntime is substantial and must not be counted).

Report mean/min/max latency per 100-frame timed run, same format as `logs/realtime/realtime_benchmark_*.txt`. Log to `logs/eval/onnx_benchmark.txt`.

### 3.4 INT8 post-training quantization

This is the step most likely to actually be a pain on a Windows/conda setup — flag this honestly rather than pretending it's a two-line change. Two realistic paths, pick one and document why:

- **Path A (lighter lift): ONNX Runtime's built-in dynamic/static quantization** (`onnxruntime.quantization.quantize_static` with a small calibration set from `data/splits/train.csv`). Works cross-platform without a separate TensorRT install. Start here.
- **Path B (heavier, "real" TensorRT): `trtexec` or `torch-tensorrt`.** Requires a matching CUDA/TensorRT toolkit install alongside the existing PyTorch CUDA wheel — verify version compatibility carefully before starting (per `01_ENVIRONMENT_SETUP.md`'s own caution about not mixing manual CUDA toolkit installs with the PyTorch wheel unless you know you need it). If Path A's INT8 numbers already tell the story you need (accuracy/latency trade-off exists and is quantifiable), Path B is optional polish, not required — don't burn a day on TensorRT toolkit installation friction for marginal additional credibility over Path A.

For whichever path is used: re-run §3.2's parity check against the INT8 model (expect measurable but bounded logit drift — this is the point of the exercise) and re-run §3.3's benchmark. Report the full **three-way table**: PyTorch fp16/fp32 → ONNX fp32 → ONNX INT8, for both latency and test-set macro-F1 (re-run `src/eval/evaluate_test.py`-equivalent scoring against the quantized model's outputs — reuse that script's metrics computation, don't hand-roll a second F1 calculator).

### 3.5 Non-negotiable constraints

1. `checkpoints/convnext_tiny/best.pt` is read-only input — export never modifies it.
2. Load exclusively via `load_inference_model()` — no second model-construction path.
3. Parity check (§3.2) is a hard gate before any benchmark or quantization number gets reported — an unverified export is worse than no export.
4. All artifacts under `exports/onnx/` (new top-level folder) and `logs/eval/onnx_*.txt` — do not write into `checkpoints/`.

### 3.6 Deliverables checklist

- [ ] `scripts/export_onnx.py` run, `exports/onnx/convnext_tiny.onnx` + `.meta.json` produced
- [ ] `scripts/verify_onnx_parity.py` run and passing (100% argmax agreement) before proceeding
- [ ] `scripts/benchmark_onnx.py` run, PyTorch vs. ONNX-fp32 latency table produced
- [ ] INT8 quantization done via Path A (minimum) or Path B, with the toolchain choice justified in writing
- [ ] Three-way accuracy/latency table (fp32/fp16 → ONNX fp32 → INT8) produced and honestly reported, including any accuracy regression
- [ ] Short write-up (`docs/phases/phase_07/onnx_export_results.md`)

---

## Stretch Goal 4 — Web UI Polish (Streamlit/Gradio)

Not this round's priority, kept lighter-detail than Goals 2–3 by design, but expanded slightly from the original one-liner since you asked for more detail generally:

- Build only after Goal 2 is settled — the UI should surface whichever smoothing tier ends up shipped, not be built against a moving target.
- Recommend **Gradio** over Streamlit for this specific use case: native `gr.Video` component handles file upload + playback without extra plumbing, and Gradio's queue-based backend maps more naturally onto the existing threaded capture/inference/render architecture than Streamlit's rerun-on-every-interaction model.
- Scope: video upload → annotated playback (reuse `src/realtime/app.py`'s `build_display_frame()` rendering logic frame-by-frame rather than reimplementing overlay drawing) → downloadable per-clip stability report (reuse `scripts/evaluate_realistic_video.py`'s per-clip CSV row schema as the report format).
- Explicitly out of scope: live webcam streaming through the web UI (latency-over-HTTP considerations the original Phase 5 doc already flagged as a reason to defer this) — file upload/playback only for this pass.

---

## Stretch Goal 5 — Domain-Specific Self-Supervised Pretraining

Not this round's priority. Slightly expanded from the original paragraph:

- Trigger condition already met: Phase 4's FUSC-checkpoint ablation was skipped as non-portable (ResNet encoder vs. convnext_tiny), documented in `docs/EXPERIMENTS.md` — this goal is the "do it properly" follow-up.
- Approach: SimCLR-style contrastive pretraining (simpler to implement correctly than DINOv2-style self-distillation, and sufficient given the systematic-review benchmark cited in `PHASE_4_KICKOFF_PROMPT.md §9` — ~3.1pp median accuracy gain from US-domain SSL pretraining over ImageNet init) over the union of all available unlabeled-as-frames ultrasound imagery: `data/raw/fetal_planes_db/Images/` (ignore labels), `data/raw/ucl_hc18/images/{HC18,UCL}/` (ignore landmark annotations — **still exclude `FP`/`MULTICENTRE`**, same leakage rationale as every other phase), and individual frames sampled from `data/raw/iugc_video/DatasetV3/*/videos/`.
- Backbone: pretrain a `convnext_tiny` (matching the winning architecture) from random init via SimCLR, not from the existing ImageNet-pretrained weights — the point is an independent domain-specific initialization to compare against `convnext_tiny.fb_in22k_ft_in1k`, not a continued-pretraining variant.
- Evaluation: fine-tune the SSL-pretrained backbone on `data/splits/train.csv`/`val.csv` with the exact same config as `configs/convnext_tiny.yaml` (same LR schedule, same class weights, same augmentation) except the pretrained-weights source, then score on `data/splits/test.csv` via the existing `src/eval/evaluate_test.py` — apples-to-apples against the 0.8927 macro-F1 baseline, same held-out test set, same metric.
- This is a multi-day-to-multi-week undertaking (SSL pretraining runs are long even on a single 4060) — scope a realistic epoch/compute budget before starting rather than open-ending it.


## Stretch Goal 6 — Second clinical track (intrapartum monitoring)

Explicitly deprioritized during scoping (see [00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md) §2) but not ruled out forever — if the primary system is complete and working well, and there's appetite to demonstrate the architecture's reusability across a second real clinical task, the IUGC dataset (already downloaded per file 02) is sitting there ready to train a second classification head + reuse the entire real-time serving pipeline (file 05, Part B) for PS/FH visibility classification and AoP/HSD measurement. Treat this as "build the second product," not "extend the first."

---

## Notes for whoever picks this up

Each stretch goal above assumes files 01-06 are fully complete, evaluated, and documented first ([EVAL_REPORT.md](../EVAL_REPORT.md) exists and the primary system demonstrably works end-to-end on video). Don't let stretch-goal scope creep delay getting the core v1 system working — that's the actual deliverable.
Phase 6's `EVAL_REPORT.md` numbers are assumed as the fixed baseline to compare against; if either goal's evaluation script produces a number that contradicts something already locked in that report (e.g., a different switches/min baseline, a different macro-F1), stop and investigate the discrepancy before reporting a "result" — this is the same audit discipline applied throughout Phases 3–6.