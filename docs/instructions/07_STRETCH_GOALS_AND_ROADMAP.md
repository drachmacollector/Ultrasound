# 07 — Stretch Goals & Future Roadmap

None of this blocks the v1 deliverable (Phases in files 01-06). Pursue only after the primary system is working and evaluated. Ordered roughly by value-to-effort ratio.

---

## Stretch Goal 1 — Detection-informed multi-task head (highest value, was our preferred design if data allowed)

Per the FAUSP-NET pattern: add a structure-detection head alongside the plane classifier, so the system can say *"Trans-thalamic plane — cavum septum pellucidum ✓, thalami ✓, midline falx not clearly visible"* instead of just a bare confidence score. This is what actually makes the tool clinically actionable rather than a black box.

**What's needed:**
- Structure-level bounding-box annotations. The UCL/HC18 dataset's landmark annotations (BPD/OFD, TAD/APAD, FL points) are a partial starting point for head/abdomen/femur — landmarks aren't boxes, but a box could be derived around each landmark cluster with a reasonable margin as a first pass.
- For thorax/cervix/other brain sub-planes, no existing annotation source was identified during planning — would require either manual annotation of a subset (time-intensive) or finding another public dataset with structure-level labels for these specific planes (worth a fresh search when this stretch goal is picked up, literature moves fast in this space).
- Architecture: shared backbone, two heads — a lightweight detection head (single-scale is probably sufficient given our anatomies are usually large in-frame, unlike FAUSP-NET's small structures like IVC) and the existing classification head, with the plane label optionally derivable from which structures are jointly detected (á la FAUSP-NET) rather than a separate softmax.

## Stretch Goal 2 — ONNX / TensorRT export and quantization

Prove the model *could* run on constrained hardware even though our deployment target didn't require it:
- Export the (re-parameterized, if RepVGG) model to ONNX, verify numerical parity with the PyTorch model on a validation batch.
- Run through `onnxruntime-gpu` and compare latency against native PyTorch.
- Try INT8 post-training quantization via TensorRT (requires an NVIDIA toolkit setup pass) and measure the accuracy/latency trade-off — this is the artifact that would let this system's design story extend credibly to an edge/Jetson deployment target, even without actually owning that hardware.

## Stretch Goal 3 — Tier-2 learned temporal module

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


## Stretch Goal 4 — Web UI polish (Streamlit/Gradio)

Once the `cv2.imshow` core loop is solid (file 05, Part B), wrap it in a small web app for easier sharing/demoing — video upload, live annotated playback, downloadable stability-metric report per clip.

## Stretch Goal 5 — Domain-specific self-supervised pretraining, done properly

If Phase 4's pretraining ablation (file 04, Step 2) found no usable public checkpoint, this becomes a real project: run SimCLR or a DINOv2-style self-supervised pass over the union of all unlabeled/labeled ultrasound frames we have access to (FETAL_PLANES_DB + UCL + IUGC frames, ignoring labels), producing our own domain-pretrained backbone checkpoint, then fine-tune on the labeled plane-classification task. This is a legitimate, portfolio-worthy sub-project on its own, referencing the label-efficient learning literature surfaced during planning (SimCLR for fetal planes, federated contrastive learning, DINOv2 fetal-US foundation models).

## Stretch Goal 6 — Second clinical track (intrapartum monitoring)

Explicitly deprioritized during scoping (see [00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md) §2) but not ruled out forever — if the primary system is complete and working well, and there's appetite to demonstrate the architecture's reusability across a second real clinical task, the IUGC dataset (already downloaded per file 02) is sitting there ready to train a second classification head + reuse the entire real-time serving pipeline (file 05, Part B) for PS/FH visibility classification and AoP/HSD measurement. Treat this as "build the second product," not "extend the first."

---

## Notes for whoever picks this up

Each stretch goal above assumes files 01-06 are fully complete, evaluated, and documented first ([EVAL_REPORT.md](../EVAL_REPORT.md) exists and the primary system demonstrably works end-to-end on video). Don't let stretch-goal scope creep delay getting the core v1 system working — that's the actual deliverable.
