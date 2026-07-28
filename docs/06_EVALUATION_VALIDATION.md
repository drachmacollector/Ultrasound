# 06 — Evaluation & Validation

`[AGENT]`-heavy for running evals, `[MANUAL]` for interpreting results and deciding if something needs another training iteration.

---

## 1. Static classification metrics (in-distribution test set — held-out FETAL_PLANES_DB patients)

- Per-class precision, recall, F1
- Macro-averaged F1 (primary headline number, matching the reference repo's own convention and the literature's standard practice for imbalanced classes)
- Confusion matrix, with particular attention to the brain sub-plane cluster (Trans-ventricular vs. Trans-thalamic is the literature-documented hardest pair — expect and report this honestly rather than being surprised by it)
- Compare against the reference repo's reported numbers (92.65% Stage 2 test accuracy) **only as a rough sanity anchor** — not an apples-to-apples comparison, since our label space, split, and possibly backbone differ.

## 2. Cross-device generalization (HC18 + UCL held-out set — `FP` and `MULTICENTRE` explicitly excluded)

**`[CORRECTION — confirmed against source data]`** The publicly released "multi-centre" bundle this evaluation was originally scoped around (arXiv 2512.16710 / *Sci Rep* s41598-026-47854-3) ships four subfolders: `FP`, `HC18`, `UCL`, and a merged `MULTICENTRE`. The paper confirms `FP` is drawn from the same Burgos-Artizzu et al. 2020 source as our own FETAL_PLANES_DB training set (matching sites, devices, and filename convention), so `FP` and `MULTICENTRE` (which contains `FP`) **must not** be used here — doing so would silently blend already-seen training data into a "generalization" number. See [00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md) §5a and [02_DATASETS.md](02_DATASETS.md) §2 for the full evidence chain.

This evaluation therefore uses only:
- **HC18** (999 images, 806 subjects, Netherlands, Voluson E8/730) — Head only
- **UCL** (424 images, 51 subjects, UCLH, single-site) — Head/Abdomen/Femur

Concretely:
- Same metrics as §1, computed **separately** per `source_subset` (HC18 vs. UCL) as well as combined, restricted to the 3 classes this set covers (map our three brain sub-planes to a single "Head" bucket, and document that mapping explicitly).
- Report the **gap** between in-distribution and cross-device performance per class — this gap *is* the generalization result, more informative than either number alone. Expect a meaningful drop; the source paper's own experiments found exactly this pattern across sites/devices. If our drop is severe, that's a legitimate, reportable finding, not a failure to hide — document it plainly.
- **Explicit scope limitation to state in [EVAL_REPORT.md](EVAL_REPORT.md):** because HC18/UCL are pure landmark-biometry datasets, every image in them is a valid standard plane by construction — there is no "Other"/non-standard example in this set. This evaluation therefore validates *plane-identity classification* cross-device (i.e., "given a standard image, does the model pick the right one of Head/Abdomen/Femur"), **not** Stage 1's standard-vs-other decision, and **not** "Thorax" or "Maternal cervix" generalization at all, since no available external dataset covers those two classes. State this plainly rather than implying the whole 8-way pipeline was cross-device validated.
- HC18's internal `Head_Train.csv`/`Head_Test.csv` (737/262) split is the multi-centre paper authors' own internal split, not the original HC18 grand-challenge split — irrelevant to us since we never train on this data; use the full 999 images as held-out evaluation data.

## 3. Realistic-video classification metric (SonoNet-style honest benchmark)

Per the SonoNet paper's own finding, live/video-realistic evaluation is dramatically harder than curated static-image test accuracy (their F1 dropped to ~0.80-0.86 on live annotation vs. much higher on curated benchmarks). To get an honest sense of this gap for our own system:
- Run the trained classifier frame-by-frame over whatever real ultrasound video we have available (synthetic ego-motion clips at minimum; any additional real fetal-anatomy video if you can source a short clip for manual demo purposes — note copyright/licensing if pulling any public video).
- Report frame-level accuracy on this video **without and with** Tier-1 smoothing applied, to explicitly quantify smoothing's contribution rather than just asserting it helps.

## 4. Video stability metrics (novel to this project, not in the reference repo — this is the actual "real-time" contribution)

- **Label-switches-per-minute** (with vs. without smoothing) — lower is better, but zero switches on a clip that genuinely changes planes would indicate the system is too sluggish, so this should be read alongside:
- **Mean latency-to-stabilize after a genuine plane change** (using the manually-marked transition points from [05_TEMPORAL_SMOOTHING_AND_REALTIME.md](05_TEMPORAL_SMOOTHING_AND_REALTIME.md) §A2) — measures responsiveness.
- **Mean dwell time on displayed label** — a sanity check that the system isn't just permanently stuck on one label (which would trivially minimize switches while being useless).

## 5. Ablations to actually run (not just discuss)

- Backbone comparison table (all 6 candidates, in-distribution + cross-device metrics) — feeds the Phase 4 backbone decision.
- Pretraining init (ImageNet vs. domain-specific, if available) — feeds the Phase 4 decision.
- Smoothing on/off, and across the tuned parameter sweep — feeds the Phase 5 decision.
- Class-weighted CE vs. focal loss, specifically on brain sub-plane F1 — only worth running if the confusion matrix from the main run shows this is still a live problem after basic class weighting.

## 6. Reporting format

Produce a single [EVAL_REPORT.md](EVAL_REPORT.md) (or notebook rendered to markdown) at the end covering all of the above, with the confusion matrix and class-distribution charts embedded as images. This is the artifact that documents "does this thing actually work and by how much," separate from the code itself.

## Deliverables checklist

- [ ] In-distribution test metrics computed and reported
- [ ] Cross-device generalization metrics computed and reported (HC18 + UCL only; `FP`/`MULTICENTRE` explicitly excluded and this exclusion documented with reasoning), gap explicitly called out
- [ ] Realistic-video frame-level accuracy measured with/without smoothing
- [ ] Video stability metrics computed (switches/min, latency-to-stabilize, mean dwell time)
- [ ] All planned ablations run
- [ ] [EVAL_REPORT.md](EVAL_REPORT.md) written, including honest statement of limitations (video-transition validation gap, cross-device set covering only 3/7 classes and containing no "Other" examples, `FP`/`MULTICENTRE` exclusion rationale, no true clinical validation)
