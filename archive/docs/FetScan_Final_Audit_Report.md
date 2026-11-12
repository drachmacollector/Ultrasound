# FetScan — Final Project Audit & Pitch-Readiness Review

**Scope:** Full cross-reference of `/docs`, `/src`, `/scripts`, `/logs`, `/checkpoints` metadata, and configs, as attached in project context.
**Audience:** Internal, pre-pitch to BYL Nair Charitable Hospital (Gynaecology/USG/Sonography department).
**Method:** Documentation claims were checked against the actual code paths that would produce them, and against the log/CSV artifacts that are supposed to be the evidence for those claims. Where a claim could not be traced to a surviving artifact, that is called out explicitly rather than given the benefit of the doubt.

---

## 0. Executive Summary

FetScan is a genuinely well-engineered ML systems project: patient-disjoint data splits with hard-assert leakage checks, a six-backbone bootstrap-significance-tested model selection, an empirically tuned two-tier temporal smoothing stack, a real threaded real-time pipeline with instrumented FPS/latency, and an honest ablation culture (focal loss and ACAM are both reported as clean negative results rather than being hidden). That engineering discipline is real and should be emphasized in the pitch.

However, the audit surfaced **one evidence-integrity problem serious enough to require fixing before any hospital sees this**, several **doc-vs-code mismatches** that would embarrass the team if a technical reviewer looked under the hood, and — more importantly — a **strategic framing problem**: the core deliverable (real-time standard-plane identification) is not obviously the highest-value thing a well-staffed hospital gynae/USG department needs, and two of the project's own evaluation results (NatalIA, cross-device gap) are, read honestly, evidence *against* real-world robustness rather than "safe fallback behaviour" as currently framed.

**Bottom line up front:**
- **Do not present this as a diagnostic or clinical-decision tool.** It is not validated for that, and the project's own docs already say so — but the pitch framing needs to say it more loudly than the current README does.
- **Do fix the latency-evidence gap and the NatalIA UI/spot-check issues below before the pitch** — these are the two things a sharp technical reviewer (or a skeptical clinician who's seen AI hype before) would poke a hole in within five minutes.
- **Reframe the pitch** around what this project actually, verifiably demonstrates: a rigorously-built, explainable, real-time computer-vision pipeline for ultrasound video, positioned as a platform for a *documentation-completeness / training-and-education / scan-QA assistant* — not as "AI that finds the right plane for you," which undersells what would actually help a hospital and oversells what the model currently does reliably outside its training distribution.

---

## 1. Critical Technical Audit

Issues are ranked **Critical → High → Medium → Low**. For each: what it is, where, why it matters, and the fix.

### 1.1 — CRITICAL: The headline "489.3 ms transition latency" claim has no surviving supporting evidence, and the one log that *is* on disk for that script shows the opposite result

**Where:** `docs/EVAL_REPORT.md` §5.1 and `README.md` ("Video stabilization latency: 489.3 ms average latency to stabilize") both state this number, attributed to `scripts/evaluate_transition_latency.py` run over "39 genuine transitions" on the 8 synthetic multiplane clips.

**What I found:** The arithmetic checks out structurally — summing the `settle` events across `multiplane_scan_01..08_annotations.json` gives exactly 39 transitions, so the number is *plausible* as a real run's output. But:

- `scripts/evaluate_transition_latency.py` hardcodes `LOG_PATH = "logs/eval/evaluate_transition_latency.txt"` and opens it in **`mode="w"`** (`logging.FileHandler(LOG_PATH, encoding="utf-8", mode="w")`) regardless of which mode (`--natalia` or default) is run.
- The log file actually present in the project (`logs/eval/evaluate_transition_latency.txt`) is unambiguously the **`--natalia`** run — it contains "Running NatalIA transition evaluation…" and per-exam-folder output.
- That surviving log shows **14/14 transitions failing to stabilize** ("Transition at frame X never stabilized on Y!" for every single one), with **Mean Latency-to-Stable = 2,750 ms**, P90 = 5,000 ms, max = 6,458 ms — i.e., the fallback "never stabilized, penalize with remaining clip duration" branch fired on every transition.
- `logs/eval/transition_latency_details.csv` corroborates this: every row has `stabilized=False`.
- There is **no surviving log, CSV, or other artifact anywhere in the provided files** that shows the synthetic-clip (non-`--natalia`) run actually producing 489.3 ms / 500 ms / 525 ms. That run's log was overwritten by the later NatalIA run because the script always truncates the same path.

**Why it matters:** This is the single number in the entire evaluation suite most likely to be quoted verbatim in the hospital pitch (it's in the README's "Results at a Glance"). Right now, if anyone asks "can I see the log for that," the answer is that the log on disk contradicts it. This isn't proof the number is fabricated — the arithmetic match on 39 transitions is a real signal that a synthetic-clip run did happen at some point — but as it stands **the claim is currently unverifiable from project evidence**, which is functionally the same problem for a hospital audience that will (correctly) expect any AI-in-healthcare claim to be traceable to evidence.

**Fix (required before pitch):**
1. Re-run `scripts/evaluate_transition_latency.py` (default, non-`--natalia` mode) today, on the current checkpoint, and save the log to a **run-stamped filename** (e.g., `logs/eval/evaluate_transition_latency_synthetic_<timestamp>.txt`), never overwriting the NatalIA run's log.
2. Patch the script so `LOG_PATH`/`CSV_PATH` are derived from the `--natalia` flag (e.g., `_synthetic.txt` vs `_natalia.txt`), so this class of evidence-loss can't recur.
3. Only then re-confirm the 489.3 ms figure in `EVAL_REPORT.md`/`README.md` — if the re-run produces a different number, update the docs to match, don't keep the old one.
4. This is also a good moment to explicitly separate, in the docs, "synthetic-clip transition latency" (optimistic, same-distribution, single-plane-swap-only) from "NatalIA transition latency" (real motion, real failure mode) — right now `EVAL_REPORT.md` §5.1 does mention the NatalIA attempt but frames it as "could not be timed reliably" due to the model mostly predicting "Other," which is true but soft-pedals that the one time this exact pipeline was run against anything resembling real free-hand probe motion, its transition-tracking behaviour was roughly **5–14× worse** than the headline number.

---

### 1.2 — HIGH: Phase 8 Stage 2.3 (NatalIA showcase clip) is marked "done" in the checklist and walkthrough, but the code does not actually wire it up

**Where:** `docs/instructions/09_MASTER_CHECKLIST.md` marks "Stage 2: NatalIA … evaluation" as `[x]` done; `docs/instructions/08_FINAL_EVALUATION_AND_POLISH.md` Task 2.3 requires the showcase clip to appear in the Streamlit UI "with a clear, non-removable UI label/caption … 'Phantom footage, untrained volunteer operator.'"

**What I found:** `scripts/build_natalia_s8howcase_clip.py` exists and was run (`outputs/demo1/annotated_demo_phantom_1.mp4` and `data/processed/natalia_showcase_clips/demo_phantom_1.mp4` both exist in the project tree). But `app_streamlit.py::_render_examples()` only globs `_SAMPLE_CLIPS_DIR = data/processed/synthetic_clips` and takes the first 3 files — it never references `natalia_showcase_clips/` at all, and there is no phantom-footage caption anywhere in `app_streamlit.py`.

**Why it matters:** If someone opens the live demo expecting to show off the NatalIA case (the most narratively useful clip for a hospital audience — "here's what happens on a completely different device/operator"), it won't be there, and if it *were* manually dropped into the picker it would be shown **without** the mandatory phantom-disclosure caption — which is precisely the honesty safeguard the project's own Phase 8 plan insisted on to avoid a clinician mistaking a plastic phantom for a real patient scan.

**Fix:** Either (a) wire `natalia_showcase_clips/` into `_render_examples()` with the required caption before the pitch, or (b) if there isn't time, remove the "done" checkmark for this sub-task and don't reference the NatalIA clip in the live demo at all — just cite the number from `EVAL_REPORT.md` verbally. Do not let it appear in the UI without the caption.

---

### 1.3 — HIGH: The NatalIA "safe fallback" framing overstates what a 4.6% standard-plane recall actually shows

**Where:** `docs/EVAL_REPORT.md` §2b and `logs/eval/evaluate_natalia.txt`.

**What the log actually shows:** Standard Plane Recall = **4.64%** (7/151 correctly identified), "Other" Precision = 99.25%, and the model's overall binary accuracy (98.6%) is *below* the trivial "always predict Other" baseline (99.2%) — i.e., on this out-of-distribution set the model performs **worse than doing nothing**.

**Why the current framing is a problem:** `EVAL_REPORT.md` calls this "safe fallback behaviour on unseen anatomy" and lists it under "partially closes the limitation." That's a generous read. A model that fails to recognize 95%+ of genuine standard planes on a different device/operator isn't demonstrating graceful degradation — it's demonstrating that its learned features do not transfer at all to meaningfully different acoustic textures. "Doesn't hallucinate a wrong anatomy" is true and worth saying, but "safely defaults" implies a designed safety behaviour, when what's actually happening is the model has no signal at all in that domain and the softmax collapses to the majority class.

**Why it matters for the hospital pitch specifically:** NatalIA uses a different device class (handheld POCUS) and non-expert operators, which is a bigger domain shift than a hospital's own clinical Voluson/Philips machine would represent — so this result is *not* directly predictive of what will happen at BYL Nair. But it **is** directly relevant to a question a sonographer will ask in the room: "what happens if I run this on our machine?" The honest answer, based on everything in this project, is: *we don't know yet — the only test with a domain shift this large showed near-total failure; the hospital's device is presumably closer to a Voluson/Aloka (like the training data) or at least closer than a phantom POCUS, so the more relevant number is the HC18/UCL cross-device gap (~83% vs ~98%, a −15pp drop), but even that has not been validated on your specific machine.*

**Fix:** Reword §2b to state the recall number and the below-trivial-baseline comparison plainly, keep the "no false anatomy hallucination" point (it's true and useful), but drop "safe fallback" as the section's characterization. Add one sentence recommending an on-site pilot capture (even 20–30 minutes of anonymized clips from BYL Nair's actual machine, run through the offline pipeline) before any performance claim is made in front of clinicians. This is also the single most persuasive thing you could offer the hospital in the pitch itself — "let us capture a short pilot session on your equipment and show you the real numbers," which is a much stronger, more credible ask than presenting numbers from a phantom.

---

### 1.4 — MEDIUM: Cross-device backbone comparison table mixes two different accuracy definitions under one ambiguous column header

**Where:** `docs/CROSS_DEVICE_BACKBONE_COMPARISON.md`, column "In-Dist (Overall)" — lists `convnext_tiny` at **95.8%**.

**What I found:** Every other in-distribution number in the project for `convnext_tiny` on the true 8-way test set is **89.27% macro-F1 / ~90% accuracy** (`EVAL_REPORT.md` §1, `checkpoints/convnext_tiny/classification_report_TEST.txt`). The 95.8% figure in the cross-device comparison table can't be the same quantity — it's almost certainly the **collapsed 3-class (Head/Abdomen/Femur only, no "Other")** in-distribution accuracy used specifically as the cross-device baseline in `scripts/evaluate_cross_device.py::run_indist_collapsed_baselines()`, which necessarily scores higher because it excludes the hardest class (`Other`, 1,678 of 5,271 test images) and the brain sub-plane discrimination task entirely (collapsed to one bucket).

**Why it matters:** A reader (or a clinician skimming the doc) could easily walk away thinking the model is "95–97% accurate in-distribution," when the actual, harder, honestly-reported number is 89.27% macro-F1. This is exactly the kind of quiet metric-inflation that erodes credibility if a reviewer cross-references two of your own documents and finds a 6-point unexplained gap.

**Fix:** Relabel the column explicitly as "In-Dist (3-class collapsed, Head/Abdomen/Femur only)" and add a footnote pointing to the true 8-way number, so the two are never confusable.

---

### 1.5 — MEDIUM: `docs/CAM_LOCALISATION_SPOTCHECK.md` cannot be verified as an actual completed manual review

**Where:** The 20-row image table uses placeholder descriptions like `*(Brain_Trans_cerebellum — correct prediction)*` instead of real file paths, and the doc claims `[MANUAL REVIEW COMPLETED]` with specific per-image ratings.

**Why it matters:** The task instructions for this deliverable (`08_FINAL_EVALUATION_AND_POLISH.md` Task 3.3) explicitly required a *human* to look at 20 real, saved images and rate each one — this was called out as something that "must be done by a human… not fabricated or approximated programmatically." Without real, referenceable image filenames in the table, there is no way to confirm that happened rather than being a plausible-sounding synthesis. Whether or not it was actually done, the artifact as it stands does not provide the audit trail the task itself demanded.

**Fix:** Before citing this in the pitch, regenerate the table with actual file paths from `scripts/generate_cam_spotcheck.py`'s output (`outputs/cam_spotcheck/`, which does contain real filenames like `07_Brain_Trans_ventricular_Patient01559_Plane3_2_of_2.png`) and re-confirm the ratings against the real images, or drop the specific 75%-plausible number from any external-facing material and speak about the feature qualitatively only ("Grad-CAM-derived approximate region, spot-checked internally, framed to users as approximate").

---

### 1.6 — MEDIUM: Sloppy/duplicated artifact files that would look bad if shown directly

**Where:** `checkpoints/convnext_tiny_focal/classification_report_TEST.txt` contains the same header ("TEST SET EVALUATION — convnext_tiny_focal…") repeated roughly 30 times before the actual table appears — almost certainly a file-append bug in whatever script wrote it (opened in append mode across multiple runs, or a loop bug). The final numbers match `EXPERIMENTS.md`'s cited 0.8785 macro-F1, so the *result* is fine, but the *artifact* is not presentable.

**Why it matters:** Low risk to correctness, moderate risk to credibility if a technical reviewer asks to see a raw report file and gets handed this.

**Fix:** Regenerate the file (single clean write) before the pitch. While doing so, audit other scripts for the same append-vs-overwrite inconsistency — the project mixes `mode="a"` (e.g., `smoke_test.py`'s log handler, `src/train/train.py`'s training-output handler) and `mode="w"` (e.g., `evaluate_transition_latency.py`) somewhat arbitrarily, which is the same root cause behind both this issue and Issue 1.1.

---

### 1.7 — LOW: Leftover Gradio references in comments/`.gitignore` after the file was removed

**Where:** `app_streamlit.py`'s own module docstring still says *"Delete the gradio code and file and update demo1_walkthrough.md accordingly"* as if it's a pending instruction, and `.gitignore` still has a comment referencing `app_gradio.py`'s output directory. The file itself is confirmed absent from the current project tree, so the deletion happened — the leftover text just wasn't cleaned up.

**Fix:** Trivial doc hygiene; remove the stale docstring line and gitignore comment. Not blocking, but a reviewer reading `app_streamlit.py` top-to-bottom would reasonably wonder if the app is only half-migrated.

---

### 1.8 — LOW: `docs/EXPERIMENTS.md` "backbone decision" reasoning double-counts a rationale

Minor: the write-up justifies keeping `convnext_tiny` partly on "cleaner ONNX export path," but ONNX export (Stretch Goal 3) was never executed for **any** backbone in this project — so this is a hypothetical tie-breaker, not a measured one. Harmless for an internal doc, but avoid repeating this specific justification verbatim in front of technical hospital staff who might ask "did you actually export it?" (answer: no, not yet).

---

## 2. What Is Actually, Verifiably Solid

To keep this balanced — the following claims **do** hold up under cross-referencing and are safe to lead with:

- **Patient-disjoint splitting is real and enforced in code**, not just documented: `scripts/verify_no_leakage.py` and `CrossDeviceDataset`'s own guardrail `assert` (in `src/data/dataset.py`) both hard-fail rather than warn, and the `FP`/`MULTICENTRE` exclusion is independently corroborated by the source paper's own description of that bundle.
- **The six-backbone comparison is statistically honest**: the 2,000-iteration paired bootstrap (`docs/phases/phase_04/bootstrap_significance_output.txt`) correctly flags `convnext_tiny` vs. `tf_efficientnetv2_s` as *statistically tied*, rather than overclaiming a win — and the tie-break reasoning (resolution, normalization simplicity) is reasonable, if slightly overstated re: ONNX (see 1.8).
- **The focal-loss and ACAM ablations are genuine negative results, reported honestly**, including the exact regression numbers. This is a real strength — most teams under deadline pressure quietly drop unfavorable ablations; this one documents them with the same rigor as the wins.
- **The Tier-1 + Tier-2a smoothing result (788.75 → 72 → 0 switches/min) is well-instrumented and the "no-op on stable clips" composability claim is actually verified in `scripts/evaluate_tier2.py`'s output**, not just asserted — the log shows 45/46 clips unaffected and the one stubborn clip fully suppressed, with the caveat about limited evidence scope (~20-frame clip) already flagged in `docs/phases/phase_07/tier2_results.md`. This is a good model of how to report a positive result honestly.
- **The real-time pipeline's FPS numbers, after the Grad-CAM feedback-loop bug was found and fixed, are internally consistent across the final validation run** (`realtime_validation_run_20260811_233812.txt`: 23.4–23.7 fps stable, thermal-flat over 120s). The earlier wildly-inconsistent runs (as low as 3.5 fps) are preserved in the logs as a debugging trail rather than deleted — good practice, but they should not be surfaced to the hospital; only the final, explained number should be shown.
- **The weakly-supervised CAM-to-bbox feature is a smart, cheap addition** (zero new training, reuses the existing trusted Grad-CAM path) and is honestly labelled in the UI as "approx. region (saliency-derived)" with a dashed box, distinct from a real detector — this is exactly the right way to add a visually compelling feature without overclaiming precision.

---

## 3. Demo-Readiness — Technical

**Will the live demo run without crashing?** Plausibly yes for the core path (upload → `render_annotated_video.py` → Streamlit playback) — this is the most exercised code path in the project (`demo1_walkthrough.md` documents three real bugs found and fixed post-launch, which is itself a sign the path has been used enough to shake out obvious issues).

**What to actually demo, and what not to:**

| Demo element | Status | Recommendation |
|---|---|---|
| Upload a clip → stable label + confidence | Solid | Lead with this |
| Grad-CAM overlay | Solid, off by default (correct choice) | Show once, explain it's for explainability not diagnosis |
| Approx.-region dashed box | Solid, honestly labelled | Fine to show, keep the "approximate" language verbally too |
| Tier-1/Tier-2a stability badge | Solid | Good visual, explain briefly what "flicker" it's solving |
| NatalIA phantom clip | **Not wired into UI (§1.2)** | Fix first, or don't show it live — mention the number only |
| Any specific latency number | **Unverifiable as-is (§1.1)** | Do not quote 489.3 ms until re-run and re-logged |
| Cross-device / "works on other machines" claim | Partially supported | Present the −15pp gap honestly; do not claim it generalizes to their machine |
| Bounding boxes as "detection" | Would be a false claim | Never call it detection; the actual detector was abandoned (1-epoch, worse than the classifier) |

---

## 4. The Central Question: Is This The Right Problem?

### 4.1 What the system verifiably does well
Given a video clip, it reliably (per in-distribution test metrics) identifies which of 7 standard planes (or "Other") is on screen, holds that label stably against probe-motion jitter, and shows an approximate saliency region. That is a real, working computer-vision capability.

### 4.2 Who actually needs "which plane am I on" told to them?
This is the crux of the strategic question, and it deserves a direct answer rather than a hedge: **a trained sonographer at a functioning hospital department almost never needs to be told which anatomical plane they are looking at** — recognizing the plane is one of the more basic, over-learned skills in the job, not a bottleneck. The literature survey the project itself conducted (`literature_survey_and_gap_analysis.md`, §2.4 and §8) independently arrives at the same conclusion when it identifies the real, underserved gap in this space:

- **Scan-completeness / checklist automation** — did the operator actually capture all the mandated standard views for a valid anomaly scan, or is a view missing/substandard? (Direct source of missed anomalies and repeat-scan burden.)
- **Diagnostic-acceptability / image-quality scoring within a plane** — is the CSP visible, is head magnification/symmetry within spec, are the correct landmarks present for a *valid* measurement — this is explicitly the top-billed, still-open gap in the project's own survey (§2.4, §8(1)).
- **Biometry assistance** (HC/AC/FL caliper placement) — this is where measurement variability and operator error concretely translate into clinical decisions (dating, growth restriction flags), and is a materially higher-value automation target than plane naming.
- **Novice/non-expert operator support** (community health workers, junior residents) — this is genuinely where "tell me what plane this is" has real value, because the user *doesn't* already know. NatalIA's own motivating use case is exactly this population. But BYL Nair's gynae/USG department is presumably staffed by trained sonographers and radiologists, not novices — so this is likely the wrong audience for that specific value proposition, even though it's the value proposition the current demo is built around.

**The honest read: the project has built a technically strong solution to a problem that is more relevant to a different user population (novice/rural operators, teaching contexts) than to the audience it's about to pitch to (an established hospital gynae/USG department).** This is not a fatal flaw — plane identification + stability + explainability is a legitimate and reusable *substrate* — but it should not be pitched as "we solve your diagnostic workflow problem," because for a well-staffed department it largely doesn't.

### 4.3 Should you pivot now, or reposition?

**Recommendation: reposition, do not attempt a technical pivot before this pitch.**

A genuine pivot toward quality-scoring or biometry-assist would require new labels (landmark presence/absence, or caliper ground truth) that do not exist in any dataset currently in this project, and building it properly is a multi-week undertaking, not something to rush before a hospital meeting. Attempting it now risks shipping something *less* validated than what already exists.

Instead, reposition the existing, real capability honestly:

1. **Lead with the engineering credibility, not the diagnostic value.** "We built a rigorously validated, explainable, real-time video AI pipeline for ultrasound — patient-disjoint validation, statistically-tested model selection, documented failure modes, honest reporting of what doesn't work yet" is a genuinely strong pitch to a hospital that has probably seen overclaiming AI vendors before. Clinicians respond well to teams that say "here is exactly what this can and cannot do."
2. **Frame the current deliverable as a documentation/training/QA assistant, not a diagnostic aid**: e.g., "automatically time-stamps and labels which standard plane was captured, for structured reporting and completeness review" and "a training tool for residents/students to get real-time feedback on plane recognition while learning." Both of these are legitimate, lower-stakes, genuinely useful applications of exactly what's already built, and neither requires the model to be more accurate than it currently is to add value.
3. **Use the pitch itself to propose the actual next-value step**: offer to run a short, consented, anonymized pilot capture on BYL Nair's own equipment (even a few clips) to (a) get real cross-device numbers instead of guessing, and (b) open a conversation about which of the four higher-value problems above (completeness checklist, quality scoring, biometry-assist, training tool) the department would actually want built next. This turns the meeting from "please believe our number" into "let's find out the real number together and co-design what's next" — a much stronger position for a hospital partnership.
4. **What remains reusable if you pursue any of the higher-value directions later:** essentially the entire engineering substrate — the backbone, the data pipeline discipline, the real-time threaded serving architecture, the temporal smoothing layer (equally useful for tracking "is a valid plane currently held" for a completeness checklist), and the Grad-CAM/explainability layer. None of that work is wasted; the pivot, if pursued later, is additive (new heads/labels on top of the existing classifier), not a rewrite.

---

## 5. Final Verdict — What's Complete vs. Functional-but-Unvalidated vs. Must-Fix vs. Future Work

**Genuinely complete and defensible:**
- In-distribution 8-way classification (89.27% macro-F1), patient-disjoint, bootstrap-validated backbone selection.
- Tier-1 + Tier-2a temporal smoothing, with honest evidence-scope caveats already documented.
- Real-time threaded pipeline with instrumented, reproducible (after the Grad-CAM bug fix) FPS numbers.
- Grad-CAM + CAM-derived approximate localization, correctly and honestly labelled as approximate.
- The negative-result ablation culture (focal loss, ACAM) — genuinely good scientific practice.

**Functional but not sufficiently validated for clinical claims:**
- Cross-device generalization (HC18/UCL): real number, real gap, but only tested on 3 of 7 planes and on datasets from a different country/era than BYL Nair's actual equipment.
- NatalIA generalization: real number, but should be described as a failure mode under large domain shift, not a safety feature, and is not predictive of BYL Nair's specific device gap.
- Transition latency: currently **unverifiable** pending the fix in §1.1 — do not quote it until re-run and re-logged.
- CAM localization spot-check: plausible but not independently traceable to real reviewed images as currently documented.

**Must-fix before the pitch (all are hours-of-work, not days):**
1. Re-run and properly log `evaluate_transition_latency.py` (synthetic mode) with a non-overwritable filename; reconcile the number in `EVAL_REPORT.md`/`README.md`. (§1.1)
2. Either wire the NatalIA showcase clip into the UI with its mandatory phantom-disclosure caption, or remove it from the live demo entirely. (§1.2)
3. Reword the NatalIA §2b framing away from "safe fallback." (§1.3)
4. Fix the ambiguous "In-Dist (Overall)" column label in the cross-device backbone comparison doc. (§1.4)
5. Regenerate the corrupted `convnext_tiny_focal` test report file before showing any raw artifacts. (§1.6)
6. Verify or redo the CAM spot-check against real, named images before quoting the 75% figure externally. (§1.5)

**Correctly scoped as future work (do not present as current capability):**
- ONNX/TensorRT export and edge deployment.
- Tier-2b learned temporal head (correctly not built — evidence didn't justify it).
- Any quality-scoring, checklist-completeness, or biometry-assist feature — these are the recommended *next* project, not something to imply already exists.
- Full multi-task detection (the 1-epoch multitask model is correctly excluded from the demo).

---

## 6. One-Paragraph Summary for Whoever Is Delivering the Pitch

You have a well-built, honestly-evaluated real-time ultrasound video classifier with genuine engineering rigor behind it — that part of the story is strong and should be told with confidence. What you do **not** yet have is evidence that this specific capability (naming the plane) is the thing a staffed hospital gynae/USG department is short on, and two of your own strongest-sounding metrics (transition latency, NatalIA "safe fallback") are currently either unverifiable or overstated relative to what the underlying logs actually show. Fix the six items above, drop the framing of this as a diagnostic tool, and walk in proposing a short on-site pilot and a conversation about which higher-value problem (scan completeness, quality scoring, biometry assist, or resident training) the department would actually want built on top of the platform you've already proven you can build well.
