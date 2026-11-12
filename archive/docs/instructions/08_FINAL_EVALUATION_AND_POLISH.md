# Phase 8 — Final Showcase-Readiness Implementation Plan

**File:** `docs/instructions/08_FINAL_EVALUATION_AND_POLISH.md` (place here on adoption)
**Status:** Proposed — ready for execution
**Author context:** Written after full review of `docs/literature_survey_and_gap_analysis.md`, all of `docs/instructions/00`–`08`, `docs/EVAL_REPORT.md`, `docs/EXPERIMENTS.md`, `docs/phases/demo1/demo1_walkthrough.md`, `docs/phases/phase_07/*`, the live repository state (`app_streamlit.py`, `app_gradio.py`, `src/realtime/pipeline.py` async Grad-CAM, `src/models/gradcam.py`, `src/models/multitask_model.py`), and `data/raw/NatalIA PBFUS1/README.md`.
**Guiding principle, stated once, binding for every stage below:** *this phase brings the project to a strong, defensible, demoable stopping point. It does not open new research directions. Every task below closes an already-identified gap using tools, data, or code the project already has or has just acquired.*

---

## 0. Pre-flight: reconciling the literature survey against ground truth

Before any implementation task, three factual corrections must be logged, because the survey's recommendations were written with the correct methodology but two specific factual premises need updating now that the actual dataset is in hand. Getting this wrong at the start would waste the highest-priority stage (Stage 2). This section is itself a `[MANUAL]`-reviewed deliverable — read it before touching code.

### 0.1 NatalIA PBF-US1 is a **phantom** dataset, not real patient video

The survey's Table in §3 and its Prioritized Recommendation #3 both describe NatalIA as letting the project "replace 'measured on synthetic multiplane clips' with 'measured on real free-hand sweep video'" and calls it "real free-hand sweep video with genuine ground-truth transitions." This is **materially inaccurate** and must not be carried into any documentation this phase produces.

Confirmed directly from `data/raw/NatalIA PBFUS1/README.md` (the file the user uploaded) and cross-checked against the dataset's Zenodo record and its companion paper ("Development and Evaluation of an AI-Driven Telemedicine System for Prenatal Healthcare," arXiv:2510.01194):

- The dataset was recorded on a **US-7a SPACE FAN phantom** (Kyoto Kagaku, Japan) — a physical mannequin simulating a 23-week pregnancy — **not a human patient**.
- 90 videos were captured by **45 volunteers with no prior ultrasound experience**, each performing 4 predefined sweep paths (1 vertical, 1 horizontal, 2 diagonal) across 4 fixed phantom poses.
- Frames: 19,407 total, from `resume.csv`, class distribution **as stated in the README** (not the secondary-source numbers the survey cited from arXiv:2605.25357, which used a *revised* annotation from a different paper and do not match this project's actual `resume.csv`):

  | Class | Count |
  |---|---|
  | Biparietal (0) | 42 |
  | Abdominal (1) | 63 |
  | Heart (2) | 61 |
  | Spine (3) | 134 |
  | Femur (4) | 46 |
  | No Plane (5) | 19,061 |
  | **Total** | **19,407** |

**What this means for the plan:** NatalIA is genuinely valuable — it is real free-hand *probe motion* by *genuinely untrained operators*, which is exactly the operator population this project's own motivating scenario (`00_PROJECT_OVERVIEW.md` intro, "assistive tool," and the NatalIA README's own stated purpose: "AI tools to assist non-experts... in low-income countries") targets. But it is **phantom anatomy, not human anatomory**, and this must be stated plainly everywhere the dataset is used, exactly the same way the project already discloses "no real patient data" for FETAL_PLANES_DB (`00_PROJECT_OVERVIEW.md` §8). Framing it as "real video" without the phantom caveat would be a credibility risk in front of a clinical audience — a sonographer would immediately recognize the phantom's artificial acoustic texture and ask why it wasn't disclosed. Framing it honestly ("real free-hand probe motion by untrained operators, on a validated obstetric phantom") is still a strong, genuinely differentiated validation story and *should be used*, just correctly labeled.

### 0.2 NatalIA's class taxonomy only partially overlaps this project's 8-way taxonomy

NatalIA has 5 standard planes + No-Plane. Only some of those planes correspond directly to this project's 8 classes:

| NatalIA class | Maps to project class? | Notes |
|---|---|---|
| Biparietal Plane | → collapse to one/all of `Brain_Trans_cerebellum` / `Brain_Trans_thalamic` / `Brain_Trans_ventricular` (see Stage 2 for the exact rule) | NatalIA doesn't distinguish the 3 brain sub-planes; "Biparietal" is a coarse head-plane label, not one of our 3 fine-grained brain classes |
| Abdominal Plane | → `Fetal_abdomen` | Direct match |
| Heart Plane | → **no match** | This project has no `Fetal_heart` / cardiac class (matches SonoNet's broader 13-class scope, which the survey correctly noted our 8-class scope excludes by design; this is an existing, already-documented scope decision, not a new gap) |
| Spine Plane | → **no match** | Not one of our 7 planes |
| Femur Plane | → `Fetal_femur` | Direct match |
| No Plane | → `Other` | Direct match |

**Consequence:** NatalIA can only validate 3 of this project's 8 classes end-to-end (`Fetal_abdomen`, `Fetal_femur`, `Other`, plus a collapsed "any brain class" check analogous to the existing HC18/UCL "Head" collapsed-scoring rule in `EVAL_REPORT.md` §2), and it can validate the standard-vs-Other decision (which HC18/UCL explicitly *cannot*, per `EVAL_REPORT.md` §5.2). `Heart Plane` and `Spine Plane` frames must be **excluded** from the accuracy-scoring manifest — they have no valid ground-truth mapping in our taxonomy and scoring them against any of our 8 classes as "correct"/"incorrect" would be fabricating a ground-truth correspondence that doesn't exist. This exclusion, and its row counts, must be logged exactly like the FP/MULTICENTRE exclusion was (`00_PROJECT_OVERVIEW.md` §5a).

### 0.3 What NatalIA is and isn't good for, given 0.1–0.2 — final scoping decision

| Candidate use | Verdict | Reasoning |
|---|---|---|
| Real-video standard-vs-Other classification accuracy (fills the §5.2 gap in `EVAL_REPORT.md`) | ✅ **Use it** — this is the single most valuable thing NatalIA adds | 19,061/19,407 frames are "No Plane" — by far the largest "Other"-analogous real-video sample this project has ever had access to |
| Real-probe-motion transition-latency validation (fills the §5.1 synthetic-clip caveat) | ⚠️ **Conditionally use it** — only if genuine plane-to-plane transitions exist within continuous per-video frame sequences; must be verified empirically before committing (Stage 2, Task 2.2) | The 90 exam folders are continuous sweeps; if frame timestamps/ordering are preserved and a folder contains both plane and no-plane frames in sequence, real transitions exist. This must be checked, not assumed — same discipline the project already applied to IUGC (Phase 5 Task 5 found *zero* usable transitions there) |
| Cross-device / cross-population generalization test, analogous to HC18/UCL | ✅ **Use it**, correctly labeled as *cross-population, non-expert-operator, phantom-anatomy* generalization, not "cross-device" in the clinical-hardware sense HC18/UCL represents | Distinct and complementary axis of generalization: HC18/UCL = same expert operators/real patients, different clinical device+site. NatalIA = different (untrained) operators, phantom anatomy, single consumer POCUS device. Reporting both, clearly distinguished, is a genuine strength |
| Training data / fine-tuning source | ❌ **Do not use for training** | Phantom anatomy has a different acoustic signature than real fetal tissue; training on it risks teaching phantom-specific artifacts. This mirrors the project's own existing, correct decision to never use IUGC frames for classifier training (`00_PROJECT_OVERVIEW.md` §5) |
| A NatalIA-derived "Demo 2" showcase clip | ✅ **Use it** — pick 1–2 of the most visually clean full sweep videos as an additional Streamlit/Gradio example clip, clearly labeled as phantom footage | Directly demonstrates the project's stated target use case (non-expert operator assistance) using footage that actually shows an untrained operator scanning, which no existing demo clip does |

This scoping is now locked. Stage 2 implements exactly this and nothing more.

---

## 1. Phase 8 Scope Summary — what is in, what is deferred

Cross-referencing the literature survey's own prioritization (§8 of the survey) against the actual codebase state, current gaps, and the "showcase-ready, not endlessly expanding" mandate:

### In scope for Phase 8 (this plan)

| # | Item | Survey ref | Why it clears the bar |
|---|---|---|---|
| 1 | NatalIA PBF-US1 acquisition, integration, and real-video evaluation | §3, §8(2)-3 | Directly closes the two most-cited honest limitations in `EVAL_REPORT.md` §5.1/§5.2; data now in hand; moderate, bounded effort |
| 2 | CAM-to-bounding-box weakly-supervised localisation | §7, §8(2)-1 | Near-zero engineering cost (reuses `src/models/gradcam.py` verbatim); unlocks a real, always-on "show me the anatomy" feature in both Demo UIs today, replacing the currently-disabled/unused multitask bbox path |
| 3 | Full 6-backbone cross-device sweep | `EVAL_REPORT.md` §5.5, already self-flagged as a Phase 7 candidate | Cheap (inference-only, no training) with existing script; closes a self-identified evaluation gap; strengthens the backbone-decision narrative for a technical audience |
| 4 | ACAM-style adaptive-contrast preprocessing ablation | §2.2, §8(2)-2 | Cheap, architecture-agnostic, single ablation against the already-locked `convnext_tiny` baseline using the existing bootstrap-CI protocol; either wins cleanly (ship it) or loses cleanly (document and discard, exactly like the focal-loss ablation) |
| 5 | Documentation, UI, and demo consolidation pass | New (repo-observed gap) | The repo currently has two web UIs (Gradio + Streamlit), inconsistent "which is primary" signaling across docs, an out-of-date `09_MASTER_CHECKLIST.md`, and `assets/fetscan_uiux_fix_brief.md` with unclear completion status — all must be resolved before a hospital demo |
| 6 | Final consolidated evaluation report, documentation audit, and acceptance test | Required by task instructions | Closes the phase; the deliverable the whole plan builds toward |

### Explicitly deferred (do not implement this phase)

| Item | Survey ref | Why deferred |
|---|---|---|
| Brain-subplane targeted intervention (GAN augmentation or COMFormer-style dedicated head) | §8(2)-4 | Real, well-motivated idea, but it is a *research experiment* with an uncertain outcome (may regress, like focal loss did), non-trivial new-code surface (GAN training pipeline or a second loss head), and no currently-broken user-facing behavior blocking a demo. Correctly bucketed by the survey itself as worth trying, not as blocking v1. Defer to a post-showcase research phase. |
| FetalCLIP / USFM foundation-model re-ablation | §2.3 | Requires downloading and adapting a CLIP-ViT or ViT-MIM encoder, a materially different architecture-integration task from anything in the current pipeline, with an uncertain payoff given the current `convnext_tiny` result is already statistically robust against 4/5 alternatives. Genuinely a "next phase" item, exactly as the survey's own §8(4) says. |
| Video-level quality/clinical-acceptability scoring layer | §2.4, survey's own §8(1) top-billed gap | The survey is correct that this is the most clinically substantive gap — but it requires either new landmark-presence detection logic or a new quality-scoring model, which is a materially new subsystem, not a bounded task. This is the correct scope boundary for "later phase" per the task's own instructions. Recorded explicitly in the final report's Limitations section as the top item for a future phase. |
| ScanAhead-style predictive guidance, TIER-LOC, memory-based video QA | §2.1 | Survey's own verdict: "not a near-term implementation item," requires data this project doesn't have. No further discussion needed. |
| ONNX/TensorRT export, INT8 quantization, mobile/edge deployment | `07_STRETCH_GOALS_AND_ROADMAP.md` Stretch Goal 3 | Already correctly scoped as a stretch goal; no clinical department demo requires edge deployment; revisit only if a real deployment target emerges. |
| Tier-2b learned temporal (GRU) module | `09_MASTER_CHECKLIST.md`, already decided | Tier-2a already achieves 100% flicker elimination on the eval suite; no evidence gate (`07_STRETCH_GOALS_AND_ROADMAP.md` §2.0) has been met that would justify building this. |
| Second clinical track (intrapartum) | `00_PROJECT_OVERVIEW.md` §2 | Explicitly rejected at project inception; nothing in this phase's findings reopens that decision. |
| Prospective/clinical validation | Survey §4, `EVAL_REPORT.md` §5.4 | Explicit, correct, and permanent non-goal for this project. Stated once more in the final report and left alone. |

---

## 2. Phase 8, Stage 1 — NatalIA PBF-US1 Acquisition, Verification, and Manifest Construction

**Goal:** Get the already-downloaded NatalIA dataset into a verified, leakage-safe, correctly-scoped manifest, following exactly the same discipline (`verify_no_leakage.py`-style hard assertions, explicit exclusion documentation) already used for FP/MULTICENTRE and HC18/UCL.

**Depends on:** Nothing — the dataset is already present at `/data/raw/NatalIA PBFUS1` per the task instructions. This is the correct starting stage of Phase 8.

### Task 1.1 — Relocate and structurally verify the dataset

- Move (do not copy, to avoid duplicate large binary storage) `/data/raw/NatalIA PBFUS1/` into the project's own `data/raw/` directory as `data/raw/natalia_pbfus1/` (lowercase, underscore convention, matching `data/raw/iugc_video/`, `data/raw/fetal_planes_db/`, `data/raw/ucl_hc18/`). Preserve the original README as `data/raw/natalia_pbfus1/README_original.md` and copy it verbatim (the project convention, per `data/raw/ucl_hc18/README-*.md`, is to preserve source READMEs unmodified alongside any project-authored documentation).
- Verify structurally against the README's stated schema:
  - Assert exactly 90 subdirectories matching the pattern `Obstetrics Exam - *`.
  - Assert `resume.csv` exists with exactly 19,407 rows and columns `file_name, studie, class, value` (and `image` if present).
  - Assert every `file_name` in `resume.csv` resolves to an actual `.jpeg` file inside its corresponding `studie` subdirectory. Log any mismatches — do not silently drop rows.
  - Assert the `value` column only contains integers 0–5 and that the per-class row counts match the README table in §0.1 exactly (42/63/61/134/46/19,061). If they do not match, **stop and report the discrepancy before proceeding** — this would indicate either a corrupted/partial download or a dataset version mismatch, and no downstream task should silently proceed on unverified data.
  - Check for `metadata.csv` (study-level demographics per the README) and inventory its columns; this is not required for the accuracy evaluation but should be captured for the documentation deliverable (Task 1.4) and may later inform an operator-experience stratification if desired.
- **[MANUAL]** Spot-check 15–20 images across at least 10 different `studie` folders by eye: confirm they visually resemble obstetric ultrasound (phantom) frames, confirm no corrupted/unreadable images, and sanity-check a handful of `class=5 (No Plane)` frames genuinely look like off-plane/transitional content rather than mislabeled standard planes.
- **Output:** `logs/natalia_structural_verification.txt` — full assertion results, row counts, any anomalies found. `docs/NATALIA_INTEGRATION.md` (started here, completed in Task 1.4).
- **Acceptance criterion:** All structural assertions pass, or all failures are explicitly logged and understood before Stage 2 begins. Zero silent data drops.

### Task 1.2 — Verify licensing and cite correctly

- Confirm CC-BY 4.0 licensing (already independently verified during planning: Zenodo record 10.5281/zenodo.14193949, "Rights: cc-by-4.0"). Record the exact citation from the README (§ References) in `docs/NATALIA_INTEGRATION.md` and add it to the project's existing license-notes location (wherever `02_DATASETS.md` / README.md currently tracks per-dataset license attributions — follow that existing convention, do not invent a new one).
- **[MANUAL]** Confirm no separate data-use agreement or additional terms exist beyond the Zenodo CC-BY-4.0 grant (check the linked GitHub repo `BiomedLabUGgt/NatalIA-PBF-US1` for any supplementary terms file). This is a quick check, not a legal review — flag anything unusual for the project owner's attention rather than making a legal determination.
- **Output:** License section appended to `docs/NATALIA_INTEGRATION.md` and to the project's dataset license inventory.
- **Acceptance criterion:** Citation and license terms recorded verbatim with source links; no unresolved licensing question remains.

### Task 1.3 — Build the canonical-class-mapped, exclusion-documented manifest

- New script: `scripts/build_natalia_manifest.py`, following the exact structural pattern of `scripts/build_cross_device_manifest.py` (same CLI conventions, same logging style, same use of `CANONICAL_CLASSES`/`CLASS_TO_IDX` from `src/data/dataset.py`).
- Mapping logic (per §0.2 above), applied row-by-row from `resume.csv`:
  - `class=0` (Biparietal) → mapped to a new manifest field `collapsed_label="Head"` (reusing the exact collapsed-scoring convention `evaluate_cross_device.py` already established for HC18's Head rows — **do not invent a second, incompatible collapsing scheme**). These rows are **not** given one of the 3 specific brain-subplane labels, exactly as HC18 Head rows aren't.
  - `class=1` (Abdominal) → `canonical_label="Fetal_abdomen"`.
  - `class=4` (Femur) → `canonical_label="Fetal_femur"`.
  - `class=5` (No Plane) → `canonical_label="Other"`.
  - `class=2` (Heart) and `class=3` (Spine) → **excluded** from the scoring manifest; write them to a separate `data/processed/natalia_excluded_no_taxonomy_match.csv` for transparency (mirrors how `FP`/`MULTICENTRE` exclusions were logged, not silently dropped).
- Preserve `studie` (exam/video ID) and frame filename/ordering info in the manifest — required for Task 2.2's transition-latency check, which needs to reconstruct per-video frame sequences.
- Add a hard assertion (mirroring `CrossDeviceDataset`'s guardrail assert in `src/data/dataset.py`) that no row in the manifest has a `canonical_label`/`collapsed_label` outside the expected set, and that excluded classes never leak into the scoring manifest.
- Because this is phantom data (single phantom, not multiple real patients), there is **no patient-leakage concern in the FETAL_PLANES_DB sense** — but there is a **volunteer/operator leakage** concern if this dataset is ever used for anything beyond held-out evaluation (it will not be, per §0.3, but document why the concern doesn't apply rather than silently omitting the check other manifests have).
- **Output:** `data/processed/natalia_manifest.csv` (scoring-eligible rows only, with `collapsed_label`/`canonical_label`, `studie`, `file_name`, original `class`/`value`), `data/processed/natalia_excluded_no_taxonomy_match.csv`, `logs/build_natalia_manifest_output.txt`.
- **Acceptance criterion:** Manifest row count = 19,407 − (Heart + Spine rows) = 19,407 − 61 − 134 = **19,212 rows**. Verify this exact arithmetic in the log output. Every included row has exactly one valid `canonical_label` or `collapsed_label`.

### Task 1.4 — Write the dataset integration documentation

- Complete `docs/NATALIA_INTEGRATION.md` covering: dataset description (phantom, operators, protocol — copied/summarized from the README), the class-mapping table from §0.2, the exclusion rationale and counts from Task 1.3, the licensing info from Task 1.2, and an explicit **"What this dataset is and is not"** section reproducing the honest phantom-not-real-patient framing from §0.1 verbatim in spirit (this exact framing must also later appear in `EVAL_REPORT.md` and any UI-facing text that surfaces NatalIA-derived numbers — Stage 2 and Stage 6 both reference back to this section).
- **[MANUAL]** Read the completed document once fully before proceeding to Stage 2; this is the one document in the whole phase most likely to get quoted directly to a clinical audience, so it is worth a careful human pass.
- **Acceptance criterion:** `docs/NATALIA_INTEGRATION.md` exists, is internally consistent with the manifest counts from Task 1.3, and has been manually reviewed.

---

## 3. Phase 8, Stage 2 — NatalIA-Based Evaluation (Standard-vs-Other, Real-Probe-Motion Transition Latency, Non-Expert-Operator Generalization)

**Depends on:** Stage 1 complete and manifest verified.

### Task 2.1 — Frame-level accuracy and standard-vs-Other evaluation on NatalIA

- New script: `scripts/evaluate_natalia.py`, structurally modeled on `scripts/evaluate_cross_device.py` (same checkpoint-loading path via `load_inference_model()`, same collapsed-scoring convention for the `Head` label, same output-file conventions).
- Run `checkpoints/convnext_tiny/best.pt` (the locked production checkpoint — **never** the `multitask` checkpoint, which per `EVAL_REPORT.md` §2's warning already has HC18/UCL contamination and must not additionally be conflated here) over all 19,212 scoring-eligible NatalIA frames.
- Report, separately and clearly labeled (do not blend into the existing HC18/UCL cross-device table — this is a structurally different generalization axis per §0.3):
  - Overall accuracy across all included classes.
  - Per-class accuracy/precision/recall for `Fetal_abdomen`, `Fetal_femur`, `Other`, and collapsed `Head` (Biparietal).
  - **Standard-vs-Other binary accuracy** — i.e., collapsing `{Head, Fetal_abdomen, Fetal_femur}` vs `{Other}` and reporting precision/recall/F1 on that binary decision. This is the single most valuable number this evaluation produces, because it is the one thing HC18/UCL structurally cannot measure (`EVAL_REPORT.md` §5.2). Given the extreme class imbalance (19,061 "No Plane" vs 246 total standard-plane frames), **report this primarily as recall-on-standard-planes and precision-on-Other**, not raw accuracy, and explicitly discuss the imbalance in the writeup — a trivial "always predict Other" baseline would score ~98.7% raw accuracy while being useless, so raw accuracy alone would be a misleading headline number here. Report the trivial-baseline accuracy explicitly, right next to the model's number, so the gap is self-evidently the meaningful signal.
  - A confusion breakdown for standard-plane frames that are misclassified (analogous to `EVAL_REPORT.md` §2's "Head Misclassification Breakdown" table): does the model still catch the *general* plane family correctly, or does it hallucinate an unrelated class?
- **Output:** `logs/eval/evaluate_natalia.txt`, `data/processed/eval_natalia/natalia_results.csv` (per-frame), `data/processed/eval_natalia/natalia_confusion.png`.
- **Acceptance criterion:** Script runs to completion over all 19,212 rows without crash; output files match the row-count arithmetic from Task 1.3 exactly; the standard-vs-Other recall/precision numbers and the trivial-baseline comparison are both present in the log.

### Task 2.2 — Real-probe-motion transition-latency investigation (conditional — verify before committing)

This task follows the exact "measure before assuming, stop if the evidence isn't there" discipline the project already applied to IUGC (Phase 5 Task 5 found zero usable transitions and the project correctly said so rather than forcing a number).

- **Step A — Investigate, do not assume.** Using the `studie`-grouped, filename-ordered NatalIA manifest from Task 1.3, reconstruct each of the 90 exam folders as an ordered frame sequence (order by the `cineframe_NNN_...` numeric index in the filename, per the README's stated naming convention). For each exam, walk the ordered `canonical_label`/`collapsed_label` sequence and identify genuine plane→plane or plane→Other→plane transitions (a transition is a maximal run of one label followed immediately by a maximal run of a *different* label, at whatever the native frame rate/sampling of the extracted-frame sequence is — note the README doesn't guarantee every video frame was extracted and labeled, only that labeled frames exist, so check the actual `cineframe` index gaps within a folder before treating the sequence as continuous).
  - **[MANUAL]** Because label sampling density may be sparse (only 246 of 19,407 frames carry a standard-plane label; the rest are "No Plane"), a human must review a sample of the folders with the most candidate transitions and confirm by eye (viewing the actual sequential frames) that a genuine, physically continuous plane-to-transition-to-different-plane event is depicted, not an artifact of sparse/non-uniform labeling within one continuous video. This mirrors exactly what Phase 5 Task 5 already did for IUGC.
- **Step B — Decision gate.** If Step A finds **zero or a statistically unusable number (<5) of genuine, visually-confirmed transitions**, do not force a latency number. Document this exactly as honestly as the existing `EVAL_REPORT.md` §3b "Honest Limitation Statement" does for IUGC, and **do not attempt** to build a new synthetic-from-NatalIA-frames clip generator as a workaround — that would just reproduce the existing synthetic-clip methodology with different source frames, adding no real validation value, and is explicitly the kind of "technically interesting but not worth doing" expansion the task instructions ask to avoid.
- **Step C (only if Step A finds ≥5 usable transitions) — measure latency.** Extend `scripts/evaluate_transition_latency.py` (reuse, do not fork, its existing latency-measurement logic — the smoothing pipeline and the definition of "settle" event should be identical to what's already validated on the synthetic clips) to accept a NatalIA exam-folder sequence as an ordered frame stream, run it through the full Tier-1+Tier-2a pipeline exactly as the synthetic clips are run, and report mean/median/P90 latency-to-stable **as a separate, clearly-labeled number**, explicitly not merged into or replacing the existing 489.3ms synthetic-clip figure in `EVAL_REPORT.md` §5.1. Both numbers should be reported side by side in the updated report (Stage 6) with a one-line explanation of what differs between them (synthetic parallax-warped single-class clips vs. real untrained-operator probe motion with sparse frame-level labels).
- **Output:** `logs/natalia_transition_investigation.txt` (Step A findings, regardless of outcome), and if Step C executes: `logs/eval/evaluate_transition_latency_natalia.txt`, updated CSV outputs alongside (not overwriting) the existing synthetic-clip transition-latency artifacts.
- **Acceptance criterion:** Step A's findings are documented either way. If Step C is skipped, the documentation states why in the same honest register as the existing IUGC limitation statement. If Step C runs, its output is clearly delineated from the synthetic-clip numbers, never silently blended.

### Task 2.3 — Select and integrate 1–2 NatalIA showcase clips into the demo UIs

- **[MANUAL]** From the 90 exam folders, manually review a shortlist (start from the folders identified as visually clean in Task 1.1's spot-check and Task 2.2's transition review) and select 1–2 full sweep videos that best demonstrate: visible probe motion, at least one standard-plane frame run, and reasonable visual/video quality. Reconstruct the selected exam's frames into a playable `.mp4` using the same frame-rate (24fps, per the README) and encoding convention `scripts/render_annotated_video.py` already expects, via a small new helper `scripts/build_natalia_showcase_clip.py` (reuses `imageio-ffmpeg`, same H.264/yuv420p settings already established in Demo 1, per `demo1_walkthrough.md` §3).
- Add the resulting clip(s) to `data/processed/synthetic_clips/` — or a new sibling directory `data/processed/natalia_showcase_clips/` if kept structurally distinct, which is preferable since these are not synthetic and should not be visually or documentation-wise conflated with the parallax-warped clips — as selectable examples in `app_streamlit.py`.
- Add a clear, non-removable UI label/caption on these specific example clips stating "Phantom footage, untrained volunteer operator (NatalIA PBF-US1 dataset)" — this is the UI-facing enforcement of the §0.1 honesty requirement, and it must not be possible for a demo viewer to mistake this for real patient footage.
- **Output:** 1–2 new `.mp4` showcase clips, UI wiring in both apps, updated example-clip picker.
- **Acceptance criterion:** Both Streamlit and Gradio apps load and correctly process the new clip(s) end-to-end (annotated output plays, labels/confidence/Grad-CAM overlay all render); the phantom-footage caption is visible in the UI without the user needing to click into any secondary panel.

---

## 4. Phase 8, Stage 3 — CAM-to-Bounding-Box Weakly-Supervised Localisation

**Depends on:** Nothing from Stage 1/2 — fully independent, can be executed in parallel by a different work-stream if desired. Ordered here for narrative flow, not a hard dependency.

**Rationale (from survey §7, verified against the actual `src/models/gradcam.py`):** The trained classifier already produces class-conditional Grad-CAM heatmaps, verified per-backbone and already wired into both the live pipeline (`src/realtime/pipeline.py`, now asynchronous) and the offline renderer (`scripts/render_annotated_video.py`). SonoNet's weakly-supervised localisation is architecturally the same family of technique (backward-pass saliency from an already-trained classifier), just carried one step further: threshold the heatmap into a binary mask, then take its bounding rectangle. This requires zero new training and does not depend on the stalled 1-epoch `multitask` detection model, which `demo1_walkthrough.md` §7.1 already documents as producing unusable boxes and which is explicitly not used in Demo 1.

### Task 3.1 — Implement CAM-to-bbox extraction

- New module: `src/models/cam_localisation.py`. Public function `cam_to_bbox(cam: np.ndarray, method: str = "otsu", percentile: float = 80.0) -> tuple[int,int,int,int] | None`, operating on the same per-pixel CAM heatmap array already produced by the existing `GradCAM` call sites (do not re-derive the CAM; take it as input from the existing `pytorch_grad_cam` output already computed in the pipeline).
  - Default method: Otsu thresholding on the normalized heatmap (standard, parameter-free, well-understood — matches the survey's suggested approach). Implement a `percentile` fallback method as a documented alternative, selectable via config, in case Otsu proves too permissive/restrictive on some classes during Task 3.2's spot-check.
  - After thresholding to a binary mask, take the **largest connected component** (via `cv2.connectedComponentsWithStats`) rather than the raw bounding box of all thresholded pixels — this avoids a box that spans the whole image when the CAM has multiple small, disconnected hot regions (a failure mode already observed and documented in `docs/EXPERIMENTS.md`'s Grad-CAM spot-check notes: "Heatmap was split across two disconnected regions... indicating diffuse uncertainty"). When no connected component survives thresholding (fully diffuse/low-confidence CAM), return `None` — the caller must handle "no meaningful box for this frame" as a valid, expected outcome, not an error.
  - Add a minimum-box-area sanity filter (configurable, default suppressing boxes below ~2% of frame area) to reject degenerate single-pixel-cluster boxes.
- Unit tests in `scripts/tests/` (following the existing test-file conventions in that directory) covering: a synthetic CAM with one clear hot blob (expect a sensible box), a synthetic CAM with two disconnected blobs (expect the larger one only), a uniform/flat CAM (expect `None`), and an all-zero CAM (expect `None`, not a crash).
- **Output:** `src/models/cam_localisation.py`, test file, `logs/cam_localisation_unit_tests.txt`.
- **Acceptance criterion:** All unit tests pass; function never raises on any of the four synthetic-CAM cases; visually sanity-checked against 2–3 real CAMs from the existing spot-check images referenced in `EXPERIMENTS.md`.

### Task 3.2 — Wire into the rendering/display pipeline, with honest framing

- Modify `build_display_frame()` (`src/realtime/app.py`) and its call sites in `src/realtime/pipeline.py` (async path) and `scripts/render_annotated_video.py` to optionally draw the CAM-derived box whenever a CAM was computed for that frame (respecting the existing `gradcam_every_n` throttling — do not compute boxes on frames where no CAM was run).
- **Critical framing requirement, directly from the survey's own caveat (§7):** this box must be visually and textually distinguished from a "real," fully-supervised detection output. Use a **dashed or semi-transparent box style** (not the same solid-line style a trained detector would imply) and a text label such as `"approx. region (saliency-derived)"` rather than anything implying measurement-grade localisation. This is not optional polish — it is the honesty requirement the survey explicitly flags as necessary given this technique is *not* as accurate as SonoNet's purpose-trained weak-localisation head.
- Add a toggle (Streamlit sidebar checkbox / Gradio checkbox, matching the existing pattern used for the Grad-CAM and Tier-2a toggles in `demo1_walkthrough.md` §4) — default **on**, since it is a genuinely new, cheap, always-available feature, but must remain user-toggleable exactly like the existing overlays.
- Update `demo1_walkthrough.md` §9 verification checklist to add: "CAM-derived approximate region box visible (dashed style, labeled 'approx. region')" as a new checked item.
- **Output:** Modified `src/realtime/app.py`, `src/realtime/pipeline.py`, `scripts/render_annotated_video.py`, `app_streamlit.py`. Updated `demo1_walkthrough.md`.
- **Acceptance criterion:** Running both `validate_pipeline_headless.py` and a live Streamlit/Gradio render shows the dashed approximate-region box overlaid correctly on frames where a CAM was computed, absent on frames where it wasn't (no crash, no box "carried over" from a stale prior frame), and the box visibly tracks the anatomy across a few frames of a stable-plane clip.

### Task 3.3 — Lightweight quantitative sanity check (not a full localisation benchmark)

- This project has no ground-truth bounding-box annotations on FETAL_PLANES_DB itself (only HC18/UCL have landmark-derived boxes, and those are already committed to the `multitask` detection line, which is explicitly out of scope for reuse here per the survey's own note that this technique should "entirely sidestep" that pipeline).
- Rather than building a new annotation effort (out of scope — this is a "final polish" feature, not a new research validation track), perform a **bounded, documented spot-check**: for 20 held-out test images spanning all 7 anatomical classes (reuse the same 10-image-plus-more Grad-CAM spot-check convention from `EXPERIMENTS.md`), a human visually rates each CAM-derived box as "plausible" / "too large/diffuse" / "too small/off-target" / "no box produced." Report the tally.
- **[MANUAL]** This rating task itself must be done by a human — it is a subjective visual-plausibility judgment, not something to fabricate or approximate programmatically.
- **Output:** `docs/CAM_LOCALISATION_SPOTCHECK.md` with the 20-image tally, a few example screenshots, and an honest summary paragraph (e.g., if `Brain_Trans_ventricular`/`Brain_Trans_thalamic` — the model's known weak pair — also produces worse boxes, say so plainly, consistent with the project's existing culture of reporting negative/mixed findings rather than only positive ones).
- **Acceptance criterion:** 20/20 images rated, tally and examples documented, no image skipped without a logged reason (e.g., corrupt file).

---

## 5. Phase 8, Stage 4 — Full 6-Backbone Cross-Device Sweep

**Depends on:** Nothing new — all 6 checkpoints already exist in `checkpoints/`, and `scripts/evaluate_cross_device.py` already exists and works (used for `convnext_tiny` in Phase 6). This is the cheapest stage in the whole plan: no training, no new code architecture, just re-running an existing, trusted script 5 more times.

### Task 4.1 — Run the existing cross-device evaluator against all 5 remaining backbones

- Run `scripts/evaluate_cross_device.py --checkpoint checkpoints/{backbone}/best.pt` for `tf_efficientnetv2_s`, `efficientnet_lite0`, `repvgg_a2`, `repvgg_a1`, `mobilenetv3_large_100` (the 5 backbones `EVAL_REPORT.md` §5.5 explicitly flags as not yet cross-device evaluated).
- No script modification should be needed — `demo1_walkthrough.md` and `EVAL_REPORT.md` both confirm the model loader already reads backbone/config from the checkpoint itself, never hardcoded, so the exact same script should work unmodified for all 5. If it does require modification (e.g., a hardcoded path), that is itself worth noting as a small bug-fix, logged the same way the project logs all such fixes (see the 14-bug log style already established in `docs/phases/phase_07/goal_1_walkthrough.md`).
- **Do not re-run for `convnext_tiny`** — its result (83.2% combined, per §2 above) is already committed and frozen; reuse it verbatim in the consolidated table (Task 4.2).
- **Output (per backbone):** `logs/eval/evaluate_cross_device_{backbone}.txt`, `data/processed/eval_cross_device/{backbone}_cross_device_results.csv`, `data/processed/eval_cross_device/{backbone}_cross_device_confusion.png`. **Do not overwrite** the existing `convnext_tiny` artifacts — this is exactly the "preserve historical evaluation artifacts" requirement from the task instructions; use backbone-suffixed filenames throughout, never the bare names already in use.
- **Acceptance criterion:** All 5 runs complete without error; each backbone's combined HC18+UCL accuracy, per-subset breakdown, and generalization-gap table are present in its log, in the identical format `EVAL_REPORT.md` §2 already established for `convnext_tiny` (so the results are directly diffable/comparable).

### Task 4.2 — Consolidate into a single 6-backbone cross-device comparison table

- Produce a single consolidated table (all 6 backbones × combined accuracy, per-subset accuracy, generalization gap vs. in-distribution) in a new `docs/CROSS_DEVICE_BACKBONE_COMPARISON.md`.
- Explicitly answer the two questions `EVAL_REPORT.md` §5.5 already poses: (a) does `convnext_tiny`'s in-distribution lead over `tf_efficientnetv2_s` hold or reverse cross-device? (b) does `tf_efficientnetv2_s`'s in-distribution `Brain_Trans_ventricular` F1 advantage (0.80 vs 0.77) translate into a cross-device Head-accuracy advantage?
- If any backbone beats `convnext_tiny` cross-device by a **non-trivial** margin, this is a genuine, reportable finding — but per the "single-checkpoint deployment target" decision already locked in `EXPERIMENTS.md` (§ Backbone decision) and reaffirmed by the project's own non-goals, **do not** treat this as grounds to switch the shipped checkpoint mid-Phase-8. Document the finding, note it as a candidate consideration for a future retraining cycle if generalization becomes a stated priority, and leave `convnext_tiny` as the shipped model. This mirrors exactly how the project already handled the `tf_efficientnetv2_s` `Brain_Trans_ventricular` in-distribution edge (documented, not acted on).
- **Output:** `docs/CROSS_DEVICE_BACKBONE_COMPARISON.md`.
- **Acceptance criterion:** All 6 backbones represented in one table; both explicit questions above answered with a stated verdict (not left open); shipped-checkpoint decision explicitly reaffirmed or reconsidered with reasoning, not silently ignored.

---

## 6. Phase 8, Stage 5 — ACAM-Style Adaptive-Contrast Preprocessing Ablation

**Depends on:** Nothing new architecturally, but this is the one stage in the plan that involves actual model retraining, so it should be scheduled with realistic time expectations (a single training run, not a sweep).

**Rationale (survey §2.2):** A validated, architecture-agnostic, lightweight preprocessing module reported consistent (if modest) gains — +1.15 to +2.02pp accuracy depending on model tier — on the **identical dataset** this project uses (FETAL_PLANES_DB, 12,400 images, six-category variant of the same taxonomy). Because it sits only in the low layers before the backbone, it is testable as a clean, single ablation against the already-locked `convnext_tiny` baseline, using the bootstrap-CI protocol the project already trusts (`docs/EXPERIMENTS.md`, `bootstrap_significance_output.txt`).

### Task 5.1 — Implement the adaptive contrast adjustment module

- New module: `src/data/acam.py` (or `src/models/acam.py` if it's implemented as a learnable pre-network layer rather than a pure preprocessing transform — **decide this explicitly based on the actual ACAM paper's design (arXiv:2509.00808), not by assumption**; the survey describes it as "a lightweight, plug-and-play preprocessing sub-network (predicts K clinically-plausible contrast transforms, fuses the resulting multi-contrast views before the backbone)," which implies a small learnable module, not a fixed transform — implement it as such, as a small conv-based module producing K contrast-adjusted views that are fused (e.g., concatenated-then-1x1-conv, or channel-attention-weighted sum — follow the paper's actual fusion mechanism as closely as feasible from the public description; if the paper's exact fusion mechanism cannot be determined with confidence from the abstract/available text, implement the most standard/defensible interpretation and **document the specific implementation choice and its rationale explicitly** in the experiment writeup, exactly as the project already does for other judgment calls like the resolution-policy decision in `EXPERIMENTS.md` §0.3).
- **[MANUAL]** Before implementation, fetch and read the ACAM paper (arXiv:2509.00808) [arXiv:2509.00808](../phases/phase_08/2509.00808_ACAM.md) in enough detail to confirm the module's exact input/output shape contract and fusion mechanism, since the survey summary alone is not sufficient implementation detail. This is a short paper-reading task, not a literature re-survey.
- Wire it as an optional module inserted between `src/data/transforms.py`'s existing preprocessing pipeline and the backbone forward pass, controlled by a new config flag (e.g., `use_acam: true/false` in a new `configs/convnext_tiny_acam.yaml`, cloned from `configs/convnext_tiny.yaml` with only this flag changed — following the exact pattern already used for `configs/convnext_tiny_focal.yaml`).
- **Output:** `src/data/acam.py` (or `src/models/acam.py`), `configs/convnext_tiny_acam.yaml`, brief inline docstring explaining the implementation choice and citing the paper.
- **Acceptance criterion:** Module is unit-testable in isolation (input tensor in, correctly-shaped tensor out, gradient flows through it), and a smoke-test forward pass through the full `convnext_tiny` + ACAM stack completes without shape errors on a single batch.

### Task 5.2 — Train and evaluate the ablation

- Train `convnext_tiny.fb_in22k_ft_in1k` + ACAM from the same initialization, same data splits (`data/splits/train.csv`/`val.csv`), same class-weighted CE loss, and same early-stopping-on-val-macro-F1 protocol already locked for the production model (`04_MODEL_TRAINING.md`), changing **only** the ACAM flag. This is a single training run, not a new backbone sweep.
- Evaluate on the exact same held-out `data/splits/test.csv` (5,271 images, 896 patients) used for every other backbone/ablation comparison in this project, using the existing `src/eval/evaluate_test.py` machinery unmodified.
- Run the same paired bootstrap significance test (2,000 iterations, matching the exact protocol already used in `EXPERIMENTS.md`'s backbone comparison and reproducible from `docs/phases/phase_04/bootstrap_significance_output.txt`) comparing ACAM-augmented `convnext_tiny` against the plain `convnext_tiny` baseline.
- **Output:** `checkpoints/convnext_tiny_acam/best.pt` (new checkpoint, does **not** overwrite `checkpoints/convnext_tiny/best.pt`), `logs/convnext_tiny_acam_training_output.txt` (following the exact naming convention of the existing `logs/convnext_tiny.fb_in22k_ft_in1k_training_output.txt`), `logs/eval/bootstrap_significance_acam.txt`.
- **Acceptance criterion:** Training completes and converges (val macro-F1 stabilizes, no NaN losses); bootstrap CI is computed and reported with a clear verdict (robust win / robust loss / statistically tied), following the exact three-way verdict language already used for the original backbone comparison.

### Task 5.3 — Decide and document, following the project's own negative-result culture

- **If ACAM produces a statistically robust win** (CI entirely above 0, matching the "Robust (CI > 0)" bar already used elsewhere in this project): promote `checkpoints/convnext_tiny_acam/best.pt` as the new production checkpoint. This requires updating every downstream reference — `EVAL_REPORT.md` §1 headline numbers, the model-loader default checkpoint path if hardcoded anywhere, `README.md`, and any UI default-checkpoint reference. **This is a significant, cascading change and must not be done casually** — re-run every evaluation that depends on the "production checkpoint" (in-distribution test metrics, cross-device sweep from Stage 4, NatalIA evaluation from Stage 2, CAM localisation spot-check from Stage 3) against the new checkpoint before finalizing, or explicitly note in the final report which evaluations still reflect the pre-ACAM checkpoint and why.
- **If ACAM is statistically tied or a net loss**: document exactly like the focal-loss ablation was documented in `docs/EXPERIMENTS.md` — a clean, dated, reasoned negative result, checkpoint retained as an artifact for reproducibility but **not** promoted, plain `convnext_tiny` remains the shipped model. This is the *expected*, perfectly acceptable outcome and should not be treated as a failure of the stage — the point of an ablation is to test the idea rigorously, not to guarantee a win.
- Because a "robust win" outcome cascades into re-running Stage 2–4's evaluations, **schedule this stage before Stage 2/4 if at all possible**, or explicitly accept and document that a late-breaking ACAM win would require a follow-up re-evaluation pass rather than retroactively rewriting already-completed stages. State this scheduling risk plainly in the plan (see §8 Sequencing below) rather than discovering it mid-execution.
- **Output:** Updated `docs/EXPERIMENTS.md` (new section, following its existing per-ablation format exactly) with the ACAM result and decision.
- **Acceptance criterion:** A clear, single-sentence, unambiguous decision statement exists ("ACAM promoted to production" or "ACAM retained as documented negative/neutral result, not shipped"), with the supporting bootstrap numbers directly beside it.

---

## 7. Phase 8, Stage 6 — Documentation, UI Consolidation, and Final Report

**Depends on:** Stages 1–5 complete (this stage consolidates their outputs). Some sub-tasks (6.1, 6.2) can start earlier since they concern pre-existing repo inconsistencies unrelated to the new work.

### Task 6.1 — Resolve the dual-UI ambiguity and stale checklist

Observed directly in the repository during planning, not from the literature survey — these are real, current inconsistencies that would confuse anyone evaluating the project fresh, including a clinical audience clicking around the repo:

- `app_streamlit.py`'s own docstring states it is "Primary web interface (Streamlit). Delete the gradio code and file and update demo1_walkthrough.md accordingly. Streamlit is to be our main focus.
- `docs/instructions/09_MASTER_CHECKLIST.md` is stale relative to the actual repo state: it shows most of Phase 6/7 as unchecked `[ ]` boxes despite `EVAL_REPORT.md`, `demo1_walkthrough.md`, and the live code (async Grad-CAM, Streamlit app, 8 synthetic multiplane clips, Tier-2a) all being complete and documented elsewhere. Update the checklist to accurately reflect completed work through the start of Phase 8, and append a new "Phase 8" section listing this plan's stages, so the checklist remains the single accurate at-a-glance status document the project's own README (§6, "the single top-to-bottom list") says it should be.
- **Output:** Updated `demo1_walkthrough.md`, updated `09_MASTER_CHECKLIST.md`
- **Acceptance criterion:** A person reading `00_PROJECT_OVERVIEW.md` → `09_MASTER_CHECKLIST.md` → whichever UI doc is now canonical, in that order, ends up with **one unambiguous answer** to "which web app do I run for the demo" and "what state is this project actually in."

### Task 6.2 — Update `README.md` for external/clinical-audience readability

- `README.md` (repo root, 8KB) should be audited and updated as the actual front door for anyone landing on the repository. **[MANUAL]** Read the current README fully first.
- Ensure it accurately reflects, what the system does, its explicit non-goals/limitations (no clinical validation, research datasets only — matching `00_PROJECT_OVERVIEW.md` §8 verbatim in spirit), how to run the primary demo (once Task 6.1 resolves which app that is), and a short "results at a glance" section (headline macro-F1, cross-device honest gap, NatalIA-derived numbers once Stage 2 completes) linking out to `docs/EVAL_REPORT.md` for full detail rather than duplicating it.
- **Output:** Updated `README.md`.

### Task 6.3 — Final consolidated `EVAL_REPORT.md` update

- Add new sections to `docs/EVAL_REPORT.md` (do not create a competing second eval report — this project already has one canonical evaluation document and Phase 8's job is to extend it, following the same "Phase 6 measures, doesn't fix" discipline the existing document models, now applied as "Phase 8 measures and closes gaps, documented in the same place"):
  - New §2b "NatalIA Non-Expert-Operator, Phantom-Anatomy Generalization" — standard-vs-Other results, per-class results, the honest phantom framing from §0.1 restated here (do not assume a reader of this section also read `docs/NATALIA_INTEGRATION.md`), and the trivial-baseline comparison from Task 2.1.
  - Updated §5.1 "Video-Transition Latency" — append the NatalIA transition-latency finding from Task 2.2 (either the honest "insufficient real transitions found" statement, or the real measured numbers, clearly delineated from the existing synthetic-clip numbers as specified in Task 2.2).
  - Updated §5.2 — note that the "no cross-device Other-class validation" limitation is now **partially closed** by NatalIA (§2b), while the "3 of 7 anatomical planes" cross-device coverage limitation (HC18/UCL's own scope) is **unchanged and still accurately stated** — do not conflate the two different generalization axes.
  - New §4.x "ACAM Preprocessing Ablation" consolidating Task 5.3's result, in the same format as the existing §4.5 focal-loss ablation entry.
  - Updated §4.2 "Backbone Comparison, Cross-Device" — replace the single-backbone table with the full 6-backbone table from Stage 4, explicitly note this closes the limitation the current §5.5 already flags.
  - New §X "Weakly-Supervised CAM Localisation" — summarizing Stage 3's spot-check findings (Task 3.3), explicitly framed as a qualitative/approximate feature addition, not a new quantitative benchmark, consistent with the honesty requirement in Task 3.2.
  - Updated §5 "Limitations" — remove/qualify the limitations Phase 8 has genuinely closed, and **add** the explicitly-deferred items from §1 of this plan (brain-subplane confusion, quality/clinical-acceptability layer, foundation-model re-ablation) as forward-looking "known, deliberately deferred" items, so the report continues to read as complete and honest rather than silently dropping topics the survey raised.
- **Output:** Updated `docs/EVAL_REPORT.md`.
- **Acceptance criterion:** Every new/updated section cross-references its source log file(s) under `logs/`, exactly matching the citation style already used throughout the existing document (e.g., "**Full log:** `logs/eval/...`").

### Task 6.4 — Light UI polish pass (only if Task 6.1 surfaces genuine open items)

- This is intentionally the smallest, most bounded task in the plan. It exists only to close out any genuinely still-open, small items surfaced by Task 6.1's review of `fetscan_uiux_fix_brief.md`, plus wiring in the CAM-box toggle (Task 3.2) and NatalIA showcase clips (Task 2.3) into whichever UI is now canonical.
- **Explicitly out of scope here:** any new UI features, redesigns, or styling work not already identified as open in the existing brief. This is consolidation, not a new design pass — new UI ideas belong in a future phase.
- **Output:** Whatever small changes Task 6.1's review determines are genuinely still open and small.
- **Acceptance criterion:** No open items remain flagged from `fetscan_uiux_fix_brief.md` without either being implemented or explicitly deferred with a stated reason.

### Task 6.5 — Full documentation and logs audit

- Walk every new file this phase created (`docs/NATALIA_INTEGRATION.md`, `docs/CAM_LOCALISATION_SPOTCHECK.md`, `docs/CROSS_DEVICE_BACKBONE_COMPARISON.md`, updated `docs/EVAL_REPORT.md`, updated `docs/EXPERIMENTS.md`) and confirm every meaningful decision, deviation, and finding referenced in this plan actually has a corresponding `.md` entry, per the task's own documentation requirement.
- Walk every `logs/` file this phase should have produced (see the per-stage "Output" lists above) and confirm each exists, is non-empty, and is named descriptively and consistently with the project's existing `logs/` naming conventions (`logs/eval/`, `logs/training/`, `logs/tuning/`, `logs/backbones/`, `logs/smoketest/`, `logs/realtime/` — place new logs in the matching existing subdirectory rather than inventing new top-level categories where an existing one fits).
- **[MANUAL]** Do one final read-through pass of `docs/EVAL_REPORT.md` end-to-end (not just the new sections) to confirm the whole document still reads coherently as a single narrative, not as a patchwork of old and new sections with inconsistent tone or contradictory statements.
- **Output:** No new files — this is a verification pass. Any gaps found should be fixed by returning to the relevant earlier stage/task, not patched over here.
- **Acceptance criterion:** Zero missing log files against the per-stage output lists in this plan; zero undocumented decisions; the final human read-through in this task raises no unresolved coherence issues.

---

## 8. Sequencing and Dependencies

```
Stage 1 (NatalIA acquisition/manifest)
   │
   ├──► Stage 2 (NatalIA evaluation)
   │
Stage 3 (CAM localisation)        ─── independent, can run in parallel with 1/2
   │
Stage 4 (6-backbone cross-device) ─── independent, can run in parallel with 1/2/3
   │
Stage 5 (ACAM ablation)           ─── independent, but SEE SCHEDULING NOTE below
   │
   ▼
Stage 6 (docs/UI consolidation + final report)  ─── depends on 1–5 all complete
```

**Scheduling note on Stage 5:** if ACAM produces a "robust win" (Task 5.3), the production checkpoint changes, which would ideally invalidate and require re-running the checkpoint-dependent parts of Stages 2–4 (NatalIA evaluation, cross-device sweep) against the new checkpoint. Two acceptable ways to handle this, either is fine, but the choice should be made deliberately and stated in the resulting docs rather than discovered as an inconsistency after the fact:

- **Option A (recommended):** Run Stage 5 *first*, before Stages 2 and 4, so that if ACAM wins, all subsequent stages evaluate against the correct final checkpoint from the start.
- **Option B:** Run stages in the order presented above, and if ACAM wins, explicitly re-run the checkpoint-dependent portions of Stages 2 and 4 as a follow-up pass before Stage 6, documenting in `EVAL_REPORT.md` which specific numbers reflect the pre- vs. post-ACAM checkpoint and why.

Whichever option is chosen, Stage 6 (Task 6.5) must confirm no numbers in the final `EVAL_REPORT.md` accidentally mix pre-ACAM and post-ACAM checkpoint results without explicit labeling.

---

## 9. Final Deliverable — Phase 8 Acceptance Test

Phase 8 is complete when, and only when, **all** of the following are independently verifiable by a person who was not involved in executing the plan:

1. **NatalIA integration verified.** `data/processed/natalia_manifest.csv` exists with exactly 19,212 rows (per Task 1.3's arithmetic); `docs/NATALIA_INTEGRATION.md` exists and correctly states the phantom-not-real-patient framing; `logs/eval/evaluate_natalia.txt` exists with standard-vs-Other precision/recall numbers and the trivial-baseline comparison both present.
2. **Transition-latency investigation resolved one way or the other.** `logs/natalia_transition_investigation.txt` exists and states plainly whether real usable transitions were found, and if so, `logs/eval/evaluate_transition_latency_natalia.txt` exists with a latency figure clearly labeled as distinct from the existing synthetic-clip figure.
3. **CAM localisation feature is live and honestly framed.** Running either canonical demo UI (per Task 6.1's resolution) on any sample clip shows a dashed, clearly-labeled "approx. region" box that tracks the anatomy across frames, toggleable off; `docs/CAM_LOCALISATION_SPOTCHECK.md` exists with a completed 20-image tally.
4. **Cross-device sweep complete for all 6 backbones.** `docs/CROSS_DEVICE_BACKBONE_COMPARISON.md` exists with all 6 backbones represented and both of Task 4.2's explicit questions answered with a stated verdict.
5. **ACAM ablation resolved with a clear decision.** `docs/EXPERIMENTS.md` contains a new dated section with an unambiguous promote/retain-as-negative-result decision, backed by the bootstrap CI numbers.
6. **Documentation is internally consistent.** Reading `00_PROJECT_OVERVIEW.md` → `09_MASTER_CHECKLIST.md` → the canonical UI walkthrough doc → `README.md`, in that order, produces no contradictions about project status, which UI is primary, or what the shipped checkpoint is.
7. **`docs/EVAL_REPORT.md` is the single, current, coherent source of truth** for all evaluation numbers in the project, old and new, each citing its underlying `logs/` file, with the Limitations section accurately reflecting what Phase 8 closed and what remains explicitly, honestly deferred.
8. **End-to-end live demo works.** A fresh clone of the repository, following only the instructions in the (now-updated) `README.md`, can: launch the canonical web UI, upload or select an example clip (including at least one NatalIA showcase clip, correctly captioned as phantom footage), and produce an annotated output video showing stable plane labels, confidence scores, a Grad-CAM overlay, and the new CAM-derived approximate-region box — without errors, crashes, or any UI element implying clinical-grade measurement precision it does not have.
9. **Nothing silently regressed.** All pre-existing evaluation artifacts (the original `convnext_tiny` in-distribution results, the original HC18/UCL cross-device numbers, the original synthetic-clip transition-latency numbers) remain present, unmodified, and correctly attributed in the repository — Phase 8 has only added and clearly labeled new evaluations alongside them, per the task's explicit "do not overwrite historical evaluation artifacts" requirement.

**When all nine conditions above hold, Phase 8 is done, and the project is in a state that can be handed to a hospital or clinical department for a live, honestly-framed technical demonstration.** Any further work — the brain-subplane intervention, the quality/clinical-acceptability layer, foundation-model re-ablation, or anything else raised by the literature survey but deferred in §1 — is explicitly out of scope until and unless the project receives clinical interest warranting a further research phase.
