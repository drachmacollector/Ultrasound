# Implementation Plan — Repository Pivot for Claude Code

**Read first:** `01_RESEARCH_PROPOSAL.md`. This document turns it into repository changes.
**Repo:** `drachmacollector/Ultrasound` @ `b854a34` · dev machine: Windows + conda env `fetalplane`, RTX 4060 8 GB · run everything from repo root.
**Audience:** Claude Code (implementer), then an independent human/Claude audit after each phase (the existing working method; keep it).
**v1.1 scope:** public datasets only, with no ethics-committee or hospital work. Human effort is limited to binary y/n image checks. Near-term publication is a JAPI review article (§8.1). Future extensions are listed in §9 and must not be implemented.

---

## 0. Ground rules for the implementer (non-negotiable)

1. **Never select, tune, or early-stop on any test set** (`data/splits/test.csv`, HC18, UCL, African). Validation = `data/splits/val.csv` only. Add a hard assertion in every training entry point that the train/val CSV paths never contain `test`, `cross_device`, `african`, `landmark`.
2. **Every evaluation writes a per-image prediction file** (`results/preds/<run_id>/<dataset>.parquet`) containing: `image_id, patient_id, dataset, source_machine, true_label, true_label_collapsed, prob_0..prob_7, pred_full, pred_restricted, correct_full, correct_restricted` (+ saliency stats when computed). All tables/figures are generated **only** from these files, never from inline inference.
3. **Every reported number has a 95% patient-clustered bootstrap CI** (2,000 replicates, seed 20261109). Model comparisons use the **paired** clustered bootstrap (same resampled patients for both models).
4. **No new metric without a unit test** that uses synthetic inputs with a hand-computed answer.
5. **Do not delete history.** Legacy code is moved to `archive/` after tagging the commit (§2). Nothing is silently overwritten; each experiment writes to a new run directory.
6. **Paths:** use `pathlib` everywhere; existing CSVs contain Windows backslash paths (`data\raw\...`) — normalise on load with `Path(p.replace("\\", "/"))`.
7. **Config is always read from the checkpoint** (existing rule; keep).
8. **No clinical claims anywhere** in code comments, docs or figures. NatalIA is phantom; never call it patient data.
9. Write a short entry in `docs/research/RESEARCH_LOG.md` at the end of each task: what was run, run_id, commit hash, outcome, deviations.

---

## 1. Component audit — verdict for every part of the repo

Legend: **RETAIN** (keep as is) · **REVISE** (keep, modify) · **ARCHIVE** (move to `archive/`, not maintained) · **REMOVE** (delete after tag; no research value) · **NEW** (create).

### 1.1 Source code (`src/`)

| Path | Verdict | Reason / action |
|---|---|---|
| `src/data/dataset.py` | **REVISE** | Keep `CANONICAL_CLASSES`, `FocalPlanesDataset`. Generalise `CrossDeviceDataset` → `ExternalDataset` (any manifest with `dataset, patient_id, plane_label_collapsed`); return `image_id` and `patient_id`; add optional preprocessing hook (sector mask/letterbox/inpaint). Remove the multitask bbox branch and `multitask_collate_fn`. |
| `src/data/transforms.py` | **REVISE** | Keep grayscale→RGB loader. Add `letterbox` resize option (aspect-preserving) and a `preprocess_mode ∈ {resize, sector_letterbox}` switch; add A4 device-style augmentations behind a config flag. Default must reproduce current behaviour exactly (baseline B0). Remove `prep_frame_grayscale_to_rgb` (video only) → archive. |
| `src/data/class_weights.py` | RETAIN | Used by training. |
| `src/data/synthetic_video.py` | **ARCHIVE** | Synthetic clips have no research role. |
| `src/models/backbone.py` | RETAIN | Add `return_features` helper (penultimate embeddings via `model.forward_features` + pooling) for E3 probes. |
| `src/models/gradcam.py` | **REVISE** | Keep layer map. Add Grad-CAM++ and a gradient-free method (occlusion or RISE) behind one interface `compute_saliency(model, x, target, method)` returning HxW float maps in model-input space. Batch support. |
| `src/models/cam_localisation.py` | RETAIN + extend | Keep `cam_to_bbox`; its outputs feed the IoU metric. |
| `src/models/acam.py` | **ARCHIVE** | Neutral ablation, not part of new question. Make `train.py` import it lazily only if `use_acam: true` (or remove the option). |
| `src/models/multitask_model.py` | **REMOVE** | 1-epoch smoke test; trained on HC18/UCL (contaminates external sets). |
| `src/train/train.py` | **REVISE** | See §5 (R5): sampler weights (A3), preprocessing mode, caliper augmentation, seed from CLI, run_id directories, test-path assertion, save val per-image preds. Lazy ACAM import. |
| `src/train/losses.py` | RETAIN (focal unused) | Keep CE path; focal can stay (tested negative). |
| `src/eval/evaluate_test.py` | **REVISE → replaced by `src/eval/predict.py`** | New generic predictor writing parquet for any (checkpoint, dataset, preprocess) triple. Keep old file in archive. |
| `src/eval/metrics_utils.py` | **REVISE** | Keep confusion-matrix plotting; move metrics to new `src/eval/stats.py`. |
| `src/realtime/model_loader.py` | **MOVE → `src/models/loading.py`** | It is imported by `scripts/evaluate_cross_device.py` and others; move before archiving `src/realtime/`. Drop multitask branch. |
| `src/realtime/{app,pipeline,capture,queues}.py` | **ARCHIVE** | Real-time serving is out of scope. |
| `src/smoothing/` | **ARCHIVE** | Temporal smoothing out of scope. |
| `app_streamlit.py`, `assets/style.css`, `assets/fetscan_uiux_fix_brief.md` | **ARCHIVE** | UI out of scope. |

### 1.2 Scripts (`scripts/`)

| Script | Verdict | Action |
|---|---|---|
| `build_manifest.py`, `patient_split.py`, `run_sanity_checks.py` | **RETAIN** | Foundations. Do **not** regenerate splits (would invalidate comparisons); add a checksum test that `data/splits/*.csv` are unchanged (record SHA-256 in `data/splits/SPLITS_SHA256.txt`). |
| `build_cross_device_manifest.py` | **REVISE** | Add `patient_id` (HC18/UCL `SubjectID`; UCL Femur uses `Patient`), `dataset`, `algo` (device field) columns; output `data/processed/external/hc18_ucl_manifest.csv`. Keep FP/MULTICENTRE exclusion for **external testing**. |
| `derive_bboxes.py` | **REVISE → `src/localisation/anatomy_masks.py`** | Replace boxes-only logic with ellipse/capsule masks (§4). |
| `evaluate_cross_device.py` | **REVISE → thin wrapper over `src/eval/predict.py` + `src/eval/external.py`** | Must save probabilities, patient_id, restricted-argmax. Keep `_assert_not_multitask_contaminated` (generalise to "training manifest must not contain any external dataset"). |
| `evaluate_test_acam.py` | **REVISE (generalise)** | Its paired bootstrap becomes `src/eval/stats.py::paired_cluster_bootstrap`; then archive the script. |
| `generate_cam_spotcheck.py` | **ARCHIVE** | Superseded by quantitative localisation (R4). |
| `lr_finder.py`, `smoke_test.py` | RETAIN | Utility. |
| `build_natalia_manifest.py`, `evaluate_natalia.py`, `verify_natalia_structural.py`, `check_natalia_transitions.py` | **ARCHIVE** (keep manifest builder usable) | Supplementary stress test only; not in primary analyses. |
| `build_natalia_showcase_clip.py`, `benchmark_pipeline.py`, `validate_pipeline_headless.py`, `render_annotated_video.py`, `save_hud_screenshot.py`, `generate_multiplane_clips.py`, `generate_sample_clips.py`, `evaluate_realistic_video.py`, `evaluate_transition_latency.py`, `evaluate_tier2.py`, `tune_tier1_smoothing.py`, `tune_tier2_mode_filter.py` | **ARCHIVE** | Video/real-time/smoothing. |
| `build_multitask_manifest.py`, `train_multitask.py`, `verify_no_multitask_leakage.py` | **REMOVE** (after tag) | Multitask line abandoned. Port the *idea* of the leakage guard into §0 rule 1. |
| `scripts/tests/test_backbone.py`, `test_transforms_parity.py`, `test_cam_localisation.py` | RETAIN → move to `tests/` | Update imports. Parity test must now cover both preprocess modes. |
| `scripts/tests/test_tier1.py`, `test_tier2_mode_filter.py`, `test_acam.py` | ARCHIVE with their modules | |

### 1.3 Configs, checkpoints, data artefacts, logs, docs

| Item | Verdict | Action |
|---|---|---|
| `configs/{convnext_tiny,tf_efficientnetv2_s,...}.yaml` | RETAIN | Baseline (B0) configs. Add `preprocess_mode: resize`, `sampler: none`, `caliper_aug_p: 0.0` explicitly. |
| `configs/{convnext_tiny_acam,convnext_tiny_focal,multitask,smoothing_*}.yaml` | ARCHIVE / REMOVE(multitask) | |
| NEW `configs/research/*.yaml` | NEW | One per arm×backbone: `b0_`, `b1_`, `b2_`, `afp_` × `convnext_tiny`, `tf_efficientnetv2_s` (§5.4). |
| `checkpoints/<6 backbones>/best.pt` (local, not in git) | **RETAIN, frozen** | Used immediately in R3/R4 as "existing models" (seed 42). Never overwrite; new runs go to `runs/`. |
| `checkpoints/convnext_tiny_acam`, `_focal` | Keep on disk, not used | |
| `checkpoints/multitask` | REMOVE | |
| `data/processed/synthetic_clips/`, `tier1_tuning/`, `manual_review/`, `eval_realistic_video/` | ARCHIVE | |
| `data/processed/eval_cross_device/*.csv` | RETAIN as historical; will be superseded by parquet preds | |
| `logs/realtime/`, `logs/tuning/`, `logs/eval/*tier2*`, `*transition*`, `*realistic_video*` | ARCHIVE | |
| `docs/instructions/00–09`, `docs/phases/`, `docs/FetScan_Final_Audit_Report.md`, `docs/NATALIA_INTEGRATION.md`, `docs/CAM_LOCALISATION_SPOTCHECK.md` | ARCHIVE → `archive/docs/` | Historical record; cite in methods history. |
| `docs/EVAL_REPORT.md`, `docs/EXPERIMENTS.md`, `docs/CROSS_DEVICE_BACKBONE_COMPARISON.md` | ARCHIVE + add header "Superseded by docs/research/RESULTS.md; numbers are image-level-bootstrap, single-seed" | |
| `docs/literature_survey_and_gap_analysis.md` | RETAIN (background) | |
| `README.md`, `CLAUDE.md` | **REWRITE** | §8 gives new CLAUDE.md. README: research-study description, no demo/clinical language, remove `file:///d:/` links. |
| `project_structure_tree.txt` | Regenerate at end of R0 | |

### 1.4 Status of existing results

| Result | Status in new study |
|---|---|
| In-dist macro-F1 0.8927 (convnext_tiny) | **Re-compute** under new eval core (clustered CI, per-machine strata). Becomes baseline B0-seed42. Gate G2 = reproduce within ±0.005. |
| 6-backbone in-dist & cross-device tables | **Re-compute** with clustered CIs + restricted-argmax → supplementary table. "repvgg_a1 winner" claim withdrawn unless CI supports it. |
| Image-level bootstrap verdicts (tied/robust) | **Superseded** (CIs too narrow). |
| Head→Other dominance under shift (180/194) | **Retained as preliminary motivation**; will be re-derived from parquet with CIs. |
| Focal / ACAM negatives | Retained as historical negatives; not re-run. |
| NatalIA 4.6% recall | Supplementary footnote only. |
| Smoothing, flicker, latency, fps, synthetic-clip accuracy | **Retired** — not relevant, and latency headline was already flagged as poorly evidenced. |
| Grad-CAM qualitative spot-check | **Retired** — replaced by R4 quantitative evaluation. |

---

## 2. Phase R0 — Repository restructuring (½–1 day)

**Tasks**
1. `git tag legacy-fetscan-demo-v1 && git push --tags`. Create branch `research/anatomy-or-acquisition`.
2. Create `archive/` and `git mv` every ARCHIVE item there, preserving relative paths (`archive/src/realtime/...`, `archive/scripts/...`, `archive/docs/...`, `archive/app_streamlit.py`). Add `archive/README.md`: "Frozen legacy FetScan demo (tag legacy-fetscan-demo-v1). Not maintained; may not run."
3. `git rm` REMOVE items (multitask model/scripts/configs).
4. Move `src/realtime/model_loader.py` → `src/models/loading.py`; drop multitask branch; update all imports (`grep -rn "src.realtime"` must return nothing outside `archive/`).
5. In `src/train/train.py`, make ACAM import lazy (inside `if use_acam:`) or remove the option; ensure `python src/train/train.py --config configs/convnext_tiny.yaml --epochs-override 1` still runs.
6. Move retained tests to `tests/`; add `pytest.ini` (`testpaths = tests`). `python -m pytest` must pass.
7. Create skeleton dirs: `src/localisation/`, `src/audit/`, `src/preprocess/`, `docs/research/`, `results/` (gitignored except `results/tables/`, `results/figures/`), `runs/` (gitignored).
8. Add to `requirements.txt`: `pyarrow`, `statsmodels`, `scikit-image`, `lightgbm` (or use sklearn `HistGradientBoostingClassifier` — prefer this, no new dep), `easyocr` (optional, only if overlay text detection needs it). Do not change torch pins.
9. Write `data/splits/SPLITS_SHA256.txt` and `tests/test_splits_frozen.py`.
10. Replace `CLAUDE.md` with §8 text; rewrite `README.md`.

**Acceptance:** tests pass; `grep -rn "streamlit\|smoothing\|realtime\|multitask" src scripts tests` returns nothing; baseline training smoke run works; tag exists.

---

## 3. Phase R1 — Data foundation (Gate G1)

### 3.1 Acquisition metadata manifest — `scripts/build_acquisition_metadata.py` (NEW)
Input: `data/splits/{train,val,test}.csv`. For each image compute and save to `data/processed/acquisition_metadata.parquet`:
`image_id (filename stem), patient_id, split, plane_label, source_machine, operator, height, width, aspect, mean_intensity, std_intensity, frac_nonzero, sector_area_frac, sector_bbox (x0,y0,x1,y1), sector_shape_class` (sector from §3.4 segmenter), `md5`.
Also emit `results/tables/E1_confounding_tables.md`: machine×class, operator×class, split×machine, (machine,operator)×Other — counts, row %, Cramér's V with bootstrap CI. Expected (already verified): train+val Other share Aloka 0.651 / E6 0.136 / S10 0.000(n=11); test S10 = 1,071 images, 51.6% Other.

### 3.2 External manifests
- `scripts/build_external_manifests.py` (NEW; supersedes cross-device builder): writes `data/processed/external/{hc18,ucl,african}.csv` with columns `image_path, image_id, patient_id, dataset, device, plane_label_collapsed ∈ {Head,Abdomen,Femur,Thorax}`.
- **African set:** user downloads `Zenodo_dataset.tar.xz` from https://zenodo.org/records/7540448 into `data/raw/african_planes/`. Parse its CSV (filename→plane, patient, country). Map `brain→Head, abdomen→Abdomen, femur→Femur, thorax→Thorax`. Record device per country from Sendra-Balcells 2023 (Malawi Mindray DC-N2; Egypt Voluson P8; Uganda Siemens Acuson X600; Ghana EDAN DUS-60; Algeria Voluson S8) **only if** the CSV gives country; otherwise `device=unknown`. Print counts; stop and report if total ≠ ~450 or patients ≠ ~100.
- Add `data/raw/african_planes/README.md` (licence CC BY 4.0, citation).

### 3.3 Landmark ground truth linkage — `scripts/link_fp_landmarks.py` (NEW)
Goal: map landmark-benchmark FP annotations onto **original FPDB test images**.
1. Load `data/raw/ucl_hc18/annotations/FP/{Head,Abdomen,Femur}.csv` (all rows, ignore their `Split`).
2. Join to FPDB by filename stem (`Patient00168_Plane2_1_of_2`). Report: matched / unmatched counts; how many matched images are in our `test.csv` (only these are used).
3. **Registration check** (critical — the benchmark paper says FP images were cropped to remove overlays): for each matched pair, load both images; if shapes equal and pixels equal → offset (0,0), scale 1. Else run `cv2.matchTemplate` (TM_CCOEFF_NORMED) of the FP image inside the FPDB image at scales {1.0, plus a coarse search 0.5–1.5 if needed}; accept if peak NCC ≥ 0.95; store `dx, dy, scale, ncc`. Transform landmark coordinates into FPDB pixel space. Images failing registration → excluded, counted, listed.
4. Output `data/processed/landmarks/fpdb_test_landmarks.parquet` (`image_id, anatomy, landmark columns in FPDB space, px_to_mm, algo, reg_ok, ncc`), and the same for HC18/UCL (native space) `external_landmarks.parquet`.
5. **Pairing discovery for E5d:** within FPDB, group images by `(patient_id, PlaneN)` from the filename pattern `Patient{ID}_Plane{N}_{M}_of_{K}`. For groups with ≥2 images, compute overlay-detector caliper score (§3.4) and SSIM after sector masking. Candidate caliper pairs = SSIM ≥ 0.9 and caliper score differs (one with, one without). Write `data/processed/audit/caliper_pairs.csv`. **Manual confirmation** of up to 50 random candidates via the §3.5 tool in pair mode. This is a single y/n per pair.

### 3.4 Sector segmentation + overlay detection — `src/preprocess/sector.py`, `src/preprocess/overlays.py` (NEW)
- `segment_sector(img_gray) -> mask (uint8 0/1)`: Otsu/low threshold → morphological close (k≈15) → largest connected component → convex hull → fill. Must handle linear (rectangular) and curvilinear (fan) sectors. Return also bbox.
- `detect_overlays(img_gray, sector_mask) -> dict(masks: {caliper, text, other_marks}, scores)`:
  - Out-of-sector: everything bright outside sector with connected-component area < 2% image = text/UI candidate; optional EasyOCR boxes.
  - In-sector calipers/markers: white top-hat (k=7) → threshold at high percentile (≥99.5) → keep thin, high-contrast structures; template-match '+' and '×' crosses (sizes 7–25 px) and dotted-line runs (≥4 collinear dots). Thresholds set on the **train** split audit images only.
  - `caliper_score` = number of detected cross/dotted components.
- `inpaint_overlays(img, overlay_mask)`: `cv2.inpaint(..., 3, cv2.INPAINT_TELEA)` on dilated mask.
- Tests: synthetic image with a drawn fan + 3 drawn '+' marks + text → detector must find all, inpaint must reduce mark-pixel intensity to within 10 grey levels of local background.

### 3.5 Manual overlay audit tool — `scripts/overlay_audit_tool.py` (NEW) — **binary checks only**
A keyboard-driven OpenCV window (no web UI). It shows **one raw image at a time**, full size, with no detector output drawn, so the human is not biased by the detector. Exactly two questions per image:
- `c` / `x`: **calipers or measurement marks present?** (yes / no)
- `t` / `n`: **any on-screen text or annotation present (anywhere)?** (yes / no)
- Optional third key after those two: the tool briefly overlays the auto sector outline and asks `s` / `f` "is the outlined sector roughly right?" (yes / no). It can be switched off with `--no-sector-check`.
- `b` goes back one image, `q` saves and quits. The tool must be resumable: progress is saved after every image.

**Samples:** 300 FPDB images stratified by machine × class (150 from **train** for detector threshold tuning, 150 from **test** for reporting). Save `data/processed/audit/overlay_audit_{train,test}.csv` with columns `image_id, calipers, text, sector_ok, timestamp`.

**Caliper-pair mode** (`--pairs data/processed/audit/caliper_pairs.csv`): shows two images side by side and asks one question, `y` / `n`: "same scan, one with calipers and one without?". Maximum 50 pairs.

**Landmark glance mode** (`--landmarks`): shows 15 images with the linked landmarks drawn as dots and asks one question, `y` / `n`: "do the dots sit on the head / abdomen / femur?".

**Explicitly not part of any tool:** drawing, boxing, outlining, measuring or timestamping. Expected total human time is about 45–60 minutes.

### Gate G1 (written decision in `docs/research/GATE_G1.md`)
Report: overlay prevalence by class × machine (from manual audit, with Wilson CIs); detector precision/recall vs manual labels; number of verified caliper pairs; FP→FPDB registration success rate; African counts.
**Decision rules:** if caliper prevalence differs between standard planes and Other by ≥10 pp → overlays are a primary axis (E4b/E5a/E5d/A2 full weight). If <10 pp → overlays secondary; emphasise device/operator/prevalence (E2/E3/E6/A3). Detector precision/recall is computed at **image level** against the binary labels. If caliper recall < 0.8 on the train audit images, tune the thresholds once more (train audit only). If it is still below 0.8, restrict E5a to test images where detector and human agree that calipers are present, and report this restriction. Never fall back to manual masks. If there are fewer than 20 confirmed caliper pairs, drop E5d.

---

## 4. Phase R2 — Evaluation core (Gate G2)

### 4.1 `src/eval/predict.py` (NEW)
`predict(checkpoint, manifest, dataset_name, preprocess_mode, out_path, batch_size=64, save_features=False)` → writes the parquet in §0 rule 2. `pred_restricted` = argmax over classes present in that dataset's collapsed label set (for FPDB test = all 8). Collapsing: brain sub-planes→Head; `Fetal_thorax→Thorax`, etc. Optional `features.npy` (penultimate pooled, float16) for E3.
CLI: `python -m src.eval.predict --ckpt ... --dataset fpdb_test|hc18|ucl|african [--preprocess resize|sector_letterbox|...] --run-id ...`.

### 4.2 `src/eval/stats.py` (NEW)
- `cluster_bootstrap(df, metric_fn, cluster_col="patient_id", n=2000, seed=...) -> (point, lo, hi)`; resample clusters with replacement, concatenate rows.
- `paired_cluster_bootstrap(df_a, df_b, metric_fn, ...)` (align on image_id; same cluster draws) → Δ point, CI, two-sided bootstrap p.
- Metrics: `macro_f1`, `balanced_accuracy`, `accuracy`, `per_class_recall`, `other_recall`, `rejection_rate` (share of external images predicted Other or absent class), `ece(probs, labels, n_bins=15)`, `auroc`, `risk_coverage_curve`.
- `multi_seed_summary(list_of_pred_dfs)` → mean over seeds, CI by bootstrapping clusters and averaging seed metrics per replicate.
- Tests: hand-computed toy examples, including a case where image-level and cluster-level CIs differ.

### 4.3 `src/eval/external.py` (NEW)
Full-argmax and restricted-argmax scoring, per-dataset and pooled; rejection vs confusion decomposition; per-device strata.

### 4.4 Re-evaluate existing frozen checkpoints — `scripts/research/r2_reevaluate_existing.py` (NEW)
For all 6 `checkpoints/<backbone>/best.pt`: predict on fpdb_test, hc18, ucl, african (preprocess = resize, i.e., exactly as trained). Produce `results/tables/R2_existing_models.md`: in-dist macro-F1 [CI], per-machine macro-F1 and Other-recall (Aloka/E6/S10/Other), external full & restricted balanced accuracy [CI], rejection rate.
**Gate G2:** convnext_tiny in-dist macro-F1 = 0.8927 ± 0.005; HC18+UCL full-argmax accuracy = 83.6% ± 0.3 pp. If not reproduced → stop and diagnose (preprocessing parity, label mapping) before anything else.

---

## 5. Phase R3 — Confounding & reliance audit (existing checkpoints)

All scripts under `scripts/research/`, all outputs to `results/tables/` + `results/figures/`, all with CIs.

| Script | Spec |
|---|---|
| `e2_metadata_only_model.py` | `HistGradientBoostingClassifier` on {machine, operator (one-hot), height, width, aspect, mean/std intensity, sector_area_frac}; fit on train+val, eval on test; tasks: 8-class (balanced acc) and Other-vs-standard (AUROC). Report all-test and excluding S10 (expected all-test AUROC ≈0.54 using machine+operator prior only; excl-S10 ≈0.735 — re-derive). Also feature-ablated variants (machine+operator only; geometry only). |
| `e3_feature_probes.py` | From `features.npy` of each backbone on **test**: logistic-regression probe (L2, C by inner CV) predicting machine (4-way) and operator (4-way), `StratifiedGroupKFold(5)` grouped by patient. Compare with ImageNet-pretrained (not fine-tuned) features of the same architecture. Report balanced acc [CI] and chance. |
| `e4_availability_train.py` | Train convnext_tiny (3 seeds, same recipe as B0, val for early stopping) on transformed inputs: **(a) non-anatomical-only** = sector region filled with per-image mean grey (outside-sector + any overlays kept); **(b) overlay-only** = black canvas + detected overlay pixels (in- and out-of-sector). Eval on equally-transformed test. Report balanced accuracy vs chance 0.125 and vs full-image model. Note limitation: sector shape itself leaks plane-independent device info — that is part of what we measure. |
| `e5_reliance_perturbation.py` | Baseline models, unchanged weights, on test images: (a) overlays inpainted; (b) out-of-sector set to 0; (c) **area-matched control**: inpaint random in-sector patches whose total area equals that image's overlay mask area (5 random draws, averaged); (d) caliper pairs (if G1 passed): compare predictions within pair. Metrics: flip rate, mean Δ p(true), Δ macro-F1, paired CIs; primary contrast (a) vs (c). Stratify by class and machine. |
| `e6_stratified_performance.py` | From R2 parquet: per-machine confusion matrices and Other-recall; **S10 test** analysis (P1c): compare Other-recall S10 vs E6 vs Aloka with clustered CIs; also per overlay-presence stratum (detector). |
| `e7_saliency_out_of_sector.py` *(bridges to R4)* | For all test images (all 8 classes): out-of-sector and overlay saliency fractions per backbone. |

**Acceptance for R3:** `results/tables/R3_*.md` exist with CIs; `docs/research/RESULTS.md` section "RQ1–RQ2" drafted with numbers pulled programmatically (a script `scripts/research/render_results.py` fills placeholders from parquet/JSON — no hand-typed numbers).

---

## 6. Phase R4 — Anatomy-grounded localisation (Gate G3)

### 6.1 `src/localisation/anatomy_masks.py` (NEW; replaces `derive_bboxes.py`)
- `head_ellipse(ofd1, ofd2, bpd1, bpd2, shape)`: center = midpoint(ofd1, ofd2); a = |ofd|/2; b = |bpd|/2; angle = atan2(ofd vector); rasterise with `cv2.ellipse`. Sanity: bpd midpoint within 0.15·a of center else flag.
- `abdomen_ellipse(tad1, tad2, apad1, apad2, shape)`: same construction (TAD as major if longer).
- `femur_capsule(fl1, fl2, shape, rel_width=0.10)`: thick line `cv2.line(thickness=max(3, rel_width·|FL|))` + round caps. Sensitivity at 0.05/0.15.
- `landmark_box(...)`: tight box with 10% margin (for IoU vs `cam_to_bbox`).
- `to_model_space(mask, preprocess_mode, img_size)`: apply **the identical geometric transform** used for model input (resize or sector-crop+letterbox) so masks align with CAMs. Unit test: transform a mask and an image with a drawn ellipse — overlap must be ≥0.98 Dice.
- Note in docstring: the column-prefix typo reported in the benchmark paper (BPD under `ofd_` prefix) — **verify column semantics automatically**: the BPD segment must be shorter than the OFD segment and roughly perpendicular to it (angle 70–110°) on ≥95% of head rows. Also check with the 15-image y/n landmark glance (§3.5) before use. Record in `GATE_G3.md`.

### 6.2 `src/localisation/metrics.py` (NEW)
`energy_pointing_game(sal, mask)` = Σ sal⁺·mask / Σ sal⁺; `pointing_hit(sal, mask)`; `box_iou(box_a, box_b)`; `fraction_in(sal, region_mask)` (used for out-of-sector and overlay fractions); all with unit tests on synthetic maps (e.g., sal = mask → EPG = 1; uniform sal → EPG = area fraction).
Always report EPG **relative to the area-fraction baseline** (EPG − area_frac, and EPG/area_frac) because a large ellipse trivially captures saliency.

### 6.3 `scripts/research/r4_localisation.py` (NEW)
For each backbone (6 existing + later R5 models) × saliency method {gradcam, gradcampp, occlusion}: on FPDB-test images with registered landmarks (Head/Abdomen/Femur), HC18, UCL: compute saliency for (i) the **true class** target and (ii) the **predicted class** target (collapse brain sub-planes: use the argmax sub-plane among the three for Head). Save per-image stats into the parquet (`epg_true, hit_true, iou_true, epg_pred, ..., out_sector_frac, overlay_frac`).
Batch on GPU; occlusion: 16×16 patch, stride 8, on a 600-image stratified subset only (cost).

### 6.4 Sanity check — `scripts/research/r4_sanity_randomisation.py` (NEW)
Cascading randomisation of the classifier head then last stage (Adebayo 2018) for convnext_tiny; compute SSIM / Spearman between original and randomised maps on 200 images. **Gate G3:** a method passes if median SSIM < 0.5 after head+last-stage randomisation. Failing methods are reported and excluded from P3 analyses. Write `docs/research/GATE_G3.md`.

### 6.5 Association analysis — `scripts/research/r4_association.py` (NEW)
On external images (HC18, UCL; African has no landmarks → use `out_sector_frac` only): `statsmodels` GLM logit `correct_full ~ epg_true + max_prob + C(dataset) + C(class)` with cluster-robust SE by patient; report OR per 0.1 EPG [CI]; AUROC of EPG alone; ΔAUROC over `max_prob` alone (paired clustered bootstrap). Also the in-distribution analogue on FPDB test. Pre-declare: this is exploratory for existing checkpoints; confirmatory for R5 models (in pre-registration).

---

## 7. Phase R5 — Anatomy-focused training protocol (after Gate G4 = OSF pre-registration)

### 7.1 Preprocessing modes — `src/preprocess/pipeline.py` (NEW)
`preprocess_mode`:
- `resize` (B0; current behaviour, bit-identical — parity test).
- `sector_letterbox` (A1): sector mask → zero outside → crop to sector bbox → letterbox to square (pad value 0) → resize.
- `sector_letterbox_inpaint` (A1+A2 inference side): plus overlay inpainting.
Applied identically at train and eval (parity test extended). Cache preprocessed images to `data/processed/cache/<mode>/` (PNG) to keep training fast.

### 7.2 Caliper augmentation — `src/preprocess/caliper_aug.py` (NEW)
Albumentations-compatible transform: with p, draw 2–4 synthetic markers ('+'/'×' size 9–21 px, intensity 230–255, optional dotted line between a pair), placed uniformly **inside the sector** for images of **all classes**. Unit test: output differs only in marker pixels.

### 7.3 Acquisition-balanced sampler (A3) — in `src/train/train.py`
Compute on train split: `w_i = p(y_i) / p(y_i | machine_i)` with Laplace smoothing (α=1); machines with <30 training images (S10: 11) pooled into "Other_machine". Use `WeightedRandomSampler(weights, num_samples=len(train), replacement=True)`. **Keep class-weighted CE off when A3 is on** (avoid double reweighting) — document the choice; additionally run one sensitivity seed with both on. Log effective class×machine prevalence of sampled batches for the first 2 epochs (assert within ±5 pp of target).

### 7.4 Device-style augmentation (A4)
Add behind `style_aug: true`: `RandomGamma(80–120)`, `RandomBrightnessContrast(0.3,0.3)`, `GaussianBlur(3–5, p=0.2)`, `MultiplicativeNoise(0.8–1.2)`, `Downscale(0.5–0.9, p=0.2)`, `CLAHE(p=0.2)`. No horizontal-flip change.

### 7.5 Config schema additions (all configs, explicit defaults)
```yaml
preprocess_mode: resize | sector_letterbox | sector_letterbox_inpaint
caliper_aug_p: 0.0
sampler: none | acquisition_balanced
use_class_weights: true
style_aug: false
seed: 42            # overridable via --seed
run_root: runs/
```
### 7.6 Training matrix — `scripts/research/r5_run_matrix.py` (NEW)
Arms × backbones × seeds {11, 22, 33}:
| Arm | preprocess | caliper_aug_p | sampler | style_aug | class weights |
|---|---|---|---|---|---|
| B0 | resize | 0 | none | false | true |
| B1 | sector_letterbox | 0 | none | false | true |
| B2 | sector_letterbox_inpaint | 0.3 | none | false | true |
| AFP | sector_letterbox_inpaint | 0.3 | acquisition_balanced | true | false |
Backbones: `convnext_tiny.fb_in22k_ft_in1k` (224), `tf_efficientnetv2_s.in21k_ft_in1k` (300, mean/std 0.5). Keep existing optimiser/LR/epochs/patience (from `configs/convnext_tiny.yaml`, `configs/tf_efficientnetv2_s.yaml`). 24 runs; each writes `runs/<arm>_<backbone>_s<seed>/{best.pt, config.yaml, train_log.txt, val_preds.parquet}`. Resume-safe (skip if `best.pt` exists).
Then evaluate every run on fpdb_test, hc18, ucl, african **with that run's own preprocessing**; also run R3-E5 and R4 metrics on all 24 models.

### 7.7 Foundation-model probe (secondary) — `scripts/research/r5_fm_linear_probe.py` (NEW)
Check FetalCLIP weight licence/availability first (GitHub `biomedia-mbzuai/fetalclip`); else USFM; else skip and log. Frozen encoder; logistic-regression probe on train features (val for C); inputs in B0 vs AFP preprocessing; same evaluation. No fine-tuning.

### 7.8 Primary analysis — `scripts/research/r5_primary_endpoint.py` (NEW)
Exactly as pre-registered: pooled external (HC18+UCL+African) full-argmax balanced accuracy, collapsed taxonomy; AFP vs B0 per backbone; seed-averaged; paired cluster bootstrap; report Δ [95% CI], p. Non-inferiority: in-dist macro-F1 Δ > −0.02 (one-sided 95% lower bound). Secondary: B1/B2 vs B0; per-dataset; rejection rate; ECE; shortcut indices (E5 flip rates); EPG.

---

## 8. Phase R6 — Analysis, reporting, reproducibility

- `docs/research/PREREGISTRATION.md` (write before R5; then lodge on OSF): hypotheses P4a–c, endpoint, analysis, exclusions, margins, seeds, stopping rules.
- `docs/research/RESULTS.md`: auto-filled by `scripts/research/render_results.py` from `results/`; sections RQ1–RQ4; every number traceable to a parquet + run_id.
- Figures (`scripts/research/make_figures.py`, matplotlib, colour-blind-safe, vector PDF + PNG):
  1. Confounding mosaic (machine × class share; train vs test; S10 highlight).
  2. Metadata-only ROC (all-test vs excl-S10).
  3. E4/E5 bar chart: availability accuracy and reliance flip rates vs control, with CIs.
  4. Saliency examples grid: image, landmark mask, Grad-CAM, EPG value — correct and incorrect, in-dist vs external (de-identified public data only).
  5. EPG distributions by class × backbone; association plot (EPG decile vs external accuracy).
  6. Forest plot of primary endpoint (B1/B2/AFP vs B0, per backbone, per dataset).
  7. Rejection-vs-confusion stacked bars per external dataset.
  8. Risk–coverage curves in-dist vs external.
- `docs/research/DATA_CARDS.md`: each dataset — source, licence, n images/patients, devices, classes, how used, exclusions.
- `docs/research/REPRODUCE.md`: ordered commands from raw data to every table.
- Determinism: `torch.backends.cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`, log versions (`pip freeze > runs/<id>/env.txt`), git hash in every run config.

---

### 8.1 JAPI review-support artefacts (no new experiments; build from R1–R4 outputs)
- `docs/research/literature_search_log.md`: databases searched (PubMed, Scopus or Google Scholar, arXiv/medRxiv), dates, exact query strings, inclusion rules, and hit counts. Re-run before submission. Claude Code may draft queries; the human runs or confirms them.
- `docs/research/review_evidence_table.csv`: one row per key study (target 30–40). Columns: `citation, year, venue, peer_reviewed (y/n), dataset(s), n_images, n_sites, devices, external_test (y/n), per_device_results (y/n), overlay_handling (none/crop/inpaint), xai_method, xai_quantitative (y/n), calibration_reported (y/n), india_data (y/n), notes`. Fill it from abstracts and full texts. Every row needs a URL/DOI. Rows not verified against the paper are marked `verified=n`.
- `docs/research/review_reference_shortlist.md`: ≤40 references ranked by necessity, in Vancouver style (JAPI uses Vancouver with square-bracket numbering, up to 6 authors, then 3 + et al.).
- `scripts/research/make_review_figures.py`: up to 4 publication figures for JAPI. (1) Shortcut-learning schematic (vector, drawn in matplotlib; no copyrighted images). (2) Box 1: machine × class confound plus metadata-only ROC. (3) Saliency grid with landmark masks, using public-data images only, with the dataset licence credited in the caption. (4) Validation checklist table rendered as a figure. 300 dpi PNG plus PDF, greyscale-safe.
- `docs/research/BOX1_ILLUSTRATIVE_ANALYSIS.md`: an auto-rendered ~200-word summary of the R1–R3 headline numbers with CIs, for direct use in the review box.

---

## 9. Out of scope now — long-term future extensions (DO NOT IMPLEMENT)

These require hospital partners and/or ethics approval and are deferred indefinitely (proposal §5.5):
- F1 Indian external validation.
- F2 Clinician key-structure box annotation.
- F3 Prospective or real-video evaluation.

The only design obligation now: keep `src/eval/predict.py` dataset-agnostic and runnable offline from a folder plus a label CSV, so that F1 would need no new evaluation code. Do not create protocols, annotation exports or site-evaluation scripts for these.

---

## 10. Testing requirements (`tests/`)

| Test | Checks |
|---|---|
| `test_splits_frozen.py` | SHA-256 of split CSVs unchanged |
| `test_no_test_leakage.py` | Training configs never reference test/external manifests; patient sets disjoint across train/val/test |
| `test_transforms_parity.py` | Train-eval geometric parity for every preprocess mode; `resize` mode bit-identical to legacy |
| `test_sector.py`, `test_overlays.py` | Synthetic fan + marks detection/inpainting |
| `test_anatomy_masks.py` | Ellipse/capsule geometry; model-space transform Dice ≥ 0.98 |
| `test_localisation_metrics.py` | EPG/hit/IoU toy cases |
| `test_stats.py` | Cluster vs image bootstrap on toy clustered data; paired bootstrap Δ=0 for identical inputs; ECE toy |
| `test_sampler.py` | Effective prevalence under A3 matches target within tolerance on synthetic metadata |
| `test_predict_schema.py` | Parquet columns/dtypes; collapsed labels; restricted argmax correctness |

---

## 11. New `CLAUDE.md` (replace the file with this)

```markdown
# CLAUDE.md

## Project (research study, not a product)
"Anatomy or Acquisition?" — auditing shortcut reliance and anatomical grounding of fetal
standard-plane classifiers trained on FETAL_PLANES_DB, with multi-device external validation
(HC18, UCL, African low-resource planes). Public data only; no ethics-approval-dependent work. Read
docs/research/01_RESEARCH_PROPOSAL.md and 02_IMPLEMENTATION_PLAN.md before changing anything.
Legacy real-time demo is frozen at tag legacy-fetscan-demo-v1 and under archive/ — do not edit or import it.

## Hard rules
- Never tune/select/early-stop on test or external sets. Validation = data/splits/val.csv only.
- data/splits/*.csv are frozen (SHA-256 test). Never regenerate.
- Every evaluation writes per-image parquet to results/preds/<run_id>/; tables/figures are built only from these.
- Every number: 95% patient-clustered bootstrap CI (src/eval/stats.py). Comparisons: paired cluster bootstrap.
- Config is always read from the checkpoint. Train/eval preprocessing must match (tests/test_transforms_parity.py).
- FP subset of the landmark benchmark = FPDB images: use ONLY as localisation ground truth for FPDB-test images; never train on it. HC18/UCL/African are external test sets only.
- NatalIA = phantom; supplementary only. No clinical or regulatory claims anywhere.
- New metrics require unit tests. Log every task in docs/research/RESEARCH_LOG.md (run_id, git hash, outcome).

## Environment
conda env `fetalplane` (Python 3.11, CUDA 12.x, RTX 4060 8 GB), Windows; run from repo root; pathlib only;
split CSVs contain backslash paths — normalise on load.

## Layout
src/data (datasets, transforms) · src/preprocess (sector, overlays, caliper_aug, pipeline) ·
src/models (backbone, loading, gradcam, cam_localisation) · src/localisation (anatomy_masks, metrics) ·
src/eval (predict, stats, external) · src/train (train.py, losses) · scripts/research (experiments E1–E7, R2–R6) ·
configs/ (B0 baselines) + configs/research/ (arms) · results/ · runs/ (gitignored) · docs/research/ · archive/

## Commands
python -m pytest
python -m src.eval.predict --ckpt <ckpt> --dataset fpdb_test|hc18|ucl|african --run-id <id>
python src/train/train.py --config configs/research/afp_convnext_tiny.yaml --seed 11
python scripts/research/r5_run_matrix.py
python scripts/research/render_results.py
```

---

## 12. Definition of done (checklist for the auditor)

- [ ] R0: legacy tagged; archive/ populated; no imports from archive; tests green; new CLAUDE.md/README.
- [ ] R1: acquisition metadata parquet; E1 tables; external manifests (HC18/UCL/African) with patient_id; FP landmark linkage + registration report; overlay detector + manual audit (300) + caliper pairs; **GATE_G1.md** decision written.
- [ ] R2: predict/stats/external modules + tests; existing 6 checkpoints re-evaluated; **G2** reproduction passed.
- [ ] R3: E2–E7 tables/figures with CIs; RESULTS.md RQ1–RQ2 auto-rendered.
- [ ] R4: masks + metrics + tests; saliency on all models; **GATE_G3.md** sanity check; association analysis.
- [ ] **G4:** PREREGISTRATION.md written and lodged (OSF link recorded) **before** R5 training starts.
- [ ] R5: 24 runs complete, all evaluated with own preprocessing; FM probe (or logged skip); primary endpoint script output.
- [ ] R6: RESULTS.md complete, 8 figures, DATA_CARDS.md, REPRODUCE.md; every number traceable to run_id.
- [ ] §8.1 review-support artefacts: search log, evidence table (30–40 rows, verified), reference shortlist (≤40), 4 review figures, Box 1 summary.
- [ ] Nothing from §9 (future extensions) was implemented.

## 13. What the human must provide / decide

1. Download the African dataset (Zenodo 7540448) into `data/raw/african_planes/`.
2. Confirm `data/raw/ucl_hc18/annotations/FP/*.csv` and `data/raw/ucl_hc18/images/FP/` are present locally (needed for R1 §3.3). The existing FP/MULTICENTRE hard assertion in the external-test dataset class stays in force; `link_fp_landmarks.py` is the only code allowed to read `FP/`, and it only reads annotations + images for registration, never feeds a training loader.
3. Do the binary overlay audit: 300 images, two y/n keys each (~30–45 min), plus ≤50 y/n pair checks and the 15-image y/n landmark glance (~15 min).
4. Lodge the OSF pre-registration before R5.
5. Confirm JAPI's current review-article format (word and reference limits, whether unsolicited reviews are accepted) with the editorial office before drafting. The instructions PDF on japi.org is undated.
6. Optional: ask a clinician colleague to read the clinical sections of the review. This needs no ethics approval.
