# R5 Walkthrough: Anatomy-Focused Training Protocol (RQ4), up to Gate G4

**Date:** 2026-11-13 · **Branch:** `main` · **Plan:** `docs/research/plans/R5_AFP_TRAINING.md`

**Status:** 🟡 **Matrix trained and primary endpoint computed (2026-11-14). Secondary analyses pending.**
- Gate G4 passed on 2026-11-13 (OSF link in `GATE_G4.md`).
- All 25 runs trained (24-run matrix + sensitivity run) and predicted on FPDB val/test, HC18, UCL and African.
- The pre-registered primary analysis ran unmodified: `results/tables/R5_primary.md` (section 6).
- Still to run: `r5_evaluate_matrix.py --saliency --probes --score`, the FetalCLIP probe, `make_figures.py`, `render_results.py`.

## 1. What was built (R5.1–R5.9)

| Plan item | Code | Tests |
|---|---|---|
| R5.1 preprocessing modes | `src/preprocess/pipeline.py`: `sector_letterbox`, `sector_letterbox_inpaint`, `prepare_native`, `geometric`, `load_prepared`, cache key and stale check. `src/data/transforms.py`: train and eval transforms built from the same `geometric()`. `scripts/research/build_cache.py`: native-scale letterbox caches for FPDB and all external sets | `tests/test_pipeline_modes.py`: crop, zeroing, inpainting; aspect ratio kept (circle within 2%); `fill=0` pad argument; train-without-random equals eval for every mode; legacy `resize` bit-identical; cache key; `load_prepared` |
| R4.2 extension | `to_model_space` for letterbox modes (crop to the sector bounding box, then the image pipeline's own geometry); `runner.SaliencyInputs` feeds each checkpoint its own preprocessing | `tests/test_anatomy_masks.py`: letterbox projection matches the eval transform, Dice ≥ 0.98 |
| R5.2 caliper augmentation | `src/preprocess/caliper_aug.py` (`CaliperAug`, native-scale, inside the sector) | `tests/test_caliper_aug.py`: only footprint pixels change; markers lie in the sector; p = 0 is the identity; deterministic; probability gate; R1 detector finds ≥ 80% (1-px strokes) |
| R5.3 sampler | `src/train/sampler.py`: q(m, y) = p(m)/K_m, Laplace α = 1, cap 10 × median, realised-joint check | `tests/test_sampler.py`: realised joint within 1 pp over 200k draws; empty cells never sampled; cap; pooling; determinism |
| R5.4 style augmentation | `src/data/transforms.py::style_augmentations` | constructed and run in `test_pipeline_modes.py` |
| R5.5 `train.py` | R5 config keys; sampler with a recording wrapper; seeded workers and generators; deterministic algorithms (warn-only); `git_hash` and `cache_key` in `best.pt`; `env.txt`, `sampler_report.json`, `DONE`; val predictions written by `src.eval.predict`; resume-safe skip | `tests/test_train_guards.py` (R5 options accepted, invalid options rejected) |
| R5.6 configs and matrix | `configs/research/{b0,b1,b2,afp}_{convnext_tiny,tf_efficientnetv2_s}.yaml` + `afp_cw_convnext_tiny.yaml` (sensitivity run, part 2); `scripts/research/r5_run_matrix.py`: G4 gate, seed-major order, VRAM check, ETA, B0 sanity check, single re-run after divergence, `--smoke` | `tests/test_r5_evaluate_matrix.py::test_run_matrix_has_the_sensitivity_run_last` |
| R5.7 evaluation | `scripts/research/r5_evaluate_matrix.py`: predictions and features; within-class machine probes (shared code, `src/audit/probes.py::within_class_machine_targets/within_class_mean`, also used by E3, with identical results); marker flip rate; Grad-CAM EPG; temperature scaling and ECE; risk–coverage; part 2 (§5) adds rejection vs confusion per arm, the confirmatory P3b and the sensitivity run | `tests/test_r5_evaluate_matrix.py` (part 2; synthetic data) |
| R5.8 FM probe | `scripts/research/r5_fm_linear_probe.py` (part 2, §5) | `tests/test_r5_fm_probe.py` |
| R5.9 primary analysis | `scripts/research/r5_primary_endpoint.py` (Holm, success rule, non-inferiority, per dataset and per centre, forest plot) | `tests/test_r5_primary_endpoint.py`: synthetic arms with a known effect, and no effect |
| G4 | `docs/research/PREREGISTRATION.md` (template sections 1–9), `docs/research/GATE_G4.md` | — |

## 2. Pre-G4 results

**Letterbox caches** cover 14,273 images: FPDB 12,400 and externals 1,873. They are keyed by the overlay-parameter hash and `r5.1-v1`.

Sector shape classes, from `build_report.json`:

| dataset | fan | linear | full |
|---|---|---|---|
| FPDB | 5,645 | 4,682 | 2,073 |
| HC18 | 666 | 313 | 20 |
| UCL | 227 | 86 | 111 |
| African | 361 | 53 | 36 |

**Smoke runs.** All 8 arm × backbone configs, seed 11, 1 epoch, in `runs/_smoke/`. They run end to end: train → val → `best.pt` → val predictions via `predict` → `DONE`.

| run | epoch-1 val macro-F1 |
|---|---|
| B0 convnext | 0.831 |
| B1 convnext | 0.822 |
| B2 convnext | 0.850 |
| AFP convnext | 0.845 |
| B0 effv2s | 0.783 |
| B1 effv2s | 0.765 |
| B2 effv2s | 0.774 |
| AFP effv2s | 0.791 |

These are one-epoch numbers and say nothing about the arms. They are val only.

**Two-epoch AFP smoke run.** The sampler's pooled realised joint was within 3.7 pp of target, with no clipping. Throughput was about 190 img/s from epoch 2, so a full convnext run should take roughly 25 min and the matrix roughly 15–20 GPU-hours.

## 3. Deviations and decisions (all also in PREREGISTRATION §8)

1. **Sampler group pooling** uses total group size < 30, not "< 30 images in any class".
   - The literal rule would merge Other_machine (254 images) and contradict the plan's expected multipliers.
2. **The realised-joint check pools epochs 1–2.**
   - Checking each epoch alone would have failed by chance in about 1 run in 8, as simulated with the real training table (12.7%).
   - The pooled false-alarm rate is 0.3%.
3. **A4 replaces both B0 photometric entries**, brightness/contrast and speckle, so speckle is not applied twice.
4. **Letterbox caches are stored at native scale**, so caliper augmentation draws in frame pixels. E4 caches stay pre-resized.
5. **Validation predictions** go to `results/preds/<run_id>/fpdb_val.parquet`, written by the evaluation predictor, not to `runs/<run_id>/val_preds.parquet`.
   - Smoke runs write to `results/preds/_smoke_<run_id>/`, so they never mix with real runs.
6. **The caliper detector failed G1**, so `_inpaint` removes only the markers the detector finds.
   - The B0 marker-sensitivity contrast is unavailable.
   - Marker sensitivity is measured for letterbox arms only.
7. **Letterbox quality depends on the R1 sector masks.**
   - FPDB G1 check: 138/150 OK.
   - 0.4% of FPDB sectors cover < 30% of the frame.
   - Externals are not audited; spot checks found UCL inset thumbnails inside the sector, and African frames that are full field-of-view crops.
8. **The R5.2 detector realism check** passes for 1-px strokes (≥ 80%). The R1.5 templates are 1 px wide and find 0% of 2-px crosses.
9. **The FM probe (R5.8)** uses FetalCLIP, licence CC BY-NC 4.0, recorded in `DATA_CARDS.md`.
   - `open_clip` is not installed yet; it will be installed only after G4.
   - The probe is part of the pre-registered secondary analyses, so it is not run before registration.

## 4. What happens after G4

1. **PI:** lodge `PREREGISTRATION.md` on OSF, quoting its commit hash, and paste `OSF: <url>` into `GATE_G4.md`. Commit.
2. `python scripts/research/r5_run_matrix.py`, in the background. It stops after B0 convnext s11 if that run's val macro-F1 is not within ±0.02 of the legacy model (legacy convnext val macro-F1 0.918; see `R2_existing_models.json → models.convnext_tiny.fpdb_val`).
3. `python scripts/research/r5_evaluate_matrix.py --predict --saliency --probes --score`.
4. `python scripts/research/r5_primary_endpoint.py`, unmodified, with its SHA-256 recorded in the output. Then `render_results.py`.

## 5. Part 2: completing the pre-registered code (2026-11-13, still before G4)

A check of PREREGISTRATION §6 against the code found three analyses that were registered but had no implementation. They are added now, *before* the registration is lodged, so the OSF record points at code that does everything it promises. `GATE_G4.md` now names this code-complete commit.

| §6 item | What was missing | Now |
|---|---|---|
| 3. Rejection vs confusion | per-arm numbers, which existed only as R2 code for legacy models | `r5_evaluate_matrix.py::errors_by_arm`: seed-averaged rejection rate and error shares per external set and pooled. Uses the R2 helper columns (`add_helper_columns`) and `multi_seed_summary` |
| 7. Confirmatory P3b on R5 models, plus the African monitor | no code (part 1 only averaged EPG) | `p3b_confirmatory`: HC18 + UCL landmark images, Grad-CAM, seeds pooled. Reuses `r4_association.glm/discrimination` unchanged. The monitor is tested on the African set (sign −1). The claim is confirmed only if OR > 1 with a CI excluding 1 for AFP on both backbones. A degenerate score (e.g. constant) is recorded, not fatal |
| 9. Sensitivity run (AFP + class weights) | not in the run matrix | `configs/research/afp_cw_convnext_tiny.yaml` (differs from AFP only in `use_class_weights` and run dirs). `r5_run_matrix.py` trains it after the 24 runs as `afp_cw_convnext_s11`. `sensitivity()` compares it with `afp_convnext_s11` (paired, descriptive) |
| 8. FM probe | script not written | `scripts/research/r5_fm_linear_probe.py`: `--extract` (frozen FetalCLIP, fp16, threaded loading), `--probe` (C on val by 8-way macro-F1; standard prediction parquets via `build_frame`), `--score` (per-mode metrics, paired letterbox − resize, legacy convnext as an unpaired reference). Refuses before G4 |

**Design points**

- **Seed pooling in P3b.** The three seeds' rows are pooled with keys `image_id|seed`; the paired bootstrap rejects duplicate keys. `patient_id` is unchanged, so every bootstrap draw and every GroupKFold fold keeps all of a patient's rows together. A test checks that the GLM reports 160 clusters for 960 pooled rows.
- **FM-probe input.** Each arm's prepared image goes through that arm's own 224-px geometry: B0 squash-resize, AFP letterbox. It then goes through FetalCLIP's eval transform, whose resize is a no-op at 224 × 224 and whose normalisation applies. The contrast is therefore exactly what the CNN arms see. The FetalCLIP README feeds PIL images straight to `preprocess_test` (repository checked 2026-11-13); we pass 3-channel PIL images.
- **Fix.** `score()` read saliency parquets from the default results root rather than the root it uses for predictions. It now passes its root explicitly. No outputs existed yet.

**Checks**

- `tests/test_r5_evaluate_matrix.py` (7 tests) covers:
  - temperature scaling, hand-computed: logits (1, 0, …), 75% class 0 → T = 1/ln 21;
  - `score()` end to end on a synthetic run, where the raw ECE equals the ECE of the stored confidences;
  - rejection shares, hand-computed: (10 + 20 + 30)/180;
  - seed-pooled P3b on synthetic data with a known effect: confirmed, with clusters equal to patients;
  - the sign of the sensitivity Δ;
  - the run matrix: 25 jobs, the sensitivity run last.
- `tests/test_r5_fm_probe.py` (7 tests) covers:
  - the letterbox geometry: 2:1 → 112 rows centred, zero padding;
  - C selection, with ties going to the smaller C;
  - a missing class raises an error;
  - a real probe on separable data;
  - scoring with hand-computed values (letterbox macro-F1 = 8/15).
- **Smoke run** of `afp_cw_convnext_s11` (1 epoch, `runs/_smoke/`): the log confirms `class_weights=True` together with the acquisition-balanced sampler, and DONE was written.

**Still open (needs the PI):** lodge `PREREGISTRATION.md` on OSF. `docs/research/OSF_REGISTRATION_GUIDE.md` gives a step-by-step guide.

## 6. Part 3: the matrix and the primary endpoint (2026-11-13 to 2026-11-14, after G4)

**Training.** `r5_run_matrix.py` started at 10:54 on 2026-11-13 from commit 7f61029 (the G4 commit).
- The built-in sanity check passed: B0 convnext seed 11 val macro-F1 matched the legacy model within the ±0.02 tolerance.
- The PI paused the matrix at 16:31 after 14 runs (a clean stop right after a run wrote DONE) and resumed it at 18:47. Runs with DONE are skipped, and each run trains from scratch with its fixed seed, so the pause changed nothing.
- All 25 runs finished by about 23:30; none diverged or needed a re-run. Per-run validation scores are in each `runs/<run_id>/DONE`.

**Predictions.** `r5_evaluate_matrix.py --predict` wrote per-image parquet for FPDB test, HC18, UCL and African (plus features, and the other-letterbox FPDB test for marker sensitivity) for every run, in about two hours.

**Primary endpoint.** `r5_primary_endpoint.py` ran unmodified. `git diff b81bf61ed3 HEAD` is empty for the script and for `src/eval/{stats,scoring}.py`; its SHA-256 is recorded in the output. Results: `results/tables/R5_primary.md`, `results/json/R5_primary.json`, `results/figures/R5_forest.pdf`.
- **convnext_tiny:** AFP beats B0 on the pooled external balanced accuracy (Holm-adjusted p < 0.05) and is non-inferior in distribution. **Pre-registered success.**
- **tf_efficientnetv2_s:** the AFP − B0 difference is positive but its CI includes 0 (Holm p > 0.05). **Not a success.** It is non-inferior in distribution.
- So the co-primary hypothesis holds for one of the two backbones. This is reported as such, not as a general result.

**Exploratory reading (secondary, no multiplicity correction).**
- **B2 is the largest gain on both backbones.** B2 is the sector letterbox plus caliper-marker inpainting and synthetic-caliper augmentation. Its pooled gain over B0 is about twice AFP's, and nearly identical across the two backbones.
- **The gain is concentrated in the African set** (several centres, low-resource devices). HC18 and UCL change little; HC18 has only the Head class.
- **AFP = B2 + acquisition-balanced sampler + device-style augmentation, with class weights off.** That AFP scores below B2 suggests that, of the extra AFP components, at least one (or dropping class weights) costs external balanced accuracy.
- Isolating which component costs the accuracy was not pre-registered. Any follow-up is labelled post hoc. The class-weight sensitivity run (`afp_cw_convnext_s11`) speaks to the class-weight part and is scored with `--score`.
- **Pooled plain accuracy is unchanged** between AFP and B0. The balanced-accuracy gain therefore comes from the minority external classes, not from the majority Head class.

**Forest plot** (`R5_forest.pdf`) was checked visually; it is produced by the frozen script and left as is.

