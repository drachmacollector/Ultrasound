# R5 Walkthrough: Anatomy-Focused Training Protocol (RQ4), up to Gate G4

**Date:** 2026-11-13 · **Branch:** `main` · **Plan:** `docs/research/plans/R5_AFP_TRAINING.md`

**Status:** ⏸ **Stopped at Gate G4. The PI needs to act.**
- Everything the plan allows before G4 is built, tested and smoke-run.
- `PREREGISTRATION.md` is committed.
- The 24-run matrix needs the OSF registration link in `GATE_G4.md`. `r5_run_matrix.py` refuses to start without it.
- No R5 model has been trained beyond one or two smoke epochs. No R5 model has been evaluated on test or external data.

## 1. What was built (R5.1–R5.9)

| Plan item | Code | Tests |
|---|---|---|
| R5.1 preprocessing modes | `src/preprocess/pipeline.py`: `sector_letterbox`, `sector_letterbox_inpaint`, `prepare_native`, `geometric`, `load_prepared`, cache key and stale check. `src/data/transforms.py`: train and eval transforms built from the same `geometric()`. `scripts/research/build_cache.py`: native-scale letterbox caches for FPDB and all external sets | `tests/test_pipeline_modes.py`: crop, zeroing, inpainting; aspect ratio kept (circle within 2%); `fill=0` pad argument; train-without-random equals eval for every mode; legacy `resize` bit-identical; cache key; `load_prepared` |
| R4.2 extension | `to_model_space` for letterbox modes (crop to the sector bounding box, then the image pipeline's own geometry); `runner.SaliencyInputs` feeds each checkpoint its own preprocessing | `tests/test_anatomy_masks.py`: letterbox projection matches the eval transform, Dice ≥ 0.98 |
| R5.2 caliper augmentation | `src/preprocess/caliper_aug.py` (`CaliperAug`, native-scale, inside the sector) | `tests/test_caliper_aug.py`: only footprint pixels change; markers lie in the sector; p = 0 is the identity; deterministic; probability gate; R1 detector finds ≥ 80% (1-px strokes) |
| R5.3 sampler | `src/train/sampler.py`: q(m, y) = p(m)/K_m, Laplace α = 1, cap 10 × median, realised-joint check | `tests/test_sampler.py`: realised joint within 1 pp over 200k draws; empty cells never sampled; cap; pooling; determinism |
| R5.4 style augmentation | `src/data/transforms.py::style_augmentations` | constructed and run in `test_pipeline_modes.py` |
| R5.5 `train.py` | R5 config keys; sampler with a recording wrapper; seeded workers and generators; deterministic algorithms (warn-only); `git_hash` and `cache_key` in `best.pt`; `env.txt`, `sampler_report.json`, `DONE`; val predictions written by `src.eval.predict`; resume-safe skip | `tests/test_train_guards.py` (R5 options accepted, invalid options rejected) |
| R5.6 configs and matrix | `configs/research/{b0,b1,b2,afp}_{convnext_tiny,tf_efficientnetv2_s}.yaml`; `scripts/research/r5_run_matrix.py`: G4 gate, seed-major order, VRAM check, ETA, B0 sanity check, single re-run after divergence, `--smoke` | — |
| R5.7 evaluation | `scripts/research/r5_evaluate_matrix.py`: predictions and features; within-class machine probes (shared code, `src/audit/probes.py::within_class_machine_targets/within_class_mean`, also used by E3, with identical results); marker flip rate; Grad-CAM EPG; temperature scaling and ECE; risk–coverage | — (runs after G4) |
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
