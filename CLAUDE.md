# CLAUDE.md

## Project (research study, not a product)
"Anatomy or Acquisition?" audits shortcut reliance and anatomical grounding of fetal
standard-plane classifiers trained on FETAL_PLANES_DB, with multi-device external validation
(HC18, UCL, African low-resource planes). Public data only; no ethics-approval-dependent work.
Read `docs/research/01_RESEARCH_PROPOSAL.md`, `docs/research/02_IMPLEMENTATION_PLAN.md` and the
phase plans in `docs/research/plans/` before changing anything. **Where the plans and 02 disagree,
the plans win** (errata in `plans/00_INDEX_AND_VERIFIED_FACTS.md §3`).
The legacy real-time demo is frozen at tag `legacy-fetscan-demo-v1` and lives under `archive/`. Do not
edit or import it.

## Hard rules
- Never tune, select or early-stop on test or external sets. Validation = `data/splits/val.csv` only.
  `src/train/train.py::assert_no_leakage` enforces the path and patient-disjointness checks.
- `data/splits/{train,val,test}.csv` are frozen (`tests/test_splits_frozen.py`). Never regenerate them.
- `checkpoints/` is frozen (six legacy seed-42 models; see `checkpoints/FROZEN.md`). Never write there.
  New training goes to `runs/<run_id>/` via `--out-dir`.
- Every evaluation writes per-image parquet to `results/preds/<run_id>/`. Tables and figures are built only
  from these (and the JSON derived from them). No hand-typed numbers in docs.
- Every number carries a 95% patient-clustered bootstrap CI (`src/eval/stats.py`; B=2000, seed 20261109).
  Comparisons use the paired cluster bootstrap.
- Config is always read from the checkpoint (`src/models/loading.py`). Train and eval preprocessing must
  match (`tests/test_transforms_parity.py`).
- The FP landmark subset is pixel-identical to FPDB images. Use it ONLY as localisation ground truth
  (val for development, test for reporting); never train on it. HC18/UCL/African are external test sets only.
- Patient IDs are dataset-qualified strings: `fpdb_<n>`, `hc18_<prefix>`, `ucl_<prefix>`, `afr_<ISO3>_<n>`.
  HC18/UCL subject IDs come from filename prefixes (the CSV SubjectID columns are empty). African patient
  numbers repeat across centres.
- NatalIA is a phantom; supplementary only. No clinical or regulatory claims anywhere.
- New metrics require unit tests with hand-computed answers. Log every task in `docs/research/RESEARCH_LOG.md`
  (run_id, git hash, outcome, deviations). Write a walkthrough per phase in `docs/research/walkthroughs/`.
- Commits: since R4 (2026-11-13), at the user's request, Claude commits each phase on `main` with plain-language messages and no attribution lines. Never push.

## Environment
Interpreter: `D:\miniconda\envs\fetalplane\python.exe` (Python 3.11, torch 2.11+cu128, timm 1.0.28,
albumentations 2.0.8). `conda` is not on PATH in agent shells, so call the interpreter by absolute path.
Windows; run everything from the repo root; use pathlib.
Resource limits learned in R3: every process that imports torch commits ~2 GB of virtual memory (WinError 1455
"paging file too small" when ~12+ run at once: training + DataLoader workers + other jobs), so do not overlap
GPU jobs that each spawn DataLoader workers. CPU-parallel sklearn work must use single-threaded BLAS
(`OMP_NUM_THREADS=1`) and torch-free worker processes; multi-threaded BLAS made probe fits 10–25× slower. Every new input shape costs ~5 s of cuDNN set-up for fp32 forward+backward, so saliency code pads CAM batches to a fixed size (`SaliencyEngine.cam_batch`). Split CSVs contain backslash paths, so normalise
on load. albumentations 2.0.x argument names (`gamma_limit`, `PadIfNeeded(fill=...)`) are pinned on purpose.

## Layout
src/data (datasets, transforms) · src/preprocess (sector, overlays, caliper_aug, pipeline) ·
src/models (backbone, loading, gradcam/saliency, cam_localisation) · src/localisation (anatomy_masks, metrics) ·
src/eval (predict, stats, metrics, external) · src/train (train.py, losses, sampler) · src/audit ·
scripts/research (experiments E1–E7, R2–R6) · configs/ (B0 baselines) + configs/research/ (arms) ·
results/ · runs/ (gitignored) · docs/research/ · archive/

## Commands
```
python -m pytest
python -m src.train.train --config configs/convnext_tiny.yaml --out-dir runs/<run_id> [--seed 11] [--epochs-override 1]
python -m src.eval.predict --ckpt <ckpt> --dataset fpdb_test|hc18|ucl|african --run-id <id>      # R2
python scripts/research/r2_reevaluate_existing.py [--score-only]                                # R2 + Gate G2
python scripts/research/e1_confounding_tables.py; e2_metadata_only_model.py; e6_stratified_performance.py  # R3 (CPU)
python scripts/research/render_box1.py                                                           # Box 1 (after E1/E2/E6)
python scripts/research/e8_error_concordance.py                                                  # E8 post-hoc error concordance (CPU, ~1 min)
python scripts/research/e3_feature_probes.py --extract --probe --score [--jobs 6]                # R3 E3
python scripts/research/e5_reliance_perturbation.py --predict --score                            # R3 E5 (GPU)
python scripts/research/build_cache.py --mode nonanat_only overlay_only sector_shape_only        # R3 E4 cache
python scripts/research/e4_availability_train.py --train --evaluate                              # R3 E4 (~2 GPU-h)
python scripts/research/r4_localisation.py --dev                                                 # R4 development (val only)
python scripts/research/r4_sanity_randomisation.py                                               # R4 Gate G3 (needs clean src/)
python scripts/research/r4_localisation.py --run; r4_summarise.py; r4_association.py              # R4 test/external (after G3)
python scripts/research/build_cache.py --mode sector_letterbox sector_letterbox_inpaint          # R5 caches (native-scale crops)
python scripts/research/r5_run_matrix.py --smoke                                                 # R5 1-epoch smoke runs (allowed before G4)
python scripts/research/r5_run_matrix.py                                                         # R5 24 runs + 1 sensitivity run (refuses before Gate G4: OSF link)
python scripts/research/r5_evaluate_matrix.py --predict --saliency --probes --score              # R5 secondary analyses
python scripts/research/r5_primary_endpoint.py                                                   # R5 pre-registered primary (never edit)
python scripts/research/r5_fm_linear_probe.py --extract --probe --score                        # R5.8 FetalCLIP probe (after G4; needs open_clip + weights)
python scripts/research/make_figures.py                                                          # R6 study figures 3/7/8 + STUDY_FIGURES.md
python scripts/research/check_orphan_json.py; check_evidence_table.py                            # R6 checks
python scripts/research/render_results.py                                                        # R6
```
