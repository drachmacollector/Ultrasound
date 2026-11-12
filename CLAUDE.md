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
- The user makes all git commits manually. Do not commit or push.

## Environment
Interpreter: `D:\miniconda\envs\fetalplane\python.exe` (Python 3.11, torch 2.11+cu128, timm 1.0.28,
albumentations 2.0.8). `conda` is not on PATH in agent shells, so call the interpreter by absolute path.
Windows; run everything from the repo root; use pathlib. Split CSVs contain backslash paths, so normalise
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
python scripts/research/r5_run_matrix.py                                                         # R5
python scripts/research/render_results.py                                                        # R6
```
