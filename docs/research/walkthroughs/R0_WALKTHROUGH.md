# R0 Walkthrough: Repository Restructure

**Date:** 2026-11-12 · **Branch:** `research/anatomy-or-acquisition` (from `79ac5a5`) · **Legacy tag:** `legacy-fetscan-demo-v1` → `b854a34` (local; not pushed)
**Plan:** `docs/research/plans/R0_REPO_RESTRUCTURE.md` · **Status:** ✅ complete, uncommitted (the user commits manually)

## 1. What was done

| Plan step | Result |
|---|---|
| R0.1 freeze | Annotated tag `legacy-fetscan-demo-v1` on `b854a34`; branch `research/anatomy-or-acquisition` created. Nothing pushed. |
| R0.2 archive / remove | 121 tracked files moved with `git mv` (history preserved) into `archive/` (src/realtime, src/smoothing, ACAM, synthetic video, legacy evaluate_test, Streamlit app, assets, 19 demo/NatalIA scripts, 3 legacy tests, 4 configs, demo data artefacts, demo logs, `docs/instructions/`, `docs/phases/`, 6 legacy reports). Untracked local bulk (mp4s, sweep CSVs, outputs/) moved alongside and gitignored. **Removed:** multitask model, scripts and config, `checkpoints/multitask/`, `data/splits/multitask_*.csv`, `src/utils/`, `project_structure_tree.txt`. `archive/README.md` explains the archive. |
| Docs | `docs/01_*`, `docs/02_*` → `docs/research/`; literature survey → `docs/research/background/`; "SUPERSEDED" header on EVAL_REPORT, EXPERIMENTS and CROSS_DEVICE_BACKBONE_COMPARISON. |
| R0.3 loader | `src/models/loading.py` (from `src/realtime/model_loader.py`): `load_model()` returns `LoadedModel` with config-driven size, normalisation and `preprocess_mode`. It **refuses** multitask checkpoints, ACAM checkpoints, and checkpoints whose train/val CSV names test or external data. `scripts/evaluate_cross_device.py` repointed (it stays until R2 replaces it). |
| R0.4 train.py | Leakage guard (`assert_no_leakage`: path tokens plus train/val/test patient disjointness), frozen-folder guard (`assert_not_frozen`; any `checkpoints/…` target is refused), `assert_supported_options` (rejects not-yet-implemented `preprocess_mode`, `sampler`, `style_aug`, `caliper_aug_p`, `use_acam` instead of silently ignoring them), `use_class_weights` honoured, `--seed` and `--out-dir` flags, effective config saved as `config.yaml`, log at `<out>/train_log.txt`. ACAM import removed. `scripts/smoke_test.py` now writes to `runs/_smoke/`. |
| Data module cleanup | `FocalPlanesDataset`: multitask bbox branch and `-100` labels removed. `multitask_collate_fn` removed. `transforms.py`: bbox params and the video-frame helper removed; the loader accepts `Path`. |
| R0.5 tests | `pytest.ini`, `tests/conftest.py` (repo root on path, `data`/`gpu` skip markers). Retained tests moved and rewritten for pytest (`test_backbone.py` was script-style and would have errored under pytest). New: `test_splits_frozen`, `test_no_archive_imports`, `test_no_test_leakage`, `test_loading`, `test_train_guards`, plus a rewritten `test_transforms_parity` (loader, RGB→grey, train/eval geometric parity, bit-identity with the legacy eval pipeline on real FPDB images). **56 tests, all passing.** |
| R0.6 configs | The 6 baseline YAMLs carry explicit B0 keys (`preprocess_mode: resize`, `caliper_aug_p: 0.0`, `sampler: none`, `use_class_weights: true`, `style_aug: false`). |
| R0.7 env | Installed into `fetalplane`: pytest 9.2.0, statsmodels 0.15.0, scikit-image 0.26.0. `requirements.txt` cleaned (demo packages removed, albumentations 2.0.8 and timm 1.0.28 pinned); `requirements-lock.txt` rewritten to match the installed environment. `.gitignore` rewritten: runs/, results/preds/ and archive bulk ignored; `data/processed/audit/**` (human labels) and `SPLITS_SHA256.txt` tracked. |
| R0.8 skeleton | `src/{preprocess,localisation,audit}/`, `scripts/research/`, `configs/research/`, `results/{tables,json,figures,preds}/`, `runs/`, `docs/research/walkthroughs/`, `docs/research/RESEARCH_LOG.md`. |
| R0.9 docs | New `CLAUDE.md` (hard rules, interpreter path, layout, the manual-commit rule) and a research `README.md`. |
| Freeze records | `data/splits/SPLITS_SHA256.txt`; `checkpoints/FROZEN.md` with the SHA-256 of all six `best.pt` files. |

## 2. Outcomes and verification

- `python -m pytest` → **56 passed**.
- Smoke training: `python -m src.train.train --config configs/convnext_tiny.yaml --epochs-override 1 --out-dir runs/_smoke/convnext_tiny` ran end to end. One epoch gave val macro-F1 0.824; this is a smoke check, not a result.
- Frozen checkpoints: SHA-256 of all six `best.pt` identical before and after R0, including after the smoke run.
- No live code imports archived modules (AST test). The remaining `multitask` / `acam` strings in `src/` are the refusal guards themselves. `scripts/evaluate_cross_device.py` keeps its legacy contamination check until R2 replaces it.

## 3. Deviations from the plan, and why

1. **Split hashes use LF-normalised bytes.** The repo has `core.autocrlf=true`, and the working-copy CSVs are CRLF. Raw-byte hashes would change with checkout settings and break the freeze test spuriously.
2. **The leakage token check inspects the repo-relative path** (or only the file name for files outside the repo), not the full absolute path. The absolute-path version, as written in the plan, false-positived on pytest's temp folder (`pytest-of-…/test_…`) and would do the same for any user folder containing "test". Covered by `test_token_check_ignores_ancestor_folders`.
3. **`--out-dir` plus a hard frozen-folder guard inside `train()`**, not just the CLI flag. `scripts/smoke_test.py` called `train()` directly with a `checkpoints/<bb>/smoke_test` target, which the flag alone would not have caught.
4. **Unimplemented protocol options raise `NotImplementedError`** rather than being accepted and ignored. R5 widens `SUPPORTED_OPTIONS`.
5. **`requirements-lock.txt` was inaccurate** (pandas 3.0.5 listed, 2.3.3 installed; gradio/imageio listed but unused). The lock now reflects the real environment.
6. The **user had already committed the proposal docs** (`79ac5a5`) before R0. Their move into `docs/research/` is staged as a rename. The branch was cut from `79ac5a5`; the tag still points at the audited demo commit `b854a34`, as planned.

## 4. Findings to carry forward

- **Training throughput is ~30 img/s** for convnext_tiny at batch 64 (about 3.4 min/epoch), and the first validation pass took ~2 min (worker spin-up on Windows). This looks bound by data loading (decoding PNGs of up to 1137×787 per sample). At ~20 epochs per run, the 24-run R5 matrix would take days rather than the planned ~24 GPU-hours. **R5's planned preprocessed-image cache must also store images pre-resized**, and the loader settings should be profiled before R5. This is noted in the R5 plan scope; no change made yet.
- `scripts/run_sanity_checks.py` and `scripts/evaluate_cross_device.py` still read `cross_device_manifest.csv`; they are handled in R1 and R2 as planned.

## 5. For the user

- Review and commit (suggested split in `R0_REPO_RESTRUCTURE.md §R0.10`). Everything is staged or untracked on `research/anatomy-or-acquisition`.
- Optional: `git push origin legacy-fetscan-demo-v1` and push the branch when you want them on the remote.
- Next phase: **R1 (data foundation)**. Human input is needed mid-phase for the binary overlay audit (~45–60 min of y/n keys).
