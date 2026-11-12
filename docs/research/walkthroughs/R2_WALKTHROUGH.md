# R2 Walkthrough: Evaluation Core (Gate G2)

**Date:** 2026-11-12 · **Branch:** `main` (base `8f6ba09`) · **Plan:** `docs/research/plans/R2_EVALUATION_CORE.md`
**Status:** ✅ complete; **Gate G2 PASSED** (`docs/research/GATE_G2.md`). Uncommitted.
**Tests:** `python -m pytest` → **131 passed** (41 new).
**Compute:** 30 prediction runs (6 checkpoints × 5 datasets) took 32 min on the RTX 4060 (fp32). Scoring with 2,000 bootstrap replicates for every number takes under 1 min (`--score-only`).

## 1. What was built

| Plan step | Implementation | Tests |
|---|---|---|
| R2.1 metrics | `src/eval/metrics.py`: confusion, accuracy, per-class recall/precision/F1, balanced accuracy, macro-F1 (restricted label sets), rejection rate, error decomposition, ECE, AUROC, risk–coverage, coverage@accuracy, bias-corrected Cramér's V | `tests/test_metrics.py`: hand-computed answers; matches sklearn |
| R2.1 statistics | `src/eval/stats.py`: `cluster_bootstrap`, `paired_cluster_bootstrap`, `multi_seed_summary`, `paired_multi_seed`, `image_bootstrap` (legacy comparison only), `wilson`. Fast paths for `CMMetric` (per-cluster confusion matrices) and `RatioMetric` (per-cluster sums); a general path for any `metric(df)`. NaN replicates are counted, and more than 1% raises. | `tests/test_stats.py`: cluster CI ≥ 2× image CI on clustered toy data; paired self-comparison gives Δ = 0, CI [0, 0], p = 1; a 10% degradation is detected; fast and general paths give identical replicates; alignment is checked; Wilson closed form |
| R2.2 predictor | `src/eval/datasets.py` (registry → one standard frame; `present_classes`) and `src/eval/predict.py` (checkpoint-driven transform, `perturb` hook on the native-resolution greyscale image with R1 masks, variant suffixes, timm pre-logit features, provenance, meta JSON). CLI supports `--dataset` or any `--manifest` (F1 compatibility). | `tests/test_predict_schema.py` (tiny resnet10t fake checkpoint), `tests/test_eval_datasets.py` |
| R2.3 external scoring | `src/eval/external.py` (`score_dataset`, `score_pooled`, `by_stratum`, `rejection_vs_confusion_table`) and `src/eval/scoring.py` (in-distribution scorer, metric factories, per-row helper columns) | `tests/test_external_scoring.py` (hand-built frames) |
| R2.4 re-evaluation | `scripts/research/r2_reevaluate_existing.py` | outputs below |
| R2.5 Gate G2 | same script → `results/json/G2.json`, `docs/research/GATE_G2.md` | — |

Refactors made along the way:
- `load_grayscale` / `gray_to_rgb` were split out of `load_and_prep_grayscale_to_rgb`, so perturbations act on the raw greyscale image.
- `validate_external_manifest` is now shared by `ExternalDataset` and the registry.
- `LoadedModel.eval_transform(mode)` accepts a preprocess override.
- `g1_report.py` uses the shared `wilson`.
- `src/__init__.py` disables albumentations' network version check.

**Outputs.**
- `results/preds/legacy_<short>_s42/{fpdb_val,fpdb_test,hc18,ucl,african}.parquet`, plus `.meta.json`, and `.features.npy` for test and externals (108 MB, gitignored).
- `results/json/R2_existing_models.json`, `results/tables/R2_existing_models.md`, `results/figures/R2_rejection_vs_confusion.{pdf,png}`.

## 2. Gate G2: reproduction

Every primary and secondary check passes (`GATE_G2.md`):
- convnext FPDB-test macro-F1 0.8929 (legacy 0.8927).
- HC18+UCL `acc_full` 83.63% (83.6%); HC18 81.38% (81.4%); UCL 88.92% (88.9%).
- Head errors 194/194, of which 180/180 were predicted Other.
- The other five backbones match their legacy macro-F1 to ≤ 0.0003.
- All six backbones match the legacy cross-device accuracy and their Head→Other counts **exactly**.

So there is no library drift (OpenCV 5.0 / albumentations 2.0.8) and no re-baselining is needed.

## 3. Results (all 95% patient-clustered CIs, B = 2000)

**In-distribution (FPDB test).**
- Macro-F1 ranges from 0.875 to 0.893. convnext is highest: 0.893 [0.883, 0.902].
- The cluster CIs are only 1.06–1.21× wider than the legacy image-level CIs. The CI correction matters little in-distribution, because there are 896 patients and errors are only weakly correlated within a patient.

**Machine confound, visible in the errors.** Recall of the "Other" class by test machine, across all six models:

| Machine | "Other" share in train+val | Other recall (test, 6 models) |
|---|---|---|
| Aloka | 65% | 0.92–0.95 |
| Voluson E6 | 14% | **0.35–0.52** |
| Voluson S10 | (unseen in train) | 0.81–0.92 |

The models find "Other" easily on the machine where "Other" was common in training, and miss half or more of it on E6, where it was rare. This is the in-distribution signature of the acquisition shortcut that R3 (E2/E3/E6) will test directly.

**External sets** (pooled HC18 + UCL + African, 1,873 images):

| | convnext | range over 6 backbones |
|---|---|---|
| `acc_full` (legacy scoring) | 81.7% [79.6, 83.9] | 80.9–83.9% |
| `bal_acc_full` | 0.728 [0.690, 0.766] | 0.691–0.765 |
| `bal_acc_restricted` (argmax over the classes in the set) | 0.957 [0.935, 0.977] | 0.879–0.957 |
| rejection rate | 18.2% [16.0, 20.4] | 15.6–18.5% |
| share of errors predicted "Other" | 95.6% [93.4, 97.7] | 72–96% |

The external drop is mostly **rejection, not anatomical confusion**:
- Restricted to the anatomies actually present, every model reaches 0.88–0.96 balanced accuracy.
- Almost all errors are "this is not a standard plane" (Other).
- True confusions between present anatomies are ≤ 3% of errors.

This is the Box-1 headline candidate. Two caveats:
- On HC18 (Head only), restricted accuracy is trivially 100%.
- Pooled balanced accuracy is pulled down by African Thorax: 75 images, recall 0.36–0.73. Balanced accuracy weights Thorax as much as Head (1,283 images).

**African centres.**
- `acc_full` ranges from 92% (Algeria, convnext) to 47–65% (Ghana, all models).
- Ghana is the centre with burned-in calipers on every audited image (R1) and no UI. Its rejection rate is 31–53%.
- The R1 overlay shift therefore lines up with the worst external centre. Testing this belongs to R3.

**Backbone comparison.** The legacy "repvgg_a1 winner" claim is **withdrawn (endpoint-dependent)**:
- On the legacy endpoint (HC18+UCL `acc_full`) repvgg_a1 is first, and significantly better than 4 of 5 backbones.
- On the study's pooled endpoint (`bal_acc_full`) it is **significantly worse than all 5** (0.691 [0.655, 0.726]): its African Thorax recall is 0.36. On HC18 it rejects fewer heads as Other, but 56% of its HC18 errors are predictions of classes absent from HC18 (mostly Thorax).
- All are single-seed models, so R5 multi-seed runs will settle any backbone ranking.

**Contradictory duplicates.**
- 8 groups of byte-identical test images (16 images) carry different labels, which forces at least 8 errors on any classifier.
- Each model makes 8–9 of its 517–615 errors on them, so the ceiling is negligible.

## 4. Deviations from the plan

1. **Module split:** the in-distribution scorer and the metric factories live in `src/eval/scoring.py`; `external.py` keeps external scoring only. This avoids an "external" module scoring FPDB.
2. **Image-level CI** is reported for all six backbones, not only convnext. The CI-correction comparison costs nothing and is more complete.
3. **Per-machine macro-F1** averages over the classes present on that machine. The "Other" machine has 57 images and lacks some classes; the whole test set still uses all 8 classes.
4. **Undefined F1 is excluded:** a class with TP+FP+FN = 0 in a resample is left out of the macro-F1 average rather than imputed. On the full test set this equals sklearn.
5. **`RatioMetric` fast path** was added next to the plan's confusion-matrix fast path. It covers accuracy-type rates and conditional error shares.
6. **Stricter G2 targets** for the other five backbones: the 4-decimal macro-F1 values from `archive/docs/EVAL_REPORT.md` instead of the 2-decimal classification reports. Secondary checks against the legacy cross-device table and Head→Other counts were added.
7. **Extra analyses:**
   - the pairwise comparison on the legacy endpoint (so the repvgg_a1 claim is tested on its own terms);
   - an HC18+UCL legacy-set table;
   - errors on contradictory duplicates, which the R1 walkthrough asked for.
8. **`src/__init__.py`** sets `NO_ALBUMENTATIONS_UPDATE=1`. albumentations made a network version check on every import, and one check timed out during the run.
9. **G1 sign-off:** you delegated the call, so I confirmed the "overlays secondary" decision with a written rationale (`GATE_G1.md §8`, generated by `g1_report.py`).

No step of the plan was skipped. G2 passed on the first run, so the diagnosis procedure was not needed.

## 5. Things to know before R3

- Every R3 experiment reads `results/preds/legacy_*_s42/*.parquet`. Penultimate features for E3 are in `*.features.npy` (row order = parquet order, ids in `*.features_ids.txt`).
- `predict(..., perturb=fn, variant="perturb-<name>")` is ready for E5. `meta["masks"]` gives the R1 `sector` / `sector_clean` / `ui` / `marker` masks.
- `paired_cluster_bootstrap` aligns on `image_id` and checks patient identity, so perturbed and clean predictions can be compared directly.
- The parquet provenance hash is `8f6ba09a68-dirty`, because R2 code was uncommitted at run time. Re-running `r2_reevaluate_existing.py --force` after the commit would stamp a clean hash (fp32 inference; results are expected to be identical, but this has not been checked by a second run).
