# R2 — Evaluation Core (ends at Gate G2)

**Goal:** one generic predictor that writes per-image parquet files, one statistics module that every number goes through (patient-clustered and paired bootstrap), one external-scoring module, and a reproduction of the legacy headline numbers from the frozen checkpoints.
**Effort:** 3–4 days (about 1 GPU-hour for re-evaluation).
**Inputs:** R0 complete; R1.1–R1.3 complete (labels, external manifests, landmark tables). R2 does **not** need the sector/overlay work.

---

## R2.1 `src/eval/stats.py` (NEW): write this first, test-driven

### Public API
```python
@dataclass(frozen=True)
class Estimate:
    point: float; lo: float; hi: float; n_images: int; n_clusters: int
    method: str = "cluster_percentile"; B: int = 2000
    def fmt(self, pct=False, dp=3) -> str          # "0.893 [0.881, 0.904]"

def cluster_bootstrap(df, metric_fn, cluster_col="patient_id", B=2000, seed=20261109, alpha=0.05) -> Estimate
def paired_cluster_bootstrap(df_a, df_b, metric_fn, key="image_id", cluster_col="patient_id",
                             B=2000, seed=20261109) -> PairedEstimate   # delta, lo, hi, p_two_sided, point_a, point_b
def multi_seed_summary(dfs: list[pd.DataFrame], metric_fn, ...) -> Estimate      # same cluster draw for all seeds per replicate; average seed metrics
def paired_multi_seed(dfs_a, dfs_b, metric_fn, ...) -> PairedEstimate           # R5 primary endpoint
def wilson(k, n, alpha=0.05) -> tuple[float, float]
def image_bootstrap(...) -> Estimate     # legacy method; ONLY for the "CI-correction" comparison column
```

### Implementation requirements
- **Fast path for confusion-matrix metrics.** Precompute per-cluster confusion matrices `C[k] ∈ ℕ^{c×c}`. Each replicate draws cluster indices `idx = rng.integers(0, K, K)`, `w = np.bincount(idx, minlength=K)`, and `CM = np.tensordot(w, C, axes=1)`, then computes the metric from `CM`. 2,000 replicates × 6 models × 5 datasets must run in seconds.
- **General path** for row-level metrics (AUROC, ECE, risk–coverage, EPG means, GLM): concatenate the row-index arrays of the drawn clusters.
- Paired: align `df_a` and `df_b` on `key`. **Assert identical key sets and identical `patient_id` per key.** Use one draw per replicate for both models.
- CI: percentile. p-value: `2 * min(mean(Δ* ≤ 0), mean(Δ* ≥ 0))`, floored at `1/(B+1)`.
- Degenerate replicates (a class absent from a resample) are handled by metric definitions (below), never by dropping replicates. Count and report any NaN replicates; assert < 1%.
- Every function records `B`, `seed`, `n_clusters` and `n_images` in its result, and these go into JSON.

### Metric definitions (`src/eval/metrics.py`, NEW; pure functions on arrays or a confusion matrix)
| name | definition |
|---|---|
| `accuracy` | trace/total |
| `balanced_accuracy(labels=L)` | mean recall over classes in `L` that have ≥ 1 true sample in the (resampled) data |
| `macro_f1(labels=L)` | mean F1 over `L`. FPDB in-distribution uses `L` = all 8 (matches sklearn on the full test set, since every class is present). External sets use `L` = classes present in that dataset. |
| `per_class_recall`, `per_class_precision` | standard; NaN if undefined |
| `other_recall` | recall of `Other` (FPDB only) |
| `rejection_rate` | share of images with `pred_full ∉ present_classes(dataset)` (i.e. predicted Other, Cervix, or a class absent from the set) |
| `error_decomposition` | among errors: `rejection_other` (pred Other) / `absent_class` (pred a non-Other class absent from the set) / `confusion` (pred a present but wrong class). Shares sum to 1. |
| `ece(conf, correct, n_bins=15)` | equal-width bins on confidence; Σ (n_b/N)·|acc_b − conf_b| |
| `auroc(score, y)` | `sklearn.metrics.roc_auc_score`; NaN if one class |
| `risk_coverage(conf, correct)` | sorted by confidence descending: coverage grid 0.05…1.0, risk = 1 − acc; also `coverage_at_accuracy(0.95)` |
| `cramers_v(table)` | bias-corrected (Bergsma 2013) |

### Tests (`tests/test_stats.py`, `tests/test_metrics.py`): hand-computed answers
- Balanced accuracy, macro-F1 and ECE on 10-row toy arrays, values computed by hand in the test docstring. Macro-F1 equals `sklearn.f1_score(average="macro")` when all classes are present.
- **Cluster vs image CI:** 20 clusters × 10 images, with correctness constant within a cluster (half the clusters all-correct). The cluster CI must be ≥ 2× wider than the image CI.
- Paired bootstrap of a dataframe against itself: Δ = 0, CI = [0, 0], p = 1.
- Paired with `df_b` = `df_a` with 10% of rows flipped to wrong: Δ < 0 and CI excludes 0.
- Fast path and general path give identical replicate distributions for accuracy (same seed).
- `error_decomposition` shares sum to 1; `rejection_rate` on a hand example.
- Wilson CI for k=0, n=10 matches the closed form.

## R2.2 `src/eval/predict.py` (NEW)

```python
def predict(ckpt: Path, dataset: str, out_dir: Path, *, preprocess_override: str | None = None,
            perturb: Callable[[np.ndarray, dict], np.ndarray] | None = None,
            batch_size=64, num_workers=4, save_features=False, amp=False) -> Path
```
- `dataset ∈ {fpdb_train, fpdb_val, fpdb_test, hc18, ucl, african}` resolves to a manifest through a registry in `src/eval/datasets.py`. `fpdb_train` is allowed **only** with `save_features=True` for probes (R5 FM probe); this is asserted.
- Model, image size, normalisation and **preprocess mode** come from the checkpoint config via `src/models/loading.py`. `preprocess_override` and `perturb` exist for R3 E5 and R5 cross-preprocessing analyses. When either is used, `out_dir` must include a suffix such as `fpdb_test__perturb-inpaint`, so files never collide.
- `perturb(gray_uint8, meta) → gray_uint8` is applied **before** the geometric transform, on the native-resolution grey image. `meta` carries the masks (R1 npz) and the `image_id`.
- **Default precision fp32** (`amp=False`). Legacy evaluation was fp32; this keeps G2 reproducible.
- `model.eval()`, `torch.no_grad()`, `cudnn.benchmark=False`.

### Parquet schema: `results/preds/<run_id>/<dataset>[__<variant>].parquet`
| column | dtype | notes |
|---|---|---|
| `image_id` | string | unique |
| `patient_id` | string | dataset-qualified |
| `dataset` | category | |
| `source_machine` | category | FPDB machine, or external `device` |
| `operator` | category | FPDB only, else `NA` |
| `centre` | category | externals |
| `true_label` | category | fine label (FPDB) or collapsed (externals) |
| `true_collapsed` | category | |
| `prob_0..prob_7` | float32 | softmax, fp32 |
| `logit_0..logit_7` | float32 | needed for temperature scaling (R5 calibration) |
| `pc_Head..pc_Other` | float32 | collapsed summed probs |
| `pred_fine` | category | 8-way argmax |
| `pred_full` | category | collapse(`pred_fine`); **primary scoring** |
| `pred_full_summed` | category | argmax over 6 summed collapsed probs |
| `pred_restricted` | category | argmax of `pc_*` over classes present in the dataset |
| `correct_fine` | bool | FPDB only (NA for externals) |
| `correct_full`, `correct_restricted` | bool | |
| `conf_max8`, `conf_full_summed` | float32 | |
| `run_id`, `ckpt_sha256`, `git_hash`, `preprocess_mode`, `variant` | string | provenance (constant per file) |

Side files: `<dataset>.meta.json` (config snapshot, timings, counts, versions); when `save_features`, `<dataset>.features.npy` (float16, N×D) plus `<dataset>.features_ids.txt`, holding `forward_head(forward_features(x), pre_logits=True)` (timm API) in the same row order as the parquet.

CLI: `python -m src.eval.predict --ckpt checkpoints/convnext_tiny/best.pt --dataset fpdb_test --run-id legacy_convnext_s42 [--features] [--preprocess ...]`.
Must stay dataset-agnostic: a `--manifest <csv> --dataset-name <name>` path also works for any CSV with `image_path, image_id, patient_id, plane_label_collapsed`. This is the F1 compatibility obligation (02 §9). No site-specific code.

Tests (`tests/test_predict_schema.py`): a tiny timm model (`resnet10t`, `pretrained=False`, 8 classes) saved as a fake checkpoint in `tmp_path`, plus a 6-row synthetic manifest of generated PNGs. Check the schema and dtypes; that `pc_*` rows sum to 1 ± 1e-5; that `pred_restricted` is always within present classes; that `pred_full == collapse(pred_fine)`; and that a features file is written with the right shape.

## R2.3 `src/eval/external.py` (NEW)

```python
def score_dataset(df) -> dict           # all metrics with Estimates: acc_full, bal_acc_full, bal_acc_restricted,
                                        # rejection_rate, error_decomposition, per-class recall (full & restricted)
def score_pooled(dfs: dict[str, df]) -> dict   # concatenated; clusters remain dataset-qualified
def by_stratum(df, col) -> pd.DataFrame # per device / centre / machine
def rejection_vs_confusion_table(...) -> pd.DataFrame
```
`present_classes(dataset)`: fpdb → all 6 collapsed (and all 8 fine); hc18 → {Head}; ucl → {Head, Abdomen, Femur}; african → {Head, Abdomen, Femur, Thorax} overall, but **per centre** Ghana and Uganda lack Thorax. The restricted argmax uses the *dataset-level* set; per-centre tables note the missing class.
> Balanced accuracy on HC18 alone is just Head recall. Report it anyway; the pooled external endpoint is the primary one.

## R2.4 Re-evaluate the six frozen checkpoints — `scripts/research/r2_reevaluate_existing.py`

- Checkpoints: `convnext_tiny, tf_efficientnetv2_s, repvgg_a1, repvgg_a2, mobilenetv3_large_100, efficientnet_lite0` → `run_id = legacy_<short>_s42`.
- Datasets: `fpdb_val, fpdb_test, hc18, ucl, african`, with `--features` on `fpdb_test` and on the externals (needed for E3).
- Outputs:
  - `results/tables/R2_existing_models.md` and `results/json/R2_existing_models.json`:
    1. In-distribution: macro-F1 (8-way), balanced accuracy, Other recall, each [cluster CI]. Extra column for convnext only: image-level CI (legacy method), to document the CI correction.
    2. Per machine (Aloka / E6 / S10 / Other): macro-F1 and Other recall [CI].
    3. External per dataset and pooled (HC18+UCL+African): acc_full, bal_acc_full, bal_acc_restricted, rejection_rate [CI], and error decomposition.
    4. African per centre: acc_full, rejection rate (n is small, so show CIs).
  - `results/figures/R2_rejection_vs_confusion.pdf`: stacked bars per dataset × backbone.
- Pairwise backbone comparison (paired cluster bootstrap) on pooled external bal_acc_full, ranked. Re-evaluates the "repvgg_a1 winner" claim: report it as supported or withdrawn.

## R2.5 Gate G2 — `docs/research/GATE_G2.md`
| check | target | tolerance |
|---|---|---|
| convnext_tiny FPDB-test macro-F1 (8-way) | 0.8927 | ± 0.005 |
| convnext_tiny HC18+UCL `acc_full` (legacy collapse(argmax) scoring) | 83.6% | ± 0.3 pp |
| HC18 / UCL separately | 81.4% / 88.9% | ± 0.5 pp |
| Head errors predicted Other (HC18+UCL) | 180 of 194 | ± 3 |
| Every other backbone: in-dist macro-F1 vs `checkpoints/<bb>/classification_report_TEST.txt` | match | ± 0.005 |

**If G2 fails, stop and diagnose in this order:**
(1) Label mapping and class order; (2) path normalisation (rows silently dropped?); (3) transform parity: compare the tensor for 5 images against legacy `evaluate_cross_device.py`, run in the same environment; (4) AMP vs fp32; (5) library drift: OpenCV is now 5.0.0 and albumentations 2.0.8 (resize interpolation or normalisation changes). If (5) is the cause, document the drift, re-baseline all legacy numbers under the current environment, and record the decision in GATE_G2.md. Do not "fix" the numbers to match.

**R2 acceptance:** `tests/test_stats.py`, `tests/test_metrics.py` and `tests/test_predict_schema.py` green; 6 × 5 parquet files exist; G2 passed or re-baselined with a documented reason; legacy `scripts/evaluate_cross_device.py` archived (or reduced to a thin wrapper over `predict` + `external`).
