# R3 — Confounding and Reliance Audit (RQ1–RQ2, existing checkpoints)

**Goal:** answer, with CIs, (RQ1) whether acquisition information predicts the label in FPDB and is encoded in trained features, and (RQ2) whether trained models *use* non-anatomical content. The first outputs (E1, E2, E6) feed **Box 1 of the JAPI review**.
**Effort:** 5–7 days; about 4–9 GPU-hours (E4 training).
**Inputs:** R1 (metadata parquet, masks, G1 decision), R2 (stats, predict, legacy parquets with features).
**Models:** the six frozen checkpoints (`legacy_*_s42`). Primary model for figures: `convnext_tiny`. Secondary: `tf_efficientnetv2_s`. The other four go in supplementary tables.

All scripts live in `scripts/research/`; reusable logic lives in `src/audit/`. Each script writes `results/json/<E>.json` (machine-readable, every number with CI) and `results/tables/<E>.md`. Figures go to `results/figures/<E>_*.pdf` (+ `.png`).

**Execution order:** E1 → E2 → E6 → *(Box 1 ready)* → E3 → E5 → E4.
E7 (saliency out-of-sector fraction) needs the saliency API and is executed in R4.

---

## E1 — Descriptive confounding tables — `e1_confounding_tables.py`

Inputs: `acquisition_metadata.parquet`.
- Tables (counts and row %): machine × class (8 and collapsed), operator × class, split × machine, (machine, operator) × Other share, and machine × image size (mode, % of images at the modal size).
- Association: Cramér's V (bias-corrected) for machine × class, operator × class, and **image size bucket × class**. 95% CI by patient-cluster bootstrap (`stats.cluster_bootstrap` with a metric that rebuilds the contingency table).
- Contrasts reported as numbers:
  - Train+val Other share Aloka − E6, with CI (expected point 0.651 − 0.136).
  - Test S10 Other share vs the train+val prior for Voluson (expected 0.516 vs 0.136).
- Figure: `E1_confound_mosaic.pdf`. Stacked bars of class share per machine, train+val vs test side by side; S10 highlighted; n above each bar.
- Assertion: the reproduced shares match 00 §3.1 to 3 dp.

## E2 — No-pixel metadata model — `e2_metadata_only_model.py`

- Model: `sklearn.ensemble.HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)`, with categorical features declared natively. **Hyperparameters fixed in advance; no tuning on test.** (Optional sanity check: GroupKFold(5) on train+val, report only.)
- Fit on train+val; evaluate on test.
- Feature sets, each a separate model:
  | set | features |
  |---|---|
  | `prior` | machine, operator |
  | `geometry` | width, height, aspect, sector_area_frac, sector bbox (normalised), sector_shape_class, touches_* |
  | `intensity` | mean, std, frac_nonzero |
  | `overlay` | text_score, caliper_score, overlay_area_frac (only if G1 validated the detector) |
  | `all` | union |
- Tasks: (i) Other vs standard plane → AUROC [CI]; (ii) 8-class → balanced accuracy [CI] (chance 0.125); (iii) collapsed 6-class balanced accuracy.
- Evaluation subsets: `all_test`; `seen_machines` (excluding S10; expected AUROC for `prior` ≈ 0.735 on 4,200 images); `S10_only` (where the prior is wrong).
- Interpretation guardrail, written into the table footnote: geometry is nearly collinear with machine (00 §3.1). `geometry` ≈ `prior` performance is **expected** and means "the image frame alone carries the device prior". It does not mean geometry adds independent label information.
- Figure: `E2_metadata_roc.pdf`. ROC curves for `prior`, `geometry`, `all` on `seen_machines` and `all_test`, with the AUROC CI in the legend.

## E6 — Stratified in-distribution performance (tests P1c) — `e6_stratified_performance.py`

Inputs: R2 parquets (`fpdb_test`) for all six backbones, plus metadata.
- Per machine (Aloka, E6, S10, Other machine): macro-F1, balanced accuracy, **Other recall**, **standard-plane recall** (collapsed, non-Other), and the confusion matrix (counts).
- **P1c contrasts** (primary convnext, then all backbones): Other recall S10 − E6 and S10 − Aloka, within the same true class (Other).
  - Use a **joint** cluster bootstrap over all test patients: in each replicate compute every per-machine metric on the resampled data and take differences. This handles patients who have images from more than one machine (report how many such patients exist).
- Same for "standard plane predicted as Other" rate per machine (the false-rejection direction).
- Adjusted model (supplementary): `statsmodels` GEE or GLM logit `correct_fine ~ C(machine) + C(true_label)`, with cluster-robust SEs by patient. Report ORs for machine.
- Overlay strata (only if G1 validated the detector): the same metrics split by `caliper_score > 0`, **within class**. Report n per cell and suppress cells with n < 20.
- Figures: `E6_other_recall_by_machine.pdf` (dot + CI per backbone × machine) and `E6_confusions_convnext.pdf` (4 small row-normalised confusion matrices).

### → Box 1 checkpoint
After E1, E2 and E6 (plus R2's external rejection numbers), run `scripts/research/render_box1.py` → `docs/research/BOX1_ILLUSTRATIVE_ANALYSIS.md` (see R6 §B). This is the first deliverable that the review can use.

## E3 — Feature decodability (P1b) — `e3_feature_probes.py`

Inputs: `results/preds/legacy_*/fpdb_test.features.npy` (fine-tuned) **and** ImageNet-only features. Extract the latter with `src/audit/probes.py::extract_imagenet_features(backbone_name)` (timm `pretrained=True, num_classes=0`, the same eval transform as the checkpoint) → `results/preds/imagenet_<short>/fpdb_test.features.npy`.
- Probe: `make_pipeline(StandardScaler(), LogisticRegression(C=?, max_iter=5000, class_weight="balanced"))`. Pick C by inner `GroupKFold(3)` from {0.01, 0.1, 1, 10}. Outer: `StratifiedGroupKFold(5, shuffle=True, random_state=0)`, grouped by `patient_id`, stratified by target.
- Targets: machine (4-way; the "Other" machine has n=57, so also report a 3-way version without it), operator (4-way).
- **Confound control (essential):** class is correlated with machine, so a probe could decode machine *through* class. Report:
  1. Unconditional machine decoding.
  2. **Within-class** machine decoding: separate probes inside `Fetal_abdomen`, `Fetal_femur`, `Brain_Trans_thalamic` and `Other`, each with enough images per machine. Use E6-style cells with n ≥ 30.
  3. Class decoding **within machine** (sanity check; it should be high).
- Metric: out-of-fold balanced accuracy, with CI by cluster bootstrap over out-of-fold predictions; chance = 1/k.
- Contrast: fine-tuned − ImageNet within-class machine decoding (paired over images) → "does fine-tuning amplify acquisition information?"
- Figure: `E3_decodability.pdf`. Grouped bars (ImageNet vs fine-tuned) × (unconditional, within-class) × backbone.

## E5 — Reliance perturbations on unchanged models (P2b, P2c) — `e5_reliance_perturbation.py`

Implement the perturbations in `src/audit/perturb.py`. Each is a `perturb(gray, meta)` callable passed to `predict(..., perturb=...)` (R2.2), so preprocessing stays identical. They act on native-resolution grey images using the R1 masks.

| ID | Perturbation | Region | Applies to |
|---|---|---|---|
| `a_marker` | inpaint in-sector markers (`marker_mask`) | in-sector | images with marker area > 0 |
| `a_text` | inpaint out-of-sector text/UI components (`text_mask`) | out-of-sector | images with text area > 0 |
| `b_zero` / `b_mean` / `b_blur` | set the whole out-of-sector region to 0 / the image's out-of-sector median / a heavily blurred copy | out-of-sector | all images |
| `c_control` | inpaint **random in-sector patches** with total area = that image's `marker_mask` area. Patches are squares of the median marker-component size, placed uniformly in the sector and not overlapping markers. **5 draws** (seeds 0–4); metrics averaged over draws. | in-sector | same images as `a_marker` |
| `d_pairs` | no perturbation: compare predictions within confirmed caliper pairs (G1) | — | confirmed pairs |

Metrics per perturbation (vs the unperturbed parquet, paired on `image_id`):
- Flip rate (`pred_fine` changed) and flip rate of `pred_full`.
- Mean Δ p(true class), and mean Δ p(Other).
- Δ macro-F1 and Δ Other recall (paired cluster bootstrap).
- Breakdown by machine × collapsed class.

**Primary reliance contrast (pre-declared):** flip rate `a_marker` − `c_control`, paired over images, clustered by patient. Reliance on markers is claimed only if the CI excludes 0.

**Out-of-sector interpretation:** by construction, the out-of-sector region contains no fetal anatomy. Any systematic flip under `b_*` is reliance on non-anatomical content **or** a distribution-shift artefact of the fill. The claim is made only if the effect is **consistent in sign and similar in size across all three fills** (`b_zero`, `b_mean`, `b_blur`). Report all three.

**Directional test for the device prior:** under `b_*`, does p(Other) fall on Aloka images and rise on E6 images? Report mean Δ p(Other) by machine, with CIs.

**`d_pairs`** (only if G1 kept E5d): within each confirmed pair, compare `p(standard)` = 1 − p(Other) and agreement of `pred_fine`. Control: an equal number of near-duplicate pairs (SSIM ≥ 0.9) where **both** have `caliper_score == 0`. Contrast the disagreement rates (cluster bootstrap over patients).

*Optional, exploratory (only if time allows; label it so):* `e_ui_transplant`. Keep the recipient's sector and replace its out-of-sector region with the out-of-sector region of a donor from a different machine, resized to the recipient frame. Measure Δ p(Other) Aloka→E6 vs E6→Aloka.

Figure: `E5_reliance.pdf`. Flip rate per perturbation, with CIs, `a_marker` next to `c_control`, by backbone.

## E4 — Availability: can a model learn the label without anatomy? (P2a) — `e4_availability_train.py`

Training inputs. These are new `preprocess_mode` values implemented in `src/preprocess/pipeline.py`; R5 reuses the same module.

| mode | construction (native resolution, before resize) |
|---|---|
| `nonanat_only` | sector interior filled with the per-image in-sector median grey; out-of-sector pixels kept; **in-sector marker pixels pasted back** on top |
| `overlay_only` | black canvas plus `text_mask ∪ marker_mask` pixels from the original |
| `sector_shape_only` | binary sector mask rendered white on black (isolates frame and sector geometry) |

- Cache the transformed images to `data/processed/cache/<mode>/<image_id>.png` (all splits, built once by `scripts/research/build_cache.py --mode ...`). The dataset class reads the cache when `preprocess_mode` names a cached mode. Training augmentations (flip, affine, brightness/contrast, noise) still apply on top.
- Recipe: identical to B0 convnext_tiny (`configs/convnext_tiny.yaml`): val-set early stopping on macro-F1, class-weighted CE. Configs: `configs/research/e4_<mode>_convnext_tiny.yaml`. **Seeds 11, 22, 33** for `nonanat_only` and `overlay_only`; seed 11 only for `sector_shape_only` (cheap, and expected to equal the prior). Runs go to `runs/e4_<mode>_convnext_s<seed>/`.
- Evaluate on the equally transformed `fpdb_test`. Report balanced accuracy (8-way; chance 0.125), Other-vs-standard AUROC (using 1 − p(Other)), and macro-F1. Multi-seed CI via `stats.multi_seed_summary`. Compare against (i) chance, (ii) the E2 `all` metadata model, and (iii) the full-image B0 model.
- Pre-declared reading: `nonanat_only` well above chance **and** above E2-`prior` means non-anatomical image content carries label information beyond the device prior, for example overlays that differ between standard planes and Other.
- Documented limitation: sector shape itself leaks device information. That is part of what is being measured, and it is why `sector_shape_only` is included.

---

## R3 acceptance
- `results/json/{E1,E2,E3,E4,E5,E6}.json` and the matching `.md` tables exist. Every number has a CI and an n.
- Figures E1, E2, E3, E5, E6 (PDF + PNG).
- New tests: `tests/test_perturb.py`. Each perturbation changes **only** its declared region (pixel-exact outside the region); `c_control` area matches marker area within 5%; perturbations are deterministic given the seed. `tests/test_audit_metadata.py` covers the E2 feature builder on a 10-row synthetic frame.
- `docs/research/RESULTS.md` section "RQ1–RQ2" rendered by `render_results.py` (R6). No hand-typed numbers.
- RESEARCH_LOG entries for each experiment, including deviations (e.g. E5d dropped by G1).
