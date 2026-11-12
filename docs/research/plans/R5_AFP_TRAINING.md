# R5 — Anatomy-Focused Training Protocol (RQ4; starts only after Gate G4)

**Goal:** test whether a simple, deployable training protocol (sector crop and mask, overlay handling, acquisition-balanced sampling, device-style augmentation) reduces shortcut reliance and narrows the external-validity gap, under a **pre-registered** primary endpoint.
**Effort:** about 2–3 weeks wall-clock. Roughly 24–36 GPU-hours for the 24-run matrix, plus evaluation.
**Inputs:** R1 (masks, overlay params, G1 decision), R2 (predict/stats/external), R3 (E5 perturbations, E3 probes), R4 (saliency pipeline, G3).

---

## R5.0 Gate G4: pre-registration (before ANY R5 training run)

1. Write `docs/research/PREREGISTRATION.md` from the template in R5.9. Commit it, recording the commit hash.
2. **Human:** lodge it on OSF (osf.io → Registrations → "OSF Preregistration"), then paste the registration DOI or URL into `PREREGISTRATION.md` and `GATE_G4.md`.
3. `scripts/research/r5_run_matrix.py` refuses to start unless `docs/research/GATE_G4.md` contains a line matching `^OSF: https?://` and the PREREGISTRATION commit hash is an ancestor of `HEAD`.
4. Anything implemented in R5.1–R5.5 *may* be built and unit-tested before G4. One-epoch smoke runs into `runs/_smoke/` are allowed. **No full training run, and no evaluation of any R5 model on test or external sets, before G4.**

## R5.1 Preprocessing pipeline — `src/preprocess/pipeline.py` (extends the R3 E4 modes)

```python
MODES = ("resize", "sector_letterbox", "sector_letterbox_inpaint",
         "nonanat_only", "overlay_only", "sector_shape_only")     # last three: R3 E4 only
def prepare_native(gray, masks, mode) -> tuple[np.ndarray, np.ndarray]   # (image_uint8, sector_mask) at native scale, mode-specific
def geometric_eval(img_size, mode) -> list[A.BasicTransform]             # resize | LongestMaxSize+PadIfNeeded(fill=0)
def get_train_transform(cfg) / get_eval_transform(cfg) -> A.Compose      # both built from the same geometric_eval()
```
- `resize`: identical to the legacy `A.Resize(S, S)`. **Bit-identical test** against the archived legacy pipeline on 20 FPDB images.
- `sector_letterbox`: zero outside the sector → crop to the sector bbox → `A.LongestMaxSize(S)` → `A.PadIfNeeded(S, S, border_mode=cv2.BORDER_CONSTANT, fill=0, position="center")`. Under albumentations 2.0.8 the pad value argument is `fill`; this is verified by a test that constructs it.
- `sector_letterbox_inpaint`: `inpaint_overlays(marker_mask)` (the R1.5 params, frozen) *before* the sector zeroing. Out-of-sector text is removed by the zeroing anyway.
- **Cache:** `scripts/research/build_cache.py --mode <m>` writes the native-scale prepared image and the cropped sector mask to `data/processed/cache/<mode>/<image_id>.png` and `..._sector.png`, for all FPDB splits and all external sets. Training and eval read the cache, so the image and mask always stay consistent. The cache is keyed by `overlay_params.json` hash plus code version; a stale cache is an error.
- External sets go through the **same** segmentation and detection. African images are mostly `full` sectors (00 §3.4); log the per-dataset `shape_class` distribution in the cache build report.
- Parity tests (`tests/test_transforms_parity.py`, extended):
  - For each mode, the train pipeline with all random augmentations disabled equals the eval pipeline (bit-exact).
  - `to_model_space(mask)` (R4.2) agrees with the transformed image geometry (Dice ≥ 0.98) for every mode.
  - Letterbox output never stretches the aspect ratio: a drawn circle stays a circle within 2% axis ratio.

## R5.2 Caliper augmentation (A2) — `src/preprocess/caliper_aug.py`

`class CaliperAug(A.ImageOnlyTransform)`. It needs the sector mask, so it runs on the **native-scale prepared image, before the geometric transform**, and is implemented as a step in `prepare_native` for training (p-gated with a seeded RNG drawn from the worker's numpy RNG):
- With probability `p` (config `caliper_aug_p`, 0.3 for B2 and AFP), draw 2–4 markers ('+' or '×', arm length 4–10 px so total size 9–21 px, stroke 1–2 px, intensity U(230, 255)). Positions are uniform inside the sector mask, eroded by the marker size. With probability 0.5, add a dotted line of 5–12 dots (spacing 4–8 px) between two of the markers.
- Applied to images of **all classes**, including Other.
- Tests (`tests/test_caliper_aug.py`): pixels change only inside the union of the drawn marker footprints; all markers lie within the sector; p=0 is the identity; deterministic with a fixed seed; the R1.5 detector finds ≥ 80% of the synthetic markers (cross-check of realism).

## R5.3 Acquisition-balanced sampler (A3) — `src/train/sampler.py`

> **Refinement vs `02 §7.3` (deviation, to be fixed in the pre-registration):** B0 uses inverse-frequency class-weighted CE (`class_weights.json`, effectively class-balanced). If A3 used `w = p(y)/p(y|m)` and also switched class weights off, AFP vs B0 would change **two** things at once: decorrelation and class balance. Instead, A3 targets a sampled joint distribution of **p(m) · Uniform(y)**. Machine and class become independent, and class balance matches B0's effective balance. Class-weighted CE is then off for AFP, since balance is handled by sampling.

- Machine groups: `Aloka`, `Voluson E6`, `Other_machine` (= `Other` ∪ `Voluson S10`, 254 training images). Groups with < 30 images in any class pool with the next group. Log the final grouping.
- Weight: `w_i = p̂(m_i) · (1/K_m) / p̂(m_i, y_i)`, with Laplace α=1 on the cell counts, where K_m is the number of classes **non-empty** in machine group m (empty cells cannot be upsampled; the target is renormalised over non-empty cells).
- **Weight cap:** clip `w_i` at 10 × median(w). Report the share of clipped samples and the realised vs target joint.
- `WeightedRandomSampler(weights, num_samples=len(train), replacement=True, generator=torch.Generator().manual_seed(seed))`.
- Training log: realised class × machine-group prevalence over epochs 1–2. **Assert** every cell is within ±5 pp of its target (absolute, as a share of the machine group).
- Expected multipliers from current train counts: Aloka-Other 1,855 → ~357 per epoch (downsampled); Aloka-ventricular 54 → ~357 (×6.6); Other_machine-Other 11 → ~32 (×2.9).
- Sensitivity run: one seed of AFP with `use_class_weights: true` **and** the sampler on (double reweighting), reported as sensitivity.
- Tests (`tests/test_sampler.py`): on synthetic metadata with a known confounded joint, draw 200k samples and check the realised joint is within 1 pp of target; empty cells get no samples; the cap is applied.

## R5.4 Device-style augmentation (A4)

Behind `style_aug: true`, appended after the geometric transform and before normalisation (albumentations **2.0.8** parameter names):
```python
A.RandomGamma(gamma_limit=(80, 120), p=0.5),
A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),   # replaces the 0.2/0.2 B0 entry
A.GaussianBlur(blur_limit=(3, 5), p=0.2),
A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=False, elementwise=True, p=0.3),
A.Downscale(scale_range=(0.5, 0.9), p=0.2),
A.CLAHE(clip_limit=2.0, p=0.2),
```
Horizontal flip and affine are unchanged from B0. A test constructs this list (so an API change fails loudly) and checks the output dtype and shape.

## R5.5 `train.py` changes (on top of R0)

- Config keys (all with defaults so legacy configs still load): `preprocess_mode`, `caliper_aug_p`, `sampler ∈ {none, acquisition_balanced}`, `use_class_weights`, `style_aug`, `seed`, `run_root: runs/`, `run_id` (derived if absent).
- Outputs → `runs/<run_id>/`: `best.pt` (config embedded, including `git_hash` and `cache_hash`), `config.yaml`, `train_log.txt`, `env.txt` (`pip freeze`), `val_preds.parquet` (the R2 schema, written at the best epoch by `predict()`, so it uses exactly the eval code path), and `sampler_report.json`.
- Determinism: seed torch, numpy and random; `worker_init_fn` seeds each worker; `torch.backends.cudnn.benchmark = False`; `torch.use_deterministic_algorithms(True, warn_only=True)`. AMP stays **on** for training (as in B0), and evaluation is fp32 (R2).
- Resume-safe: if `runs/<run_id>/best.pt` and `DONE` exist, skip. Write `DONE` only after `val_preds.parquet` is written.
- Leakage guard and test-path assertion (R0.4) unchanged.
- Early stopping and model selection: val macro-F1 (8-way), patience 8, max 60 epochs. The **same for all arms** (pre-registered).

## R5.6 Configs and run matrix

`configs/research/{b0,b1,b2,afp}_{convnext_tiny,tf_efficientnetv2_s}.yaml`, each inheriting the B0 backbone hyperparameters verbatim (lr, batch, wd, scheduler, image size, normalisation):

| Arm | preprocess_mode | caliper_aug_p | sampler | style_aug | use_class_weights |
|---|---|---|---|---|---|
| B0 | resize | 0 | none | false | true |
| B1 | sector_letterbox | 0 | none | false | true |
| B2 | sector_letterbox_inpaint | 0.3 | none | false | true |
| AFP | sector_letterbox_inpaint | 0.3 | acquisition_balanced | true | false |

> If G1 decided "overlays secondary", B2 and AFP keep A2 unchanged (it is cheap and harmless), but the write-up de-emphasises the overlay interpretation. Record this in the pre-registration either way.

- Seeds {11, 22, 33} → 24 runs. `run_id = <arm>_<convnext|effv2s>_s<seed>`.
- `scripts/research/r5_run_matrix.py`: runs sequentially, ordered by seed first (all arms of seed 11, then 22, then 33), so a partial matrix is still balanced. Each run is a subprocess with the log tee'd. Checks free VRAM before each run. Prints the ETA from the completed runs.
- Run it in the background (Claude Code `run_in_background`), and check `runs/*/DONE` rather than polling logs.
- Sanity check after the first B0 seed: the B0 convnext seed-11 val macro-F1 should fall within the range of the legacy seed-42 val macro-F1 ± 0.02. If not, stop and diagnose before continuing.

## R5.7 Evaluation of all runs — `scripts/research/r5_evaluate_matrix.py`

For each run, with **its own preprocessing** (from the checkpoint config):
1. `predict` on `fpdb_val` (calibration), `fpdb_test`, `hc18`, `ucl`, `african`, with `--features` on `fpdb_test` and the externals.
2. Shortcut indices for P4b:
   - **Machine decodability within class** (R3 E3 protocol) on `fpdb_test` features. This is the primary shortcut index and is comparable across arms.
   - **Marker sensitivity:** flip rate of predictions between `sector_letterbox` and `sector_letterbox_inpaint` inputs, on images with markers. For B0 use `resize` with and without `a_marker` inpainting. Contrast with the area-matched control (R3 E5 `c_control`).
   - **EPG** (R4 pipeline, Grad-CAM true and predicted targets) on the landmark images; `out_sector_frac` is structurally ≈ 0 for sector-masked arms and is reported, not tested.
3. Calibration: fit a temperature on `fpdb_val` logits per run (LBFGS on NLL). Report ECE (15 bins, on `conf_full_summed`) in- vs out-of-distribution, before and after scaling.
4. Selective prediction: risk–coverage curves, plus the coverage needed for 95% accuracy per external set.

## R5.8 Foundation-model linear probe (secondary) — `scripts/research/r5_fm_linear_probe.py`

1. **Licence first.** Read the LICENSE in `github.com/BioMedIA-MBZUAI/FetalCLIP` and the Hugging Face model card. Record the licence text and URL in `docs/research/DATA_CARDS.md`. Non-commercial research use is acceptable; if no licence is stated or use is not permitted → USFM (check its licence the same way) → else **skip and log**.
2. Frozen image encoder, fp16 inference (8 GB VRAM; ViT-L/14 at 224 fits at batch ≤ 32). Use the model's own normalisation and resize; the "B0 vs AFP input" contrast feeds `resize` vs `sector_letterbox_inpaint` prepared images into that pipeline.
3. Probe: multinomial logistic regression on `fpdb_train` features. C is chosen on `fpdb_val` from {0.01, 0.1, 1, 10} by macro-F1 (val only). Evaluate on test and externals with the same scoring.
4. Report as a secondary table. It is not part of the primary endpoint.

## R5.9 Primary analysis — `scripts/research/r5_primary_endpoint.py` (exactly as pre-registered)

**Primary endpoint:** pooled external (HC18 + UCL + African) **full-argmax balanced accuracy** over the collapsed classes present in the pool {Head, Abdomen, Femur, Thorax}.
- Comparison: AFP vs B0, **per backbone** (two co-primary comparisons), seed-averaged: `stats.paired_multi_seed`, the same patient-cluster draw for all 6 runs in each replicate, B = 2,000, seed 20261109.
- Multiplicity: Holm–Bonferroni over the 2 backbones at family-wise α = 0.05. Report 95% CIs and Holm-adjusted p.
- Success (per backbone): Holm-adjusted p < 0.05 **and** Δ > 0.
- **Non-inferiority** (in-distribution): Δ macro-F1 (8-way, FPDB-test), with the lower bound of the one-sided 95% interval (= 5th percentile of the bootstrap Δ) > −0.02.
- Known property, stated in advance: Thorax contributes only 75 African images from 3 centres but carries 1/4 of the balanced-accuracy weight, so the CIs will be wide. Pooled accuracy is a secondary endpoint.
- Secondary (no multiplicity correction; labelled exploratory): B1 vs B0 and B2 vs B0 (an incremental ladder), per-dataset and per-centre results, rejection rate, error decomposition, ECE, shortcut indices (P4b), EPG, the confirmatory P3b association on R5 models, and the FM probe.
- Output: `results/json/R5_primary.json`, `results/tables/R5_primary.md`, and the forest plot `results/figures/R5_forest.pdf` (B1/B2/AFP − B0 per backbone, pooled and per dataset).

### `PREREGISTRATION.md` template (fill in, commit, lodge)
1. Title, authors, date; links to the proposal and to commit hashes of the code that will run.
2. Background and the specific hypotheses P4a (primary), P4b and P4c (secondary), and the confirmatory P3b on R5 models.
3. Data: exact datasets and counts (from R1 outputs), exclusions (QC-flagged landmarks, failed sector segmentation), FPDB splits frozen (SHA-256 values).
4. Arms, backbones, seeds, recipe (verbatim config tables above), sampler target definition, model-selection rule (val macro-F1, patience 8).
5. Primary endpoint, comparison, CI method, multiplicity, success criterion, non-inferiority margin.
6. Secondary and exploratory analyses (listed).
7. Stopping rules: none for efficacy; runs that crash are re-run with the same seed; a run diverging (val macro-F1 < 0.5 at the end) is re-run once and reported.
8. Deviations policy: any deviation is logged in RESEARCH_LOG and in a "Deviations" section of RESULTS.md.
9. What results R1–R4 were already known at registration time (transparency about the legacy-model findings).

---

## R5 acceptance
- G4 recorded **before** the timestamp of the first full-run directory (the auditor checks `GATE_G4.md` commit time < `runs/*/train_log.txt` first-line time).
- 24 runs with `DONE`; all evaluated; parquets present.
- FM probe done or skip logged with the reason.
- `R5_primary.json` produced by the unmodified pre-registered script (hash recorded).
- Tests: `test_caliper_aug.py`, `test_sampler.py`, extended `test_transforms_parity.py`, `test_pipeline_modes.py` green.
