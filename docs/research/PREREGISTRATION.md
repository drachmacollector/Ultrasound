# Pre-registration: Anatomy-Focused Training Protocol (AFP) for fetal standard-plane classifiers (R5, Gate G4)

**Study:** "Anatomy or Acquisition? Shortcut reliance and anatomical grounding of fetal standard-plane classifiers" (`docs/research/01_RESEARCH_PROPOSAL.md`).
**Date written:** 2026-11-13.
**Authors:** the study PI (name on the OSF record). Implementation assisted by an AI coding assistant.
**Registration status:** written and committed before any R5 training run. To be lodged on OSF ("OSF Preregistration") by the PI; the OSF link goes into `GATE_G4.md`.

**Code that will run** (commit hashes are recorded in `GATE_G4.md` once this file is committed):
- training: `src/train/train.py`, `src/train/sampler.py`, `src/preprocess/{pipeline,caliper_aug}.py`, `src/data/transforms.py`;
- run matrix: `scripts/research/r5_run_matrix.py`;
- evaluation: `scripts/research/r5_evaluate_matrix.py`;
- primary analysis: `scripts/research/r5_primary_endpoint.py`, whose SHA-256 is written into its own output.

---

## 1. Background

R1–R4 audited six fetal standard-plane classifiers. They were fine-tuned on FETAL_PLANES_DB (FPDB) with one recipe ("B0"). The audit found the following.
1. In FPDB, acquisition (machine, operator, frame geometry) is strongly entangled with the plane label.
2. A model with no pixels predicts "standard plane vs Other" well above chance.
3. A model trained on images with the anatomy blanked out still learns the 8-way label far above chance.
4. The classifiers' errors on external devices are mostly rejections of standard planes as "Other".

The study asks whether a simple, deployable protocol reduces reliance on acquisition and narrows the external-validity gap. The protocol is:
- sector crop and mask;
- overlay handling;
- acquisition-balanced sampling;
- device-style augmentation.

## 2. Hypotheses

- **P4a (primary).** AFP improves pooled external balanced accuracy over B0 (HC18 + UCL + African; full-argmax prediction; 4 classes present).
- **P4b (secondary).** AFP lowers the shortcut indices and raises on-anatomy saliency.
  - Shortcut indices: within-class machine decodability from features, and marker sensitivity.
  - On-anatomy saliency: EPG − area on expert-landmark images.
- **P4c (secondary, non-inferiority).** AFP's in-distribution 8-way macro-F1 on FPDB-test is within −0.02 of B0.
- **P3b (confirmatory for R5 models).** On external landmark images (HC18 + UCL), per-image on-anatomy saliency predicts correctness after adjusting for confidence. The specification is in §6.

## 3. Data

All data are public and de-identified (`docs/research/DATA_CARDS.md`).

**Training, validation and test (FPDB).** Frozen patient-disjoint splits:

| split | images | patients |
|---|---|---|
| train | 6,106 | 772 |
| val | 1,023 | 124 |
| test | 5,271 | 896 |

SHA-256 values are in `data/splits/SPLITS_SHA256.txt`, and `tests/test_splits_frozen.py` enforces them. Model selection uses val only.

**External test sets (never trained on, never used for selection):**

| dataset | images | patients | classes |
|---|---|---|---|
| HC18 | 999 | 806 | Head |
| UCL | 424 | 51 | Head, Abdomen, Femur |
| African (5 centres, 5 devices) | 450 | 127 | Head, Abdomen, Femur, Thorax |

**Exclusions.**
- Landmark rows with `qc_flag` are excluded from EPG analyses: FPDB test 5, UCL 2. They are included in a sensitivity analysis.
- No image is excluded from classification endpoints.
- Images whose sector segmentation is empty fall back to the full frame (crop = whole image). Their number is reported.

## 4. Arms, backbones, seeds and recipe

| Arm | preprocess_mode | caliper_aug_p | sampler | style_aug | use_class_weights |
|---|---|---|---|---|---|
| B0 | resize | 0 | none | false | true |
| B1 | sector_letterbox | 0 | none | false | true |
| B2 | sector_letterbox_inpaint | 0.3 | none | false | true |
| AFP | sector_letterbox_inpaint | 0.3 | acquisition_balanced | true | false |

- **Backbones.** convnext_tiny (224 px, batch 64, lr 1e-4) and tf_efficientnetv2_s (300 px, batch 24, lr 1.5e-3). Other hyperparameters are copied verbatim from the B0 configs: AdamW, cosine schedule, weight decay 0.01, AMP. Configs: `configs/research/{b0,b1,b2,afp}_{convnext_tiny,tf_efficientnetv2_s}.yaml`.
- **Seeds.** 11, 22, 33, giving 24 runs. They run seed-major (all arms of seed 11 first).
- **Model selection.** Best val 8-way macro-F1. Early-stopping patience 8, maximum 60 epochs. The same rule applies to all arms.

**Preprocessing.**
- `sector_letterbox`:
  1. zero outside the R1 `sector_clean` mask;
  2. crop to its bounding box;
  3. longest side → input size;
  4. zero padding to a square, centred.
- `_inpaint`: Telea inpainting of the R1 caliper-marker mask (frozen R1.5 detector parameters), applied before zeroing.
- Train and eval use the same geometric transform; this is tested. Caches are keyed by the overlay-parameter hash and the pipeline version.

**A2 caliper augmentation.**
- With p = 0.3, draw 2–4 '+'/'×' markers on the native-scale crop: arm 4–10 px, stroke 1–2 px, intensity 230–255.
- Positions are uniform inside the eroded sector.
- With probability 0.5, add a dotted line of 5–12 dots between two markers.
- Applied to all classes.

**A3 acquisition-balanced sampler.**
- Machine groups: Aloka, Voluson E6, Other_machine (= Other + Voluson S10).
- Target sampled joint: q(m, y) = p(m) · 1/K_m, over classes present in group m.
- Weights: w = q / p̂(m, y), with Laplace α = 1, capped at 10 × the median weight.
- WeightedRandomSampler with replacement, N = |train| per epoch, seeded.
- The realised joint is checked on the draws of epochs 1–2 pooled: every cell must be within ±5 pp of target, as a share of its machine group.
  - Pooling is required: in a single epoch the smallest group draws ~32 images per class.
  - Simulated false-alarm rate with the real training table: 12.7% for one epoch, 0.3% for two epochs pooled.
- Class-weighted cross-entropy is off for AFP; balance comes from sampling. This avoids changing two things at once relative to B0, which is effectively class-balanced through class weights.

**A4 style augmentation.**
- RandomGamma (80–120, p 0.5)
- RandomBrightnessContrast (0.3/0.3, p 0.5)
- GaussianBlur (3–5, p 0.2)
- MultiplicativeNoise (0.8–1.2, p 0.3)
- Downscale (0.5–0.9, p 0.2)
- CLAHE (clip 2, p 0.2)

These replace B0's brightness/contrast (0.2/0.2) and speckle (0.9–1.1) entries. Flip and affine are unchanged.

**Determinism.**
- Seeded torch, numpy and random; seeded workers and generators.
- cudnn.benchmark off; deterministic algorithms (warn-only).
- AMP training; fp32 evaluation.

## 5. Primary endpoint and analysis (P4a)

- **Endpoint.** Pooled external (HC18 + UCL + African) balanced accuracy over {Head, Abdomen, Femur, Thorax}. The prediction is `pred_full`, the collapse of the 8-way argmax. A Cervix or Other prediction counts as an error.
- **Comparison.** AFP vs B0, separately per backbone (two co-primary comparisons), seed-averaged.
  - Inference uses a paired patient-cluster bootstrap: one cluster draw is applied to all 6 runs of a backbone in each replicate.
  - B = 2,000, seed 20261109, percentile 95% CI.
  - Two-sided p = 2 · min(P(Δ* ≤ 0), P(Δ* ≥ 0)).
- **Multiplicity.** Holm–Bonferroni over the 2 backbones, family-wise α = 0.05.
- **Success, per backbone.** Holm-adjusted p < 0.05 **and** Δ > 0.
- **Non-inferiority (P4c).** Δ 8-way macro-F1 on FPDB-test (AFP − B0, seed-averaged, paired cluster bootstrap). Non-inferior if the 5th percentile of the bootstrap Δ (one-sided 95% lower bound) exceeds −0.02.
- **Known property, stated in advance.** Thorax has only 75 African images from 3 centres but carries 1/4 of the balanced-accuracy weight, so CIs will be wide. Pooled accuracy is a secondary endpoint.
- **Code.** `scripts/research/r5_primary_endpoint.py`. Its unit test, `tests/test_r5_primary_endpoint.py`, checks Holm, success and non-inferiority on synthetic data. The script will not be modified after registration.

## 6. Secondary and exploratory analyses

None of these are multiplicity-corrected. All are labelled exploratory except P3b on R5 models.
1. B1 − B0 and B2 − B0 on the primary endpoint (incremental ladder).
2. Per-dataset (HC18, UCL, African) and per-African-centre balanced accuracy, and pooled accuracy.
3. Rejection rate and error decomposition (rejection vs confusion), as in R2.
4. Calibration.
   - Temperature scaling fitted on FPDB-val logits per run (LBFGS on NLL).
   - ECE (15 bins, on the summed collapsed confidence), in- vs out-of-distribution, before and after scaling.
5. Selective prediction: risk–coverage curves, and the coverage needed for 95% accuracy per external set.
6. **P4b shortcut indices.**
   - (a) Within-class machine decodability from penultimate features on FPDB-test. This is the R3 E3 protocol: nested patient-grouped CV linear probe, mean over the 4 classes with ≥ 30 images per machine. It is the primary shortcut index.
   - (b) Marker sensitivity: flip rate between `sector_letterbox` and `sector_letterbox_inpaint` inputs on test images with a detected marker. Letterbox arms only; see §8.
   - (c) EPG − area (Grad-CAM, true and predicted target) on expert-landmark images, R4 pipeline. `out_sector_frac` is structurally ≈ 0 for sector-masked arms; it is reported, not tested.
7. **P3b (confirmatory on R5 models).** On HC18 + UCL landmark images, per backbone × arm, seed-pooled with patient clusters:
   - model: GLM Binomial `correct_full ~ (EPG − area)/0.1 + conf_full_summed + C(dataset) + C(true_collapsed)`, patient-cluster-robust SEs;
   - discrimination: AUROC of EPG − area alone, and ΔAUROC (confidence + EPG, 5-fold GroupKFold out-of-fold) − confidence alone, by paired cluster bootstrap;
   - confirmatory claim: OR > 1 with 95% CI excluding 1 for the AFP models of both backbones;
   - the ground-truth-free monitor (share of the predicted-class Grad-CAM energy outside the sector) is tested the same way on the African set, where it needs no landmarks. R4 showed that predicted-class EPG against the expert mask is favoured by construction for wrong predictions, so it is reported but never used as evidence.
8. Foundation-model linear probe (R5.8), FetalCLIP (CC BY-NC 4.0).
   - Frozen ViT-L/14 image encoder.
   - Multinomial logistic regression on FPDB-train features, with C ∈ {0.01, 0.1, 1, 10} chosen on FPDB-val macro-F1.
   - Inputs: `resize` vs `sector_letterbox_inpaint` prepared images.
   - Same scoring. Reported as a secondary table.
9. Sensitivity run: one seed of AFP with class weights **and** the sampler on (double reweighting).

## 7. Stopping and re-run rules

- No stopping for efficacy.
- A crashed run is re-run with the same seed.
- A run that ends with val macro-F1 < 0.5 (diverged) is re-run once and reported.
- **Sanity check after B0 convnext seed 11:** its val macro-F1 must be within ±0.02 of the legacy seed-42 checkpoint (fp32, from the evaluation parquets). Otherwise the matrix stops for diagnosis before any further run, and the diagnosis is logged.

## 8. Deviations from the original plan, decided before registration

- **Overlays are a secondary axis (Gate G1).**
  - The caliper-marker detector failed validation: train recall 0.20 after one tuning round.
  - Consequently the `_inpaint` step removes only the markers the detector finds; on FPDB most real calipers stay.
  - A2 augmentation and B2/AFP are kept unchanged: cheap, harmless, and calipers are frequent externally. The write-up de-emphasises the overlay interpretation.
  - The B0 marker-sensitivity contrast (`a_marker` vs area-matched control) is not available.
- **Sampler group pooling** uses total group size < 30, not "< 30 images in any class". The latter would merge Other_machine (254 images, 11 'Other') and contradict the plan's own expected multipliers.
- **A4** replaces both B0 photometric entries, not only brightness/contrast, so speckle is not applied twice.
- **Letterbox caches** are stored at native scale, so caliper augmentation draws in real frame pixels.
- **Letterbox quality depends on the R1 sector masks.**
  - G1 sector-outline check on FPDB: 138/150 images OK; device-consensus recheck 38/40.
  - 0.4% of FPDB images have a sector mask covering < 30% of the frame.
  - External images were not audited for sector quality. Spot checks found three failure modes: on some UCL frames, inset thumbnails and measurement tables lie inside the detected sector; African frames are mostly full-frame crops, so in-image text stays; and occasionally the sector is too small and crops anatomy.
  - The protocol is run as specified, without manual masks. This limitation is reported.
- **African sector shapes.** The plan expected African images to be mostly 'full'. In practice the cache build reports fan 361, linear 53 and full 36 (of 450), because the authors cropped to the field of view, not to the sector.
- **Validation predictions** of each run are written by the evaluation predictor to `results/preds/<run_id>/fpdb_val.parquet`, not `runs/<run_id>/val_preds.parquet`.
- **Policy for deviations after registration:** any deviation is logged in `RESEARCH_LOG.md` with a `deviation:` tag, and listed in the Deviations section of `RESULTS.md`.

## 9. What was already known at registration

All R1–R4 results on the **legacy** checkpoints (B0 recipe, seed 42) were known when this was written. No R5 model had been trained or evaluated.

- **R1 (data).**
  - The FP landmark images are pixel-identical to FPDB.
  - HC18 and UCL patient IDs come from filename prefixes.
  - African patient IDs are centre-qualified.
  - The caliper detector failed validation (train recall 0.20), so overlays became a secondary axis (Gate G1).
  - The UI/text detector validated.
- **R2 (legacy evaluation).**
  - The legacy numbers reproduce (Gate G2).
  - External errors are mostly rejections of standard planes as "Other".
- **R3 (confounding and reliance).**
  - 'Other' is 65% of Aloka vs 14% of Voluson E6 training images.
  - A no-pixel model reaches AUROC 0.735 for standard vs Other on seen machines (0.87 with frame geometry).
  - The machine is decodable from features within one class (0.85–0.97), and fine-tuning *reduces* this relative to ImageNet features.
  - Training without anatomy reaches 8-way balanced accuracy 0.48.
  - Removing out-of-sector content changes 0.6–2.5% of predictions.
  - 'Other' recall is 0.35–0.52 on Voluson E6 vs 0.92–0.95 on Aloka.
- **R4 (localisation).** All three saliency methods passed Gate G3.
  - Saliency is on the anatomy above chance for all classes.
  - "Femur lowest" is a CAM-resolution effect.
  - Incorrect predictions have lower on-anatomy saliency.
  - Out-of-sector Grad-CAM energy is below the uniform baseline but higher on Aloka.
  - P3b was not supported for the primary legacy model (convnext Grad-CAM).
  - The ground-truth-free out-of-sector monitor reached AUROC 0.78 vs confidence 0.66 on the African set (exploratory).

These results motivated the AFP arms (A1 sector crop against frame and geometry shortcuts; A3 sampler against the machine × class confound). They do not change the endpoint, which was fixed in the proposal (`01_RESEARCH_PROPOSAL.md`, P4a) before R1.
