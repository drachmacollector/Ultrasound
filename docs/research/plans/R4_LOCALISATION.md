# R4 — Anatomy-Grounded Localisation (RQ3; ends at Gate G3)

**Goal:** measure, against expert-landmark ground truth, whether saliency falls on fetal anatomy. Verify that the saliency methods pass sanity checks. Test whether on-anatomy saliency predicts correctness on external devices.
**Effort:** 5–7 days, with GPU saliency runs of a few hours (occlusion is the expensive part).
**Inputs:** R1.3 landmark tables (val for development, test for reporting; HC18/UCL external), R1 masks, R2 predictor and parquets.
**Datasets with landmarks:** FPDB-test 765 Head (all trans-thalamic) / 350 Abdomen / 371 Femur; HC18 999 Head; UCL 159/130/135. African: **no landmarks**, so only GT-free metrics.

---

## R4.1 Saliency API — `src/models/saliency.py` (NEW; `gradcam.py` keeps the layer map)

```python
METHODS = ("gradcam", "gradcampp", "occlusion")
def compute_saliency(model, x: Tensor[B,3,H,W], targets: list[int | list[int]], method: str,
                     backbone_name: str, **kw) -> np.ndarray[B,H,W] float32   # model-input space
```
- `gradcam` and `gradcampp`: `pytorch_grad_cam.GradCAM` / `GradCAMPlusPlus` with `get_target_layer`. Create the CAM object **once per model** and reuse it over batches (context-manager hooks per call are slow).
  - **No per-image min-max normalisation inside the API.** Return the raw ReLU'd CAM upsampled to H×W. Metrics normalise as needed. (The library's default scaling normalises to [0,1] per image. Keep that if it cannot be disabled, but then document that EPG is scale-invariant, which it is: a ratio of sums.)
- **Multi-class target (Head):** for a Head image the target is the *set* {0,1,2}. Implement a custom target `SumOfLogitsTarget(idxs)` (Grad-CAM of the summed logits). Also store the single-subclass variant (argmax brain sub-plane) for the sensitivity analysis.
- `occlusion`: gradient-free. Patch 16×16 px at model-input resolution, stride 8, fill = the normalised value of the image's in-sector median grey, batched (≥ 64 occluded copies per forward pass). Saliency = drop in the target's softmax probability (or the summed probability for a set target), accumulated per pixel and averaged over the patches covering it.
  - Cost: (224/8)² ≈ 784 forwards per image. Run only on the **600-image stratified subset** (200 per anatomy from FPDB-test, plus 100 HC18 and 100 UCL), at fixed seed. For EfficientNetV2-S (300 px) use stride 10.
- Tests (`tests/test_saliency.py`): a tiny model whose logit is the mean of a fixed image region (a hand-built conv). Grad-CAM and occlusion maps must peak inside that region. The output shape equals the input H×W. `SumOfLogitsTarget` equals the sum of the single-target logits.

## R4.2 Anatomy masks — `src/localisation/anatomy_masks.py` (NEW; replaces archived `derive_bboxes.py`)

All functions work in **native image pixel space** and return uint8 masks of shape `(H, W)`.
```python
def head_ellipse(ofd1, ofd2, bpd1, bpd2, shape) -> np.ndarray
def abdomen_ellipse(tad1, tad2, apad1, apad2, shape) -> np.ndarray     # major = longer of the two axes
def femur_capsule(fl1, fl2, shape, rel_width=0.10) -> np.ndarray       # thickness = max(3, rel_width*|FL|), round caps
def landmark_box(mask, margin=0.10) -> tuple[int,int,int,int]
def anatomy_mask_for_row(row, shape, femur_rel_width=0.10) -> np.ndarray
def to_model_space(mask, *, preprocess_mode, img_size, sector: Sector | None) -> np.ndarray[float32]
```
- Ellipse construction: centre = midpoint of the major-axis landmarks; semi-axes a = |major|/2, b = |minor|/2; angle = `atan2` of the major vector; `cv2.ellipse(..., thickness=-1)`.
- QC (already computed in R1.3): rows with `qc_flag` are **excluded from the primary analysis** and counted. Sensitivity analysis: include them.
- **`to_model_space` must apply the exact geometric transform the model saw.** For `resize`, use `cv2.resize(mask_float, (S, S), INTER_LINEAR)` (a soft mask in [0,1]) and threshold at 0.5 for binary metrics. For `sector_letterbox*` (R5), crop to the sector bbox, letterbox and resize, using the **same function** as the image pipeline (import it; do not re-implement). The parity test enforces this.
- Tests (`tests/test_anatomy_masks.py`):
  - A synthetic ellipse drawn into an image, landmarks taken at its axis endpoints → mask Dice ≥ 0.98 vs the drawn ellipse, at 0°, 30° and 90° rotations.
  - The capsule area is within 5% of `|FL|·w + π(w/2)²`.
  - `to_model_space(mask)` vs transforming an image with that ellipse drawn in through the real eval transform → Dice ≥ 0.98, for both `resize` and `sector_letterbox`.
  - Clipped landmarks still give a valid mask.

## R4.3 Localisation metrics — `src/localisation/metrics.py` (NEW)

```python
def energy_pointing_game(sal, mask) -> float        # Σ relu(sal)·mask / Σ relu(sal); NaN if Σ relu(sal) == 0
def epg_relative(sal, mask) -> dict                 # {"epg", "area_frac", "epg_minus_area", "epg_over_area"}
def pointing_hit(sal, mask, tol_px=0) -> bool       # argmax inside mask (ties: any max inside)
def cam_box_iou(sal, mask) -> float                 # cam_to_bbox(sal) vs landmark_box(mask); 0 if cam_to_bbox is None
def fraction_in(sal, region) -> float               # out-of-sector fraction, overlay fraction, generic
```
- **EPG is always reported alongside its area-fraction baseline.** A uniform saliency map gives EPG = area_frac. Primary metric: `epg_minus_area` (0 = no better than uniform). Secondary: `epg_over_area`.
- Sector-aware variant: `epg_in_sector = Σ sal·mask / Σ sal·sector`, with the baseline `area(mask)/area(sector)`. This separates "looks at anatomy" from "doesn't look outside the fan".
- Tests (`tests/test_localisation_metrics.py`): sal = mask → EPG = 1 and hit = True. Uniform sal → EPG = area_frac and `epg_minus_area` = 0. Sal outside the mask → 0. `fraction_in` with a complementary region pair sums to 1. IoU of identical boxes = 1.

## R4.4 Development on val, then the locked run on test

1. **Development (val only):** run the R4.6 pipeline end to end on the 238 val-split landmark images (126/49/63) for convnext_tiny, all methods. Inspect a 24-image grid (`results/figures/R4_dev_grid_val.pdf`). Fix bugs here. **No test-image saliency is computed until the code is frozen** (commit hash recorded in GATE_G3.md).
2. **Landmark glance** (R1.7 `--mode landmarks`, 15 val images) uses the R4.2 masks. The human answers y/n.

## R4.5 Sanity check (Gate G3) — `scripts/research/r4_sanity_randomisation.py`

Model-parameter randomisation test (Adebayo et al. 2018), cascading:
- Stages for convnext_tiny: (0) trained; (1) head re-initialised; (2) + `stages[3]` re-initialised; (3) + `stages[2]`. For tf_efficientnetv2_s: head, `blocks[5]`, `blocks[4]`.
- Re-initialise with the module's own `reset_parameters()`, or timm `init_weights` where available, else `kaiming_normal_` for convs. Seed 0.
- 200 FPDB-test images, stratified by class (25 per class). Targets are the true class (set target for Head).
- Similarity between the trained map and each randomised map: SSIM (`skimage`, data_range from the trained map), Spearman rank correlation, and HOG-feature Pearson (as in Adebayo). Report the median and IQR at each stage.
- **Pass rule (pre-declared):** median SSIM < 0.5 **and** median Spearman < 0.5 after stage (2) (head + last stage randomised). Methods that fail are reported and **excluded** from R4.6 claims and from the R4.7 association.
- Also data-randomisation lite (optional): skip. It needs label-shuffled retraining, which is out of budget.
- Output: `docs/research/GATE_G3.md` (tables, a figure of example cascades, the pass/fail decision, the code commit hash). Running the sanity check on test images is allowed: it measures the method, not the model, and is not used for selection.

## R4.6 Localisation run — `scripts/research/r4_localisation.py`

For each backbone (6 legacy now; R5 models later) × each method that passed G3:
- Datasets: FPDB-test landmark images (1,486), HC18 (999), UCL (424); occlusion only on the 600-image subset.
- For each image compute saliency for (i) the **true-class target** and (ii) the **predicted-class target** (from the R2 parquet).
- Also, for **all** FPDB-test images (all 8 classes, 5,271) and all externals (including African), compute GT-free metrics with the Grad-CAM true target: `out_sector_frac = fraction_in(sal, 1 − sector)`, `overlay_frac = fraction_in(sal, text ∪ marker)`. **This is E7 from R3.**
- Store per-image stats in `results/preds/<run_id>/<dataset>__saliency-<method>.parquet`, keyed on `image_id`: `epg_true, epg_minus_area_true, epg_in_sector_true, hit_true, iou_true, epg_pred, …, area_frac, out_sector_frac, overlay_frac, femur_width` (femur rows: also at 0.05 and 0.15), `head_target` (`set` | `argmax_sub`).
- Save the raw maps only for the figure subset: 48 images, `float16 .npz`.
- Analyses (`scripts/research/r4_summarise.py` → `results/json/R4_localisation.json`):
  - P3a: mean `epg_minus_area` and hit rate by anatomy × backbone × method, with cluster CIs. Femur vs Head contrast. Images with vs without detected overlays (within anatomy).
  - In-distribution vs external: the same metrics on HC18/UCL vs FPDB-test, within anatomy.
  - Correct vs incorrect predictions: `epg_minus_area_true` difference (descriptive).
  - Out-of-sector saliency by machine and class (E7), and on African centres.
- Figures: `R4_epg_by_class_backbone.pdf` (violin + mean with CI), `R4_saliency_grid.pdf` (image | landmark mask outline | Grad-CAM | EPG value; correct and incorrect; FPDB vs HC18/UCL; public data only, with licence credit in the caption).

## R4.7 Association: does on-anatomy saliency predict external correctness? (P3b) — `scripts/research/r4_association.py`

- Data: HC18 + UCL images (they have landmarks), one model at a time. Primary model: convnext_tiny, Grad-CAM, true target.
- Model: `statsmodels` GLM Binomial `correct_full ~ epg_minus_area_true + conf_full_summed + C(dataset) + C(true_collapsed)`, `cov_type="cluster"`, groups = patient_id. Report the OR per +0.1 EPG with CI.
- Discrimination:
  - AUROC of `epg_minus_area_true` alone, with CI.
  - AUROC of `conf_full_summed` alone.
  - AUROC of a logistic model with both (fit by 5-fold GroupKFold out-of-fold predictions, so there is no in-sample optimism).
  - ΔAUROC (both − confidence alone) via **paired** cluster bootstrap.
- Note the target-leakage caveat. `epg_true` uses the *true* label, which is unknown at deployment. Also report the deployable variant with `epg_pred` (predicted-class saliency). For African images (no landmarks), use `out_sector_frac_pred` as a GT-free monitor and test the same association.
- In-distribution analogue on the FPDB-test landmark images.
- Pre-declared status: **exploratory** for legacy checkpoints; **confirmatory** for R5 models (written into the pre-registration).
- Figure: `R4_association.pdf` (external accuracy by EPG decile, with CIs; ROC curves).

---

## R4 acceptance
- Tests: `test_saliency.py`, `test_anatomy_masks.py`, `test_localisation_metrics.py` green.
- `GATE_G3.md` with the pass/fail decision per method and the frozen code hash.
- Saliency parquets for 6 backbones × passing methods × datasets; `R4_localisation.json`, `R4_association.json`; figures.
- RESULTS.md section "RQ3" auto-rendered.
