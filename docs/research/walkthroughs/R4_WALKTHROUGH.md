# R4 Walkthrough: Anatomy-grounded localisation (RQ3, Gate G3)

**Date:** 2026-11-13 · **Branch:** `main` · **Plan:** `docs/research/plans/R4_LOCALISATION.md`
**Code frozen before any test-image saliency:** `512cede`. A loader fix followed in `2e9c2c2`; it does not change the maps or scores (verified). Gate G3 ran at `2e9c2c2`, recorded in `GATE_G3.md`.
**Status:** ✅ G3 passed for all three methods. The locked run covered 6 backbones × 3 methods × 4 datasets, plus the summaries and the association analysis.

**Compute** (one GPU job at a time):

| step | time |
|---|---|
| G3 | ≈ 50 min |
| Test run | ≈ 75 min |
| Association bootstrap (CPU) | 13 min |

## 1. What was built

| Plan item | Code | Tests |
|---|---|---|
| R4.1 saliency API | `src/models/saliency.py`: `SaliencyEngine`, `compute_saliency`, `SumOfLogitsTarget`, raw (not min-max-scaled) Grad-CAM / Grad-CAM++, batched occlusion | `tests/test_saliency.py`: a hand-built model whose evidence region is known; maps peak inside it; raw-scale check; set target = sum of members; chunking and padding invariance |
| R4.2 anatomy masks | `src/localisation/anatomy_masks.py`: head/abdomen ellipses, femur capsule, `landmark_box`, `to_model_space` | `tests/test_anatomy_masks.py`: Dice ≥ 0.98 at 0°/30°/90°; capsule area within 5%; projection matches the real eval transform; clipped landmarks |
| R4.3 metrics | `src/localisation/metrics.py`: EPG with the area baseline, sector-aware EPG, pointing hit, CAM-box IoU, `fraction_in` | `tests/test_localisation_metrics.py`: hand-computed values |
| R4.6 runner | `src/localisation/runner.py`: per-image metrics for true, predicted and single-sub-plane targets; GT-free metrics (E7); figure and occlusion subsets chosen by correctness only | `tests/test_localisation_runner.py` |
| R4.4 development | `scripts/research/r4_localisation.py --dev` (val only) → `results/figures/R4_dev_grid_val.pdf` | |
| R4.5 Gate G3 | `scripts/research/r4_sanity_randomisation.py` → `GATE_G3.md`, `results/{json,tables}/G3.*`, `G3_*` figures | |
| R4.6 locked run | `r4_localisation.py --run` → `results/preds/legacy_*_s42/<dataset>__saliency-<method>.parquet` (+ `.maps.npz`, 48 figure images) | |
| R4.6 analyses | `scripts/research/r4_summarise.py` → `results/{json,tables}/R4_localisation.*`, `R4_epg_by_class_backbone`, `R4_saliency_grid`, `R4_out_sector_by_machine` | |
| R4.7 association | `scripts/research/r4_association.py` → `results/{json,tables}/R4_association.*`, `R4_association` | |

**Test count:** `python -m pytest` reports all tests passing, including 4 new R4 test files.

## 2. Gate G3 (saliency sanity check)

All three methods pass on both backbones. After re-initialising the head and the last stage, maps no longer resemble the trained map:

| method | median SSIM | median Spearman |
|---|---|---|
| Grad-CAM | 0.25–0.27 | 0.02–0.03 |
| Grad-CAM++ | 0.28–0.31 | 0.09–0.17 |
| occlusion | 0.30 | 0.04 |

The threshold for both measures is < 0.5. So the maps depend on the learned weights and can support the R4 claims (P3c supported).

## 3. Results (95% patient-clustered CIs; the full tables are in `results/tables/R4_*.md`)

**P3a: saliency on the anatomy varies by class and backbone. Supported, but the "lowest for femur" part depends on the method.**
- Grad-CAM and Grad-CAM++ put more energy inside the expert-landmark anatomy than a uniform map would, for every class and backbone. EPG − area on FPDB-test:
  - Head 0.25–0.44
  - Abdomen 0.16–0.61
  - Femur 0.03–0.16
- Femur − Head is negative for every backbone under both CAMs (−0.11 to −0.41).
- Occlusion, at a finer resolution, reverses this for 5 of 6 backbones: Femur − Head is +0.07 to +0.27. Femur EPG − area under occlusion is 0.16–0.41.
- The femur capsule covers about 3% of the input, while a 7×7 CAM cell covers about 2%. The low femur score under CAMs is therefore mainly a resolution limit, not evidence that the models ignore the femur. On the ratio scale (EPG/area, Grad-CAM), femur is 2.4–7.1× against head 1.8–2.5×. Femur is the *highest* for 5 of 6 backbones; the exception is tf_efficientnetv2_s, about equal.
- **Reading:** classifiers attend to the anatomy well above chance. Class differences are real but metric- and method-dependent, which matters for any paper that ranks "explainability by class".

**External vs in-distribution (within anatomy).**
- For most backbones, saliency on external images is equally or slightly less on-anatomy than on FPDB: Δ EPG − area mostly between −0.09 and +0.01.
- The exception is convnext Grad-CAM on HC18 heads: +0.09 (+0.11 under Grad-CAM++).
- There is no sign that external failures come from saliency leaving the anatomy wholesale.

**Correct vs incorrect.** Incorrect predictions have lower on-anatomy saliency for 11 of 12 Grad-CAM backbone × set cells, and for all occlusion cells. Example, HC18 + UCL:

| backbone | Δ (Grad-CAM) |
|---|---|
| effv2s | +0.26 |
| efflite0 | +0.22 |
| convnext | +0.01, n.s. |

**E7 (from R3): out-of-sector saliency, ground-truth-free.**
- Grad-CAM energy outside the fan sector is **below** the uniform-map baseline everywhere: FPDB 0.04–0.14 vs 0.18. The models do not preferentially look at the frame or UI.
- It is consistently higher on Aloka than on Voluson E6 (+0.04 to +0.08), including within the 'Other' class (+0.06 to +0.11). This matches R3 E5: the small reliance on out-of-sector content sits on Aloka images.

**P3b: does on-anatomy saliency predict external correctness beyond confidence? Not supported for the pre-declared primary model; mixed elsewhere (exploratory).**
- **Primary, convnext Grad-CAM, true target:**
  - OR per +0.1 EPG − area: 0.99 [0.86, 1.15]
  - AUROC: 0.52 [0.47, 0.57], against 0.82 for confidence
  - The out-of-fold combined model is *worse* than confidence alone: −0.04 [−0.07, −0.01]
- **Other backbones:** EPG alone discriminates (AUROC 0.57–0.91) and is associated after adjusting for confidence (ORs 1.5–6.5). The increment over confidence is small and significant in only 4 of 18 backbone × method cells (+0.04 to +0.10). Saliency and confidence carry largely overlapping information.
- **Ground-truth-free monitor:** the share of the predicted-class Grad-CAM energy outside the sector, on the 450 African images (no landmarks needed):
  - AUROC 0.78 [0.72, 0.83], against confidence 0.66
  - Δ AUROC of the combined model +0.13 [+0.06, +0.21]
  - OR 0.31 per +0.1
  - This is the most promising deployment signal in R4. It is exploratory and from one model; a confirmatory test on R5 models is pre-registered.

## 4. Deviations from the plan

1. **Raw CAMs.** `pytorch_grad_cam`'s per-image min-max scaling is overridden (`_RawCAM`), so maps are the raw ReLU'd CAM. The plan preferred this when possible.
2. **Exact mask rasterisation.** The plan's `cv2.ellipse` drawing was replaced by pixel-centre rasterisation. `cv2` boundary pixels inflated thin femur capsules by more than 5% and failed the plan's own area test.
3. **Occlusion geometry for EfficientNetV2-S (300 px):** patch 20 / stride 10, keeping the 2:1 ratio of the plan's 16 / 8 at 224 px. Occlusion runs under fp16 autocast. This changes EPG by ≤ 0.0015 relative to the first fp32-chunked implementation on val.
4. **Speed fixes, none of which change the outputs** (all verified):
   - CAM batches are padded to a fixed size (each new input shape cost about 5 s of cuDNN set-up);
   - CAM backward stops at the target layer (other parameters frozen during CAMs);
   - occlusion is vectorised;
   - inputs are loaded by threads, not DataLoader worker processes (about 30 s start-up each on Windows). This was committed as `2e9c2c2` before G3 and before any test saliency.
5. **Overlay fraction** uses the R1 `ui` mask only. The caliper `marker` mask failed G1. The plan's "with vs without detected overlays" split was not run, because UI is on every FPDB frame and calipers are not reliably detected.
6. **Sector-aware EPG** uses mask ∩ sector in both the numerator and the baseline. The plan wrote area(mask)/area(sector).
7. **The plan's "deployable variant" (`epg_pred`) is not deployable.** It still uses the expert mask of the *true* anatomy, so wrong predictions get low EPG by construction (AUROC 0.97). It is reported with that warning, never as evidence. The ground-truth-free monitor is `out_sector_frac_pred`.
8. **Agreement with the R2 predictions.** Saliency forward passes reproduce the R2 argmax for 100% of images, except one repvgg_a1 image (fp32 near-tie) and one mobilenetv3 occlusion image (fp16). Targets always use the R2 predictions.
9. **Zero-energy maps** (Grad-CAM map with no positive part: 4 repvgg_a1 HC18 images, 2 efflite0 HC18 images) have undefined EPG and are excluded, as counted in `R4_localisation.json → coverage`.
10. **Landmark glance (R4.4).** The 24 val dev-grid images were inspected by the vision rater used in G1: 24/24 masks fit the anatomy. The R1 glance had already passed 15/15.
11. **Deferred to R6.** The RESULTS.md "RQ3" section is rendered there, by `scripts/research/render_results.py`, which now exists and is tested.

## 5. Things to know before R5

- `SaliencyEngine` and `runner.compute` work for any checkpoint. For letterbox models, `runner.SaliencyInputs` must read the prepared crop and project masks with the sector bounding box. This is added with the R5 pipeline.
- For R5 claims, use the ground-truth-free `out_sector_frac_pred` as the deployment monitor and the true-target EPG as the grounding measure. Do not use `epg_pred` as evidence.
