# R1 Walkthrough: Data Foundation (Gate G1)

**Date:** 2026-11-12 · **Branch:** `main` (base `0de1ec0`) · **Plan:** `docs/research/plans/R1_DATA_FOUNDATION.md`
**Status:** ✅ complete; Gate G1 decided (`docs/research/GATE_G1.md`, PI sign-off pending). Uncommitted.
**Tests:** `python -m pytest` → **90 passed**.

## 1. What was built

| Plan step | Implementation | Output |
|---|---|---|
| R1.1 labels/paths | `src/data/labels.py` (collapsed taxonomy, `COLLAPSE_MATRIX`, `collapse_probs`), `src/data/paths.py` (path normalisation, FPDB stem parsing, numeric prefixes) | `tests/test_labels.py` |
| R1.2 external manifests | `scripts/build_external_manifests.py`; `ExternalDataset` replaces `CrossDeviceDataset`; `FocalPlanesDataset(return_meta=True, prepare=...)`; `run_sanity_checks.py` repointed; legacy `build_cross_device_manifest.py`, `evaluate_cross_device.py` and `cross_device_manifest.csv` archived | `data/processed/external/{hc18,ucl,african,external_all}.csv`, `results/tables/R1_external_counts.md`, `data/raw/african_planes/README.md` |
| R1.3 landmark GT | `scripts/link_fp_landmarks.py` (identity check, QC geometry, val+test only) | `data/processed/landmarks/{fpdb,external}_landmarks.parquet`, `results/{tables,json}/R1_landmark_linkage.*` |
| R1.4 sector segmentation | `src/preprocess/sector.py` (background-level threshold, open/close, largest CC hull; `build_consensus`, `refine_with_consensus`) + `scripts/build_sector_consensus.py` | 12 train-only consensus cores in `data/processed/masks/consensus/` |
| R1.5 overlays | `src/preprocess/overlays.py` (UI = out-of-sector bright + glyphs-on-background; markers = top-hat + cross template matching; Telea inpainting), frozen `OverlayParams` | `data/processed/audit/overlay_params.json` |
| R1.6 metadata + masks | `src/audit/metadata.py`, `scripts/build_acquisition_metadata.py` (two passes, parallel, resumable) | `acquisition_metadata.parquet` (12,400), `external_metadata.parquet` (1,873), 14,273 mask npz files (bit-packed, 96 MB), `results/json/R1_metadata_summary.json`, `R1_exact_duplicates.json` |
| R1.7 binary audit | `scripts/audit_sheets.py` (stratified sample builder, contact-sheet rendering for overlay/sector/re-check/landmark/external audits, rating import) | `data/processed/audit/overlay_audit_{sample,train,test}.csv`, `external_audit*.csv`, `ratings_raw/*.txt` (all tracked) |
| R1.8 caliper pairs | `scripts/find_caliper_pairs.py` | `near_duplicate_pairs.csv`, `caliper_pairs.csv` (empty), `results/json/R1_caliper_pairs.json` |
| R1.9 Gate G1 | `scripts/research/g1_report.py` (applies the pre-specified rules) | `docs/research/GATE_G1.md`, `results/json/G1.json` |

## 2. Results

**External sets.** HC18 999 images / 806 patients; UCL 424 / 51; African 450 / 127 (thorax only in Algeria, Egypt, Malawi). Each African centre has one fixed image size; Egypt images are RGB.

**Landmarks.**
- All 3,090 FP images are pixel-identical to FPDB.
- Kept: test 765 Head / 350 Abdomen / 371 Femur; val 126 / 49 / 63.
- Head QC flag rate 0% (BPD shorter than and perpendicular to OFD), Abdomen 1.4%, Femur 0%.
- Landmark glance: 15/15 correct.

**Confounding reproduced exactly.** Train+val "Other" share: Aloka 0.6506, E6 0.1357. S10-test: 0.5163. Modal image sizes differ by machine (Aloka 640×392, E6 959×661, S10 1137×787, Other 1137×787).

**Binary audit** (rated from contact sheets; see §3.1):

| | FPDB (300 images) | External (42 images, 6 per centre) |
|---|---|---|
| Calipers / measurement marks | 9/300 = 3.0% [1.6, 5.6]; standard 1.7% vs Other 6.5% (test) | Ghana 6/6, Uganda 6/6, UCL 4/6; others 0/6 |
| On-screen text / UI | 300/300 = 100% | Egypt 0/6, Ghana 0/6; others 5–6/6 |

**Detector validation** (image level):

| Detector | Precision | Recall |
|---|---|---|
| UI/text, test | 1.00 | 0.97 |
| Calipers, train (tuned) | 1.00 | **0.20** |
| Calipers, test | 0.00 | 0.00 |

FPDB calipers are faint, small, dark or dotted. The high-precision template detector misses almost all of them. On UCL and Ghana it fires frequently (28% and 97% of all images).

**Sector segmentation.**
- First pass: Aloka 91%, Other 95%, **E6 88%**, S10 97% judged correct.
- After the consensus prior, on 40 new images: **E6 93%**, Aloka 100%.

**Near-duplicates and exact duplicates.**
- FPDB test: 436 within-acquisition pairs with SSIM ≥ 0.90, involving 658 images.
- Across all splits: **863 byte-identical images** in 384 groups, always within one patient and split. **8 test-split groups give contradictory labels** for the same file (e.g. abdomen vs cervix).
- `PlaneN` groups are label-heterogeneous in 315 of 943 test groups, which confirms that PlaneN is an acquisition index, not a class.

**Gate G1 decisions** (pre-specified rules; details in GATE_G1.md):

| Rule | Outcome |
|---|---|
| Caliper prevalence difference ≥10 pp | −4.8 pp → **overlays are a secondary axis**; emphasise E2/E3/E6/A3 |
| Caliper detector recall ≥0.8 | 0.20 → FAIL → E5a restricted to detector∧audit-positive images, of which there are **0** → **E5a infeasible** |
| Confirmed caliper pairs ≥20 | 0 candidates → **E5d dropped** |
| Sector check ≥90% per machine | E6 88% → consensus prior enabled → re-check **93% PASS** |
| Landmark glance / Head QC | 15/15, 0% → **R4 may proceed** |

**Consequence for later phases.** The testable overlay axis in FPDB is UI/out-of-sector content: E4 and E5 `a_text` / `b_*` perturbations. External sets show the opposite overlay shift: calipers frequent, little UI.

## 3. Deviations from the plan

1. **Audit rater.** At your request, a vision-capable model rated the binary audit instead of a human. It viewed numbered contact sheets of raw images, and was blind to detector output for the caliper and text questions. The rater is recorded as `claude-vision` in every CSV. All sheets can be regenerated (`scripts/audit_sheets.py`) for a human agreement check, which would be worth doing before publication.
2. **Rendered contact sheets instead of the planned interactive OpenCV tool.** Same y/n questions, no drawing. The sample and the frozen ratings are tracked.
3. **No registration step.** The FP images are pixel-identical to FPDB (00 §3.2), so an exact identity check replaces template matching.
4. **HC18/UCL patient IDs come from filename prefixes**, because the annotation `SubjectID` columns are empty. This gives 806 HC18 patients, matching the documentation.
5. **Abdominal landmark QC bound is 0.30**, not 0.15. TAD/APAD skin-edge caliper pairs do not bisect each other; median offset 0.08, p95 0.19.
6. **Sector threshold.** The background-level threshold replaced the planned `0.5 × Otsu`, because Otsu cut away dark fluid regions inside the fan (verified visually).
7. **Glyph-based UI detection.** I added detection of glyphs sitting on a black background, so device labels that lie inside the fan's convex hull (e.g. "Voluson E6" at the top) are caught. I added a `sector_clean` mask without those glyphs.
8. **Consensus prior = union with a 0.95-coverage "core", not intersection with a majority mask.**
   - The observed failures were *under*-inclusion: dark fan regions cut off. Intersection would make that worse.
   - A 0.5-majority mask covered 92–95% of the frame for the main E6 and Aloka geometries, because zoomed frames fill the screen. It would have swallowed the UI strips.
   - The core is computed from train images only, is applied only to FPDB, and also stops interior tissue fragments from being labelled as UI.
   - Groups with mixed geometry produce empty cores. S10 has none, because it has only 11 training images.
9. **Caliper detector.** Dotted-line scoring is disabled: on the train audit, every dotted-run detection was speckle. The final parameters (NCC 0.78, intensity ≥200) are the best F1 on the train audit. Per the rules, there was one tuning round and no manual masks.
10. **Extra audits added:** an external-set caliper/text audit (42 images) and a 40-image sector re-check. Both quantify issues found during R1.

## 4. Things to know before R2

- Mask files are the first input to the R3/R5 perturbations: `data/processed/masks/<dataset>/<image_id>.npz` holds `sector`, `sector_clean`, `ui` and `marker`.
- The `ui` mask is reliable. The `marker` mask is not, except on Ghana.
- External images have different framing: African images are cropped to the field of view and have little UI. This is expected to drive part of the B0 external drop.
- Exact duplicates and the 8 label conflicts stay in the frozen test split. R2 should report how many of the in-distribution errors fall on conflicting duplicates, so that this ceiling is visible.
- Runtime: metadata and masks take about 20 minutes per pass (6 workers). The caliper-pair SSIM search takes about 12 minutes.
