# Research Log: Anatomy or Acquisition?

One entry per task (format: `docs/research/plans/00_INDEX_AND_VERIFIED_FACTS.md §2.5`).
Numbers are quoted from `results/json/*.json`, never typed by hand. Lines starting with
`deviation:` are collected into RESULTS.md → Deviations.

---

## 2026-11-12 — R0 — Repository restructure
commit: `0de1ec0`; legacy tag `legacy-fetscan-demo-v1` → `b854a34`
run_ids: `_smoke/convnext_tiny` (1-epoch smoke run; not a result)
outputs: `archive/`, `tests/` (56 tests), `src/models/loading.py`, `data/splits/SPLITS_SHA256.txt`, `checkpoints/FROZEN.md`, new `CLAUDE.md`/`README.md`
outcome: legacy demo archived; multitask line removed; leakage and frozen-checkpoint guards added to training; test suite green; smoke training OK (frozen checkpoint hash unchanged).
deviation: split hashes use LF-normalised bytes (repo has core.autocrlf=true; raw-byte hashes would depend on checkout settings).
deviation: forbidden-token leakage check inspects the repo-relative path (or only the file name for files outside the repo) instead of the full absolute path; the absolute-path version false-positived on temp folders such as `pytest-…`.
deviation: `requirements-lock.txt` listed pandas 3.0.5 but the environment has 2.3.3; the lock now reflects the installed environment.

---

## 2026-11-12 — R1 — Data foundation (Gate G1)
commit: `8f6ba09`
run_ids: none (no model runs)
outputs: `data/processed/external/*.csv`, `data/processed/landmarks/*.parquet`, `data/processed/metadata/*.parquet`, `data/processed/masks/` (14,273 npz + 12 consensus cores), `data/processed/audit/*`, `results/json/{R1_*,G1}.json`, `results/tables/R1_*.md`, `docs/research/GATE_G1.md`
outcome: external manifests HC18 999/806 pts, UCL 424/51, African 450/127; FP landmarks 3,090/3,090 pixel-identical to FPDB, 1,486 test + 238 val kept; Head QC flags 0%; confounding shares reproduced exactly (Aloka 0.6506, E6 0.1357, S10-test 0.5163). Binary audit (300 FPDB + 42 external + 150 sector + 40 re-check + 15 landmark): calipers 3.0% [1.6, 5.6] of FPDB, text 100%; G1 → overlays SECONDARY, E5a infeasible, E5d dropped, consensus prior enabled (E6 88% → 93% after), R4 may proceed. 863 byte-identical FPDB images (384 groups, within-patient), 8 test groups with conflicting labels.
deviation: binary audit rated by a vision model (`claude-vision`) from contact sheets instead of a human, at the PI's request; rater was blind to detector output for caliper/text ratings.
deviation: FP↔FPDB registration replaced by an exact identity check (images are pixel-identical).
deviation: HC18/UCL patient IDs from filename prefixes (annotation SubjectID columns empty).
deviation: abdominal landmark QC uses a mid-point offset bound of 0.30 (head 0.15); TAD/APAD caliper pairs do not bisect each other.
deviation: sector consensus prior = union with a 0.95-coverage train-only "core" (plan: intersection with a majority mask). Observed failures were under-inclusion, and a majority mask covered 92–95% of the frame.
deviation: caliper detector recall 0.20 (train) after one tuning round; dotted-line scoring disabled (fired on speckle). Per the pre-specified rule, no manual masks were used.

---

## 2026-11-12 — R2 — Evaluation core and re-evaluation of the six frozen checkpoints (Gate G2)
commit: `739dd78` (parquet provenance records `8f6ba09a68-dirty`: run before the commit)
run_ids: `legacy_convnext_s42`, `legacy_effv2s_s42`, `legacy_repvgga1_s42`, `legacy_repvgga2_s42`, `legacy_mnv3l_s42`, `legacy_efflite0_s42`
outputs: `src/eval/{metrics,stats,datasets,predict,scoring,external}.py`, `results/preds/legacy_*_s42/` (30 parquet + meta + features), `results/json/{R2_existing_models,G2}.json`, `results/tables/R2_existing_models.md`, `results/figures/R2_rejection_vs_confusion.{pdf,png}`, `docs/research/GATE_G2.md`
outcome: G2 PASSED. All 6 primary and 12 secondary reproduction checks match: convnext test macro-F1 0.8929 (legacy 0.8927); HC18+UCL acc_full 83.63% (83.6%); HC18 81.38% / UCL 88.92%; Head→Other 180/194. The other 5 backbones match their macro-F1 to ≤0.0003, and their cross-device accuracy and Head→Other counts match exactly.
outcome: convnext FPDB-test macro-F1 0.893 [0.883, 0.902] (cluster CI 1.06–1.21× wider than the legacy image-level CI). Pooled external (1,873 images): acc_full 81.7% [79.6, 83.9], bal_acc_full 0.728 [0.690, 0.766], bal_acc_restricted 0.957 [0.935, 0.977]. 95.6% [93.4, 97.7] of external errors are 'Other' rejections (72–96% across backbones).
outcome: Other recall by machine, all 6 models: Aloka 0.92–0.95, Voluson E6 0.35–0.52, S10 0.81–0.92.
outcome: legacy "repvgg_a1 winner" claim WITHDRAWN (endpoint-dependent). It is best on HC18+UCL acc_full (significantly better than 4 of 5) but significantly worse than all 5 on pooled bal_acc_full (0.691 [0.655, 0.726]).
outcome: 8–9 test errors per model fall on the 8 contradictory duplicate groups (floor 8).
deviation: in-distribution scorer and metric factories are in `src/eval/scoring.py`; `external.py` only holds external scoring (plan put everything in `external.py`).
deviation: the image-level CI column is reported for all six backbones, not only convnext. Per-machine macro-F1 averages over the classes present on that machine.
deviation: macro-F1 excludes classes whose F1 is undefined in a resample (TP+FP+FN = 0). This equals sklearn on the full test set.
deviation: a `RatioMetric` fast path (sum/sum ratios) was added next to the confusion-matrix fast path.
deviation: G2 macro-F1 targets for the five other backbones use the 4-decimal values in archive/docs/EVAL_REPORT.md, not the 2-decimal classification reports. This is stricter.
deviation: extra analyses not in the plan: pairwise comparison on the legacy endpoint (HC18+UCL acc_full), HC18+UCL legacy-set table, and errors on contradictory duplicates.
deviation: `src/__init__.py` added; it sets `NO_ALBUMENTATIONS_UPDATE=1`, because albumentations' import-time version check made network calls and once timed out during the run.

---

## 2026-11-12 — R3.E1 — Descriptive confounding
commit: 63e1082 (base `739dd78`)   run_ids: none   outputs: `results/{json,tables}/E1.*`, `results/figures/E1_confound_mosaic.*`
outcome: shares match 00 §3.1 to 3 dp (asserted). Train+val 'Other' share Aloka − Voluson E6 = 51.5 pp [45.0, 57.4]; test S10 'Other' share 51.6% [44.0, 58.1] vs E6 training prior 13.6% [9.7, 18.0]. Cramér's V (train+val): machine × class 0.32 [0.30, 0.43], operator × class 0.31 [0.28, 0.35], image-size bucket × machine 0.79 [0.78, 0.97].
deviation: strata with < 10 patients get no CI (S10 train+val: 11 images from 1 patient).

## 2026-11-12 — R3.E2 — No-pixel metadata model (P1a)
commit: 63e1082   outputs: `results/{json,tables}/E2.*`, `results/figures/E2_metadata_roc.*`, `results/preds/e2_metadata/`
outcome: P1a supported. `prior` (machine + operator) AUROC standard vs Other on seen machines 0.735 [0.714, 0.751] (reproduces the proposal's preliminary 0.735); `geometry` 0.872 [0.854, 0.887]; `all` 0.880 [0.862, 0.894]; full-image B0 0.974. Geometry − prior, paired: +0.137 [+0.122, +0.154]. On S10 the prior is uninformative (0.500).
deviation: `overlay` features use only the validated UI detector (G1); caliper features excluded. The planned footnote "geometry ≈ prior is expected" was replaced by a data-driven sentence plus paired AUROC contrasts, because geometry clearly exceeds the prior.

## 2026-11-12 — R3.E6 — Stratified performance (P1c)
commit: 63e1082   outputs: `results/{json,tables}/E6.*`, `results/figures/E6_*`
outcome: 'Other' recall (6 backbones) Aloka 0.92–0.95, Voluson E6 0.35–0.52, S10 0.81–0.92. P1c (S10 recall lower than E6 and Aloka) NOT supported vs E6: S10 − E6 = +40 to +51 pp (all CIs > 0). Partly supported vs Aloka: S10 − Aloka = −3 to −11 pp (CI < 0 for 5 of 6 backbones). False rejection of standard planes is low everywhere (0–7%). 2 test patients have images from > 1 machine (joint bootstrap).
deviation: overlay strata not run (G1: caliper detector not validated; UI present on 100% of images). GEE (exchangeable) odds ratios reported as supplementary.

## 2026-11-12 — R3 Box 1 — `render_box1.py`
outputs: `docs/research/BOX1_ILLUSTRATIVE_ANALYSIS.md` (auto-rendered from E1/E2/E6/R2 JSON; ~220 words; the S10 sentence is chosen by the sign of the contrast, so it states the refuted prediction).

## 2026-11-12 — R3.E3 — Feature decodability (P1b)
commit: 63e1082   run_ids: `imagenet_<short>` (feature extraction only)   outputs: `results/{json,tables}/E3.*`, `results/figures/E3_decodability.*`
outcome: P1b supported. Machine decodable within one true class from fine-tuned features: mean balanced accuracy 0.85–0.97 (chance 0.33); unconditional 4-way 0.82–0.89 (chance 0.25); operator 0.76–0.85. ImageNet-only features decode the machine even better (within class 0.99–1.00): fine-tuning REDUCES within-class machine decodability by 2–14 pp (all paired CIs < 0). Fine-tuning does not amplify acquisition information.
deviation: probe solver unchanged (sklearn lbfgs) but run with single-threaded BLAS in torch-free worker processes; a torch L-BFGS solver was written, validated and then removed when BLAS oversubscription turned out to be the cause of slowness.

## 2026-11-12 — R3.E5 — Reliance perturbations (P2b)
commit: 63e1082   outputs: `results/preds/legacy_*_s42/fpdb_test__perturb-*.parquet`, `results/{json,tables}/E5.*`, `results/figures/E5_reliance.*`
outcome: removing out-of-sector content (median 15% of the frame; all detected UI) flips 0.6–2.5% of 8-way predictions and lowers p(Other) by 0–2.4 pp, consistently across `b_zero`/`b_mean`/`b_blur` (same sign, CIs exclude 0, size ratio ≤ 1.5) for 5 of 6 backbones (not mobilenetv3). `a_text` (1.3% of the frame) ≈ `b_*`. Largest for tf_efficientnetv2_s (Other recall −4.5 to −4.8 pp). Effects concentrate on Aloka images. Exploratory UI transplant: an Aloka frame on Voluson E6 images raises p(Other) by +2.3 pp (convnext) / +2.6 pp (effv2s). Reliance on out-of-sector content is real but small.
deviation: `a_marker`, `c_control` (the pre-declared primary contrast) and `d_pairs` not run (G1). Text drawn on top of the anatomy is not detected at pixel level and is left in place; E5 measures out-of-sector reliance only (coverage table in E5.md).

## 2026-11-12 — R3.E4 — Availability: training without anatomy (P2a)
commit: 63e1082   run_ids: `e4_nonanat_only_convnext_s{11,22,33}`, `e4_overlay_only_convnext_s{11,22,33}`, `e4_sector_shape_only_convnext_s11` (`runs/`, ~2 GPU-h)
outputs: `configs/research/e4_*`, `data/processed/cache/` (gitignored), `results/preds/e4_*`, `results/{json,tables}/E4.*`
outcome: P2a supported. 8-way balanced accuracy without anatomy: `nonanat_only` 0.480 [0.466, 0.492], `overlay_only` 0.421 [0.408, 0.433], `sector_shape_only` 0.436 [0.419, 0.452] (chance 0.125; E2 prior 0.198; E2 all 0.431; full image 0.901). AUROC standard vs Other 0.81 / 0.79 / 0.76. Most of the signal is the sector outline (zoom, depth, framing).
deviation: cached modes are stored pre-resized to 224 px (identical pixels to on-the-fly resizing; tested) instead of native resolution, for training speed. `nonanat_only` keeps one in-sector grey level per image.
deviation: RESULTS.md "RQ1–RQ2" section is rendered in R6 (`render_results.py` does not exist yet); all numbers are in results/json/E1–E6.json. E7 (saliency out-of-sector fraction) runs in R4 as planned.

## 2026-11-13 — R4.1–R4.4 — Saliency API, anatomy masks, metrics, development on val
commit: 512cede (code frozen; first test-image saliency computed after it), 2e9c2c2 (thread loader; outputs unchanged)
run_ids: `legacy_convnext_s42` (`fpdb_val__saliency-*`, 238 val landmark images)   outputs: `results/figures/R4_dev_grid_val.pdf`
outcome: Grad-CAM, Grad-CAM++ and occlusion run end to end on val; masks fit the anatomy on 24/24 dev-grid images (vision rater, as in G1). Saliency forward passes reproduce the R2 argmax on 100% of val images.
deviation: raw CAMs (library min-max scaling overridden); exact pixel-centre rasterisation of masks instead of `cv2.ellipse` (cv2 boundary pixels inflated thin femur capsules by > 5%, failing the plan's area test); EfficientNetV2-S occlusion patch 20 / stride 10 (2:1 ratio kept); occlusion under fp16 autocast.
deviation: speed fixes that leave outputs unchanged (verified): fixed-size CAM batches (each new input shape cost ~5 s of cuDNN set-up), CAM backward stops at the target layer, vectorised occlusion, thread-based input loading (DataLoader workers took ~30 s each to start on Windows).

## 2026-11-13 — R4.5 — Gate G3 (model-parameter randomisation)
commit: 2e9c2c2 (recorded in GATE_G3.md)   outputs: `docs/research/GATE_G3.md`, `results/{json,tables}/G3.*`, `results/figures/G3_*`
outcome: all three methods PASS on convnext_tiny and tf_efficientnetv2_s (stage 2 median SSIM 0.25–0.31, Spearman 0.02–0.17; thresholds 0.5). P3c supported.
deviation: ConvNeXt layer-scale `gamma` vectors keep trained values during re-initialisation (block-level parameters, not sub-modules); maps compared after positive part and per-map max scaling.

## 2026-11-13 — R4.6 — Locked localisation run + E7 (legacy checkpoints)
commit: 2e9c2c2   run_ids: `legacy_<short>_s42` (6 backbones x {gradcam, gradcampp, occlusion} x {fpdb_test, hc18, ucl, african})
outputs: `results/preds/legacy_*_s42/*__saliency-*.parquet`, `results/json/R4_subsets.json`, `results/{json,tables}/R4_localisation.*`, `results/figures/R4_{epg_by_class_backbone,saliency_grid,out_sector_by_machine}.*`
outcome: P3a supported with a caveat. All classes and backbones beat a uniform map (Grad-CAM EPG − area: Head 0.25–0.44, Abdomen 0.16–0.61, Femur 0.03–0.16). Femur − Head < 0 under both CAMs, but > 0 under occlusion for 5/6 backbones, and femur has the highest EPG/area ratio for 5/6: the "femur lowest" finding is a CAM-resolution effect. Incorrect predictions have lower on-anatomy saliency (11/12 Grad-CAM cells, all occlusion cells). External vs FPDB within anatomy: small differences (mostly −0.09 to +0.01). E7: out-of-sector Grad-CAM energy is below the uniform baseline everywhere (FPDB 0.04–0.14 vs 0.18) and higher on Aloka than E6 (+0.04 to +0.08, also within 'Other').
deviation: overlay fraction uses the R1 UI mask only; the "with vs without overlays" split was not run (G1); sector-aware EPG uses mask ∩ sector in numerator and baseline; 6 zero-energy Grad-CAM maps (HC18) excluded as undefined.

## 2026-11-13 — R4.7 — Association with external correctness (P3b, exploratory)
commit: 2e9c2c2   outputs: `results/{json,tables}/R4_association.*`, `results/figures/R4_association.*`
outcome: P3b not supported for the pre-declared primary (convnext Grad-CAM, true target: OR 0.99 [0.86, 1.15] per +0.1; AUROC 0.52; combined − confidence −0.04 [−0.07, −0.01]). Other backbones: EPG alone AUROC 0.57–0.91, ORs 1.5–6.5 after adjusting for confidence, but a significant increment over confidence in only 4/18 cells. Ground-truth-free monitor (out-of-sector share of the predicted-class map) on the African set: AUROC 0.78 [0.72, 0.83] vs confidence 0.66; Δ combined − confidence +0.13 [+0.06, +0.21].
deviation: the plan's "deployable" `epg_pred` still uses the expert mask of the true anatomy and is favoured by construction for wrong predictions (AUROC 0.97); it is reported with that warning and not interpreted. `--report-only` added to re-render without the 13-min bootstrap.
