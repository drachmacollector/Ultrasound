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

## 2026-11-13 — R5.1–R5.6, R5.9 — AFP protocol implementation (pre-G4)
commit: 91a54aedbc   run_ids: `runs/_smoke/<arm>_<backbone>_s11` (1 epoch), `runs/_smoke/afp_convnext_s11_e2` (2 epochs)
outputs: `src/preprocess/{pipeline,caliper_aug}.py`, `src/train/{sampler,train}.py`, `src/data/{transforms,dataset}.py`, `configs/research/{b0,b1,b2,afp}_*.yaml`, `scripts/research/r5_{run_matrix,evaluate_matrix,primary_endpoint}.py`, `data/processed/cache/sector_letterbox*/` (gitignored), `docs/research/PREREGISTRATION.md`, `docs/research/GATE_G4.md`
outcome: all 8 arm configs train, validate and write val predictions end to end (1-epoch smoke). Letterbox caches for 14,273 images. Sampler realised joint within 3.7 pp of target over epochs 1-2 (no clipping). Stopped at Gate G4: the 24-run matrix waits for the OSF registration (PI action).
deviation: sampler pooling rule = total group size < 30 (the "< 30 in any class" reading would merge Other_machine and contradict the plan's expected multipliers); realised-joint check pools epochs 1-2 (single-epoch false-alarm rate 12.7% in simulation, pooled 0.3%); A4 replaces both B0 photometric entries; letterbox caches at native scale; val predictions written by `predict` to results/preds/<run_id>/; R5.2 detector realism check on 1-px strokes only (R1 templates find 0% of 2-px crosses).
deviation: letterbox quality depends on the R1 sector masks (FPDB G1 138/150 ok; externals not audited; UCL insets and African full-frame text remain inside the crop); African sector shapes are mostly 'fan' (361/450), not 'full' as the plan expected.

## 2026-11-13 — R6 (part 1) — Reporting and review artefacts independent of R5
outputs: `scripts/research/{render_results,make_review_figures,check_evidence_table}.py`, `docs/research/templates/RESULTS.template.md`, `docs/research/{RESULTS,REPRODUCE,DATA_CARDS,literature_search_log,review_reference_shortlist}.md`, `docs/research/review_evidence_table.csv`, `results/figures/review_fig{1..4}_*.pdf`
outcome: RESULTS.md renders every RQ1–RQ3 number from JSON (8 R5 placeholders pending). Six PubMed searches logged (56/114/60/65/73/24 hits). Evidence table: 25 rows, 10 verified against PubMed records. Four review figures drawn and checked in greyscale. FetalCLIP licence = CC BY-NC 4.0 (repository LICENSE).
deviation: JSON keys containing '|' are addressed with '~' in templates; review figure 3 reuses the R4 grid function.

## 2026-11-13 — R5 (part 2, pre-G4) — every pre-registered analysis coded; FM probe; sensitivity run
commit: b81bf61ed3 (the PREREGISTRATION commit in GATE_G4.md)   run_ids: `runs/_smoke/afp_cw_convnext_s11` (1 epoch)
outputs: `scripts/research/r5_fm_linear_probe.py`; `r5_evaluate_matrix.py` (errors_by_arm, p3b_confirmatory, sensitivity); `r5_run_matrix.py` (+1 sensitivity run); `configs/research/afp_cw_convnext_tiny.yaml`; `tests/test_r5_{evaluate_matrix,fm_probe}.py`; `docs/research/OSF_REGISTRATION_GUIDE.md`; `PREREGISTRATION.md` (code list, clarifications)
outcome: every analysis in PREREGISTRATION §5–6 now has code, tested on synthetic data with hand-computed answers (temperature T = 1/ln 21; seed-averaged rejection shares; seed-pooled P3b with patient clusters; FM probe C selection and scoring). The sensitivity config trains end to end (class weights + sampler; 1-epoch smoke). No R5 model was trained beyond smoke epochs or evaluated on test/external data.
deviation: found before lodging the registration: PREREGISTRATION §6 items 3 (rejection/error decomposition per arm), 7 (confirmatory P3b and the African monitor on R5 models) and 9 (sensitivity run) had no code in part 1; added now, and the PREREGISTRATION commit in GATE_G4.md moves to this code-complete commit. The sensitivity run is fixed to convnext_tiny seed 11.
deviation: P3b pools the three seeds as rows keyed `image_id|seed` (clusters remain patients); the FM probe feeds each arm's own 224-px geometry into the FetalCLIP eval transform (its resize is then a no-op). Both are recorded in PREREGISTRATION §8 before lodging.

## 2026-11-13 — R6 (part 2) — study figures, literature screening, clean-up checks
outputs: `scripts/research/{make_figures,check_orphan_json}.py`, `results/figures/F{3,7,8}_*.pdf`, `results/figures/STUDY_FIGURES.md`, `docs/research/review_evidence_table.csv`, `literature_search_log.md`, `review_reference_shortlist.md`, `templates/RESULTS.template.md`, `DATA_CARDS.md`
outcome: 7 of 8 study figures present (6, the R5 forest plot, waits for R5). All six PubMed searches screened (376 unique records); 19 rows added and 10 existing rows newly verified, so 39 of 44 evidence rows are verified (target 30). Reference shortlist: 39 references, 5 preprints (13%). Orphan check: every results JSON is used. "FetScan/real-time/demo" wording occurs only in historical context. RESULTS.md now also renders R1 data quality and Gates G1/G2; its 32 pending placeholders are all R5.
deviation: study figures 1, 2, 4, 5 and 6 stay in the scripts that produce their data (one implementation per panel); `make_figures.py` draws the cross-phase figures 3, 7, 8 and the index. Geirhos 2020 was verified against its Semantic Scholar record (publisher page behind a login). Five references stay unverified (reasons in the search log). The landmark benchmark is cited as its published Sci Rep 2026 version.

## 2026-11-13 — Gate G4 passed
outputs: `docs/research/GATE_G4.md`
outcome: the PI lodged PREREGISTRATION.md on OSF (https://osf.io/635sf/overview; pending automatic approval at commit time), quoting commit b81bf61ed3. The R5 matrix starts after this commit.

## 2026-11-13 — E8 (post-hoc, exploratory) — error concordance across the six legacy architectures
outputs: `scripts/research/e8_error_concordance.py`, `src/eval/metrics.py::{fleiss_kappa_binary, all_wrong_ratio}` (hand-computed tests in `tests/test_metrics.py`), `results/{json,tables}/E8.*`, `results/figures/E8_error_concordance.pdf`
outcome: errors pile up on the same images: Fleiss' kappa 0.56 FPDB test, 0.72 HC18, 0.48 UCL, 0.53 African (pooled external 0.62 [0.58, 0.65]). 6.5% of external images are wrong for all six models, far above the independence expectation within dataset x class. On external images all six get wrong, most models answer 'Other' in 78% (100% on UCL and African), against 6% in distribution: shared external failures are rejections, not confusions.
deviation: E8 is a post-hoc addition (not in the plans or the pre-registration), run on legacy R2 predictions only, while the R5 matrix trains; it is reported as exploratory.

## 2026-11-13 — R5 matrix paused (14 of 25 runs done)
run_ids: `runs/{b0,b1,b2,afp}_{convnext,effv2s}_s11`, `runs/{b0,b1,b2}_{convnext,effv2s}_s22` (all with DONE)
outcome: matrix started 10:54 at commit 7f61029; sanity check passed (B0 convnext s11 val macro-F1 0.9186 vs legacy 0.9180). Stopped at 16:31 by the PI's request (laptop shutdown), right after b2_effv2s_s22 wrote DONE; no partial run was left. Resume with `python scripts/research/r5_run_matrix.py` (runs with DONE are skipped); 11 runs remain (afp s22, all s33, afp_cw_convnext_s11).
deviation: none; the pause does not change any run (each run trains from scratch with its fixed seed).

## 2026-11-14 — R5 matrix complete; pre-registered primary endpoint
commit: (this commit)   run_ids: `runs/{b0,b1,b2,afp}_{convnext,effv2s}_s{11,22,33}`, `runs/afp_cw_convnext_s11`
outputs: `results/preds/<run_id>/{fpdb_val,fpdb_test,hc18,ucl,african}.parquet` (+ features), `results/{json,tables}/R5_primary.*`, `results/figures/R5_forest.pdf`
outcome: all 25 runs trained (resumed 2026-11-13 18:47 after the pause; finished about 23:30), none diverged. Primary (seed-averaged, paired patient-cluster bootstrap, Holm over 2 backbones): convnext_tiny AFP − B0 = +0.037 [+0.016, +0.058], Holm p = 0.001, success, non-inferior in distribution; tf_efficientnetv2_s +0.016 [−0.002, +0.034], Holm p = 0.096, not a success, non-inferior. Exploratory: B2 − B0 = +0.076 (convnext) and +0.079 (effv2s), with the gain concentrated on the African set.
deviation: none. The primary script and its dependencies are byte-identical to the PREREGISTRATION commit b81bf61ed3. open_clip_torch 3.3.0 was installed with timm pinned at 1.0.28 (REPRODUCE.md); FetalCLIP weights SHA-256 recorded.

## 2026-11-14 — R5 secondary analyses, FetalCLIP probe, final render
outputs: `results/preds/<run_id>/{*__saliency-gradcam,probe_within_class_machine}.parquet`, `results/preds/fm_fetalclip_{resize,sector_letterbox_inpaint}/`, `results/{json,tables}/R5_{evaluation,fm_probe}.*`, `results/figures/F{3,7,8}_*.pdf`, `docs/research/RESULTS.md`
outcome: Grad-CAM for 24 runs, within-class machine probes, calibration, rejection vs confusion, P3b and the African monitor on R5 models, sensitivity run, FetalCLIP probe. P3b confirmatory claim for AFP: not supported (OR < 1 for convnext, > 1 for effv2s). AFP lowers within-class machine decodability (convnext 0.976 → 0.883). Sensitivity: AFP + class weights − AFP = −0.029 [−0.051, −0.009] pooled external. FetalCLIP probe: letterbox-inpaint − resize = +0.089 [+0.058, +0.120] pooled external. RESULTS.md: 21 JSON files, 0 pending; orphan check clean; 8 of 8 study figures.
deviation: none. FetalCLIP probe C chosen on FPDB val only (resize 0.1, letterbox-inpaint 0.01).

## 2026-11-14 — R7 (post hoc): AFP component ablation (E9), shared failures (E10), input-cleaning tool (E11), key-findings brief
commits: 416af4b (plan + configs + runner, before any run), then code, results and docs   run_ids: `runs/{b2smp,b2sty}_{convnext,effv2s}_s{11,22,33}`
outputs: `results/{json,tables}/E9.*`, `E10.*`, `results/figures/E9_cleaning_and_components.pdf`, `src/preprocess/clean.py`, `scripts/clean_ultrasound.py`, `docs/research/{INPUT_CLEANING,KEY_FINDINGS}.md`, `docs/research/walkthroughs/R7_WALKTHROUGH.md`
outcome: 12 runs trained 12:49–18:54, none diverged. Components vs B2 (pooled external balanced accuracy, Holm over 6): acquisition-balanced sampler −0.063 [−0.080, −0.048] convnext, −0.110 [−0.132, −0.090] effv2s; style augmentation −0.038 [−0.052, −0.024] convnext, −0.017 [−0.038, +0.006] effv2s; AFP −0.039 / −0.063. The sampler raises external 'Other' rejections (effv2s 23.4% vs 16.0%) and does not lower machine decodability. Marker split: marker inpainting helps marker-bearing external frames (effv2s B2 − B1 +0.140 [+0.102, +0.186]); sector crop helps marker-free frames. E10: shared failures persist under every arm; B2's net reduction is not robust across seeds. Cleaning tool is pixel-identical to the cached B2 input on 24 external images.
deviation: post hoc throughout (R7 plan). Marker stratification added before any E9 result. Interim analysis runs on partial seeds were made for progress updates (analysis fixed; nothing tuned). E10 seeds disagree; reported.
