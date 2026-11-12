# R3 Walkthrough: Confounding and Reliance Audit (RQ1–RQ2)

**Date:** 2026-11-12 · **Branch:** `main` (base `739dd78`) · **Plan:** `docs/research/plans/R3_CONFOUNDING_RELIANCE_AUDIT.md`
**Status:** ✅ E1–E6 complete; Box 1 rendered (`docs/research/BOX1_ILLUSTRATIVE_ANALYSIS.md`). E7 runs in R4, as planned. Uncommitted.
**Tests:** `python -m pytest` → **157 passed** (26 new).
**Compute:**
- 7 E4 training runs: about 2 GPU-hours.
- 26 perturbed prediction runs: about 1.5 h.
- 96 nested-CV probes: about 25 min on 6 CPU workers.
- Everything else runs in minutes.

## 1. What was built

| Experiment | Code | Outputs |
|---|---|---|
| E1 confounding tables | `scripts/research/e1_confounding_tables.py` | `results/{json,tables}/E1.*`, `E1_confound_mosaic` |
| E2 no-pixel model | `src/audit/meta_features.py` (feature sets; failed caliper features excluded), `scripts/research/e2_metadata_only_model.py` | `E2.*`, `E2_metadata_roc`, `results/preds/e2_metadata/` |
| E6 stratified performance | `scripts/research/e6_stratified_performance.py` (joint bootstrap contrasts, GEE) | `E6.*`, `E6_other_recall_by_machine`, `E6_confusions_convnext` |
| Box 1 | `scripts/research/render_box1.py` (all numbers from JSON; direction-dependent wording) | `docs/research/BOX1_ILLUSTRATIVE_ANALYSIS.md` |
| E3 feature probes | `src/audit/probes.py` (ImageNet feature extraction, nested-CV probe), `scripts/research/e3_feature_probes.py` | `E3.*`, `E3_decodability`, `results/preds/imagenet_*/` |
| E5 perturbations | `src/audit/perturb.py` (`a_text`, `b_zero`, `b_mean`, `b_blur`, exploratory `UITransplant`), `scripts/research/e5_reliance_perturbation.py` | `E5.*`, `E5_reliance`, `results/preds/legacy_*/fpdb_test__perturb-*.parquet` |
| E4 availability | `src/preprocess/pipeline.py` (modes `nonanat_only`, `overlay_only`, `sector_shape_only`; also used by R5), `scripts/research/build_cache.py`, `configs/research/e4_*.yaml`, `scripts/research/e4_availability_train.py` | `E4.*`, `runs/e4_*` (gitignored) |

**Supporting changes:**
- `stats.py` gained `SumsMetric` (a fast path for differences of rates within one bootstrap draw) and two-axis contingency tables in `CMMetric`.
- Shared figure conventions live in `src/audit/plotting.py`, using a validated colour palette.
- `train.py`, `dataset.py`, `loading.py` and `predict.py` accept cached preprocess modes.
- `FrameDataset` is now public.
- `CLAUDE.md` gained the R3 commands and the resource lessons below.

**New tests:**
- `test_perturb.py`: every perturbation changes only its declared region, pixel-exactly, and is deterministic.
- `test_preprocess_pipeline.py`: mode rendering, and cached input equals on-the-fly input.
- `test_probes.py`.
- E2 feature builder (in `test_audit_metadata.py`).
- `SumsMetric` and contingency tables (in `test_stats.py`).

## 2. Results (95% patient-clustered CIs, B = 2000)

| Prediction | Result | Verdict |
|---|---|---|
| **RQ1 confounding** (E1) | 'Other' is 65% of Aloka training images vs 14% on Voluson E6 (gap 51.5 pp [45.0, 57.4]); image size identifies the machine (Cramér's V 0.79–0.83) | descriptive |
| **P1a** no-pixel model beats chance (E2) | AUROC on seen machines: machine + operator 0.735 [0.714, 0.751] (reproduces the proposal's 0.735); + frame/sector geometry 0.872; all metadata 0.880; full image 0.974 | **supported** |
| **P1b** machine decodable from features (E3) | within one true class, balanced accuracy 0.85–0.97 (chance 0.33). ImageNet-only features: 0.99–1.00. Fine-tuning *reduces* decodability by 2–14 pp | **supported**, but fine-tuning does not amplify it |
| **P1c** S10 'Other' recall lower than E6 and Aloka (E6) | S10 − E6 **+40 to +51 pp** (opposite direction); S10 − Aloka −3 to −11 pp (CI < 0 for 5 of 6) | **not supported** vs E6; partly vs Aloka |
| **P2a** label learnable without anatomy (E4) | balanced accuracy (chance 0.125): `nonanat_only` 0.480 [0.466, 0.492]; `overlay_only` 0.421; `sector_shape_only` 0.436; device prior 0.198 | **supported** |
| **P2b** models *use* non-anatomical content (E5) | removing all out-of-sector content flips 0.6–2.5% of predictions and lowers p(Other) by 0–2.4 pp, consistent across fills (5 of 6 backbones); largest effect: efficientnetv2, Other recall −4.5 to −4.8 pp | real but **small**; the pre-declared marker contrast was infeasible |

**What this says, in plain terms:**
1. In FPDB, the machine and the image frame predict the label without seeing any anatomy. A model trained on an image with the anatomy blanked out still gets about half the classes right, mostly from the shape and zoom of the ultrasound sector.
2. The trained classifiers carry this information. However, generic ImageNet features carry even more of it, and fine-tuning reduces it rather than adding to it.
3. At test time, the frozen classifiers depend only slightly on content outside the sector, such as on-screen text, rulers and panels. Removing it changes about 1–2.5% of predictions, though always in the same direction: fewer "Other" calls, mostly on Aloka images.
4. The models' clearest failure is machine-specific: "Other" frames on the Voluson E6 are missed half the time.
   - The pre-specified prediction that the unseen S10 would be worst was wrong: the S10 behaves like the Aloka.
   - The simplest reading is a class prior learned *per machine* from the E6 training images. It is not a "Voluson means standard plane" rule.
5. Together with R2 (external errors are almost all "Other" rejections), the problem centres on the standard-vs-Other decision, which is the decision most entangled with acquisition.

**Exploratory UI transplant (E5):** putting an Aloka-style screen frame on Voluson E6 images raises p(Other) by 2.3 pp (convnext) and 2.6 pp (efficientnetv2). That is the direction a learned "Aloka screen means Other" association predicts. It is labelled exploratory.

## 3. Deviations from the plan

1. **E5 scope (Gate G1).** `a_marker` / `c_control` (the pre-declared primary reliance contrast) and `d_pairs` were not run: the caliper detector failed validation and no caliper pairs exist. The remaining `a_text` and `b_*` perturbations measure reliance on *out-of-sector* content only. Text drawn on top of the anatomy (common on zoomed frames) is not detected at pixel level and stays in place. A coverage table in E5.md shows what was reached: median 15% of the frame and 100% of the detected UI.
2. **E2 footnote.** The plan pre-wrote "geometry ≈ prior is expected". The data contradict it (paired Δ AUROC +0.137 [+0.122, +0.154]), so the footnote is now generated from the paired contrast.
3. **E2 overlay features** are UI-only (G1 validated the UI detector, not the caliper detector). E6 overlay strata were not run, for the same reason and because UI is universal.
4. **E4 cache** is stored pre-resized to 224 px instead of native resolution. The pixels are identical, since the train and eval transforms both start with that resize (tested). Training ran at about 25 s per epoch instead of minutes.
5. **E3 runtime.** The plan's sklearn probe is used unchanged, with two practical changes:
   - Probes run in torch-free worker processes with single-threaded BLAS. Multi-threaded BLAS made each fit 10–25× slower.
   - I briefly wrote a torch L-BFGS solver, validated it against sklearn, then removed it once the real cause was found.
6. **Resource failures.** Running E4 training, E5 inference and E3 probes at once exhausted Windows virtual memory: each torch process commits about 2 GB (WinError 1455). One E5 job hung and one probe batch died. Both were re-run sequentially; finished outputs are skipped on re-run, so nothing was lost or mixed. The lesson is recorded in CLAUDE.md.
7. **Box 1 length** is about 220 words against the ~200 target; R6 does the final edit. RESULTS.md "RQ1–RQ2" is rendered in R6, because `render_results.py` is an R6 deliverable. E7 is R4.
8. **Extras:**
   - E1 adds the size-bucket × machine association;
   - E2 adds paired AUROC contrasts and saves per-image predictions;
   - E5 adds a coverage table and the exploratory transplant;
   - E6 adds false-rejection rates.

## 4. Things to know before R4

- The pipeline modes in `src/preprocess/pipeline.py` and the cache builder are what R5's `sector_letterbox` / `_inpaint` modes should extend.
- Use `predict(..., perturb=...)` with the R1 masks for any test-time intervention. Do not run more than one DataLoader-spawning GPU job at a time on this machine.
- E7 (out-of-sector saliency fraction) can reuse `src.audit.perturb.outside(masks)` as the out-of-sector region definition, so E5 and E7 measure the same area.
- Pre-specified predictions so far: P1a, P1b and P2a supported; P2b small; **P1c refuted vs E6**. The review's Box 1 already states P1c as refuted.
