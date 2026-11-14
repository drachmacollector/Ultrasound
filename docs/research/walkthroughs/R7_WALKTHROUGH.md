# R7 Walkthrough: post-hoc follow-up (E9, E10, E11) and the key-findings brief

**Date:** 2026-11-14 · **Branch:** `main` · **Plan:** `docs/research/plans/R7_POSTHOC_FOLLOWUP.md` (committed 416af4b, before any R7 run)

**Status:** ✅ Complete. Everything in R7 is **post hoc and exploratory**. It is not part of the OSF registration and is labelled so in RESULTS.md and KEY_FINDINGS.md.

## 1. Why R7

R5 left three facts pointing the same way:
- AFP > B0 held for convnext only;
- input cleaning alone (B2) gave the largest external gain on both backbones;
- the same cleaning helped a frozen FetalCLIP probe.

AFP = B2 + training-time device invariance, and it lost part of B2's gain. R7 asks three questions:
- which component costs accuracy (E9);
- whether cleaning shrinks the shared failure set found in E8 (E10);
- how to hand the cleaning to other groups (E11).

It closes with a findings brief for ultrasound-AI researchers.

## 2. What was built

| Item | Output |
|---|---|
| E9 arms | `configs/research/b2smp_*.yaml` (B2 + acquisition-balanced sampler, class weights off), `b2sty_*.yaml` (B2 + device-style augmentation); otherwise identical to B2 |
| E9 runner | `scripts/research/e9_run_ablation.py`: trains, then predicts each run in a separate process; resume-safe; writes progress (runs done, %, ETA) to stdout and `runs/E9_progress.json` |
| E9 analysis | `scripts/research/e9_ablation.py` → `results/{json,tables}/E9.*`, `results/figures/E9_cleaning_and_components.pdf` |
| E10 | `scripts/research/e10_shared_failures.py` → `results/{json,tables}/E10.*`; test `tests/test_e10_shared_failures.py` (hand-computed rescued/new shares) |
| E11 tool | `src/preprocess/clean.py`, `scripts/clean_ultrasound.py`, `docs/research/INPUT_CLEANING.md`; `tests/test_clean.py` (pixel-identical to the cached B2 input on HC18/UCL/African; synthetic and CLI tests) |
| Brief | `docs/research/templates/KEY_FINDINGS.template.md` → `docs/research/KEY_FINDINGS.md`, rendered by `render_results.py` from JSON only |

## 3. Runs

12 runs: 2 arms × 2 backbones × seeds 11/22/33. They ran 2026-11-14 from 12:49 to 18:54. Logs are in `runs/E9_stdout.log`.
- Run times ranged from 17 to 67 min. Style augmentation roughly doubles EfficientNetV2-S epoch time, because it runs on the CPU.
- No run diverged.
- `b2smp_effv2s_s11` peaked at epoch 2 (val macro-F1 0.880) and early-stopped at epoch 10. That is valid under the R5 rules (above the divergence threshold) and is kept.
- The sampler arms on EfficientNetV2-S had the lowest val scores of the ablation (0.876–0.910).

## 4. Results (numbers in `results/tables/E9.md`, `E10.md`, RESULTS.md → R7)

**E9 components vs B2** (pooled external balanced accuracy; Holm over 6):
- **The acquisition-balanced sampler costs most, on both backbones** (Holm p < 0.01). With EfficientNetV2-S it ends *below plain-resize B0*.
- It raises pooled-external "Other" rejections, mainly in the minority external classes (Abdomen, Femur, Thorax).
- Its target class mix is uniform, the same effective balance that B2's class weights give. So the cost comes from resampling with replacement and/or from balancing by device. The ablation does not separate these.
- **Device-style augmentation** costs on convnext. On EfficientNetV2-S its CI includes 0.
- The costs do not add: AFP (both) costs about as much as style augmentation alone on convnext.
- **Within-class machine decodability does not track external accuracy.** The sampler slightly *raises* it (convnext), and AFP lowers it most, yet both generalise worse than B2.

**Input-cleaning ladder and marker split** (already in RESULTS.md before E9 finished; R5 runs only):
- sector letterbox (B1 − B0) helps on frames without a detected measurement marker;
- marker inpainting + synthetic calipers (B2 − B1) helps on frames with one (large for EfficientNetV2-S).
- Measurement markers are detected on 17% of African and 28% of UCL images, against about 2% of FPDB.

**E10 shared failures.**
- Under every arm, the two backbones fail on the same external images far more often than independence predicts (ratio about 2.6–4.3).
- B2 rescues a share of B0's shared failures but creates new ones. The net change is negative for seeds 11 and 33 and positive for seed 22, so it is **not robust**.
- The sampler arm adds shared failures for every seed.

## 5. Decisions and deviations

- **Marker stratification was added to the E9 analysis on 2026-11-14.** This was after the R5 results but before any E9 result. It reuses the R1 detector output (`n_crosses > 0`) with no new tuning, and is recorded in the script header and here.
- **E10 uses seed 11 as primary, as planned.** Seeds 22 and 33 are reported as robustness checks; they disagree, and that is stated.
- **Interim E9 looks.** The analysis code was run on partial seeds while training continued (convnext sampler first) to give the PI progress updates. The analysis code was fixed beforehand and nothing was tuned, so these looks change nothing. They are mentioned for transparency.
- **The E11 tool uses the per-image sector path.** It has no device-consensus prior, which is exactly how the external sets were processed. Its known limitations are documented in `INPUT_CLEANING.md`:
  - low caliper recall on faint markers;
  - dotted traces are not removed;
  - it assumes a dark background.
- **Brief framing was corrected while drafting.** ImageNet features decode the scanner *better* than fine-tuned features, and zeroing off-sector content flips only about 1% of predictions. So the brief says the device signal is *available* in the pixels and that direct reliance on off-sector content is small. It does not say that models "learn the acquisition".

## 6. Checks

- 288 tests pass (new: `test_clean.py`, `test_e10_shared_failures.py`).
- `render_results.py`: RESULTS.md and KEY_FINDINGS.md each have 0 pending placeholders.
- `check_orphan_json.py`: all 23 JSON files are used.
- Figure `E9_cleaning_and_components.pdf` was checked visually.

## 7. Open (not done)

- **Separate resampling from device balancing.** A class-balanced resampling arm without device balancing would tell which one drives the sampler's cost. It is post hoc, about 4–5 GPU-h.
- **Human visual QC of the cleaning tool on a sample from each external centre.**
- **Manuscript**, built from KEY_FINDINGS.md and RESULTS.md.
