# R7: Post-hoc follow-up after R5 (written 2026-11-14, before any R7 run)

**Status of everything here: post hoc and exploratory.** R7 was planned after the pre-registered R5 results were known. It is not part of the OSF registration, and every R7 output is labelled post hoc in RESULTS.md, the log and any manuscript. The hard rules in CLAUDE.md apply unchanged:
- validation-only selection;
- frozen splits and checkpoints;
- per-image parquet;
- 95% patient-cluster bootstrap CIs (B = 2000, seed 20261109);
- tests for new metrics;
- no clinical claims.

## Why R7

R5 ended with three results that point the same way:
1. The pre-registered AFP > B0 holds for convnext only.
2. B2 (input cleaning: sector letterbox + caliper-marker inpainting + synthetic-caliper augmentation) gives the largest external gain on both backbones.
3. A frozen fetal-ultrasound foundation model (FetalCLIP) gains by a similar amount from the same input cleaning.

AFP adds two training-time components on top of B2 and loses part of B2's gain. The class-weight sensitivity run rules out class weights as the cause.

So R7 does three things:
- finds which training-time component costs accuracy (E9);
- tests whether input cleaning reduces the *shared* external failures found in E8 (E10);
- packages input cleaning as a tool other groups can use (E11).

## E9: AFP component ablation (GPU, about 5–6 h)

**Arms** (configs derived from B2; only the arm keys differ):

| arm | preprocessing | caliper aug | sampler | style aug | class weights |
|---|---|---|---|---|---|
| B2 (R5) | sector_letterbox_inpaint | 0.3 | none | off | on |
| **B2SMP** | sector_letterbox_inpaint | 0.3 | acquisition_balanced | off | off |
| **B2STY** | sector_letterbox_inpaint | 0.3 | none | on | on |
| AFP (R5) | sector_letterbox_inpaint | 0.3 | acquisition_balanced | on | off |

**Runs.**
- 2 new arms × 2 backbones × seeds 11/22/33 = 12 runs.
- Same hyperparameters, early stopping on val macro-F1, and run directory layout as R5.
- `scripts/research/e9_run_ablation.py` trains and predicts each run; it is resume-safe.

**Analysis** (`scripts/research/e9_ablation.py`). This is fixed here, before any E9 run.
- **Endpoint.** The R5 primary endpoint: pooled external full-argmax balanced accuracy over {Head, Abdomen, Femur, Thorax}. Seed-averaged, `paired_multi_seed`, B = 2000, seed 20261109.
- **Component contrasts, per backbone:**
  - B2SMP − B2 (cost of the sampler);
  - B2STY − B2 (cost of the style augmentation);
  - AFP − B2 (cost of both).
  - Holm over these 6 tests (3 contrasts × 2 backbones). This is reported for honesty; the analysis stays exploratory.
- **Decomposition of the input-cleaning gain**, using existing R5 runs:
  - B1 − B0 (sector letterbox);
  - B2 − B1 (marker inpainting + synthetic caliper augmentation).
  - Each also per external dataset.
- **Mechanism readouts per arm** (seed-averaged):
  - pooled-external rejection rate ("Other");
  - within-class machine decodability from penultimate features (R3 E3 protocol, FPDB test);
  - FPDB-test 8-way macro-F1 (in-distribution cost).
- **No pass/fail criterion.** E9 is descriptive; the CIs carry the evidence.

## E10: do input-cleaned models share fewer external failures? (CPU)

Extends E8 to the R5 arms, using seed 11 and both backbones.

**Per arm** (B0, B1, B2, AFP; plus B2SMP and B2STY once trained):
- the share of pooled-external images wrong for *both* backbones;
- the two-model all-wrong ratio against stratified independence (the E8 metric with m = 2).

**Paired contrast:** B2 − B0 in the share of images both backbones get wrong. Patient-cluster bootstrap.

**Question.** Does input cleaning remove part of the "shared, image-level" failure set E8 found, or only move errors between models?

## E11: a reusable input-cleaning tool (no new results)

`scripts/clean_ultrasound.py` takes a folder of 2-D B-mode images. It writes:
- the sector-cropped, caliper-inpainted, aspect-preserving letterbox at a chosen size;
- the sector and overlay masks;
- a JSON log per image.

It uses the same `src/preprocess` code as the B2 arm, so a model trained with B2 sees exactly this input. A test checks that the tool's output is identical to the cached B2 input for a few FPDB images.

**Documentation:** `docs/research/INPUT_CLEANING.md`. It covers:
- what the tool does;
- the known failure cases (from the R1 sector-detection QC);
- the evidence that motivated it, linked to the results files.

## Synthesis

`docs/research/KEY_FINDINGS.md` is rendered from JSON like RESULTS.md: no hand-typed numbers. It is a short brief for ultrasound-AI researchers. It separates:
- the pre-registered results (R5 primary, P3b);
- the post-hoc results (E8–E10);
- practical recommendations, which are reporting and preprocessing recommendations, not clinical claims.
