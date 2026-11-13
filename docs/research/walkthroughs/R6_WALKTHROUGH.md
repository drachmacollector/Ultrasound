# R6 Walkthrough: Reporting, reproducibility and review support (in progress)

**Date:** 2026-11-13 · **Branch:** `main` · **Plan:** `docs/research/plans/R6_REPORTING_AND_REVIEW.md`

**Status:** 🟡 Every part that does not depend on R5 results is built and checked. RQ4 renders as `⟨pending⟩` (32 placeholders, all R5) until the R5 matrix runs, which waits on Gate G4.

## 1. What was built

| Plan item | Output | State |
|---|---|---|
| A. Results rendering | `scripts/research/render_results.py`, `docs/research/templates/RESULTS.template.md` → `docs/research/RESULTS.md` | ✅ (R5 numbers pending) |
| B.1 study figures (8) | Figures 1, 2, 4, 5 and 6 come from their producing scripts; 3, 7 and 8 come from `scripts/research/make_figures.py`. Index: `results/figures/STUDY_FIGURES.md` | ✅ 7 of 8 (6 = R5 forest plot, pending) |
| B.2 Box 1 | `render_box1.py` (from R3) | ✅ |
| C.1 literature search log | `docs/research/literature_search_log.md` | ✅ PubMed screened; 🟡 Scopus/Scholar/arXiv searches for the human |
| C.2 evidence table | `docs/research/review_evidence_table.csv` + `scripts/research/check_evidence_table.py` | ✅ 44 rows, 39 verified (target ≥ 30) |
| C.3 reference shortlist | `docs/research/review_reference_shortlist.md` | ✅ 39 references, 5 preprints (13%) |
| C.4 review figures | `scripts/research/make_review_figures.py` → `results/figures/review_fig{1..4}_*` | ✅ |
| D.1 data cards | `docs/research/DATA_CARDS.md` | ✅ |
| D.2 reproduction | `docs/research/REPRODUCE.md` | ✅ |
| D.3 final clean-up | `scripts/research/check_orphan_json.py`; wording scan; archive check | ✅ for the current outputs; re-run after R5 |

## 2. Part 1 (earlier on 2026-11-13)

- **Results rendering (A).**
  - The renderer reads `results/json` only.
  - A missing key is a hard error; `--allow-missing` gives a draft instead.
  - The header lists the git hash, render time and the SHA-256 of every JSON file used.
  - Deviations are collected from the research log.
  - Tests: `tests/test_render_results.py`.
- **Review figures (C.4).** Four figures, each with a greyscale copy, all inspected visually.
  1. Schematic.
  2. Box 1.
  3. Saliency 2×3 grid.
  4. Pre-adoption checklist.
- **Data cards (D.1) and reproduction guide (D.2).**

## 3. Part 2 (this session)

**Study figures (B.1).** `make_figures.py` draws three figures:
- **F3: availability (E4) and reliance (E5).**
  - E4: 8-way balanced accuracy from non-anatomical input only, against chance, the no-pixel metadata models and the full-image legacy model.
  - E5: flip rates for the out-of-sector perturbations.
- **F7: outcome shares on each external set.** Correct, predicted "Other", absent class, or wrong present class. Legacy convnext and effv2s now; the R5 arms are added automatically once all three seeds exist.
- **F8: risk–coverage curves.** FPDB test against pooled external, per backbone. Legacy now; R5 B0 and AFP, seed-averaged, later.

Every PDF names its sources in the Subject metadata field; the shared `save()` gained a `sources` argument. All three figures were inspected visually and two layout issues were fixed:
- F3: panel titles overlapped and the legend covered data.
- F8: the y-axis showed 2.5% as "2%"; it now uses `PercentFormatter`.

**Literature (C.1–C.3).**
- All six PubMed searches were re-run and every title was screened: 376 unique records out of 393 hits.
- Abstracts were read for each candidate.
- Each included record is a table row with its PMID, and the per-search "included" counts are computed from those PMIDs.
- Non-PubMed items were verified against their arXiv abstract pages, or, for Geirhos 2020, its Semantic Scholar record.
- The landmark benchmark is now published (*Sci Rep* 2026). Its table row, the shortlist and the data card cite that version.
- Five references remain unverified: publisher login, a placeholder page, or the article not found. The reasons are logged; the human can verify them with institutional access.

**RESULTS template (A).**
- New Data block rendered from R1/G1/G2: per-machine counts, exact duplicates, near-duplicates, landmark linkage, Gate G1 and G2 outcomes. With it, every results JSON is used.
- The full pre-registered secondary set for RQ4 is now in place:
  - B1/B2 − B0;
  - the non-inferiority Δ with its 90% interval;
  - within-class machine decodability, B0 vs AFP;
  - pooled rejection rate;
  - confirmatory P3b;
  - FM probe.
- Limitations no longer hand-type dataset sizes.

**Clean-up (D.3).**
- `check_orphan_json.py`: all 17 JSON files are used (test: `tests/test_check_orphan_json.py`).
- `rg "FetScan|real-time|realtime|clinical-grade|demo"` over README, CLAUDE.md and docs/research: historical mentions only. These are the legacy tag, the archive note, the proposal's background, R0 records, and the SonoNet title.
- `scripts/evaluate_cross_device.py` and `build_cross_device_manifest.py` are already archived (`archive/scripts/`). `data/processed/cross_device_manifest.csv` no longer exists. `data/processed/eval_cross_device/*.csv` stay as historical records, as the 02 plan says.

**OSF help.** `docs/research/OSF_REGISTRATION_GUIDE.md` gives the PI step-by-step instructions, from the OSF help pages read on 2026-11-13. It covers:
- the account and project;
- "Add a Registration" → "OSF Preregistration";
- a field-by-field map to PREREGISTRATION.md;
- the existing-data answer;
- embargo options;
- the 48-hour approval;
- where to find the URL;
- how to unblock the gate.

## 4. Decisions

- **Figure ownership.** Figures 1, 2, 4, 5 and 6 stay in the scripts that compute their data, so each panel has one implementation. `make_figures.py` only draws the figures that combine phases, plus the index.
- **The "Existing data" OSF answer.** The guide recommends the conservative option, *prior to analysis*, with §9 pasted in. This is because the public images were analysed with the legacy models in R1–R4.
- **Benchmark-only FPDB papers.** Recent FPDB plane-classifier papers that report only in-distribution accuracy (several from Indian groups) are included as a group. They are the review's evidence that external, per-device and calibration reporting is rare.

## 5. Remaining (after G4 and R5)

- Run R5, then `make_figures.py` (adds the R5 arms to F7/F8), `r5_primary_endpoint.py` (F6) and `render_results.py` **without** `--allow-missing`.
- Run `check_orphan_json.py` again; the R5 JSON files are already referenced by the template.
- **Human:**
  - lodge the OSF registration;
  - run the Scopus/Scholar/arXiv searches;
  - verify the five pending references;
  - confirm the JAPI article format with the editorial office.
