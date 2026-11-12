# R6 Walkthrough: Reporting, reproducibility and review support (in progress)

**Date:** 2026-11-13 · **Branch:** `main` · **Plan:** `docs/research/plans/R6_REPORTING_AND_REVIEW.md`

**Status:** 🟡 Every part that does not depend on R5 results is built. RQ4 renders as `⟨pending⟩` until the R5 matrix runs, which waits on Gate G4.

## 1. What was built

| Plan item | Output | State |
|---|---|---|
| A. Results rendering | `scripts/research/render_results.py`, `docs/research/templates/RESULTS.template.md` → `docs/research/RESULTS.md` | ✅ |
| B.1 study figures | Written by each experiment script (E1–E6, R2, R4, G3) as PDF (tracked) and PNG | ✅ for R2–R4; R5 forest plot pending |
| B.2 Box 1 | `render_box1.py` (from R3) | ✅ |
| C.1 literature search log | `docs/research/literature_search_log.md` | 🟡 |
| C.2 evidence table | `docs/research/review_evidence_table.csv` + `scripts/research/check_evidence_table.py` | 🟡 |
| C.3 reference shortlist | `docs/research/review_reference_shortlist.md` | 🟡 |
| C.4 review figures | `scripts/research/make_review_figures.py` → `results/figures/review_fig{1..4}_*` | ✅ |
| D.1 data cards | `docs/research/DATA_CARDS.md` | ✅ |
| D.2 reproduction | `docs/research/REPRODUCE.md` | ✅ |
| D.3 final clean-up pass | — | ⬜ After R5 |

**Notes on each item**

- **Results rendering (A).**
  - The renderer reads `results/json` only. A missing key is a hard error; `--allow-missing` gives a draft instead.
  - The header lists the git hash, render time and the SHA-256 of every JSON file used.
  - Deviations are collected from the research log.
  - Tests: `tests/test_render_results.py`.
  - Current draft: 8 R5 placeholders are pending; everything else renders.
- **Literature search log (C.1).** Six PubMed searches are logged with exact queries and hit counts: 56, 114, 60, 65, 73 and 24. Screening and the non-PubMed databases are still to do.
- **Evidence table (C.2).** 25 seed rows, 10 verified against PubMed records. The target is ≥ 30 verified rows. The checker passes.
- **Reference shortlist (C.3).** 10 citable references in Vancouver style, all verified. The others are listed as pending.
- **Review figures (C.4).**
  1. Schematic, drawn with vector shapes and no images.
  2. Box 1: E1 confound bars and E2 no-pixel ROC.
  3. Saliency 2×3 grid with landmark outlines and a licence credit.
  4. Pre-adoption checklist.

  Each figure also has a greyscale copy for the greyscale-safety check. All four were inspected visually.

## 2. Decisions

- Figure 3 reuses the R4 grid drawing code, so there is one implementation for the same panel.
- The R4 "femur" sentence in RESULTS is written to stay true for any data. The rendered table shows convnext_tiny and tf_efficientnetv2_s, and the text points to `R4_localisation.md` for the other four backbones. This avoids a hand-typed claim that one model contradicts.
- JSON keys that contain '|' are written with '~' in templates, for example `fpdb_test~Head`.

## 3. Remaining (after G4 and R5)

- Render RESULTS.md without `--allow-missing`.
- R5 forest plot and risk–coverage figures.
- Screen searches #1–#6 and run the non-PubMed searches, then verify at least 20 more evidence rows.
- D.3 clean-up: orphan-JSON check, and a scan for "FetScan" / "demo" wording.
- **Human:** confirm the JAPI article format with the editorial office.
