# R6 — Reporting, Reproducibility and JAPI Review Support

**Goal:** every number in every document is generated from stored per-image predictions. Figures are publication-ready. Data cards and a one-command reproduction guide exist. The artefacts needed for the JAPI review are built from R1–R4 outputs without new experiments.
**Effort:** 1–2 weeks. Parts A and B start as soon as R3 E1/E2/E6 exist; part C runs in parallel from R1 onward.

---

## A. Results rendering — `scripts/research/render_results.py`

- Inputs: `results/json/*.json` only (written by R2–R5 scripts). **Never** reads parquets directly, and never runs inference.
- Template: `docs/research/templates/RESULTS.template.md`, with placeholders of the form `{{E2.prior.seen_machines.auroc}}`. `Estimate` objects render as `point [lo, hi]` followed by `(n = <images> images, <clusters> patients)`, e.g. `0.735 [lo, hi] (n = 4,200 images, … patients)`. Percentages render with 1 dp.
- Missing key → **hard error** (no silent blanks). A `--allow-missing` flag lets a partial render mark them `⟨pending⟩`. CI runs without the flag at the end.
- Output: `docs/research/RESULTS.md`, with a header listing the git hash, render time, and the SHA-256 of every JSON consumed.
- Sections: Data (counts; R1), RQ1 (E1, E2, E3, E6), RQ2 (E4, E5), RQ3 (R4), RQ4 (R5 primary, secondary, FM probe), Deviations (pulled from RESEARCH_LOG entries tagged `deviation:`), Limitations (template text).
- Test (`tests/test_render_results.py`): a toy template plus toy JSON → expected markdown; a missing key raises.

## B. Figures

### B.1 Study figures — `scripts/research/make_figures.py`
matplotlib, colour-blind-safe palette (Okabe–Ito), sans-serif 8–9 pt, single-column 85 mm / double-column 175 mm. Each figure is saved as vector **PDF** (tracked in git) plus 300-dpi PNG (gitignored, regenerated). Each figure function reads JSON or parquet produced earlier and states its sources in a PDF metadata field.

| # | Figure | Source | Phase |
|---|---|---|---|
| 1 | Confounding mosaic: class share per machine, train+val vs test, S10 highlighted | E1 | R3 |
| 2 | Metadata-only ROC: `prior`/`geometry`/`all` on all-test vs seen-machines | E2 | R3 |
| 3 | Availability (E4) and reliance (E5) bars with CIs, `a_marker` vs `c_control` | E4, E5 | R3 |
| 4 | Saliency grid: image, landmark mask, Grad-CAM, EPG; in-distribution vs external; correct vs incorrect | R4 | R4 |
| 5 | EPG (minus area) by class × backbone; association plot (EPG decile vs external accuracy) | R4 | R4 |
| 6 | Forest plot: B1/B2/AFP − B0 per backbone, pooled and per dataset | R5 | R5 |
| 7 | Rejection vs confusion stacked bars per external dataset (legacy and R5 arms) | R2, R5 | R2/R5 |
| 8 | Risk–coverage curves, in-distribution vs external | R5 | R5 |

Image content in figures: public datasets only, with the licence credited in each caption (FPDB: Zenodo record licence; landmark benchmark: CC BY-NC-SA 4.0; African: CC BY 4.0; HC18: challenge terms). No patient identifiers appear. FPDB filenames contain patient numbers, so they must not be printed on figures.

### B.2 Box 1 summary — `scripts/research/render_box1.py` → `docs/research/BOX1_ILLUSTRATIVE_ANALYSIS.md`
A ~200-word auto-rendered paragraph plus a 4-row table:
1. Machine × class entanglement (Aloka vs E6 Other share, CI; Cramér's V).
2. A no-pixel model's AUROC for Other vs standard plane (seen machines; all-test).
3. The S10 "confound-broken" subset: Other recall vs E6/Aloka (E6, legacy convnext).
4. External: full-argmax accuracy and rejection share of errors (R2, convnext; HC18/UCL/African).

Heading text: "Illustrative re-analysis of publicly available datasets; not a clinical study."

## C. JAPI review-support artefacts (no new experiments)

> The human confirms the current JAPI review-article format (word and reference limits, whether unsolicited reviews are accepted) with the editorial office. Public instruction PDFs are undated, and a 2026-11-12 search found no stated limits. Until confirmed, work to the conservative budget in proposal §9.1: ~2,000 words, ≤ 40 references, ≤ 4 figures, abstract ≤ 200 words, Vancouver style.

### C.1 `docs/research/literature_search_log.md`
- One block per search: date, database (PubMed, Scopus or Google Scholar, arXiv, medRxiv), the **exact query string**, filters (years 2015–2026, English), hit count, and the number screened and included.
- Starter queries (Claude drafts; the human runs or confirms them in the databases they have access to; PubMed can also be queried via the PubMed connector if authorised):
  1. `("fetal" OR "foetal") AND ultrasound AND ("standard plane" OR "plane classification" OR "plane detection") AND ("deep learning" OR "convolutional")`
  2. `("shortcut learning" OR "confound*" OR "spurious correlation*") AND (ultrasound OR radiograph* OR dermatolog*) AND "deep learning"`
  3. `("external validation" OR "domain shift" OR "generali*") AND fetal AND ultrasound AND ("artificial intelligence" OR "deep learning")`
  4. `("saliency" OR "Grad-CAM" OR "explainab*") AND ("sanity check" OR "faithfulness" OR "localization") AND medical imaging`
  5. `India AND (ultrasound OR sonograph*) AND ("artificial intelligence" OR "deep learning") AND (fetal OR antenatal OR obstetric)`
  6. `"FETAL_PLANES_DB" OR "Burgos-Artizzu"` (citation tracking of the benchmark)
- Inclusion rules written once at the top. Redo all searches within 4 weeks before submission and log the delta.

### C.2 `docs/research/review_evidence_table.csv`
Columns (02 §8.1): `citation, year, venue, peer_reviewed, doi_or_url, dataset(s), n_images, n_sites, devices, external_test, per_device_results, overlay_handling, xai_method, xai_quantitative, calibration_reported, india_data, notes, verified`.
- Target 30–40 rows. Seed it with the proposal's §2 and §12 references.
- `verified=y` only after the row has been checked against the paper's full text or abstract. The review cites only `verified=y` rows.
- `scripts/research/check_evidence_table.py` validates the schema, DOI/URL presence and duplicates, and counts verified rows.

### C.3 `docs/research/review_reference_shortlist.md`
≤ 40 references ranked by necessity (Must / Should / Optional), in Vancouver style (up to 6 authors, then the first 3 + "et al."; numbered in square brackets in order of first citation). Each entry links to its evidence-table row. Preprints are flagged "(preprint)", and the review keeps them to ≤ 15% of references.

### C.4 Review figures — `scripts/research/make_review_figures.py` (≤ 4)
1. Schematic: "acquisition signals vs anatomy". Drawn in matplotlib with vector shapes, using no copyrighted images or real patient images: a fan sector, an anatomy ellipse, calipers, burned-in text, a device badge, and arrows into a "model" box.
2. Box 1 figure: the E1 confound bars plus the E2 ROC (two panels).
3. Saliency example grid (2×3) with landmark mask outlines, from public data, with the licence credited.
4. A pre-adoption checklist rendered as a table figure (8–10 questions: per-device results? external sites? rejection rate reported? overlay robustness? saliency sanity-checked? calibration? local pilot plan? PCPNDT-compliant data handling?).
Output: 300-dpi PNG + PDF, greyscale-safe (check by converting to greyscale and confirming the series stay distinguishable).

## D. Data cards and reproduction

### D.1 `docs/research/DATA_CARDS.md`
One card per dataset (FPDB; landmark benchmark FP/HC18/UCL; African; NatalIA as supplementary): source and DOI, licence, n images / n patients (from R1 outputs), devices, classes, how it is used in this study (train/val/test/external/GT-only), exclusions, known quirks (00 §3: identity of FP to FPDB, empty SubjectIDs, centre-qualified African IDs, author cropping), and citation. Also the foundation-model weights licence (R5.8).

### D.2 `docs/research/REPRODUCE.md`
Ordered, copy-pasteable commands from raw data to every table and figure:
```
# 0 env
D:\miniconda\envs\fetalplane\python.exe -m pip install -r requirements.txt
python -m pytest
# 1 data (R1)
python scripts/build_external_manifests.py
python scripts/link_fp_landmarks.py
python scripts/build_acquisition_metadata.py
python scripts/research/g1_report.py
# 2 eval core (R2)
python scripts/research/r2_reevaluate_existing.py
# 3 audit (R3)
python scripts/research/e1_confounding_tables.py  ...  e6_stratified_performance.py
# 4 localisation (R4)
python scripts/research/r4_sanity_randomisation.py; r4_localisation.py; r4_summarise.py; r4_association.py
# 5 training (R5)
python scripts/research/build_cache.py --mode sector_letterbox  (and _inpaint)
python scripts/research/r5_run_matrix.py; r5_evaluate_matrix.py; r5_primary_endpoint.py
# 6 reporting (R6)
python scripts/research/render_results.py; make_figures.py; render_box1.py; make_review_figures.py
```
Each step lists its expected runtime and outputs. Note which steps need human input (the G1 audit) and which need the GPU.

### D.3 Final cleanup pass
- Archive `scripts/evaluate_cross_device.py` and `scripts/build_cross_device_manifest.py` if not already done; delete `data/processed/cross_device_manifest.csv` only after confirming no live reference (`rg cross_device_manifest src scripts tests`).
- `rg -n "FetScan|real-time|realtime|clinical-grade|demo"` in `README.md`, `CLAUDE.md` and `docs/research/` → only historical mentions.
- Confirm every `results/json/*.json` is referenced by RESULTS.md or a figure (orphan check script).

---

## Definition of done (auditor checklist; mirrors 02 §12 with plan-specific checks)
- [ ] R0: tag `legacy-fetscan-demo-v1` at b854a34; `archive/` populated; `tests/test_no_archive_imports.py` green; frozen checkpoints untouched (`git status checkpoints` clean, SHA-256 of `best.pt` files unchanged vs a list recorded in R0).
- [ ] R1: manifests with asserted counts (HC18 806 patients; African 450 images / 127 IDs); landmark identity check 3,090/3,090; masks for all images; audit CSVs; **GATE_G1.md** signed.
- [ ] R2: stats/metrics/predict tests; 6 × 5 parquets; **GATE_G2.md** (reproduced or re-baselined with reason).
- [ ] R3: E1–E6 JSON/tables/figures with CIs; Box 1 rendered.
- [ ] R4: **GATE_G3.md**; localisation and association JSON; E7 out-of-sector fractions.
- [ ] **G4**: PREREGISTRATION.md committed and lodged (OSF link) before the first R5 run.
- [ ] R5: 24 runs DONE and evaluated; primary endpoint from the pre-registered script; FM probe done or skip logged.
- [ ] R6: RESULTS.md renders with no `⟨pending⟩`; 8 figures; DATA_CARDS; REPRODUCE; review artefacts (search log, ≥ 30 verified evidence rows, ≤ 40 references, 4 review figures, Box 1).
- [ ] Nothing from 02 §9 (future extensions F1–F3) was implemented.
