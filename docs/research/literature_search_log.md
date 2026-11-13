# Literature search log (R6 C.1)

**Inclusion rules (fixed before screening):**
- Peer-reviewed articles or preprints, 2015–2026, English.
- Included if the paper does one of these:
  1. trains or evaluates a deep-learning fetal-ultrasound plane classifier or detector;
  2. reports shortcut, confounding or spurious-correlation behaviour of medical-imaging deep learning (ultrasound, radiograph or dermatology);
  3. reports external, multi-device or multi-site validation of fetal-ultrasound AI;
  4. evaluates saliency faithfulness, sanity or localisation against ground truth in medical imaging;
  5. reports fetal or obstetric ultrasound AI developed or validated in India.
- Excluded:
  - non-imaging studies;
  - editorials without data;
  - duplicates;
  - conference abstracts with no full text.
- A row enters `review_evidence_table.csv` with `verified=y` only after its full text or abstract has been checked.

Each search block lists the date, database, exact query, filters and hit count. The **screened** and **included** columns are filled in during the screening pass.

| # | date | database | exact query | filters | hits | screened | included |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-13 | PubMed (via PubMed connector) | `("fetal" OR "foetal") AND ultrasound AND ("standard plane" OR "plane classification" OR "plane detection") AND ("deep learning" OR "convolutional")` | publication date 2015–2026 | 56 | 56 | 13 |
| 2 | 2026-11-13 | PubMed | `("shortcut learning" OR "confounding" OR "spurious correlation") AND (ultrasound OR radiograph OR dermatology) AND "deep learning"` | 2015–2026 | 114 (115 on the same-day re-run) | 115 | 8 |
| 3 | 2026-11-13 | PubMed | `("external validation" OR "domain shift" OR "generalization" OR "generalisation") AND fetal AND ultrasound AND ("artificial intelligence" OR "deep learning")` | 2015–2026 | 60 | 60 | 5 (3 also found by #1) |
| 4 | 2026-11-13 | PubMed | `("saliency" OR "Grad-CAM" OR "explainability" OR "explainable") AND ("sanity check" OR "faithfulness" OR "localization" OR "localisation") AND "medical imaging"` | 2015–2026 | 65 | 65 | 1 |
| 5 | 2026-11-13 | PubMed | `India AND (ultrasound OR sonography) AND ("artificial intelligence" OR "deep learning") AND (fetal OR antenatal OR obstetric)` | 2015–2026 | 73 | 73 | 5 (3 also found by #1) |
| 6 | 2026-11-13 | PubMed | `"FETAL_PLANES_DB" OR "Burgos-Artizzu XP"[Author]` | 2015–2026 | 24 | 24 | 2 (+ the 2022 author correction, noted on the FPDB row) |

**Screening pass (2026-11-13).** All 376 unique records (393 hits, 17 of them duplicates across searches) were retrieved through the PubMed connector, and every title was screened.
- Abstracts were read for each candidate.
- An included record becomes an evidence-table row with its PMID in `notes`. Per-search counts come from those PMIDs.
- The usual exclusion reasons:
  - search #2: the wrong modality or task (MRI/CT/omics studies with no shortcut analysis);
  - search #4: saliency used only as an illustration, with no faithfulness or localisation test;
  - search #5: Global Burden of Disease papers and non-ultrasound oncology, which matched "India" incidentally;
  - search #6: author-name matches in other fields (dentistry, preterm birth).
- Fetal-plane classifiers on FPDB that report only in-distribution accuracy were included as a group (search #1). They are the review's evidence that external, per-device and calibration reporting is rare.

**Non-PubMed verification (2026-11-13).**
- arXiv abstract pages: Adebayo 2018, Mikolaj 2023, Wong 2025, Barrientos 2026 and Maani 2025.
- Di Vece is now published in *Sci Rep* 2026 (PMID 41981094), and its row was updated to that version.
- Geirhos 2020 was checked against its Semantic Scholar record, because the publisher page needs a login.
- Still unverified, with reasons:
  - DeGrave 2021 (*Nat Mach Intell*): publisher login; the PubMed record 32995822 is the 2020 medRxiv preprint;
  - Saporta 2022: publisher login, and the metadata-API request was not permitted;
  - Lim 2024 (*WFUMB Ultrasound Open*): the publisher returned only a placeholder page;
  - Phutke 2018 (IJME): the article page was not found;
  - ICMR 2023: the guideline PDF has not been located.
- The human can verify these through institutional access.

**Notes**
- Wildcards (`confound*`, `generali*`, `sonograph*`, `explainab*`) are not accepted by the connector. They were expanded to the explicit word forms shown above. PubMed's automatic term mapping adds MeSH synonyms; the full translated queries are kept in the connector output.
- Scopus, Google Scholar, arXiv and medRxiv have not been *searched* yet (arXiv records above were looked up, not searched). The plan leaves these to the human, who has access. arXiv matters for computer-vision work on shortcut learning and saliency sanity checks, such as Adebayo 2018 and Geirhos 2020, which PubMed does not index.
- All searches must be re-run within the 4 weeks before submission, with the change in results recorded here.
