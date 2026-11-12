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
| 1 | 2026-11-13 | PubMed (via PubMed connector) | `("fetal" OR "foetal") AND ultrasound AND ("standard plane" OR "plane classification" OR "plane detection") AND ("deep learning" OR "convolutional")` | publication date 2015–2026 | 56 | pending | pending |
| 2 | 2026-11-13 | PubMed | `("shortcut learning" OR "confounding" OR "spurious correlation") AND (ultrasound OR radiograph OR dermatology) AND "deep learning"` | 2015–2026 | 114 | pending | pending |
| 3 | 2026-11-13 | PubMed | `("external validation" OR "domain shift" OR "generalization" OR "generalisation") AND fetal AND ultrasound AND ("artificial intelligence" OR "deep learning")` | 2015–2026 | 60 | pending | pending |
| 4 | 2026-11-13 | PubMed | `("saliency" OR "Grad-CAM" OR "explainability" OR "explainable") AND ("sanity check" OR "faithfulness" OR "localization" OR "localisation") AND "medical imaging"` | 2015–2026 | 65 | pending | pending |
| 5 | 2026-11-13 | PubMed | `India AND (ultrasound OR sonography) AND ("artificial intelligence" OR "deep learning") AND (fetal OR antenatal OR obstetric)` | 2015–2026 | 73 | pending | pending |
| 6 | 2026-11-13 | PubMed | `"FETAL_PLANES_DB" OR "Burgos-Artizzu XP"[Author]` | 2015–2026 | 24 | pending | pending |

**Notes**
- Wildcards (`confound*`, `generali*`, `sonograph*`, `explainab*`) are not accepted by the connector. They were expanded to the explicit word forms shown above. PubMed's automatic term mapping adds MeSH synonyms; the full translated queries are kept in the connector output.
- Scopus, Google Scholar, arXiv and medRxiv have not been searched yet. The plan leaves these to the human, who has access. arXiv matters for computer-vision work on shortcut learning and saliency sanity checks, such as Adebayo 2018 and Geirhos 2020, which PubMed does not index.
- All searches must be re-run within the 4 weeks before submission, with the change in results recorded here.
