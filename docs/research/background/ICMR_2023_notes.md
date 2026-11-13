# ICMR (2023) ethical guidelines for AI in biomedical research and healthcare: notes for this study

**Record.** Indian Council of Medical Research. *Ethical Guidelines for Application of Artificial Intelligence in Biomedical Research and Healthcare.* New Delhi: ICMR (DHR-ICMR AI Cell); 2023. ISBN 978-93-5811-343-3.
- Checked against the PDF supplied by the user (`Ethical_Guidelines_AI_Healthcare_2023.pdf`, gitignored) on 2026-11-13.
- The publication forbids reproduction, so these notes **paraphrase** it and give section numbers. The full-text conversion (`ICMR_AI_Ethics_2023.txt`) stays local and is not committed.
- Page numbers are approximate; they come from the printed footers.

## Points that bear on this study

| Section (approx. page) | What it asks for (paraphrase) | How this study speaks to it |
|---|---|---|
| §1.3 Trustworthiness (pp. 16–17) | Results should be reliable across clinical settings, and explainable on scientific grounds. The "black box" deters adoption | R4: are heat-maps on the anatomy, and are they sanity-checked (G3)? |
| §1.5 Accountability (p. 21) | Regular internal and external audits, with reports made public; systems open to independent review | The whole study is an independent, public, pre-registered audit of published classifiers |
| §1.6 Data quality (pp. 23–24) | Bias must be checked before deployment; data should represent the target population; **pre-deployment testing at each new site** | E1 (machine × label confounding), E6 (per-device results), external sets from other devices and countries |
| §1.10 Validity (p. 29) | Rigorous clinical and field validation before use on patients; differences in training data make algorithms diverge | No clinical claims; external validation shows the divergence |
| §2.1 xiii Development (p. 33) | Consent can be waived for anonymised retrospective data | Public, de-identified datasets only |
| §2.2 Validation phase (pp. 34–35) | Validate on **independently collected data**, cover all population segments, prefer prospective validation, and report standard metrics (AUROC, F1 and others) | Patient-clustered CIs; external sets; a prospective local pilot is named as the necessary next step, not done here |
| §2.3 ii Deployment (pp. 35–36) | Local validation on the target population before deployment | Pre-adoption checklist (review figure 4): "per-device results" and "local pilot" |
| §3 D Ethics review (p. 40) | The type of ethics review (exempt, expedited or full) depends on risk | Public de-identified data; no new data collection |

## Use in the review

The guideline supports the review's central recommendation: a model's published accuracy does not carry over to a new site or device. Before adoption it needs per-device external results and local testing, which the guideline itself asks for (§1.6, §2.2, §2.3).

The guideline does not mention prenatal sex determination; PC-PNDT obligations come from the PC-PNDT Act, not from this document. The review's checklist item on data governance should cite the Act directly.
