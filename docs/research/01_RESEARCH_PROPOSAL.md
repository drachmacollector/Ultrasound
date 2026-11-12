# Research Proposal — Anatomy or Acquisition?

**Working title:** *Anatomy or Acquisition? Auditing shortcut reliance and anatomical grounding of deep-learning fetal standard-plane classifiers, with multi-device external validation.*

**Repository:** `drachmacollector/Ultrasound` (audited at commit `b854a34`, 2026-11-09)
**Companion document:** `02_IMPLEMENTATION_PLAN.md` (repository changes and step-by-step instructions for Claude Code)
**Status:** Proposal v1.1 (2026-11-09). Pre-registration draft. No results in this document are final.

> **v1.1 scope corrections (from Nakul):**
> 1. **No ethics-committee or hospital involvement at this stage.** All current work uses public, de-identified datasets under their published licences, and collects no new human data. The Indian external validation and any clinician annotation are moved to a **long-term future extension** (§5.5). Nothing in the current plan depends on them.
> 2. **Near-term publication target is a review article in JAPI** (§9). It will be supported by an illustrative analysis on public data from this study. The full technical study remains a later, separate output.
> 3. **Manual work is limited to quick binary checks** (yes/no per image: calipers present? on-screen text present?). No region drawing, box annotation or frame timestamping. Where a task would need that, it is replaced by an automatic method or dropped (§5.6).

---

## 0. Summary

**The pivot in one paragraph.** The repository built a good engineering system: a real-time fetal plane classifier with smoothing, Grad-CAM and a UI. It has not yet produced a novel, publishable scientific finding. The audit below shows that the repository's own data already contains the seed of one, and it is not where the project was looking. FETAL_PLANES_DB (FPDB) is the public benchmark used by most fetal-plane AI papers. In FPDB, the class label is strongly entangled with *how* the image was acquired. In the training pool, 65.1% of Aloka images are "Other", against 13.6% of Voluson E6 images. Operator "Op. 3" scanned only on the Aloka. The Voluson S10 appears almost only in the test split (11 training images, 1,071 test images). A classifier given **no pixels at all**, only the machine and operator fields, separates "Other" from standard planes with AUROC 0.735 on the 4,200 test images from machines seen in training (computed in this audit from `data/splits/*.csv`). Our own model loses 10–15 percentage points on external devices, and most of those errors are standard-plane heads relabelled "Other". One explanation fits all of this: models trained on FPDB may partly learn *acquisition signatures* (device texture, operator habits, on-screen calipers and text, image geometry) instead of fetal anatomy. Published accuracies of 90–99% would then overstate real-world validity. This matters most where the devices differ from the Spanish training machines, which includes most Indian imaging settings.

**What we will do.** We will (1) quantify acquisition confounding in FPDB and how much trained classifiers depend on it. (2) Measure, against real anatomical ground truth, whether saliency-based "localisation" actually falls on fetal anatomy. (3) Test whether a model's reliance on anatomy predicts its accuracy on unseen devices. (4) Test whether a simple, deployable *anatomy-focused training protocol* reduces shortcut reliance and narrows the external-validity gap, on European and African low-resource devices. All datasets are public. No ethics approval is needed for the current scope.

**Why it is defensible.** It needs no new model architecture. It answers a question clinicians and regulators ask: *"will this tool work on our machines?"* It uses public data, so it can be reproduced. Its conclusions hold whichever way the results go. If shortcut reliance is large, the benchmark literature is miscalibrated. If it is small, we will have shown that FPDB-trained models are anatomy-grounded, and that the external drop comes from device physics rather than shortcuts. Either result is new.

---

## 1. Where the project stands, and why it must change

### 1.1 What the existing work established (verified from repo files)

| Finding | Source in repo | Scientific value |
|---|---|---|
| ConvNeXt-Tiny test macro-F1 0.8927 on FPDB official test split (5,271 images, 896 patients) | `docs/EVAL_REPORT.md §1`, `checkpoints/convnext_tiny/classification_report_TEST.txt` | Baseline, not novel. Single seed. Backbone chosen with knowledge of test results (disclosed). |
| Cross-device collapsed accuracy 83.6% (HC18 81.4%, UCL 88.9%). 180 of 194 Head errors predicted "Other" | `docs/EVAL_REPORT.md §2`, `logs/eval/evaluate_cross_device.txt` | **Key preliminary evidence.** The failure mode is "rejection", not confusion. This is consistent with a learned acquisition prior. |
| 6-backbone cross-device sweep: gap −10.7 to −16.6 pp | `docs/CROSS_DEVICE_BACKBONE_COMPARISON.md` | Useful. No CIs, single seed, so the "winner" (`repvgg_a1`) is not statistically established. |
| Bootstrap model comparison (2,000 iterations) | `docs/phases/phase_04/bootstrap_significance_output.txt` | Method is good, but **image-level**. Images are clustered within patients, so CIs are too narrow. Must be redone with patient-clustered resampling. The bootstrap script itself is not in the repo. |
| ACAM and focal-loss ablations: neutral/negative | `docs/phases/phase_08/2509.00808_ACAM.md`, `docs/EXPERIMENTS.md` | Valid negative results. Not relevant to the new question. Supplementary at most. |
| NatalIA phantom: standard-plane recall 4.6%; the trivial "always Other" baseline beats the model | `docs/EVAL_REPORT.md §2b` | Shows extreme-shift failure. Phantom anatomy plus 98% "No plane" prevalence make it unsuitable as primary evidence. |
| Smoothing: 90.9% flicker reduction. Transition latency 489 ms (synthetic) vs 2,750 ms with 14/14 failed (NatalIA) | `docs/EVAL_REPORT.md §3–5` | Engineering results, not research contributions. The audit (`docs/FetScan_Final_Audit_Report.md §1.1`) already found the latency headline poorly evidenced. Retire. |
| CAM-box spot check (20 images, qualitative) | `docs/CAM_LOCALISATION_SPOTCHECK.md` | Says "no ground-truth boxes exist on FPDB". **That is no longer true** (see §5.2). Becomes a quantitative experiment. |

**Bottom line.** The existing work has **not** yet produced a finding that is novel, clinically meaningful and peer-review-proof. It has produced (a) solid infrastructure: patient-disjoint splits, manifests, six trained backbones, cross-device evaluation and Grad-CAM; (b) one important unexplained observation, the "Other-fallback" under device shift; and (c) the metadata needed to explain it. The new direction builds on all three.

### 1.2 Why the original goal (real-time plane assistant) is not the research target

- **Clinical need is weak for the original audience.** The project's own audit (`FetScan_Final_Audit_Report.md §4.2`) concluded that trained sonographers rarely need to be told which plane they are on. We agree.
- **No real-video ground truth exists.** No public dataset has annotated plane transitions in real patient video. NatalIA is a phantom. So the temporal claims cannot be validated on real data.
- **Plane classification on FPDB is saturated** as an accuracy race: dozens of CNN, ensemble, attention and foundation-model papers report 90–99%. Another architecture would not be novel.
- **What is not saturated** is whether those numbers mean what they appear to mean. That is the gap this proposal targets.

---

## 2. Literature review and gap analysis

Searches were run live on 2026-11-09. The existing `docs/literature_survey_and_gap_analysis.md` (SonoNet lineage, ScanAhead, COMFormer, FetalCLIP/USFM, quality assessment) is still valid background and is not repeated here. This section covers what decides the new direction.

### 2.1 The benchmark and its use

- **FETAL_PLANES_DB** (Burgos-Artizzu et al., *Sci Rep* 2020). 12,400 images, 1,792 patients, two Barcelona hospitals, several operators, Voluson E6/S10 and Aloka devices. Per-image `Operator` and `US_Machine` fields are included. These fields have, as far as our search found, never been analysed as confounders.
- A 2026 systematic-review **preprint** (Lundström-Imanov & Günalp, arXiv:2601.00990, 78 studies) reports that 56% of fetal-plane AI studies used this dataset. Only 35% tested on an unseen centre, with a median accuracy drop of 0.064. Only 28% ran any quantitative test of explanation faithfulness, and only 24% reported calibration. *Caveat: an unreviewed preprint. Use it as a pointer, not a citation of record.*

### 2.2 Shortcut learning: established in medicine, thin in fetal ultrasound

| Work | What it showed | Relevance |
|---|---|---|
| Geirhos et al., *Nat Mach Intell* 2020; Zech et al., *PLoS Med* 2018; DeGrave et al., *Nat Mach Intell* 2021; Winkler et al., *JAMA Dermatol* 2019 | Deep models exploit hospital, scanner, text tokens and skin markings. Performance collapses when the confound breaks. | General principle. Well established. |
| **Mikolaj et al., "Removing confounding information from fetal ultrasound images"**, arXiv:2303.13918 (Copenhagen; Feragen/Tolsgaard group) | On private Danish screening data, embedded text and calipers confound plane classification. Removing them by masking, noise fill, inpainting, U-Net or GAN improved accuracy on marker-free test images. | **Closest prior work.** It establishes that the phenomenon exists in fetal ultrasound. It used private data, did not study device or operator confounding, and did not link the effect to saliency or external generalisation. |
| Wong et al., "Influence of classification task and distribution shift type on OOD detection in fetal ultrasound", **MICCAI 2025** | A *scanner-classification* model is a better OOD detector for image-characteristic shift than a plane model. Plane accuracy falls from private data → BCNatal (FPDB) → African sets. | Shows scanner signal is strongly present in fetal ultrasound images. Did not test whether plane classifiers *use* it. |
| Wong et al., "Weight Space Correlation Analysis", arXiv:2512.13144 | A way to test whether a clinical head relies on scanner metadata (preterm-birth task). | A method we can borrow as a secondary probe. |
| Echo-DM (Wang et al., 2026) | Diffusion-based removal of calipers and text in ultrasound, motivated by shortcut bias. | Possible mitigation tool. Code not yet released. |

**Gap 1, largely unexplored on the public benchmark:** No study we found measures, on FPDB, (i) how strongly the class label is confounded with device, operator and site, (ii) whether trained models rely on that confound or on overlays, and (iii) how this relates to external failure. Mikolaj et al. is the closest work, but on private data with no device/operator analysis. *Novelty status: explored in a related setting, but not on the benchmark that most published claims rest on, and never with device/operator confounding.*

### 2.3 Localisation and explanation faithfulness

- **SonoNet** (Baumgartner et al., *IEEE TMI* 2017) and **Schlemper et al.** (arXiv:1808.00793, DLMIA 2018) derived weakly supervised boxes from classifiers. They evaluated them on limited manually boxed subsets, on private data.
- In general medical imaging, saliency maps often fail localisation and sanity tests: Adebayo et al., NeurIPS 2018; Arun et al., *Radiol AI* 2021; Saporta et al., *Nat Mach Intell* 2022.
- **Key enabling resource we found:** the **multi-centre landmark biometry benchmark** (Di Vece et al., arXiv:2512.16710, CC BY-NC-SA 4.0). Its **FP subset is FPDB itself**: 3,090 images, 1,047 subjects, with expert landmarks for BPD/OFD, TAD/APAD and FL, plus per-image pixel spacing and a device field. Its HC18 and UCL subsets carry the same landmarks. The project already has this bundle (`data/raw/ucl_hc18/`), but excluded FP to prevent train/test leakage in the cross-device study. That exclusion was correct for that purpose. **For localisation, FP is exactly what we need:** anatomical ground truth on our own test images. The authors also report that FP images were cropped to remove overlays, and that "in most cases" images were saved both with and without calipers. If confirmed, that gives a natural paired counterfactual (§5.1, E5).

**Gap 2, partly explored but inadequately validated:** quantitative, anatomy-grounded evaluation of plane-classifier saliency on the public benchmark, and whether "looking at anatomy" predicts generalisation to new devices. We found no study that tests this association.

### 2.4 External validity and low-resource settings

- **Sendra-Balcells et al., *Sci Rep* 2023.** FPDB-trained models degraded in five African countries (Malawi Mindray DC-N2, Egypt Voluson P8, Uganda Siemens X600, Ghana EDAN DUS-60, Algeria Voluson S8). Fine-tuning on a handful of local patients recovered most of the loss. The dataset is public (Zenodo 7540448, CC BY 4.0, classes: abdomen, brain, femur, thorax). **This is a ready-made low-resource external test set that covers thorax**, closing a gap the repo lists as "no external dataset available" (`EVAL_REPORT.md §5.2`).
- **Barrientos et al., arXiv:2605.27796 (2026)** benchmarked ultrasound foundation models (USFM, MOFO, UltraSAM, FetalCLIP) against CNNs on FPDB → African data. Foundation models generalise well, but no confounding or saliency analysis was done. *Implication: a "which model is best externally" paper is already taken. Our contribution must be the mechanism, not the leaderboard.*
- **FetalCLIP** (Maani et al., 2025) pretrained on about 208k images from one Abu Dhabi hospital, with fan cropping and inpainting of coloured overlays during preprocessing. That is itself evidence that the field treats overlays as a nuisance. Weight licence must be checked before use.

### 2.5 The Indian context

| Fact | Source | Implication for us |
|---|---|---|
| 85.9% of recent mothers received an ultrasound in NFHS-5 (2019–21), with ultrasound showing the largest wealth inequality (33.3 pp) of all ANC components | Chi et al., *Sci Rep* 2024 | Getting a scan is not the main problem. Quality and consistency of scans across very different facilities are. |
| ~1 radiologist per 100,000 population (vs ~1/10,000 in US), as reported in the press | ThePrint, June 2024 (press figure; verify before citing) | Pressure toward AI-assisted QC in diverse, often lower-end equipment environments. |
| **GAUGE** (GARBH-Ini cohort, Gurugram; Voluson E8): an India-specific deep model for gestational age. On external validation, **62% of external images were rejected as out-of-distribution**, partly attributed to lower resolution | medRxiv 10.1101/2024.05.13.24305466 (IIT Madras/THSTI) | **Direct Indian evidence that device and acquisition shift is the main barrier to deploying ultrasound AI in India.** This is the strongest local motivation for our question. |
| Garbhini-GA2 shows Western dating formulas (Hadlock) are biased in Indian women; Indian population-specific EFW models outperform 12 published formulas | THSTI/IIT-M (Lancet Reg Health SE Asia, 2023); Hiwale et al., *Ultrasonography* 2019 | Indian ultrasound AI must be validated locally. Population and device both matter. |
| **PCPNDT Act (1994; amended 2003):** registration of every machine and facility, mandatory records for at least 2 years, restricted sale of machines, only qualified/registered persons may scan, sex determination prohibited | ThePrint 2024; IJME 2018 (Phutke et al.); press coverage of 2026 reform debate (*The Hindu*, June 2026, via secondary summaries) | (a) Indian image data must exclude anything that could reveal fetal sex, and be handled under facility records rules. (b) Task-shifting to untrained operators (blind-sweep, NatalIA-style) is legally constrained in India, which weakens the NatalIA direction for an Indian paper. (c) Reform proposals discuss "AI-safeguarded" imaging. Validated, auditable AI behaviour is relevant to policy. Regulatory status of AI in ultrasound is currently ambiguous (conflicting press reports), so make no regulatory claims. |
| ICMR *Ethical Guidelines for AI in Biomedical Research and Healthcare* (2023) | ICMR | Not triggered by the current public-data scope. Relevant to the review's discussion of how Indian institutions should validate AI tools, and to the future extension (§5.5). |
| A 2026 narrative review of AI fetal ultrasound in low-resource settings found **no India-only AI fetal-ultrasound plane study** | Kumari & Meena, *Cureus* 2026 | A gap the JAPI review can state explicitly. Filling it with an Indian validation is a future extension (§5.5), not current work. |

### 2.6 Novelty ledger: be precise about what is new

| Claim | Status | How we phrase it |
|---|---|---|
| Deep models can exploit calipers/text in fetal US | **Established** (Mikolaj 2023) | "Consistent with prior private-data findings…" |
| Scanner identity is strongly encoded in fetal US images | **Established** (Wong 2025) | Replicate briefly as a sanity check. Do not claim it. |
| FPDB-trained models degrade on external/LMIC devices | **Established** (Sendra-Balcells 2023; Barrientos 2026) | Used as the *outcome*, not the contribution |
| Device/operator × class-prevalence confounding exists in FPDB, and trained models rely on it | **Unexplored, to the best of our search** | Primary contribution 1 |
| Paired caliper / no-caliper counterfactual test on public data | **Not done on public data** | Contribution 1b, if the FP pairing is confirmed |
| Quantitative saliency faithfulness against expert landmarks on FPDB | **Explored on private data, inadequately validated** | Contribution 2 |
| Saliency-on-anatomy predicts external-device correctness | **Unexplored, to the best of our search** | Contribution 3 (the most novel; also the riskiest) |
| An anatomy-focused protocol (fan masking + overlay handling + acquisition-balanced sampling) narrows the external gap | **Components explored separately; the combination is untested against a confounding-aware endpoint** | Contribution 4 (incremental but practical) |
| Indian external validation of a plane classifier | **Unexplored** | Future extension only (§5.5). In the review it is stated as a gap, not claimed as work done. |
| A review synthesising shortcut learning, external validity and explainability for fetal-plane AI, with Indian deployment considerations (PCPNDT, device diversity, GAUGE OOD evidence) | **No such review found** (existing reviews cover applications or LMIC access, not validity) | JAPI review article (§9) |

> **Honesty rule for the paper:** never say "first" without a dated, documented search. Prefer "to our knowledge, no prior study has…", keep the search log in `docs/research/literature_search_log.md`, and redo it before submission.

---

## 3. The research problem

### 3.1 Problem statement

Fetal standard-plane classifiers report near-expert accuracy on public benchmarks but degrade on unseen devices and sites. It is unknown how much of their benchmark accuracy comes from fetal anatomy, and how much from acquisition-related signals that are correlated with class labels in the training data: device texture and settings, operator and site practice, on-screen calipers and text, image geometry. It is also unknown whether the "localisation" heatmaps used as evidence of trustworthiness actually point at anatomy, and whether they can warn us before a model fails on a new machine.

### 3.2 Why it matters clinically

1. **Procurement and deployment decisions.** Hospitals and public programmes in India (district hospitals, PMSMA camps, private diagnostic chains) use very different machines from the Spanish ones behind FPDB. A tool's benchmark accuracy is only useful if it transfers. GAUGE's 62% OOD rejection rate shows this is the main barrier in India.
2. **Silent vs loud failure.** A plane classifier inside a completeness checker or auto-biometry pipeline that relabels true standard planes as "Other" causes missed or repeated measurements. One that hallucinates the wrong anatomy is worse. Clinicians need to know which failure mode happens, on which devices.
3. **Explainability is used as reassurance.** Heatmaps appear in AI products and papers as evidence that "the model looks at the right thing". If they do not track anatomy, that reassurance is false. This matters for clinician trust and for regulators.
4. **Scientific hygiene of the field.** If the dominant benchmark rewards shortcut use, the field's progress measure is biased. A reproducible audit, with a corrected evaluation protocol, is a lasting contribution.

### 3.3 Central hypothesis

> **H0 (central):** A measurable share of the accuracy of FPDB-trained fetal plane classifiers is driven by non-anatomical acquisition signals that are confounded with class labels. The degree of this reliance, measured by anatomy-grounded saliency and counterfactual tests, is associated with performance loss on external devices. Training that removes or decorrelates these signals reduces reliance and narrows the external-validity gap without reducing in-distribution accuracy beyond a pre-specified margin.

### 3.4 Research questions and testable predictions

**RQ1. Confounding: is acquisition information predictive of the label in FPDB, and is it encoded in the classifiers?**
- P1a: a no-pixel metadata model (machine, operator, image size) predicts Standard-vs-Other above chance. *Preliminary result: AUROC 0.735 on test images from machines seen in training.*
- P1b: machine and operator identity can be decoded from frozen classifier features with high accuracy under patient-grouped CV.
- P1c: on the **Voluson S10** test subset, where the training prior "Voluson ⇒ standard plane" is wrong (51.6% Other, vs 13.6% Other for Voluson E6 in training), "Other" recall is lower than on Aloka or E6 test images with the same true class.

**RQ2. Reliance: do classifiers use non-anatomical content?**
- P2a (*availability*): a model trained on **non-anatomical input only** (everything outside the ultrasound sector, plus detected overlays, with the sector blanked) reaches above-chance balanced accuracy.
- P2b (*reliance*): removing overlays by inpainting, or masking the out-of-sector region, changes the baseline model's predictions more often than an anatomy-preserving control perturbation of equal pixel area.
- P2c (*paired counterfactual*): for the same scan plane saved with and without calipers, predictions or confidence differ systematically. Only if such pairs are confirmed (Gate G1).

**RQ3. Grounding: does saliency localise anatomy, and does that predict external success?**
- P3a: the share of saliency energy inside the expert-landmark anatomy region varies by class and backbone, and is lowest for femur and for images with overlays.
- P3b: on external images (HC18, UCL, African), per-image on-anatomy saliency predicts correctness (AUROC > 0.5, cluster-robust) after adjusting for model confidence.
- P3c: saliency maps pass the model-randomisation sanity check. Otherwise they cannot support P3a–P3b.

**RQ4. Mitigation: does an anatomy-focused training protocol (AFP) help?**
- P4a (**pre-registered primary endpoint**): AFP improves pooled external balanced accuracy over the baseline protocol (paired, patient-clustered bootstrap, 3 seeds × 2 backbones).
- P4b: AFP lowers the shortcut indices (P2a, P2b) and raises on-anatomy saliency (P3a).
- P4c (non-inferiority): AFP's in-distribution macro-F1 is within −0.02 of baseline.

*(The former RQ5, Indian external validity, is now a future extension. See §5.5.)*

---

## 4. Data

| Dataset | Role | Size (verify) | Classes usable | Notes |
|---|---|---|---|---|
| **FETAL_PLANES_DB** | Train / val / in-distribution test | 12,400 images, 1,792 patients | All 8 | Keep existing patient-disjoint splits (`data/splits/*.csv`). Per-image `source_machine`, `operator`. The S10 sub-test is a built-in "confound-broken" subset. |
| **Landmark benchmark: FP subset** | **Localisation ground truth on FPDB images** | 3,090 images (Head 1,637 / Abdomen 693 / Femur 760) | Head, Abdomen, Femur | **Never used for training.** Use only images whose filenames are in our `test.csv`. Must verify image identity and coordinate frame vs FPDB (cropping). |
| **Landmark benchmark: HC18** | External test + localisation GT | 999 images, 806 subjects, Voluson E8/730, NL | Head | Already in `cross_device_manifest.csv` |
| **Landmark benchmark: UCL** | External test + localisation GT | 424 images, 51 pregnancies, UCLH | Head, Abdomen, Femur | Already in manifest |
| **African planes** (Zenodo 7540448) | **New external test** (LMIC devices) | ~450 images, ~100 patients, 5 countries | Brain (→Head), Abdomen, Femur, **Thorax** | Download and verify counts. Has its own CSV with patient IDs. |
| NatalIA PBF-US1 | Supplementary only (extreme-shift stress test) | 19,212 frames | Head/Abdomen/Femur/Other | Phantom. Do not use for primary claims. |
| IUGC clips | **Retire** | — | none | Intrapartum task, no plane labels |

All current datasets are public and de-identified, and are used under their licences (FPDB: per its Zenodo record; landmark benchmark: CC BY-NC-SA 4.0; African: CC BY 4.0; HC18: challenge terms). Non-commercial terms are fine for research and for a journal article. No new human data are collected. **Ethics statement for publications:** "This study used only publicly available, de-identified datasets released for research under their respective licences and original ethics approvals; no new data were collected from human participants." JAPI's instructions to contributors do not mention an IEC requirement. If a journal still asks, an institutional "exempt/not human-subjects research" note is the usual answer. That is a one-page letter, not a committee review.

**Label harmonisation for external sets.** We use a collapsed 4-class anatomical taxonomy {Head, Abdomen, Femur, Thorax}, plus Cervix where available. The three brain sub-planes collapse to Head. We report two scorings:
1. **Full-argmax:** predicting "Other" or an absent class is an error. This is the clinically relevant measure, and it is what the repo does now.
2. **Restricted-argmax:** argmax over the classes present in that external set. This isolates *anatomy confusion* from *rejection*, as in Barrientos et al.

The **gap between the two** is our "rejection rate", a key descriptive outcome.

---

## 5. Methods

All experiments follow these fixed rules:
- **Patient-disjoint everything.** The test sets are locked. Model selection uses validation only. Seeds are fixed.
- **Every reported metric has a 95% CI** from a **patient-clustered bootstrap** (resample patients, not images; 2,000 replicates). Model comparisons use the **paired** clustered bootstrap.
- **Per-image predictions are stored** (probabilities, plus CAM statistics) for every model × dataset, so every number can be regenerated without inference.
- Analyses are **pre-registered** (OSF) before running RQ4.

### 5.1 RQ1–RQ2: confounding and reliance audit

| ID | Experiment | Output metric | Notes |
|---|---|---|---|
| **E1** | Descriptive confounding tables: machine × class, operator × class, split × machine; image geometry (width, height, aspect ratio, sector shape) × class | Cramér's V; tables | Already partly computed in this audit (§0) |
| **E2** | **No-pixel metadata model:** gradient-boosted trees on {machine, operator, H, W, aspect ratio, mean intensity, % pixels outside sector}, trained on train+val, tested on test. Report with and without S10. | AUROC (Standard-vs-Other), balanced accuracy (8-class) | Quantifies how much label information is available without anatomy |
| **E3** | **Feature decodability:** linear probe on frozen penultimate features of each trained backbone to predict machine and operator (patient-grouped 5-fold CV on test features) | Balanced accuracy vs chance; compare with an ImageNet-only (untrained-head) feature baseline | Shows whether fine-tuning *amplifies* acquisition information |
| **E4** | **Availability test:** train the same backbone on (a) *non-anatomical-only* images (sector blanked to mean grey, overlays and outside-sector kept) and (b) *overlay-only* images (blank canvas plus detected overlay pixels) | Test balanced accuracy vs chance (12.5%) and vs the full-image model | Above chance means the shortcut is available |
| **E5** | **Reliance test on the baseline model:** evaluate the unchanged baseline on (a) overlay-inpainted test images, (b) out-of-sector-masked images, (c) **area-matched control**: inpaint random in-sector patches of the same total area. If FP pairs are confirmed: (d) paired with-caliper vs without-caliper images of the same plane | Prediction-flip rate, Δ confidence, Δ macro-F1, with paired CIs | Reliance = (a) or (b) flip rate significantly greater than (c) |
| **E6** | **Stratified performance:** in-distribution metrics by machine (Aloka, E6, **S10**, Other) and by overlay presence (from the overlay detector) | Per-stratum macro-F1, "Other" recall, confusion matrices | Tests P1c directly |

**Overlay detection** (needed for E4–E6 and mitigation) is a small classical pipeline: sector segmentation (threshold + largest contour + convex hull); within-sector caliper/marker detection (morphological top-hat + template matching for cross and plus shapes, dotted-line detection); outside-sector text via OCR (EasyOCR/PaddleOCR) or connected components. It is **validated at image level against a quick binary manual audit**: 300 stratified FPDB images, each marked y/n for calipers and y/n for text, with precision and recall reported (§5.6). Without that validation it cannot be used.

### 5.2 RQ3: anatomy-grounded localisation

**Ground-truth anatomy regions** come from expert landmarks (FP/HC18/UCL):
- **Head:** ellipse centred at the OFD midpoint, semi-major axis OFD/2 along the OFD direction, semi-minor axis BPD/2. HC18's original ellipse masks are available as a cross-check.
- **Abdomen:** ellipse from TAD and APAD, built the same way.
- **Femur:** a capsule (line segment between FL endpoints, dilated by 10% of FL length). Report a sensitivity analysis at 5% and 15%.

**Saliency methods.** Grad-CAM (existing), Grad-CAM++, and one perturbation method (occlusion sensitivity or RISE) as a gradient-free reference.

**Metrics per image:**
- **Energy-based pointing game (EPG):** share of positive saliency mass inside the anatomy region.
- **Pointing-game hit:** is the saliency maximum inside the region?
- **CAM-box IoU** (existing `cam_to_bbox`) against the landmark-derived box.
- **Out-of-sector saliency fraction:** needs no landmarks, so it is available for all 8 classes.
- **Overlay saliency fraction:** saliency mass on detected overlay pixels.

**Sanity check (gating).** Model-parameter randomisation test (Adebayo 2018). The SSIM between maps from the trained and randomised models must be low. Methods that fail are reported and excluded.

**Association analysis (P3b).** On external images, a logistic regression with patient-cluster-robust SEs: correct ~ EPG + max-softmax + dataset + class. Report the AUROC of EPG alone, and the incremental AUROC over confidence. *This is the riskiest and most novel analysis. A null result is reportable.*

**No manual region annotation.** All anatomy regions come automatically from the existing expert landmarks. For classes without landmarks (brain sub-planes beyond the head outline, thorax, cervix, Other), we use only metrics that need no ground truth: out-of-sector and overlay saliency fractions. Clinician-drawn key-structure boxes (CSP, cerebellum and so on) would add depth but count as region annotation, so they are deferred to the future extension (§5.5).

### 5.3 RQ4: anatomy-focused training protocol (AFP)

The protocol is kept to four components, each cheap and deployable:

| Component | What | Rationale |
|---|---|---|
| **A1. Sector crop and mask** | Segment the ultrasound sector, zero everything outside it, crop to the sector bounding box, **letterbox** to the square input. The current pipeline resizes and distorts aspect ratio. | Removes UI, text and border signatures. Removes device-specific geometry distortion. |
| **A2. Overlay handling** | Inpaint detected calipers and markers (cv2 Telea; Echo-DM if released). **Plus caliper augmentation:** paste synthetic calipers at random positions on images of *all* classes, including "Other", with p = 0.3. | Breaks the "calipers ⇒ standard plane" correlation, following Mikolaj et al. |
| **A3. Acquisition-balanced sampling** | Importance weights w(x) = p(y) / p(y \| machine) for each training sample, so that within the sampler, class prevalence is the same for every machine | Removes the prevalence confound (Aloka 65% Other vs E6 14%) without discarding data |
| **A4. Device-style augmentation** | Speckle, gain, dynamic-range and blur randomisation, with wider ranges than now | Reduces reliance on device texture. Kept standard and simple. |

**Arms:** B0 (current protocol) · B1 (A1) · B2 (A1+A2) · **AFP** (A1+A2+A3+A4).
**Backbones:** `convnext_tiny` (current best in-distribution) and `tf_efficientnetv2_s` (statistically tied in-distribution, better on the hardest class). Plus a **frozen foundation-model linear probe** (FetalCLIP if the licence allows, otherwise USFM) on B0 vs AFP inputs. This tests whether large-scale pretraining already removes the shortcut.
**Seeds:** 3 per arm per CNN backbone, giving 24 CNN runs (about 0.5–1 GPU-hour each on an RTX 4060).

**Pre-registered primary endpoint:** pooled external balanced accuracy (full-argmax, collapsed taxonomy) over HC18 + UCL + African. AFP vs B0, averaged over seeds, per backbone. Paired patient-clustered bootstrap. Success means the CI excludes 0. Non-inferiority margin for in-distribution macro-F1: −0.02.

**Power (rough).** With ~1,870 external images and an expected discordance between paired models of about 10%, a paired comparison has ~80% power to detect a ~2.5 pp difference at α = 0.05 (McNemar approximation, ~1,250 images needed). Patient clustering will reduce effective n, so expect detectable effects of roughly 3–4 pp. Effects smaller than this will be reported as "not detected", not as "no effect".

### 5.4 Secondary analyses (kept deliberately small)

- **Calibration under shift:** ECE and reliability diagrams in- vs out-of-distribution, before and after temperature scaling fitted on validation.
- **Selective prediction:** risk–coverage curves using max-softmax. Report the coverage needed for 95% accuracy on each external set. This puts the "Other-fallback" into clinical terms: how often a clinician would have to take over.
- **Few-shot local adaptation (optional):** replicate Sendra-Balcells' finding by fine-tuning on k = 5 African patients, and compare the gain for AFP vs B0. *Do this only if time allows.*

### 5.5 Long-term future extensions (out of current scope; consider only if the first study works out)

These need a hospital partner and/or ethics approval, so they are **not planned or implemented now**. They are recorded here so the current design stays compatible with them.

- **F1. Indian external validation.** A retrospective, de-identified diagnostic-accuracy study of archived Indian standard-plane images from at least 2 device makes. Needs IEC approval under ICMR 2023 guidelines and a PCPNDT-compliant protocol: exclude any image showing fetal genitalia, collect no sex data, redact burned-in text at source. Reference standard: 2 sonologists plus an adjudicator. Rough size: about 200 images per class, from 1.96² × 0.85 × 0.15 / 0.05² ≈ 196. *Compatibility now:* keep the evaluation code dataset-agnostic and runnable offline, so it could later run on-site on a hospital machine.
- **F2. Clinician key-structure annotation.** Boxes for ISUOG structures, to test whether saliency sits on clinically used landmarks. This is region annotation, so it is deferred.
- **F3. Prospective or real-video evaluation.** Needs clinical data. Deferred.

### 5.6 Manual effort budget (current scope)

| Task | What the human does | Type | Estimated time |
|---|---|---|---|
| Overlay audit | For ~300 images shown one at a time, press y/n for **calipers present** and y/n for **on-screen text present** | Binary per image | ~30–45 min |
| Caliper-pair confirmation | For ≤50 auto-proposed image pairs, press y/n: "same scan, one with calipers and one without?" | Binary per pair | ~10 min |
| Landmark sanity glance | For 15 images with landmarks auto-plotted, press y/n: "do the dots sit on the head / abdomen / femur?" | Binary per image | ~5 min |
| Sector-mask glance | Shown together with the overlay audit: y/n "is the outlined sector roughly right?" | Binary, optional | 0 extra |

**Not required anywhere:** drawing regions, boxes or outlines, measuring anything, labelling planes, or timestamping frames. Where the earlier draft needed these, they were replaced by automatic methods: landmark-derived masks, template-matching registration, and automatic overlay detection validated against the binary audit. Otherwise they were dropped.

---

## 6. Expected results and what each outcome would mean

| Outcome | Likelihood (our prior) | Interpretation | Is it publishable? |
|---|---|---|---|
| Strong confounding (E2 AUROC ≳ 0.7), clear reliance (E5 flip ≫ control), S10 "Other" recall depressed, AFP improves external balanced accuracy by 3–8 pp | moderate | Benchmark accuracies overstate validity. Simple protocol changes help. | Yes, the strongest paper |
| Confounding present (E1/E2) but models show little reliance (E5 ≈ control), saliency anatomy-grounded, AFP ≈ B0 | moderate | FPDB-trained models are more robust to confounding than feared. The external drop is device physics or texture, not shortcuts. Points to local fine-tuning and data diversity, not preprocessing. | Yes. A careful negative result with a public, reproducible audit protocol |
| EPG predicts external correctness beyond confidence (P3b) | low–moderate | Saliency becomes a *deployment monitoring* tool, not decoration | Yes, high-novelty secondary finding |
| Saliency fails sanity checks | low | Heatmap-based "localisation" claims in this literature are unreliable | Yes, a caution with practical consequences |

**Realistic magnitudes**, based on our preliminary data and the literature:
- External collapsed accuracy for baselines: 80–88% (our current 82–87%).
- Rejection ("Other") share of errors: >70%.
- Machine decodability from features: >90%.
- AFP external gain: 0–6 pp.
- In-distribution change: within ±0.02 macro-F1.

---

## 7. Contributions to the literature

1. A **quantified confounding analysis of FPDB**, the most-used public fetal-plane benchmark: label–device/operator entanglement, its availability to models, and models' reliance on it. Released as a reusable audit protocol and code.
2. The first **anatomy-grounded, quantitative saliency evaluation** for fetal plane classifiers on public data, using expert biometry landmarks as ground truth, with sanity checks.
3. A test of whether **anatomical grounding predicts external-device failure**, linking explainability to deployment safety.
4. A **simple, reproducible anatomy-focused training and evaluation protocol** (sector masking, overlay handling, acquisition-balanced sampling, clustered-bootstrap external evaluation with rejection-vs-confusion scoring), evaluated on European and African low-resource devices.
5. A **review for physicians** (JAPI) that pulls together the evidence on shortcut learning, external validity and explainability in fetal-plane AI, illustrated with our public-data audit, and translated into a practical validation checklist for Indian institutions.

What we explicitly do **not** claim: a new state-of-the-art architecture, clinical readiness, real-time clinical benefit, or regulatory suitability.

---

## 8. Clinical applications and broader implications

- **Pre-deployment validation checklist** for hospitals and procurement committees adopting ultrasound AI: per-device external testing, rejection-rate reporting, overlay-robustness and saliency sanity checks. Directly usable by Indian institutions and ethics committees evaluating vendors.
- **Benchmark reform:** recommend that FPDB results be reported per machine and with overlay-controlled inputs, and that external tests use clustered CIs and two-way scoring.
- **Downstream safety:** plane recognition is the gate for automated biometry, completeness checks and QC. Showing *where* it fails (rejection on unfamiliar devices) tells system designers to fall back to a human, not to an automatic measurement.
- **Indian policy relevance:** PCPNDT reform debates mention AI-safeguarded imaging. Any such system depends on plane and anatomy recognition that is robust across Indian devices. Our protocol is one way to evidence that.
- **Equity:** the devices most likely to fail (lower-end, older, portable) are concentrated in poorer settings. NFHS-5 already shows ultrasound has the largest wealth gap among ANC components. Unvalidated AI could widen it.

---

## 9. Publication strategy

### 9.1 Primary near-term output: JAPI review article

**Why a review fits.** JAPI (general medicine; PubMed/Scopus indexed; Vancouver references) is read by physicians, not imaging specialists. A review aimed at clinicians, on **how AI imaging tools fail and how to judge them before adoption**, fits that readership better than a technical original article. Fetal ultrasound serves as the worked example.

**Format constraints** (from JAPI's *Instruction to Contributors* PDF; the PDF is undated and may be outdated, so **confirm with the editorial office**):
- Main text up to ~2,000 words, ≤16 pages including tables and figures.
- Abstract ≤200 words.
- **≤40 references.**
- 3–10 MeSH keywords.
- Up to 4 illustrations free.
- No IEC requirement is stated. Our ethics statement is in §4.

**Working title:** *"Is the algorithm seeing the fetus or the machine? Shortcut learning, external validity and explainability in AI for fetal ultrasound, with implications for Indian practice."*

**Type:** narrative review with a documented search strategy (databases, dates, terms, inclusion rules). This is stronger than a free narrative and achievable without PRISMA-level screening. It includes **one illustrative box/figure from our own public-data analysis**, clearly labelled as an illustrative re-analysis of public datasets and not a clinical study.

**Outline and word budget (~2,000 words):**

| Section | Words | Content |
|---|---|---|
| Introduction | 250 | Ultrasound in Indian ANC: 85.9% coverage but large quality and wealth gaps; AI promises; why validity matters to physicians |
| How AI recognises fetal planes | 250 | Plain-language CNN and benchmark basics; FETAL_PLANES_DB dominance; reported accuracies |
| When AI learns the wrong thing: shortcut learning | 400 | General medical examples (Zech, DeGrave, Winkler); fetal-US evidence (Mikolaj; Wong); **Box 1: our FPDB confounding analysis** (machine × class table, metadata-only AUROC, S10 effect, external "Other"-fallback) |
| External validity across devices and settings | 350 | Sendra-Balcells (Africa); Barrientos (foundation models); GAUGE's 62% OOD rejection in India; rejection vs confusion |
| Do heatmaps prove the AI looks at anatomy? | 300 | Saliency pitfalls (Adebayo; Saporta); our anatomy-grounded EPG result, if available by then |
| Indian context | 250 | PCPNDT constraints (sex-determination safeguards, operator qualification, records); device heterogeneity; ICMR 2023 AI ethics guidelines; absence of Indian validation studies |
| Practical checklist for physicians and institutions | 150 | **Table: 8–10 questions to ask before adopting an ultrasound AI tool** (per-device results? external sites? rejection rate? overlay robustness? calibration? local pilot?) |
| Conclusion | 50 | |

**Figures (≤4):** (1) Schematic of shortcut learning in ultrasound (acquisition signals vs anatomy). (2) Box 1 figure: machine × class confound plus metadata-only ROC. (3) Saliency example grid with landmark masks, from public data. (4) Checklist table.

**References:** curate ≤40. A ranked shortlist is built during the work in `docs/research/review_reference_shortlist.md`. Since JAPI's guidance prefers references "not older than 1995", that is no constraint here.

**Dependency on the study:** Box 1 needs only the R1–R3 results (confounding plus the existing-model audit), which are achievable in the first ~6 weeks. The saliency figure needs R4. The review does **not** wait for R5 (mitigation training).

**Authorship:** a clinician co-author (obstetrics or radiology) would strengthen the review's credibility for a physician readership. This needs no ethics approval, only their review of the clinical sections. Optional.

### 9.2 Later output: the full technical study

- The full audit plus mitigation (RQ1–RQ4) goes to a technical venue: MICCAI ASMUS/PIPPI workshop, then *Sci Rep*, *Computers in Biology and Medicine* or *Medical Image Analysis*. Alternatively, an Indian imaging journal (*IJRI*, *JOGI*) as an original article.
- **Reporting guidelines:** CLAIM 2024; STARD-AI (*Nat Med* 2025) for external-validation parts; TRIPOD+AI (*BMJ* 2024) where relevant. OSF pre-registration of RQ4.
- **Order:** the JAPI review can cite the study's preprint (arXiv/medRxiv) if the preprint is posted first. Post the preprint once R5 is complete.

### 9.3 Long-term

F1–F3 (§5.5) would support a clinical original article later, if the first study works out.

---

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| FPDB images contain few overlays/calipers | E4b, E5d, A2 become weak | Device/operator/prevalence confounding (E1–E3, E6, A3) stands on its own. Gate G1 decides the emphasis early. |
| FP landmark images are cropped or rescaled vs FPDB originals | Localisation GT misaligned | Template-matching registration with residual check. Images that fail registration are excluded and counted. |
| African dataset is small (~450 images) or classes are imbalanced | Wide CIs | Pool external sets for the primary endpoint. Report per-set results as secondary. |
| FetalCLIP weights are non-commercial or unavailable | Foundation-model arm missing | Use USFM, or drop the arm. It is secondary. |
| AFP gives no gain | Primary endpoint null | Pre-registered. Reported as such. Section 6 explains why this is still informative. |
| JAPI scope or format differs from the old PDF | Review needs reshaping | Confirm current article types and limits with the editor before drafting. The review content is portable to other Indian journals. |
| Binary overlay audit is too coarse | Detector validation weaker | Binary flags are enough to compute detector precision/recall at image level. Pixel-level accuracy is not needed for E5 because the area-matched control uses the same masks. |
| Agent-implemented code has silent bugs (a history of this in the repo) | Wrong numbers | Unit tests with synthetic ground truth for every metric. Independent audit of every phase (existing working method). Per-image prediction stores allow recomputation. |

---

## 11. Timeline (indicative, ~14–16 weeks part-time)

| Weeks | Work | Gate |
|---|---|---|
| 1 | R0 repo restructure; tag legacy | — |
| 2–3 | R1 data foundation; binary overlay audit (~45 min); FP↔FPDB linkage; African set | **G1:** overlay prevalence and FP pairing known; emphasis fixed |
| 4 | R2 evaluation core; reproduce baseline with clustered CIs | **G2:** baseline reproduced (macro-F1 within ±0.005 of 0.8927) |
| 5–6 | R3 confounding and reliance audit (E1–E6) on existing checkpoints | Box 1 for the review is available |
| 7–8 | R4 localisation (masks, metrics, sanity checks, association) | **G3:** saliency sanity check passed |
| 8 | Write and lodge OSF pre-registration for RQ4 | **G4:** pre-registered before any AFP training |
| 9–11 | R5 AFP training matrix and evaluation | — |
| 12–13 | R6 analysis, figures, results document; review-support artefacts | — |
| parallel (from week 2) | Literature search log + reference shortlist for the JAPI review; confirm JAPI format with the editor | — |
| after week 8 | JAPI review drafting can start (uses R1–R4 results) | — |

---

## 12. References (verified during this review unless marked)

- Burgos-Artizzu XP et al. Evaluation of deep CNNs for automatic classification of common maternal fetal ultrasound planes. *Sci Rep* 2020;10:10200. doi:10.1038/s41598-020-67076-5
- Baumgartner CF et al. SonoNet. *IEEE TMI* 2017. arXiv:1612.05601
- Schlemper J et al. Weakly supervised localisation for fetal ultrasound images. DLMIA 2018. arXiv:1808.00793
- Mikolaj K et al. Removing confounding information from fetal ultrasound images. arXiv:2303.13918 (2023)
- Wong CK et al. Influence of classification task and distribution shift type on OOD detection in fetal ultrasound. MICCAI 2025. arXiv:2509.18326
- Wong CK et al. Weight Space Correlation Analysis. arXiv:2512.13144 (2025/26)
- Sendra-Balcells C et al. Generalisability of fetal ultrasound deep learning models to low-resource imaging settings in five African countries. *Sci Rep* 2023;13:2728. Dataset: Zenodo 7540448
- Barrientos L, Du Y, Dvornek NC. Benchmarking ultrasound foundation models for fetal plane classification. arXiv:2605.27796 (2026)
- Maani F et al. FetalCLIP. arXiv:2502.14807 (2025); Saeed N et al. MobileFetalCLIP. arXiv:2603.05421 (2026)
- Di Vece C et al. A multi-centre, multi-device benchmark dataset for landmark-based comprehensive fetal biometry. arXiv:2512.16710 (2025)
- van den Heuvel TLA et al. Automated measurement of fetal head circumference (HC18). *PLoS One* 2018 *(from prior knowledge)*
- Wang Z et al. Echo-DM: ultrasound marker removal via conditional latent diffusion (2026, preprint)
- Lim A et al. Automatic standard plane and diagnostic usability classification in obstetric ultrasounds. *WFUMB Ultrasound Open* 2024;2:100050
- Lundström-Imanov G, Günalp Ö. Uncertainty-calibrated XAI for fetal ultrasound plane classification: a systematic review. arXiv:2601.00990 (preprint; treat with caution)
- Geirhos R et al. Shortcut learning in deep neural networks. *Nat Mach Intell* 2020 *(prior knowledge)*; Zech JR et al. *PLoS Med* 2018 *(prior knowledge)*; DeGrave AJ et al. *Nat Mach Intell* 2021 *(prior knowledge)*; Winkler JK et al. *JAMA Dermatol* 2019 *(prior knowledge)*
- Adebayo J et al. Sanity checks for saliency maps. NeurIPS 2018; Arun N et al. *Radiol Artif Intell* 2021; Saporta A et al. *Nat Mach Intell* 2022 *(prior knowledge)*
- GAUGE / GARBH-Ini deep-learning GA model. medRxiv 10.1101/2024.05.13.24305466 (2024)
- Garbhini-GA2, THSTI/IIT Madras, *Lancet Reg Health SE Asia* 2023 *(verify exact citation)*; Hiwale S et al. Indian population-based fetal weight models. *Ultrasonography* 2019;38:50–57
- Chi H et al. Socioeconomic and geographic inequalities in antenatal and postnatal care components in India, 2016–2021. *Sci Rep* 2024;14:10221
- Phutke G et al. Ultrasound in rural India: a failure of the best intentions. *Indian J Med Ethics* 2018
- Kumari P, Meena R. AI-assisted fetal ultrasound in low-resource settings (narrative review). *Cureus* 2026;18(8):e114172
- ICMR. Ethical Guidelines for Application of AI in Biomedical Research and Healthcare, 2023
- Press: ThePrint (12 Jun 2024), The South First (8 Feb 2026), The Hindu (26 Jun 2026, via secondary summaries) on PCPNDT and AI. Background only. Do not cite as evidence.
- Reporting: CLAIM 2024 update; STARD-AI (*Nat Med* 2025); TRIPOD+AI (*BMJ* 2024)
- NatalIA PBF-US1. González D et al. Zenodo 10.5281/zenodo.14193949 (2024)
