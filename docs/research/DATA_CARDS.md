# Data cards (R6 D.1)

- All counts come from R1 outputs: `results/json/R1_*.json`, `results/tables/R1_external_counts.md`, and the frozen split CSVs.
- No new human data were collected. Every dataset is public and de-identified.
- **Licence status.**
  - "verified" means the licence text was read at the source.
  - "per proposal" means it comes from `01_RESEARCH_PROPOSAL.md` §8.
  - The human checks every "per proposal" entry against the dataset record before submission.

---

## 1. FETAL_PLANES_DB (FPDB): training, validation and in-distribution test

| field | value |
|---|---|
| Source | Burgos-Artizzu XP, et al. *Sci Rep* 2020;10:10200. doi:10.1038/s41598-020-67076-5 (Zenodo record linked from the paper) |
| Licence | Per its Zenodo record (`data/raw/fetal_planes_db/LICENSE_NOTE.txt`); cite the paper. Status: per proposal, to verify on Zenodo |
| Centre | BCNatal, Barcelona (two hospitals) |
| Images / patients | 12,400 images. Frozen patient-disjoint splits: train 6,106 / 772 patients; val 1,023 / 124; test 5,271 / 896 (`data/splits/SPLITS_SHA256.txt`) |
| Devices | Aloka (5,114), Voluson E6 (5,862), Voluson S10 (1,082), Other (342) |
| Classes | 8: three trans-axial brain planes (trans-cerebellum, trans-thalamic, trans-ventricular), abdomen, femur, thorax, maternal cervix, Other |
| Use | Training, model selection (val only) and in-distribution test |
| Exclusions | None |
| Known quirks | Machine and class are confounded (R3 E1): 'Other' is 65% of Aloka training images vs 14% of Voluson E6. Voluson S10 is almost absent from train+val (11 images) but is 1,071 test images. Image size nearly identifies the machine. Frames carry burned-in device names, settings text, depth rulers and orientation marks. Filenames contain patient numbers, so they are never printed on figures |

## 2. Fetal-biometry landmark benchmark: FP, HC18 and UCL subsets

| field | value |
|---|---|
| Source | Multicentre fetal-biometry landmark benchmark (`data/raw/ucl_hc18/README-general.md`): 4,513 images, 1,904 subjects, 4 sites, 7 devices |
| Citation | Di Vece C, Mao Z, Avisdris N, et al. A multicentre benchmark dataset for comprehensive landmark-based fetal ultrasound biometry. *Sci Rep* 2026;16. doi:10.1038/s41598-026-47854-3 (PMID 41981094; preprint arXiv:2512.16710). The published abstract gives the same counts (4,513 images, 1,904 subjects, four sites, seven devices); checked 2026-11-13 |
| Licence | CC BY-NC-SA 4.0. Status: per proposal; the local copy has no LICENSE file, so verify against the repository record. The HC18 subset is also subject to the HC18 challenge terms (van den Heuvel TLA, et al. *PLoS One* 2018;13:e0200412) |
| FP subset | 3,090 images, **pixel-identical to FPDB images** (verified 3,090/3,090). Used **only as localisation ground truth**: val (238) for development, test (1,486) for reporting. Never trained on |
| HC18 subset | 999 Head images, 806 patients (filename prefix), GE Voluson E8/730, Radboud UMC (NL). External test set |
| UCL subset | 424 images (Head 159, Abdomen 130, Femur 135), 51 pregnancies (filename prefix; one cluster per prefix across anatomies), GE Voluson (model unspecified), UCLH (UK). External test set |
| Landmarks | Head OFD + BPD (trans-thalamic only); Abdomen TAD + APAD (the longer axis is used as the ellipse major axis); Femur FL |
| Exclusions | Landmark rows with `qc_flag` are excluded from the primary R4 analyses and counted: FPDB test 5, FPDB val 1, UCL 2. A sensitivity analysis includes them |
| Known quirks | `SubjectID`/`Patient` columns are empty, so patient ID = numeric filename prefix. About 2.7% of FP Head rows have a landmark outside the image (clipped and flagged). The FP `Split` column does not match our splits and is ignored |

## 3. African fetal standard planes: external test set

| field | value |
|---|---|
| Source | Sendra-Balcells C, et al. *Sci Rep* 2023 (DOI 10.5281/zenodo.7540448) |
| Licence | CC BY 4.0. Status: per proposal |
| Images / patients | 450 images, 127 centre-qualified patient IDs. Algeria (Voluson S8) 100, Egypt (Voluson P8) 100, Ghana (EDAN DUS 60) 75, Malawi (Mindray DC-N2) 100, Uganda (Siemens ACUSON X600) 75 |
| Classes | Head (brain) 125, Abdomen 125, Femur 125, Thorax 75 (Thorax absent in Ghana and Uganda) |
| Use | External test only. All 450 images are used; the authors' `Train` column is ignored |
| Known quirks | `Patient_num` repeats across centres, so IDs are centre-qualified (`afr_<ISO3>_<n>`). Image geometry is fixed per centre. **The authors cropped images to the field of view and removed vendor logos and controls**, so out-of-sector UI is largely absent: a framing shift relative to FPDB. Calipers are frequent in Ghana and Uganda |

## 4. NatalIA (PBFUS1 phantom): supplementary only

| field | value |
|---|---|
| Use | Supplementary only (a phantom, not patients). It is in no primary or secondary analysis of this study |
| Note | `data/processed/natalia_excluded_no_taxonomy_match.csv` lists the clips with no match in the 8-class taxonomy |

## 5. Foundation-model weights (R5.8 secondary probe)

| field | value |
|---|---|
| Model | FetalCLIP (BioMedIA-MBZUAI). Image tower ViT-L/14, 224 px (`FetalCLIP_config.json`: width 1024, 24 layers, patch 14) |
| Code / weights | https://github.com/BioMedIA-MBZUAI/FetalCLIP; weights `FetalCLIP_weights.pt` from https://huggingface.co/numansaeed/fetalclip-model |
| Licence | **CC BY-NC 4.0** (the repository LICENSE file; checked 2026-11-13). Non-commercial research use with attribution is permitted, so it is admissible for this study. The Hugging Face model card has not been checked for additional terms; check it before download |
| Caveat | FetalCLIP was pre-trained on about 210k images from one Abu Dhabi hospital. Its overlap with FPDB, HC18, UCL and African images is undocumented. Its README evaluates zero-shot on FPDB ("zero_shot_planes_db"), which suggests FPDB was not in pre-training, but this is unconfirmed. Report the probe as secondary, with this caveat |
| Citation | Maani F, et al. FetalCLIP: a visual-language foundation model for fetal ultrasound image analysis. arXiv:2502.14807 (preprint) |
