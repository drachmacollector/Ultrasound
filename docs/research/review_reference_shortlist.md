# Review reference shortlist (R6 C.3)

**Rules**
- **Style.** Vancouver: list all authors when there are six or fewer; when there are more, list the first 3 followed by "et al.".
- **Numbering.** References are numbered in square brackets in order of first citation. The numbers below are a ranked draft; the final order is set by the manuscript text.
- **Limits.** At most 40 references, with preprints at most 15% of them.
- **Eligibility.** Only rows marked `verified=y` in `review_evidence_table.csv` may be cited. Each entry gives its PMID or source so it can be matched to its row.

**Status (2026-11-13)**
- The evidence table has 44 rows, of which 42 are verified (`python scripts/research/check_evidence_table.py`).
- This shortlist has 40 references: 17 Must, 15 Should, 8 Optional.
- Preprints: 4 of 40 (10%), all flagged "[Preprint]". This is within the 15% limit.
- If the 40-reference budget gets tight, drop from Optional first.

## Must

1. Burgos-Artizzu XP, Coronado-Gutiérrez D, Valenzuela-Alcaraz B, et al. Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes. Sci Rep. 2020;10:10200. doi:10.1038/s41598-020-67076-5 *(PMID 32576905; Author Correction: Sci Rep. 2022;12:1950)*
2. Sendra-Balcells C, Campello VM, Torrents-Barrena J, et al. Generalisability of fetal ultrasound deep learning models to low-resource imaging settings in five African countries. Sci Rep. 2023;13:2728. doi:10.1038/s41598-023-29490-3 *(PMID 36792642)*
3. Baumgartner CF, Kamnitsas K, Matthew J, et al. SonoNet: real-time detection and localisation of fetal standard scan planes in freehand ultrasound. IEEE Trans Med Imaging. 2017;36:2204-15. doi:10.1109/TMI.2017.2712367 *(PMID 28708546)*
4. Geirhos R, Jacobsen JH, Michaelis C, et al. Shortcut learning in deep neural networks. Nat Mach Intell. 2020;2:665-73. doi:10.1038/s42256-020-00257-z *(Semantic Scholar record)*
5. Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLoS Med. 2018;15:e1002683. doi:10.1371/journal.pmed.1002683 *(PMID 30399157)*
6. Badgeley MA, Zech JR, Oakden-Rayner L, et al. Deep learning predicts hip fracture using confounding patient and healthcare variables. NPJ Digit Med. 2019;2:31. doi:10.1038/s41746-019-0105-1 *(PMID 31304378)*
7. DeGrave AJ, Janizek JD, Lee SI. AI for radiographic COVID-19 detection selects shortcuts over signal. Nat Mach Intell. 2021;3:610-9. doi:10.1038/s42256-021-00338-7 *(full text checked)*
8. Winkler JK, Fink C, Toberer F, et al. Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning convolutional neural network for melanoma recognition. JAMA Dermatol. 2019;155:1135-41. doi:10.1001/jamadermatol.2019.1735 *(PMID 31411641)*
9. Yi PH, Malone PS, Lin CT, Filice RW. Deep learning algorithms for interpretation of upper extremity radiographs: laterality and technologist initial labels as confounding factors. AJR Am J Roentgenol. 2022;218(4):714-5. doi:10.2214/AJR.21.26882 *(PMID 34755522)*
10. Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps [Preprint]. arXiv:1810.03292. 2018. *(arXiv record; NeurIPS 2018. Replace with the proceedings citation once that record is checked.)*
11. Arun N, Gaw N, Singh P, et al. Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging. Radiol Artif Intell. 2021;3:e200267. doi:10.1148/ryai.2021200267 *(PMID 34870212)*
12. Di Vece C, Mao Z, Avisdris N, et al. A multicentre benchmark dataset for comprehensive landmark-based fetal ultrasound biometry. Sci Rep. 2026;16:17405. doi:10.1038/s41598-026-47854-3 *(PMID 41981094)*
13. van den Heuvel TLA, de Bruijn D, de Korte CL, van Ginneken B. Automated measurement of fetal head circumference using 2D ultrasound images. PLoS One. 2018;13(8):e0200412. doi:10.1371/journal.pone.0200412 *(PMID 30138319)*
14. Fiorentino MC, Villani FP, Di Cosmo M, Frontoni E, Moccia S. A review on deep-learning algorithms for fetal ultrasound-image analysis. Med Image Anal. 2023;83:102629. doi:10.1016/j.media.2022.102629 *(PMID 36308861)*
15. Tejani AS, Klontzas ME, Gatti AA, et al. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 update. Radiol Artif Intell. 2024;6(4):e240300. doi:10.1148/ryai.240300 *(PMID 38809149)*
16. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378 *(PMID 38626948)*
17. Indian Council of Medical Research. Ethical guidelines for application of artificial intelligence in biomedical research and healthcare. New Delhi: ICMR; 2023. ISBN 978-93-5811-343-3 *(full text checked; notes in `background/ICMR_2023_notes.md`)*

## Should

18. Souza R, Wilms M, Camacho M, et al. Image-encoded biological and non-biological variables may be used as shortcuts in deep learning models trained on multisite neuroimaging data. J Am Med Inform Assoc. 2023;30(12):1925-33. doi:10.1093/jamia/ocad171 *(PMID 37669158)*
19. Shu HJ, Chang ST, Gameiro RR, Gichoya JW, Celi LA, Kuo PC. Impact of exposure parameters on deep learning models in chest radiography and implications for deployment. Radiol Artif Intell. 2026;8(4):e250731. doi:10.1148/ryai.250731 *(PMID 42340188)*
20. Luo L, Chen H, Xiao Y, et al. Rethinking annotation granularity for overcoming shortcuts in deep learning-based radiograph diagnosis: a multicenter study. Radiol Artif Intell. 2022;4(5):e210299. doi:10.1148/ryai.210299 *(PMID 36204545)*
21. Maron RC, Hekler A, Krieghoff-Henning E, et al. Reducing the impact of confounding factors on skin cancer classification via image segmentation: technical model study. J Med Internet Res. 2021;23(3):e21695. doi:10.2196/21695 *(PMID 33764307)*
22. Saporta A, Gui X, Agrawal A, et al. Benchmarking saliency methods for chest X-ray interpretation. Nat Mach Intell. 2022;4:867-78. doi:10.1038/s42256-022-00536-x *(full text checked)*
23. Schlemper J, Oktay O, Schaap M, et al. Attention gated networks: learning to leverage salient regions in medical images. Med Image Anal. 2019;53:197-207. doi:10.1016/j.media.2019.01.012 *(PMID 30802813)*
24. Bashir Z, Lin M, Feragen A, et al. Clinical validation of explainable AI for fetal growth scans through multi-level, cross-institutional prospective end-user evaluation. Sci Rep. 2025;15(1):2074. doi:10.1038/s41598-025-86536-4 *(PMID 39820804)*
25. Sridar P, Kumar A, Quinton A, Nanan R, Kim J, Krishnakumar R. Decision fusion-based fetal ultrasound image plane classification using convolutional neural networks. Ultrasound Med Biol. 2019;45(5):1259-73. doi:10.1016/j.ultrasmedbio.2018.11.016 *(PMID 30826153)*
26. Tehsin S, Alshaya H, Bouchelligua W, Nasir IM. Hybrid state-space and vision transformer framework for fetal ultrasound plane classification in prenatal diagnostics. Diagnostics (Basel). 2025;15(22):2879. doi:10.3390/diagnostics15222879 *(PMID 41300902)*
27. Montero A, Bonet-Carne E, Burgos-Artizzu XP. Generative adversarial networks to improve fetal brain fine-grained plane classification. Sensors (Basel). 2021;21(23):7975. doi:10.3390/s21237975 *(PMID 34883977)*
28. Chi H, Jung S, Subramanian SV, Kim R. Socioeconomic and geographic inequalities in antenatal and postnatal care components in India, 2016–2021. Sci Rep. 2024;14:10221. doi:10.1038/s41598-024-59981-w *(PMID 38702357)*
29. Gadekar VP, Damaraju N, Xavier A, et al. Development and external validation of Indian population-specific Garbhini-GA2 model for estimating gestational age in second and third trimesters. Lancet Reg Health Southeast Asia. 2024;25:100362. doi:10.1016/j.lansea.2024.100362 *(PMID 39021476)*
30. Self A, Chen Q, Desiraju BK, et al. Developing clinical artificial intelligence for obstetric ultrasound to improve access in underserved regions: protocol for a Computer-Assisted Low-Cost Point-of-Care UltraSound (CALOPUS) study. JMIR Res Protoc. 2022;11(9):e37374. doi:10.2196/37374 *(PMID 36048518)*
31. Maani F, Saeed N, Saleem T, et al. FetalCLIP: a visual-language foundation model for fetal ultrasound image analysis [Preprint]. arXiv:2502.14807. 2025. *(arXiv record)*
32. Mikolaj K, Lin M, Bashir Z, et al. Removing confounding information from fetal ultrasound images [Preprint]. arXiv:2303.13918. 2023. *(arXiv record)*

## Optional

33. Zhao Q, Adeli E, Pohl KM. Training confounder-free deep learning models for medical applications. Nat Commun. 2020;11(1):6010. doi:10.1038/s41467-020-19784-9 *(PMID 33243992)*
34. Hill BG, Koback FL, Schilling PL. The risk of shortcutting in deep learning algorithms for medical imaging research. Sci Rep. 2024;14(1):29224. doi:10.1038/s41598-024-79838-6 *(PMID 39587148)*
35. Bonnard J, Dapogny A, Zsamboki R, et al. Prior-guided attribution of deep neural networks for obstetrics and gynecology. IEEE J Biomed Health Inform. 2024;28(2):870-80. doi:10.1109/JBHI.2023.3337511 *(PMID 38019619)*
36. Migliorelli G, Fiorentino MC, Di Cosmo M, Villani FP, Mancini A, Moccia S. On the use of contrastive learning for standard-plane classification in fetal ultrasound imaging. Comput Biol Med. 2024;174:108430. doi:10.1016/j.compbiomed.2024.108430 *(PMID 38613892)*
37. Wong CK, Christensen AN, Bercea CI, Schnabel JA, Tolsgaard MG, Feragen A. Influence of classification task and distribution shift type on OOD detection in fetal ultrasound [Preprint]. arXiv:2509.18326. 2025. *(arXiv record; MICCAI 2025 according to the record)*
38. Abdillahi MA, Baker MR, Doğru A, Buyrukoğlu S. Enhanced fetal plane classification in ultrasound imaging via prototypical networks and few-shot learning. J Imaging Inform Med. 2025;39(4):3132-49. doi:10.1007/s10278-025-01753-7 *(PMID 41251863)*
39. Sivasubramanian A, Sasidharan D, Sowmya V, Ravi V. Efficient feature extraction using light-weight CNN attention-based deep learning architectures for ultrasound fetal plane classification. Phys Eng Sci Med. 2025;48(3):1079-93. doi:10.1007/s13246-025-01566-6 *(PMID 40437331)*
40. Priyadharshni S, Ravi V. Dynamic graph-based quantum feature selection for accurate fetal plane classification in ultrasound imaging. Sci Rep. 2025;15(1):41743. doi:10.1038/s41598-025-26835-y *(PMID 41274967)*

## Not citable yet

- Lim 2024 (*WFUMB Ultrasound Open*) and Phutke 2018 (IJME): not verified. Neither is needed for the shortlist.

**Notes**
- Authors, volumes and pages come from the PubMed records, fetched on 2026-11-13, except for these:
  - the arXiv items, taken from their arXiv abstract pages;
  - Geirhos 2020, taken from its Semantic Scholar record;
  - DeGrave 2021, Saporta 2022, Di Vece 2026 and ICMR 2023, checked against the full-text PDFs the user supplied (`docs/research/background/`).
- Tehsin 2025's article number (2879) comes from its MDPI DOI, because the PubMed record has no page field.
