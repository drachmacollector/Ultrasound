# African fetal planes (external low-resource test set)

**Source:** Sendra-Balcells C, Campello VM, Torrents-Barrena J, et al. *Generalisability of fetal
ultrasound deep learning models to low-resource imaging settings in five African countries.*
Sci Rep. 2023;13:2728. doi:10.1038/s41598-023-29490-3
**Data:** Zenodo, doi:10.5281/zenodo.7540448 (v1.0, `Zenodo_dataset.tar.xz`). **Licence: CC BY 4.0.**

## Contents (verified 2026-11-12)
- `African_planes_database.csv`: `Patient_num, Plane, Train, Center, Filename` (no file extension).
- `<Center>/<Filename>.png`: 450 images.

| Centre | Device (paper Table 1) | Images | Planes | Geometry |
|---|---|---|---|---|
| Algeria (EPH Kouba / Clinique des Lilas) | GE Voluson S8 | 100 | abdomen, brain, femur, thorax | 500×430 grey |
| Egypt (Sayedaty Center) | GE Voluson P8 | 100 | abdomen, brain, femur, thorax | 400×450 **RGB** |
| Ghana (KBTH Polyclinic, Accra) | EDAN DUS 60 | 75 | abdomen, brain, femur (**no thorax**) | 500×500 grey |
| Malawi (Queen Elizabeth Central Hospital) | Mindray DC-N2 | 100 | abdomen, brain, femur, thorax | 500×400 grey |
| Uganda (Mulago National Referral Hospital) | Siemens ACUSON X600 | 75 | abdomen, brain, femur (**no thorax**) | 600×500 grey |

## Quirks that matter for this study
- **`Patient_num` repeats across centres** (e.g. `patient001_DZA` and `patient001_EGY` are different
  women). Patient IDs are centre-qualified as `afr_<ISO3>_<Patient_num>`, giving 127 IDs. The paper
  reports 25 patients per centre; Malawi shows 27 IDs because IDs 14 and 30 have one image each.
- `Train` is the authors' patient-disjoint 50/50 split. **In this study all 450 images are an external
  test set.** The optional few-shot analysis would fine-tune on `Train==1` only and report on `Train==0` only.
- The authors **cropped images to the field of view, excluding the vendor logo and control indicators**
  (paper §2.1.3). These images therefore carry little out-of-sector UI, unlike FETAL_PLANES_DB frames.
- The manifest used by the code is `data/processed/external/african.csv`
  (`scripts/build_external_manifests.py`).
