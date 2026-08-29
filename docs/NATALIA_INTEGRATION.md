# NatalIA PBF-US1 Dataset Integration

**File:** `docs/NATALIA_INTEGRATION.md`
**Dataset:** [NatalIA PBF-US1](https://zenodo.org/records/14193949)

## 1. What This Dataset Is and Is Not

> [!WARNING]
> **Phantom Data**
> The NatalIA PBF-US1 dataset was recorded on a **US-7a SPACE FAN phantom** (Kyoto Kagaku, Japan)—a physical mannequin simulating a 23-week pregnancy—**not a human patient**.

This dataset contains real free-hand probe motion captured by 45 volunteers with *no prior ultrasound experience*. They performed predefined sweep paths across the phantom.

This directly aligns with this project's stated goal: an assistive tool for non-expert operators. It allows us to validate the classifier against untrained operator sweeps and real-time off-plane (transitional) frames, but all metrics reported on it must be strictly contextualized as **phantom anatomy**. It must never be presented as real patient data.

## 2. Licensing and Citation

**License:** Creative Commons Attribution 4.0 International (CC-BY 4.0).
No supplementary data-use agreements exist that restrict non-commercial or academic use.

**Citation:**
> F. J. P. Londoño, A. E. C. S. C. S. C. S. C. S. C. S. (2025). Development and Evaluation of an AI-Driven Telemedicine System for Prenatal Healthcare (arXiv:2510.01194). Zenodo. https://doi.org/10.5281/zenodo.14193949

## 3. Class Taxonomy and Exclusions

The dataset originally contained 19,407 labeled frames across 5 standard planes + No-Plane. 
We mapped the classes to this project's taxonomy as follows:

| NatalIA class | Mapped Project Class / Label | Notes |
|---|---|---|
| Biparietal Plane (0) | `collapsed_label="Head"` | NatalIA does not distinguish the 3 sub-planes of the brain. |
| Abdominal Plane (1) | `Fetal_abdomen` | Direct match |
| Heart Plane (2) | **Excluded (61 rows)** | Our 8-class taxonomy has no Fetal_heart class. |
| Spine Plane (3) | **Excluded (134 rows)** | Our taxonomy has no Spine class. |
| Femur Plane (4) | `Fetal_femur` | Direct match |
| No Plane (5) | `Other` | Direct match |

**Final Manifest Count:** 19,407 - 195 (excluded) = **19,212 eligible rows** saved to `data/processed/natalia_manifest.csv`. Excluded rows are explicitly logged in `data/processed/natalia_excluded_no_taxonomy_match.csv`.

## 4. Volunteer/Operator Leakage
Since all images come from a single physical phantom rather than multiple distinct patients, patient-leakage (like in FETAL_PLANES_DB) does not exist here. However, volunteer operator leakage exists. Because this dataset is used strictly as a held-out evaluation set and never used for training or fine-tuning, this leakage does not affect our trained classifier. 
