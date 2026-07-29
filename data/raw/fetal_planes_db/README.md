# Fetal_Planes_DB

This repository contains the **FETAL_PLANES_DB**, a large dataset of routinely acquired maternal-fetal screening ultrasound images.

**Citation:**
Burgos-Artizzu, X.P., Coronado-Gutiérrez, D., Valenzuela-Alcaraz, B. et al. Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes. Sci Rep 10, 10200 (2020). [https://doi.org/10.1038/s41598-020-67076-5](https://doi.org/10.1038/s41598-020-67076-5)

## Overview

The dataset contains **over 12,400 de-identified ultrasound images** from **1,792 unique subjects**. The data was collected from two different hospitals by several operators using multiple ultrasound machines (e.g., Voluson E6/S10, Aloka). 

All images were manually labeled by an expert maternal fetal clinician (B.V-A.). The dataset is designed for automatic classification of common maternal fetal ultrasound planes.

### Anatomical Categories

The images are divided into 6 main classes plus a general "Other" category:

- **Fetal Brain**: Includes 3 sub-classes (Trans-thalamic, Trans-cerebellum, Trans-ventricular) and an "Other" brain category.
- **Fetal Abdomen**
- **Fetal Femur**
- **Fetal Thorax**
- **Maternal Cervix**: Widely used for prematurity screening.
- **Other**: A general category to include any other less common or off-plane images.

## Dataset Structure

The repository is organized as follows:

```text
fetal_planes_db/
├── Images/                    # Folder containing all the ultrasound image files (.png)
├── FETAL_PLANES_DB_data.csv   # Metadata, labels, and splits (semi-colon separated)
├── FETAL_PLANES_DB_data.xlsx  # Same metadata in Excel format
└── README.md                  # This file
```

## Data Organization

### Image Files

All image files are provided in `.png` format inside the `Images/` directory.

- `Images/Patient[ID]_Plane[N]_[M]_of_[K].png` (e.g., `Patient00001_Plane1_1_of_15.png`)

### CSV Annotation Format

The dataset labels and metadata are provided in `FETAL_PLANES_DB_data.csv`. The file uses a semi-colon (`;`) separator. 

```csv
Image_name;Patient_num;Plane;Brain_plane;Operator;US_Machine;Train 
```

**Columns Description:**

- **Image_name**: The corresponding image filename (without the `.png` extension).
- **Patient_num**: De-identified numerical patient identifier.
- **Plane**: The main anatomical plane category. Possible values:
  - `Fetal brain`
  - `Fetal abdomen`
  - `Fetal femur`
  - `Fetal thorax`
  - `Maternal cervix`
  - `Other`
- **Brain_plane**: Sub-categorization for fetal brain planes. Possible values:
  - `Trans-thalamic`
  - `Trans-cerebellum`
  - `Trans-ventricular`
  - `Other` (Brain plane not matching the 3 standard ones)
  - `Not A Brain` (For all non-brain planes)
- **Operator**: Identifier for the operator who acquired the image.
- **US_Machine**: Ultrasound device identifier (e.g., `Aloka`, `Voluson E6`, `Voluson S10`).
- **Train**: Data split indicator. 
  - `1`: Assigned to the Training split.
  - `0`: Assigned to the Testing split.

## Data Splits

The dataset provides standardized train/test splits to ensure fair and reproducible evaluation. The splits are indicated by the `Train` column in the CSV file. 

⚠️ **Important**: 
The splits are **patient-disjoint** (subject-disjoint). Images from the same patient (`Patient_num`) appear exclusively in either the training set or the testing set, but never both. This is crucial for preventing data leakage during model training.

## License

Please refer to the accompanying paper and any provided license documentation for terms of use.

If you find this dataset useful, please cite:

```bibtex
@article{Burgos-ArtizzuFetalPlanesDataset,
  title={Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes},
  author={Burgos-Artizzu, X.P. and Coronado-Gutiérrez, D. and Valenzuela-Alcaraz, B. and Bonet-Carne, E. and Eixarch, E. and Crispi, F. and Gratacós, E.},
  journal={Nature Scientific Reports}, 
  volume={10},
  pages={10200},
  doi={10.1038/s41598-020-67076-5},
  year={2020}
} 
```
