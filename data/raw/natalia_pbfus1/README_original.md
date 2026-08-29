# NatalIA: PBF-US1 (Phantom Blind-sweeps for Fetal Ultrasound Scanning)

## Overview
This dataset, **NatalIA PBF-US1**, is designed to support the development of AI-based tools for detecting relevant fetal planes in ultrasound videos captured by non-trained personnel (e.g., midwives, nurses). This is particularly relevant for low-income countries or remote communities with a shortage of trained sonographers.

The dataset features 19,407 ultrasound frames extracted from 90 videos of a fetal ultrasound phantom, recorded through free-hand sweeps by non-experts.

## Dataset Details
* **Phantom Used**: US-7a SPACE FAN phantom (Kyoto Kagaku, Japan), simulating a 23-week gestation pregnancy.
* **Device**: Clarius C3 HD3 point-of-care ultrasound (POCUS) device (Clarius, Canada).
* **Settings**: Obstetric mode, maximum depth of 16 cm, captured at 24 frames per second (fps).
* **Operators**: 45 volunteers with *no prior ultrasound experience*, simulating real-world non-expert variability.
* **Protocol**: Each volunteer performed four predefined scanning paths:
  * 1 Vertical sweep
  * 1 Horizontal sweep
  * 2 Diagonal sweeps
* **Poses**: The phantom was configured in four different fetal poses:
  * Occiput Posterior (OP)
  * Sacrum Posterior (SP)
  * Occiput Anterior (OA)
  * Sacrum Anterior (SA)
* **Annotations**: Image annotation was performed by a radiologist, an obstetrician, and pre-trained medical students using the Labelbox platform. Labels by medical students were reviewed and validated by the radiologist.

## Class Distribution
The dataset contains five standard fetal planes and a background "no plane" class:

| Class | Plane Description | Count | Value (Encoding) |
|---|---|---|---|
| 0 | Biparietal Plane | 42 | 0 |
| 1 | Abdominal Plane | 63 | 1 |
| 2 | Heart Plane | 61 | 2 |
| 3 | Spine Plane | 134 | 3 |
| 4 | Femur Plane | 46 | 4 |
| 5 | No Plane | 19,061 | 5 |
| **Total** | | **19,407** | |

## Understanding `resume.csv` and the Exam Folders
The dataset is structured as 90 individual studies or examinations, represented by 90 subdirectories named `Obstetrics Exam - [Date]_[Time]`. Each folder corresponds to one of the 90 videos captured and contains all the `.jpeg` frames extracted from that specific sweep (e.g., `cineframe_100_...jpeg`, `cineframe_101_...jpeg`).

The `resume.csv` file provides metadata and frame-level annotations mapping every image in these 90 folders to its correct standard plane class. It contains rows mapping all 19,407 frames, with the following columns:
1. `file_name`: The filename of the extracted frame (e.g., `cineframe_100_2024-05-03T12-19-10.jpeg`).
2. `studie`: The parent study/exam folder (e.g., `Obstetrics Exam - 03-May-2024_1216_PM`). This matches the exact name of the directory where the frame is stored.
3. `class`: The textual description of the labeled plane (e.g., `Biparietal standard plane`, `No plane`).
4. `value`: The integer encoding for the class (0-5) as shown in the distribution table.
5. `image`: An optional column (sometimes empty in raw releases) designed to hold the relative path to the image, such as `./data/Obstetrics Exam...`.

*Note: For deeper study-level demographics (e.g., volunteer age, gender, education level, and the specific sweep protocol/position used), refer to the included `metadata.csv`.*

## Usage & Integration
This dataset is vital for evaluating AI models on out-of-distribution (OOD) data and non-expert sweeps. Its primary use case is to test whether models trained on curated clinical datasets can generalize to real-world, noisy, non-expert ultrasound sweeps.

## References
* **GitHub Repository**: [https://github.com/BiomedLabUGgt/NatalIA-PBF-US1](https://github.com/BiomedLabUGgt/NatalIA-PBF-US1)
* **Zenodo**: [https://doi.org/10.5281/zenodo.14193949](https://doi.org/10.5281/zenodo.14193949)
* **Citation**: González, D., Barrientos, J. P., Perez, M., Fajardo, J., Reyna, F., & Lara, A. (2024). NatalIA: PBF-US1 (Phantom Blind-sweeps for Fetal Ultrasound Scanning) (1.0.0) [Data set]. Zenodo.
