# Real-Time Sonography Assistant

## Project Overview
This project is an assistive tool for sonographers to perform real-time standard anatomical plane detection. Given a live video stream (such as a webcam standing in for an ultrasound probe, or a pre-recorded video file), the system continuously classifies each moment of the scan into one of 7 standard anatomical planes, or "Other" (non-standard/transitional). 

It provides:
- A stable (non-flickering) on-screen label
- A confidence score
- A Grad-CAM style visual explanation highlighting the anatomical regions driving the classification

### Target Classes
1. Brain — Trans-cerebellum
2. Brain — Trans-thalamic
3. Brain — Trans-ventricular
4. Fetal abdomen
5. Fetal femur
6. Fetal thorax
7. Maternal cervix
8. Other (non-standard / transitional / off-plane)

## Datasets Used
The models in this project are trained and evaluated using the following public datasets:
- **FETAL_PLANES_DB** (Burgos-Artizzu et al., 2020): Used as the primary dataset for training, validation, and in-distribution testing across all 8 classes.
- **UCL/HC18 Cross-Device Benchmark**: A combination of the HC18 and UCL subsets used specifically as a held-out test set to measure cross-device and cross-site generalization.

## Architecture
The system is designed to run locally on consumer hardware (e.g., an RTX 4060 GPU) with low perceived latency, rather than relying on a cloud backend. 

Key architectural components include:
- **Backbone**: A lightweight CNN (e.g., RepVGG, EfficientNet, MobileNetV3) for fast feature extraction.
- **Classification Head**: A single 8-way softmax classifier processing features in a single forward pass.
- **Interpretability**: A throttled Grad-CAM module that provides visual explanations without severely impacting real-time throughput.
- **Temporal Stabilization**: A smoothing layer that uses confidence-weighted moving averages, hysteresis, and minimum dwell-times to stabilize the per-frame predictions and eliminate flickering.
- **Serving**: A local Python real-time loop utilizing OpenCV for video capture, preprocessing, inference, and overlay rendering.

## How It Works
1. **Video Capture**: A local thread captures frames from a webcam or a video file using OpenCV.
2. **Preprocessing**: Frames are resized, normalized, and converted into 3-channel tensors.
3. **Inference**: The CNN backbone and classification head process the frame, producing raw class probabilities. Periodically, Grad-CAM maps are generated.
4. **Temporal Smoothing**: Raw probabilities and class predictions are fed into a smoothing buffer to suppress spurious frame-level noise.
5. **UI Rendering**: The final stable prediction, confidence score, and Grad-CAM overlay are drawn onto the video frame and displayed to the user.

## Instructions to Run

### 1. Prerequisites
- **Hardware**: An NVIDIA GPU (e.g., RTX 4060) is recommended.
- **Drivers**: Update NVIDIA drivers to support CUDA 12.x.
- **Python**: Python 3.10 or 3.11.

### 2. Environment Setup
Create a Conda environment and install the required packages:

```bash
conda create -n ultrasound_env python=3.11 -y
conda activate ultrasound_env
pip install -r requirements.txt
```
*(Ensure that you install the version of PyTorch that matches your system's CUDA version. Check `requirements.txt` for dependencies.)*

### 3. Data Preparation
- Download the **FETAL_PLANES_DB** dataset and the **UCL/HC18** datasets.
- Place the raw datasets into `data/raw/`.
- Run the provided data pipeline scripts to build manifests, verify data, and generate train/validation/test splits:
  ```bash
  python scripts/build_manifest.py
  python scripts/patient_split.py
  python scripts/build_cross_device_manifest.py
  python scripts/run_sanity_checks.py
  ```

### 4. Training
To train the model from scratch on the processed dataset:
```bash
python src/train/train.py --config configs/convnext_tiny.yaml
```
*(You can use any of the provided per-backbone configuration files in the `configs/` directory.)*
*Note: You can also run `python scripts/smoke_test.py` to verify the pipeline before starting full training.*

### 5. Evaluation
To evaluate the trained model on the held-out test set:
```bash
python scripts/evaluate_test.py --weights checkpoints/convnext_tiny/best.pt
```

### 6. Real-Time Inference (WIP)
*(Note: Real-time inference scripts are currently being built in Phase 5 and are not yet fully implemented.)*
Run the real-time application using your trained model checkpoint:
```bash
python src/realtime/app.py --weights checkpoints/convnext_tiny/best.pt --source 0
```
*(Use `--source 0` for your primary webcam, or provide a path to an `.mp4` video file.)*
