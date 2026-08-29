# Phase 8 Walkthrough

**Final Evaluation and Polish** | 2026-08-29

## Overview
This document summarizes the completion of Stages 3, 4, and 5 from Phase 8. The ultrasound classifier is now fully evaluated for cross-device generalization across 6 backbones, and the Grad-CAM bounding box approximate localisation feature has been successfully integrated across all pipelines.

---

## 1. Stage 4: Cross-Device Backbone Sweep

The `evaluate_cross_device.py` infrastructure was updated to prevent artifact overwriting, allowing us to safely evaluate the 5 new backbones alongside the original `convnext_tiny`.

### Key Findings
All six backbones comfortably passed the 50% Head-class accuracy gate check on unseen devices, proving that the domain shift does not catastrophically break the visual representations learned on FETAL_PLANES_DB.

**Winner: `repvgg_a1`**
`repvgg_a1` demonstrated the highest cross-device Head accuracy (**88.9%**) and the smallest generalization gap (**-10.7%**) compared to the `convnext_tiny` baseline (-14.8% gap). It also had the fewest catastrophic "Other" misclassifications (56 compared to the baseline's 180). Given that it is heavily optimized for inference speed (via structural re-parameterization), it represents a Pareto improvement over `convnext_tiny` for the real-time edge deployment profile: it is both faster and more robust to clinical domain shift.

The full breakdown table is available in:
[`docs/CROSS_DEVICE_BACKBONE_COMPARISON.md`](file:///d:/Ultrasound/docs/CROSS_DEVICE_BACKBONE_COMPARISON.md)

---

## 2. Stage 3: CAM-to-Bounding-Box Localisation

We implemented weakly-supervised approximate bounding box extraction from Grad-CAM activation maps.

### Implementation Details
- **Module:** [`src/models/cam_localisation.py`](file:///d:/Ultrasound/src/models/cam_localisation.py)
  - Uses `cv2.THRESH_OTSU` to binarize the heatmap.
  - Extracts the largest connected component to avoid drawing boxes around disconnected noise (addressing the split-blob failure mode documented in EXPERIMENTS.md).
  - Filters out tiny boxes (area < 2%).
  - Fully tested via `scripts/tests/test_cam_localisation.py` (4/4 tests passing).
- **Integration:** The `GradCamWorker` in [`src/realtime/pipeline.py`](file:///d:/Ultrasound/src/realtime/pipeline.py) now yields a `cam_bbox` alongside the `grayscale_cam`.
- **Rendering:** `build_display_frame()` in [`src/realtime/app.py`](file:///d:/Ultrasound/src/realtime/app.py) was updated to render the box.
  - **Honesty Framing:** The box is drawn as a **dashed orange rectangle** and explicitly labeled `"approx. region (saliency-derived)"` to visually distinguish it from a proper ground-truth object detector box. It is only shown when the Grad-CAM overlay is active.
- **Web UI:** [`app_streamlit.py`](file:///d:/Ultrasound/app_streamlit.py) received a new sidebar toggle: `"Approx. region box (saliency-derived)"`.


### Manual Spot-Check
The 20-image manual spot-check template is ready for your completion at:
[`docs/CAM_LOCALISATION_SPOTCHECK.md`](file:///d:/Ultrasound/docs/CAM_LOCALISATION_SPOTCHECK.md)

---

## 3. Stage 5: ACAM Ablation
This stage was completed earlier. The ACAM contrast preprocessing ablation was investigated and formally documented as a negative result (tied/insignificant performance difference) in [`docs/phases/phase_08/2509.00808_ACAM.md`](file:///d:/Ultrasound/docs/phases/phase_08/2509.00808_ACAM.md).

---

## Next Steps
We are now ready to proceed to **Stage 6 (Final Polish)**, which will involve code cleanup and preparing the final `docs/PROJECT_SHOWCASE_README.md`.
