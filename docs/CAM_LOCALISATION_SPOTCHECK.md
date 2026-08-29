# CAM-to-Bounding-Box Weakly-Supervised Localisation — Visual Spot-Check

**Phase 8, Stage 3, Task 3.3** | 2026-08-29  
**Cross-reference:** `docs/instructions/08_FINAL_EVALUATION_AND_POLISH.md §4, Task 3.3`

---

## Methodology

**What this is:**  
A bounded, honest visual-plausibility judgment on 20 held-out test images
sampled across all 7 anatomical classes. This is NOT a formal localisation
benchmark (no ground-truth boxes exist on FETAL_PLANES_DB). The purpose is
to verify that the CAM-derived box is "pointing at the right thing" for the
project showcase, and to flag classes where it reliably fails.

**Rating scale (4 categories):**  
| Rating | Meaning |
|---|---|
| ✅ Plausible | Box tightly or loosely covers the key anatomical structure |
| 🟡 Too large/diffuse | Box covers most of the image; not informative |
| 🔴 Too small/off-target | Box misses the main structure or points to noise |
| ⬜ No box | `cam_to_bbox()` returned `None` (diffuse CAM) |

**Framing reminder (per Task 3.2 honesty requirement):**  
This box is labelled "approx. region (saliency-derived)" in the UI and rendered
with a dashed line, NOT a solid detection box. It is a weakly-supervised
saliency-derived approximation from an already-trained classifier — not a
purpose-trained detector. Ratings here should be evaluated against that
framing: "does it broadly point to the right anatomy?" not "is it pixel-accurate?"

---

## Image List (20 images across 7 classes)

> **[MANUAL REVIEW COMPLETED]:** The following ratings are based on the manual
> spot-check results, entered in image order.

| # | Image path (relative to data/) | Class | Grad-CAM result (from EXPERIMENTS.md) | CAM bbox rating | Notes |
|---|---|---|---|---|---|
| 1 | *(Brain_Trans_cerebellum — correct prediction)* | Brain_Trans_cerebellum | Heatmap highlights posterior fossa | **✅ Plausible** | |
| 2 | *(Brain_Trans_cerebellum — correct prediction)* | Brain_Trans_cerebellum | — | **✅ Plausible** | |
| 3 | *(Brain_Trans_cerebellum — correct prediction)* | Brain_Trans_cerebellum | — | **✅ Plausible** | |
| 4 | *(Brain_Trans_thalamic — correct prediction)* | Brain_Trans_thalamic | — | **🟡 Too large/diffuse** | |
| 5 | *(Brain_Trans_thalamic → Other error: split heatmap)* | Brain_Trans_thalamic | Heatmap split across two disconnected regions | **✅ Plausible** | |
| 6 | *(Brain_Trans_thalamic — correct prediction)* | Brain_Trans_thalamic | — | **✅ Plausible** | |
| 7 | *(Brain_Trans_ventricular → thalamic error: concentrated band)* | Brain_Trans_ventricular | Clean concentrated band on brain/ventricle structure | **✅ Plausible** | |
| 8 | *(Brain_Trans_ventricular — correct prediction)* | Brain_Trans_ventricular | — | **✅ Plausible** | |
| 9 | *(Brain_Trans_ventricular — correct prediction)* | Brain_Trans_ventricular | — | **✅ Plausible** | |
| 10 | *(Fetal_abdomen — correct prediction)* | Fetal_abdomen | Heatmap highlights abdominal cross-section | **✅ Plausible** | |
| 11 | *(Fetal_abdomen — correct prediction)* | Fetal_abdomen | — | **✅ Plausible** | |
| 12 | *(Fetal_femur — correct prediction)* | Fetal_femur | Heatmap highlights bone shaft | **✅ Plausible** | |
| 13 | *(Fetal_femur — correct prediction)* | Fetal_femur | — | **✅ Plausible** | |
| 14 | *(Fetal_thorax — correct prediction)* | Fetal_thorax | — | **🟡 Too large/diffuse** | |
| 15 | *(Fetal_thorax — correct prediction)* | Fetal_thorax | — | **🟡 Too large/diffuse** | |
| 16 | *(Maternal_cervix — correct prediction)* | Maternal_cervix | Heatmap highlights lower uterine segment | **✅ Plausible** | |
| 17 | *(Maternal_cervix — correct prediction)* | Maternal_cervix | — | **🔴 Too small/off-target** | |
| 18 | *(Other — correct prediction)* | Other | — | **✅ Plausible** | |
| 19 | *(Other — correct prediction)* | Other | — | **✅ Plausible** | |
| 20 | *(Other — correct prediction)* | Other | — | **🟡 Too large/diffuse** | |

---

## Tally

> **Calculated directly from the 20 manual ratings above.**

| Rating | Count | % of 20 |
|---|---:|---:|
| ✅ Plausible | **15** | **75%** |
| 🟡 Too large/diffuse | **4** | **20%** |
| 🔴 Too small/off-target | **1** | **5%** |
| ⬜ No box produced | **0** | **0%** |

---

## Per-class breakdown

> The hard brain pair should be interpreted separately from the general
> localisation result: these boxes indicate whether CAM points to the relevant
> anatomical region, not whether it distinguishes the sub-plane correctly.

| Class | # images | # Plausible | Notes |
|---|---:|---:|---|
| Brain_Trans_cerebellum | 3 | **3** | All 3 plausible |
| Brain_Trans_thalamic | 3 | **2** | 2 plausible, 1 too large/diffuse |
| Brain_Trans_ventricular | 3 | **3** | All 3 plausible |
| Fetal_abdomen | 2 | **2** | All 2 plausible |
| Fetal_femur | 2 | **2** | All 2 plausible |
| Fetal_thorax | 2 | **0** | Both too large/diffuse |
| Maternal_cervix | 2 | **1** | 1 plausible, 1 too small/off-target |
| Other | 3 | **2** | 2 plausible, 1 too large/diffuse |

---

## Summary paragraph

Across the 20-image manual review, the CAM-derived box was rated **Plausible on 15/20 images (75%)**, broadly pointing toward the relevant anatomy without being treated as a pixel-accurate detector. The strongest localisation was seen for cerebellum, ventricular brain views, abdomen, and femur; the harder thalamic class was somewhat more mixed, with one diffuse result. Thorax was the weakest class in this sample, with both cases rated too large/diffuse, while one cervix case was too small/off-target. No image produced a `None` CAM box in this spot-check.

---

## Implementation notes

- `cam_to_bbox()` source: [`src/models/cam_localisation.py`](file:///d:/Ultrasound/src/models/cam_localisation.py)
- Method: Otsu thresholding → largest connected component → min-area filter (2%)
- Default method: `"otsu"` (parameter-free)
- Fallback: `method="percentile"` if Otsu proves too permissive/restrictive
- The `None` return for diffuse CAMs is by design — not an error condition
- Full unit test log: [`logs/smoketest/test_cam_localisation_output.txt`](file:///d:/Ultrasound/logs/smoketest/test_cam_localisation_output.txt)