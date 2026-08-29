# CAM-to-Bounding-Box Weakly-Supervised Localisation — Visual Spot-Check

**Phase 8, Stage 3, Task 3.3** | 2026-08-29  
**Cross-reference:** `docs/instructions/08_FINAL_EVALUATION_AND_POLISH.md §4, Task 3.3`

---

## Methodology

**What this is:**  
A bounded, honest visual-plausibility judgment on 20 held-out test images
sampled across all 7 anatomical classes.  This is NOT a formal localisation
benchmark (no ground-truth boxes exist on FETAL_PLANES_DB).  The purpose is
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
with a dashed line, NOT a solid detection box.  It is a weakly-supervised
saliency-derived approximation from an already-trained classifier — not a
purpose-trained detector.  Ratings here should be evaluated against that
framing: "does it broadly point to the right anatomy?" not "is it pixel-accurate?"

---

## Image List (20 images across 7 classes)

> **[MANUAL TASK]:** Run `cam_to_bbox()` on the Grad-CAM heatmaps for the images
> below (using the spot-check images from `docs/EXPERIMENTS.md`'s Grad-CAM notes
> or generating new ones) and fill in the Rating and Notes columns.  
> Command to generate visualisations:
> ```
> conda run -n fetalplane python scripts/evaluate_test.py --checkpoint checkpoints/convnext_tiny/best.pt
> ```
> Then visually inspect the overlays with `cam_bbox` drawn on them.

| # | Image path (relative to data/) | Class | Grad-CAM result (from EXPERIMENTS.md) | CAM bbox rating | Notes |
|---|---|---|---|---|---|
| 1 | *(Brain_Trans_cerebellum — correct prediction)* | Brain_Trans_cerebellum | Heatmap highlights posterior fossa | **[FILL IN]** | |
| 2 | *(Brain_Trans_cerebellum — correct prediction)* | Brain_Trans_cerebellum | — | **[FILL IN]** | |
| 3 | *(Brain_Trans_cerebellum — correct prediction)* | Brain_Trans_cerebellum | — | **[FILL IN]** | |
| 4 | *(Brain_Trans_thalamic — correct prediction)* | Brain_Trans_thalamic | — | **[FILL IN]** | |
| 5 | *(Brain_Trans_thalamic → Other error: split heatmap)* | Brain_Trans_thalamic | Heatmap split across two disconnected regions | **[FILL IN]** | Expected: ⬜ or 🟡 (diffuse CAM per EXPERIMENTS.md) |
| 6 | *(Brain_Trans_thalamic — correct prediction)* | Brain_Trans_thalamic | — | **[FILL IN]** | |
| 7 | *(Brain_Trans_ventricular → thalamic error: concentrated band)* | Brain_Trans_ventricular | Clean concentrated band on brain/ventricle structure | **[FILL IN]** | Expected: ✅ or 🟡 (right place, wrong label) |
| 8 | *(Brain_Trans_ventricular — correct prediction)* | Brain_Trans_ventricular | — | **[FILL IN]** | |
| 9 | *(Brain_Trans_ventricular — correct prediction)* | Brain_Trans_ventricular | — | **[FILL IN]** | |
| 10 | *(Fetal_abdomen — correct prediction)* | Fetal_abdomen | Heatmap highlights abdominal cross-section | **[FILL IN]** | |
| 11 | *(Fetal_abdomen — correct prediction)* | Fetal_abdomen | — | **[FILL IN]** | |
| 12 | *(Fetal_femur — correct prediction)* | Fetal_femur | Heatmap highlights bone shaft | **[FILL IN]** | |
| 13 | *(Fetal_femur — correct prediction)* | Fetal_femur | — | **[FILL IN]** | |
| 14 | *(Fetal_thorax — correct prediction)* | Fetal_thorax | — | **[FILL IN]** | |
| 15 | *(Fetal_thorax — correct prediction)* | Fetal_thorax | — | **[FILL IN]** | |
| 16 | *(Maternal_cervix — correct prediction)* | Maternal_cervix | Heatmap highlights lower uterine segment | **[FILL IN]** | |
| 17 | *(Maternal_cervix — correct prediction)* | Maternal_cervix | — | **[FILL IN]** | |
| 18 | *(Other — correct prediction)* | Other | — | **[FILL IN]** | |
| 19 | *(Other — correct prediction)* | Other | — | **[FILL IN]** | |
| 20 | *(Other — correct prediction)* | Other | — | **[FILL IN]** | |

---

## Tally

> **[FILL IN after completing the 20-image review]**

| Rating | Count | % of 20 |
|---|---|---|
| ✅ Plausible | **[?]** | **[?]%** |
| 🟡 Too large/diffuse | **[?]** | **[?]%** |
| 🔴 Too small/off-target | **[?]** | **[?]%** |
| ⬜ No box produced | **[?]** | **[?]%** |

---

## Per-class breakdown

> **[FILL IN]** — Especially note whether `Brain_Trans_ventricular` /
> `Brain_Trans_thalamic` (the model's known hard pair) also produces worse boxes.
> If so, say so plainly. If not, say so plainly. This is the project culture.

| Class | # images | # Plausible | Notes |
|---|---|---|---|
| Brain_Trans_cerebellum | 3 | **[?]** | |
| Brain_Trans_thalamic | 3 | **[?]** | |
| Brain_Trans_ventricular | 3 | **[?]** | |
| Fetal_abdomen | 2 | **[?]** | |
| Fetal_femur | 2 | **[?]** | |
| Fetal_thorax | 2 | **[?]** | |
| Maternal_cervix | 2 | **[?]** | |
| Other | 3 | **[?]** | |

---

## Summary paragraph

> **[FILL IN]** — A 2–4 sentence honest summary.  
> E.g.: "For the 5 anatomical classes with distinctive structural signatures
> (Femur, Cervix, Abdomen, Thorax, Cerebellum), the CAM-derived box was rated
> Plausible on X/Y images, broadly tracking the correct anatomical region.
> For the hard Brain pair (Thalamic/Ventricular), results were more mixed:
> the same concentrated-but-misclassified CAM pattern documented in EXPERIMENTS.md
> produced boxes that pointed to the right region but the wrong sub-plane (rated
> Plausible under the 'right anatomy' framing, but not useful for sub-class
> discrimination).  Other class boxes were frequently diffuse (⬜ or 🟡),
> consistent with the diffuse-heatmap failure mode documented in EXPERIMENTS.md."

---

## Implementation notes

- `cam_to_bbox()` source: [`src/models/cam_localisation.py`](file:///d:/Ultrasound/src/models/cam_localisation.py)
- Method: Otsu thresholding → largest connected component → min-area filter (2%)
- Default method: `"otsu"` (parameter-free)
- Fallback: `method="percentile"` if Otsu proves too permissive/restrictive
- The `None` return for diffuse CAMs is by design — not an error condition
- Full unit test log: [`logs/smoketest/test_cam_localisation_output.txt`](file:///d:/Ultrasound/logs/smoketest/test_cam_localisation_output.txt)
