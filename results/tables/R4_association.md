# R4.7 (P3b): does on-anatomy saliency predict correctness? (exploratory, legacy checkpoints)

Primary: `convnext_tiny`, gradcam, HC18 + UCL landmark images. 95% patient-cluster CIs.

## Primary (true-class saliency; uses the true label, so not deployable)

* OR 0.99 [0.86, 1.15] per +0.1 epg_minus_area_true (p = 0.932); n = 1421 images / 857 patients, 233 incorrect. `correct ~ s10 + conf_full_summed + C(dataset) + C(true_collapsed)`
* AUROC epg_minus_area_true: 0.520 [0.473, 0.565]; confidence: 0.823 [0.787, 0.858]; both (OOF): 0.784 [0.742, 0.826]; Δ both − confidence -0.039 [-0.066, -0.013] (p = 0.004)

## Predicted-class saliency against the expert mask (NOT deployable; favoured by construction)

The landmark mask marks the true anatomy: when the prediction is wrong, the predicted-class map is scored against an anatomy it does not belong to, so a low EPG for errors is built in. Reported, not interpreted.

* OR 10.65 [5.65, 20.08] per +0.1 epg_minus_area_pred (p = 2.64e-13); n = 1421 images / 857 patients, 233 incorrect. `correct ~ s10 + conf_full_summed + C(dataset) + C(true_collapsed)`
* AUROC epg_minus_area_pred: 0.966 [0.948, 0.982]; confidence: 0.823 [0.787, 0.858]; both (OOF): 0.972 [0.957, 0.985]; Δ both − confidence +0.149 [+0.112, +0.186] (p = 0.0005)

## African (no landmarks): out-of-sector energy of the predicted-class map (ground-truth-free monitor)

* OR 0.31 [0.18, 0.54] per +0.1 out_sector_frac_pred (p = 3.67e-05); n = 450 images / 127 patients, 109 incorrect. `correct ~ s10 + conf_full_summed + C(centre) + C(true_collapsed)`
* AUROC out_sector_frac_pred: 0.778 [0.718, 0.832]; confidence: 0.660 [0.593, 0.725]; both (OOF): 0.793 [0.732, 0.846]; Δ both − confidence +0.133 [+0.059, +0.207] (p = 0.0005)
* sign −1: less out-of-sector energy scored as more likely correct

## In-distribution analogue (FPDB-test landmark images)

* OR 1.04 [0.74, 1.48] per +0.1 epg_minus_area_true (p = 0.804); n = 1481 images / 532 patients, 57 incorrect. `correct ~ s10 + conf_full_summed + C(source_machine) + C(true_collapsed)`
* AUROC epg_minus_area_true: 0.652 [0.583, 0.721]; confidence: 0.866 [0.802, 0.921]; both (OOF): 0.860 [0.801, 0.914]; Δ both − confidence -0.006 [-0.056, +0.045] (p = 0.796)

## All backbones (HC18 + UCL, true target)

| backbone | method | AUROC EPG − area | AUROC confidence | AUROC both (OOF) | Δ both − conf | OR per +0.1 |
|---|---|---|---|---|---|---|
| `convnext` | gradcam | 0.520 [0.473, 0.565] | 0.823 [0.787, 0.858] | 0.784 [0.742, 0.826] | -0.039 [-0.066, -0.013] | 0.99 [0.86, 1.15] |
| `effv2s` | gradcam | 0.873 [0.839, 0.902] | 0.928 [0.909, 0.946] | 0.937 [0.917, 0.956] | +0.009 [-0.004, +0.022] | 3.34 [2.61, 4.28] |
| `repvgga1` | gradcam | 0.809 [0.763, 0.856] | 0.944 [0.928, 0.958] | 0.939 [0.911, 0.964] | -0.005 [-0.031, +0.018] | 3.40 [2.55, 4.55] |
| `repvgga2` | gradcam | 0.765 [0.717, 0.810] | 0.846 [0.811, 0.879] | 0.904 [0.878, 0.928] | +0.058 [+0.018, +0.095] | 2.84 [2.31, 3.50] |
| `mnv3l` | gradcam | 0.580 [0.531, 0.631] | 0.816 [0.785, 0.847] | 0.799 [0.755, 0.839] | -0.017 [-0.057, +0.024] | 1.83 [1.47, 2.29] |
| `efflite0` | gradcam | 0.848 [0.804, 0.894] | 0.883 [0.854, 0.908] | 0.922 [0.891, 0.950] | +0.040 [+0.004, +0.077] | 3.77 [2.87, 4.96] |
| `convnext` | gradcampp | 0.673 [0.629, 0.719] | 0.823 [0.787, 0.858] | 0.818 [0.787, 0.848] | -0.005 [-0.035, +0.023] | 1.49 [1.30, 1.71] |
| `effv2s` | gradcampp | 0.822 [0.780, 0.857] | 0.928 [0.909, 0.946] | 0.933 [0.913, 0.951] | +0.004 [-0.008, +0.016] | 2.25 [1.95, 2.59] |
| `repvgga1` | gradcampp | 0.570 [0.506, 0.630] | 0.939 [0.922, 0.954] | 0.937 [0.916, 0.954] | -0.002 [-0.021, +0.015] | 2.50 [1.71, 3.66] |
| `repvgga2` | gradcampp | 0.582 [0.518, 0.641] | 0.845 [0.809, 0.879] | 0.857 [0.824, 0.885] | +0.011 [-0.028, +0.049] | 2.39 [1.76, 3.24] |
| `mnv3l` | gradcampp | 0.597 [0.555, 0.636] | 0.816 [0.785, 0.847] | 0.805 [0.776, 0.835] | -0.011 [-0.034, +0.014] | 1.97 [1.57, 2.48] |
| `efflite0` | gradcampp | 0.841 [0.790, 0.892] | 0.883 [0.854, 0.908] | 0.912 [0.876, 0.945] | +0.029 [-0.009, +0.070] | 3.50 [2.67, 4.59] |
| `convnext` | occlusion | 0.790 [0.716, 0.851] | 0.867 [0.791, 0.926] | 0.902 [0.847, 0.946] | +0.035 [-0.032, +0.107] | 3.34 [2.22, 5.03] |
| `effv2s` | occlusion | 0.892 [0.798, 0.965] | 0.902 [0.855, 0.945] | 0.908 [0.821, 0.969] | +0.006 [-0.088, +0.077] | 6.52 [3.09, 13.75] |
| `repvgga1` | occlusion | 0.845 [0.766, 0.910] | 0.898 [0.825, 0.956] | 0.954 [0.925, 0.978] | +0.056 [+0.006, +0.122] | 5.66 [2.83, 11.31] |
| `repvgga2` | occlusion | 0.774 [0.692, 0.842] | 0.899 [0.832, 0.949] | 0.922 [0.871, 0.961] | +0.023 [-0.029, +0.082] | 3.36 [2.28, 4.95] |
| `mnv3l` | occlusion | 0.907 [0.857, 0.946] | 0.815 [0.725, 0.896] | 0.914 [0.866, 0.956] | +0.099 [+0.034, +0.173] | 4.99 [2.74, 9.10] |
| `efflite0` | occlusion | 0.703 [0.597, 0.794] | 0.877 [0.796, 0.943] | 0.916 [0.863, 0.954] | +0.039 [-0.027, +0.109] | 2.23 [1.43, 3.47] |
