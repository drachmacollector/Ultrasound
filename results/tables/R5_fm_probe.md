# R5.8 Foundation-model linear probe (FetalCLIP, frozen; secondary, exploratory)

FPDB test: 8-way macro-F1. Externals: full-argmax balanced accuracy over present classes of {Head, Abdomen, Femur, Thorax}. 95% patient-clustered bootstrap CIs (B = 2000).

| input | C (val) | fpdb_test | hc18 | ucl | african | pooled |
|---|---|---|---|---|---|---|
| FetalCLIP + resize | 0.1 | 0.868 [0.857, 0.878] | 0.871 [0.847, 0.893] | 0.649 [0.585, 0.710] | 0.651 [0.597, 0.701] | 0.673 [0.634, 0.711] |
| FetalCLIP + sector_letterbox_inpaint | 0.01 | 0.888 [0.878, 0.898] | 0.838 [0.811, 0.862] | 0.759 [0.700, 0.821] | 0.738 [0.685, 0.785] | 0.762 [0.725, 0.795] |
| Δ letterbox-inpaint − resize (paired) | – | +0.020 [+0.012, +0.029] | -0.033 [-0.049, -0.018] | +0.110 [+0.054, +0.173] | +0.087 [+0.046, +0.126] | +0.089 [+0.058, +0.120] |
| reference: `legacy_convnext_s42` (fine-tuned, unpaired) | – | 0.893 [0.883, 0.902] | 0.814 [0.784, 0.840] | 0.886 [0.836, 0.936] | 0.727 [0.679, 0.773] | 0.728 [0.690, 0.766] |
