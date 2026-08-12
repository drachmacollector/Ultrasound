# Phase 6 Evaluation Report: Fetal Standard Plane Real-Time Detection

## 1. In-Distribution Performance

Performance on the held-out FETAL_PLANES_DB test set (5,271 images from 896 patients) for the primary selected backbone (`convnext_tiny.fb_in22k_ft_in1k`, class-weighted CE).

- **Macro-F1:** 0.8927
- **Overall Accuracy:** 0.90

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Brain_Trans_cerebellum | 0.82 | 0.92 | 0.87 | 339 |
| Brain_Trans_thalamic | 0.84 | 0.85 | 0.85 | 765 |
| Brain_Trans_ventricular | 0.83 | 0.73 | 0.77 | 302 |
| Fetal_abdomen | 0.88 | 0.97 | 0.92 | 358 |
| Fetal_femur | 0.87 | 0.92 | 0.89 | 524 |
| Fetal_thorax | 0.94 | 0.93 | 0.94 | 660 |
| Maternal_cervix | 1.00 | 0.99 | 0.99 | 645 |
| Other | 0.93 | 0.88 | 0.90 | 1678 |

### Confusion Matrix

![Confusion Matrix](../checkpoints/convnext_tiny/confusion_matrix_TEST.png)

> [!NOTE]
> `Brain_Trans_ventricular` remains the weakest class (F1=0.77, recall=0.73), consistent with the literature-documented hard pair and with the focal-loss ablation's negative result (`docs/EXPERIMENTS.md`).
