# R5 primary endpoint (pre-registered)

Script SHA-256: `333d778d021e40b7f0d305572b194fa3ecae2067170d87d08437f372d92be46e`

Endpoint: pooled external full-argmax balanced accuracy over {Head, Abdomen, Femur, Thorax}. Δ = arm − B0, seed-averaged (3 seeds), paired patient-cluster bootstrap.

| backbone | B0 | AFP | Δ AFP − B0 [95% CI] | p | Holm p | success | NI: Δ macro-F1 (5th pct) |
|---|---|---|---|---|---|---|---|
| `convnext_tiny` | 0.761 | 0.798 | +0.037 [+0.016, +0.058] | 0.0005 | 0.0010 | yes | +0.006 (+0.001; non-inferior) |
| `tf_efficientnetv2_s` | 0.707 | 0.723 | +0.016 [-0.002, +0.034] | 0.0960 | 0.0960 | no | -0.011 (-0.017; non-inferior) |

Secondary (exploratory, no multiplicity correction):

| backbone | B1 − B0 | B2 − B0 | pooled accuracy AFP − B0 |
|---|---|---|---|
| `convnext_tiny` | +0.039 [+0.017, +0.059] | +0.076 [+0.056, +0.096] | -0.006 [-0.015, +0.004] |
| `tf_efficientnetv2_s` | +0.052 [+0.028, +0.077] | +0.079 [+0.058, +0.099] | -0.001 [-0.009, +0.008] |
