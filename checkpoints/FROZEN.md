# checkpoints/ is frozen

These six legacy classifiers (seed 42, trained before the research pivot; tag
`legacy-fetscan-demo-v1`) are the "existing models" analysed in R2–R4. They are
**never overwritten or retrained in place**. `src/train/train.py` refuses any
`checkpoint_dir` / `log_dir` inside this folder. New training goes to `runs/<run_id>/`.

`run_id` used in results: `legacy_<short>_s42`.

| backbone dir | timm name | SHA-256 of `best.pt` (recorded 2026-11-12) |
|---|---|---|
| `convnext_tiny` | convnext_tiny.fb_in22k_ft_in1k | `3bc6867b3e4e5a1836b201dda4b4878127bb609f316d5aef6742b486c697f0bf` |
| `tf_efficientnetv2_s` | tf_efficientnetv2_s.in21k_ft_in1k | `6d59986dff79de366819c16c28d0c2f030d534a9805e1df8aa87714807b03a50` |
| `repvgg_a1` | repvgg_a1 | `123e1633ac308af96d5ea5aa1a1568674bc57cf74444a711c1c4b3b975c278f4` |
| `repvgg_a2` | repvgg_a2 | `be12718e84bf0af301f5f8926ff571d89e87826c4339fd0471732975d65405d1` |
| `mobilenetv3_large_100` | mobilenetv3_large_100 | `c5f4f245ea9b965e01aba793f948e3eee45001d6d4db364ed7e2678fce83baa4` |
| `efficientnet_lite0` | efficientnet_lite0 | `ec0e378e89bf804577dce7b65e7812024c026696e161fbf0b2a4375a1cd0670e` |

`convnext_tiny_acam/` and `convnext_tiny_focal/` are kept on disk for provenance only.
`load_model` refuses ACAM checkpoints. The multitask checkpoint was deleted because it
was trained on HC18/UCL, which are external test sets here.
