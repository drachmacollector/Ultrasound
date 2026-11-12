# Results: Anatomy or Acquisition?

Every number below is rendered from `results/json/*.json` by `scripts/research/render_results.py`. Estimates are point [95% patient-clustered bootstrap CI] (B = 2,000). Δ values are paired.
*Illustrative re-analysis of publicly available datasets; not a clinical study.*

## Data

FETAL_PLANES_DB: {{E1.n_images}} images from {{E1.n_patients}} patients. External sets:

| set | images | patients |
|---|---|---|
| HC18 | {{R2_existing_models.models.convnext_tiny.external.hc18.n_images}} | {{R2_existing_models.models.convnext_tiny.external.hc18.n_patients}} |
| UCL | {{R2_existing_models.models.convnext_tiny.external.ucl.n_images}} | {{R2_existing_models.models.convnext_tiny.external.ucl.n_patients}} |
| African (5 centres) | {{R2_existing_models.models.convnext_tiny.external.african.n_images}} | {{R2_existing_models.models.convnext_tiny.external.african.n_patients}} |

Details are in `DATA_CARDS.md`.

## Legacy models in and out of distribution (R2)

**convnext_tiny (B0 recipe, seed 42)**

| | value |
|---|---|
| FPDB-test 8-way macro-F1 | {{R2_existing_models.models.convnext_tiny.fpdb_test.macro_f1_8}} |
| Pooled external accuracy, full argmax | {{R2_existing_models.models.convnext_tiny.external.pooled.acc_full}} |
| Pooled external accuracy, restricted to the classes each set contains | {{R2_existing_models.models.convnext_tiny.external.pooled.acc_restricted|bare}} |
| Share of external errors that are "Other" rejections | {{R2_existing_models.models.convnext_tiny.external.pooled.errors.err_other|pct,bare}} |

## RQ1: is acquisition entangled with the label, and do models encode it?

- **Confound (E1).** 'Other' is {{E1.other_share.trainval.Aloka|pct,bare}} of Aloka training images vs {{E1.other_share.trainval.Voluson E6|pct,bare}} of Voluson E6 images. The difference (Aloka − E6) is {{E1.contrasts.trainval_other_share_aloka_minus_e6|pct,bare}}. Cramér's V (machine × class): {{E1.cramers_v.trainval.machine_x_class8|bare}}.
- **No-pixel model (E2, P1a).** AUROC for standard plane vs Other on seen machines:
  - machine + operator: {{E2.results.prior.seen_machines.auroc_standard_vs_other}}
  - adding frame geometry: {{E2.results.geometry.seen_machines.auroc_standard_vs_other|bare}}
  - Δ for geometry: {{E2.paired_auroc_vs_prior.seen_machines.geometry}}
  - full-image classifier, for reference: {{E2.b0_full_image_auroc.seen_machines|bare}}
- **Decodability (E3, P1b).** Machine decoded within one true class (convnext_tiny features): {{E3.models.convnext_tiny.finetuned.within_class_mean}}. ImageNet features: {{E3.models.convnext_tiny.imagenet.within_class_mean|bare}}. Fine-tuned − ImageNet: {{E3.models.convnext_tiny.finetuned_minus_imagenet_within_class}}.
- **Stratified performance (E6, P1c).** 'Other' recall (convnext_tiny):

  | machine | 'Other' recall |
  |---|---|
  | Aloka | {{E6.models.convnext_tiny.per_machine.Aloka.other_recall|bare}} |
  | Voluson E6 | {{E6.models.convnext_tiny.per_machine.Voluson E6.other_recall|bare}} |
  | Voluson S10 | {{E6.models.convnext_tiny.per_machine.Voluson S10.other_recall|bare}} |

  S10 − E6: {{E6.models.convnext_tiny.contrasts.other_recall_s10_minus_e6}}. The pre-specified direction of P1c is refuted.

## RQ2: is the label learnable from, and do models use, non-anatomical content?

- **Availability (E4, P2a).** 8-way balanced accuracy, where chance is 0.125:
  - with the anatomy blanked: {{E4.modes.nonanat_only.summary.bal_acc_8}}
  - sector shape alone: {{E4.modes.sector_shape_only.summary.bal_acc_8|bare}}
  - no-pixel prior: {{E4.e2_metadata.prior.bal_acc_8|bare}}
- **Reliance (E5, P2b).** Zeroing everything outside the fan sector changes {{E5.models.convnext_tiny.variants.perturb-b_zero.flip_fine|pct,bare}} of convnext_tiny's 8-way predictions.

## RQ3: does saliency fall on the anatomy, and does that predict correctness?

- **Gate G3.** The method passed the randomisation sanity check: Grad-CAM {{G3.decision.gradcam.pass}}, Grad-CAM++ {{G3.decision.gradcampp.pass}}, occlusion {{G3.decision.occlusion.pass}}.
- **On-anatomy saliency (P3a).** EPG − area for convnext_tiny Grad-CAM on FPDB-test:

  | anatomy | EPG − area |
  |---|---|
  | Head | {{R4_localisation.p3a.gradcam.convnext_tiny.fpdb_test~Head.epg_minus_area|bare}} |
  | Abdomen | {{R4_localisation.p3a.gradcam.convnext_tiny.fpdb_test~Abdomen.epg_minus_area|bare}} |
  | Femur | {{R4_localisation.p3a.gradcam.convnext_tiny.fpdb_test~Femur.epg_minus_area|bare}} |

  Femur − Head:

  | model | Grad-CAM | occlusion (16-px patches) |
  |---|---|---|
  | convnext_tiny | {{R4_localisation.p3a.gradcam.convnext_tiny.fpdb_test~femur_minus_head|bare}} | {{R4_localisation.p3a.occlusion.convnext_tiny.fpdb_test~femur_minus_head|bare}} |
  | tf_efficientnetv2_s | {{R4_localisation.p3a.gradcam.tf_efficientnetv2_s.fpdb_test~femur_minus_head|bare}} | {{R4_localisation.p3a.occlusion.tf_efficientnetv2_s.fpdb_test~femur_minus_head|bare}} |

  Under occlusion the sign reverses for every backbone except convnext_tiny (`R4_localisation.md`). The "femur lowest" pattern therefore depends on the resolution of the saliency method.
- **E7: out-of-sector saliency (convnext_tiny).**
  - FPDB-test: {{R4_localisation.e7.convnext_tiny.all.out_sector_frac|bare}}, against a uniform-map baseline of {{R4_localisation.e7.convnext_tiny.all.out_sector_area}}
  - Aloka − E6: {{R4_localisation.e7.convnext_tiny.aloka_minus_e6|bare}}
- **Association (P3b, exploratory).** EPG − area against external correctness, primary model:
  - OR per +0.1: {{R4_association.primary.glm.or|dp=2}} [{{R4_association.primary.glm.or_lo|dp=2}}, {{R4_association.primary.glm.or_hi|dp=2}}]
  - AUROC: {{R4_association.primary.discrimination.auroc_score|bare}}, against confidence {{R4_association.primary.discrimination.auroc_conf|bare}}
- **Ground-truth-free monitor on the African set.** Out-of-sector share of the predicted-class map:
  - AUROC: {{R4_association.african_out_sector_pred.discrimination.auroc_score|bare}}
  - Δ AUROC over confidence: {{R4_association.african_out_sector_pred.discrimination.delta_both_minus_conf}}

## RQ4: does the anatomy-focused protocol help? (R5, pre-registered)

Primary endpoint: pooled external full-argmax balanced accuracy, AFP − B0.

| backbone | Δ | Holm-adjusted p | success | non-inferior |
|---|---|---|---|---|
| convnext_tiny | {{R5_primary.per_backbone.convnext_tiny.afp_vs_b0}} | {{R5_primary.per_backbone.convnext_tiny.holm_p|dp=4}} | {{R5_primary.per_backbone.convnext_tiny.success}} | {{R5_primary.per_backbone.convnext_tiny.non_inferiority_macro_f1_8.non_inferior}} |
| tf_efficientnetv2_s | {{R5_primary.per_backbone.tf_efficientnetv2_s.afp_vs_b0}} | {{R5_primary.per_backbone.tf_efficientnetv2_s.holm_p|dp=4}} | {{R5_primary.per_backbone.tf_efficientnetv2_s.success}} | {{R5_primary.per_backbone.tf_efficientnetv2_s.non_inferiority_macro_f1_8.non_inferior}} |

## Deviations

{{@deviations}}

## Limitations

- **Public data only.** FPDB comes from one centre and four device groups; external sets are small (UCL 51 pregnancies; African 5 centres, 450 images). Nothing here is a clinical claim.
- **Overlay detection.** The caliper detector failed validation (Gate G1), so the overlay analyses are limited to UI and text outside the sector.
- **Saliency.** Saliency metrics depend on method resolution and mask area. EPG is reported with its area baseline, and "deployable" EPG variants still need expert masks.
- **Prior knowledge.** The legacy-model results (R1–R4) were known before the R5 protocol was registered.
