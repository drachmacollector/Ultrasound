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

FPDB images per machine: Aloka {{R1_metadata_summary.fpdb_by_machine.Aloka.n}}, Voluson E6 {{R1_metadata_summary.fpdb_by_machine.Voluson E6.n}}, Voluson S10 {{R1_metadata_summary.fpdb_by_machine.Voluson S10.n}}, other {{R1_metadata_summary.fpdb_by_machine.Other.n}}. Details are in `DATA_CARDS.md`.

**Data quality (R1).**
- **Exact duplicates.** {{R1_exact_duplicates.exact_duplicate_images}} FPDB images fall in {{R1_exact_duplicates.exact_duplicate_groups}} groups of byte-identical files. Of these groups, {{R1_exact_duplicates.groups_spanning_splits}} span splits and {{R1_exact_duplicates.groups_with_label_conflict}} carry conflicting labels.
- **Near-duplicates.** FPDB test images with a near-duplicate in the same patient-and-plane acquisition group (SSIM ≥ 0.90 within the sector): {{R1_caliper_pairs.near_duplicate_images}}.
- **Expert landmarks.** Landmarks link pixel-identically to FPDB images for {{R1_landmark_linkage.fp_report.Head.total}} head, {{R1_landmark_linkage.fp_report.Abdomen.total}} abdomen and {{R1_landmark_linkage.fp_report.Femur.total}} femur images. Identity failures: {{R1_landmark_linkage.fp_report.Head.identity_failures}} / {{R1_landmark_linkage.fp_report.Abdomen.identity_failures}} / {{R1_landmark_linkage.fp_report.Femur.identity_failures}}.

**Gates.**
- **G1 (data audit).** The caliper detector found {{G1.detector.train.calipers.tp}} and missed {{G1.detector.train.calipers.fn}} of the caliper-marked audit images, so it did not pass (passes = {{G1.decisions.caliper_detector_passes}}). Overlays are therefore not a primary analysis axis.
- **G2 (legacy results reproduced).** {{G2.gate.passed}}.
- **G3.** See RQ3.
- **G4 (pre-registration).** See `GATE_G4.md`.

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
- Occlusion maps and the figure examples use fixed image subsets drawn with seed {{R4_subsets.seed|raw}} (`R4_subsets.json`).

## Post-hoc exploratory: do different architectures fail on the same images? (E8)

*Added after Gate G4; not pre-registered; legacy checkpoints only.*

- **Agreement.** Agreement between the six legacy architectures on which images they get wrong (Fleiss' κ):
  - FPDB test: {{E8.by_set.fpdb_test.fleiss_kappa|bare}}
  - HC18: {{E8.by_set.hc18.fleiss_kappa|bare}}
  - UCL: {{E8.by_set.ucl.fleiss_kappa|bare}}
  - African: {{E8.by_set.african.fleiss_kappa|bare}}
  - pooled external: {{E8.by_set.pooled_external.fleiss_kappa|bare}}
- **Images all six get wrong.** These are {{E8.by_set.pooled_external.all_wrong_share|pct,bare}} of external images.
- **How they fail.** Of the external images all six get wrong, {{E8.by_set.pooled_external.all_wrong_majority_other|pct}} are rejected as "Other" by most models, against {{E8.by_set.fpdb_test.all_wrong_majority_other|pct}} in distribution.
- **Reading.** External failures are largely shared across architectures, so they are a property of the images rather than of one model. Figure: `E8_error_concordance.pdf`.

## RQ4: does the anatomy-focused protocol help? (R5, pre-registered)

Primary endpoint: pooled external full-argmax balanced accuracy, AFP − B0.

| backbone | Δ | Holm-adjusted p | success | non-inferior |
|---|---|---|---|---|
| convnext_tiny | {{R5_primary.per_backbone.convnext_tiny.afp_vs_b0}} | {{R5_primary.per_backbone.convnext_tiny.holm_p|dp=4}} | {{R5_primary.per_backbone.convnext_tiny.success}} | {{R5_primary.per_backbone.convnext_tiny.non_inferiority_macro_f1_8.non_inferior}} |
| tf_efficientnetv2_s | {{R5_primary.per_backbone.tf_efficientnetv2_s.afp_vs_b0}} | {{R5_primary.per_backbone.tf_efficientnetv2_s.holm_p|dp=4}} | {{R5_primary.per_backbone.tf_efficientnetv2_s.success}} | {{R5_primary.per_backbone.tf_efficientnetv2_s.non_inferiority_macro_f1_8.non_inferior}} |

**Secondary analyses** (exploratory; no multiplicity correction; PREREGISTRATION §6).

Incremental ladder on the primary endpoint, and in-distribution 8-way macro-F1:

| backbone | B1 − B0 | B2 − B0 | AFP − B0, FPDB-test macro-F1 (90% CI; lower end = one-sided 95% bound) |
|---|---|---|---|
| convnext_tiny | {{R5_primary.per_backbone.convnext_tiny.b1_vs_b0|bare}} | {{R5_primary.per_backbone.convnext_tiny.b2_vs_b0|bare}} | {{R5_primary.per_backbone.convnext_tiny.non_inferiority_macro_f1_8|bare}} |
| tf_efficientnetv2_s | {{R5_primary.per_backbone.tf_efficientnetv2_s.b1_vs_b0|bare}} | {{R5_primary.per_backbone.tf_efficientnetv2_s.b2_vs_b0|bare}} | {{R5_primary.per_backbone.tf_efficientnetv2_s.non_inferiority_macro_f1_8|bare}} |

- **Shortcut index (P4b).** Within-class machine decodability, B0 vs AFP:
  - convnext_tiny: {{R5_evaluation.by_arm.b0~convnext_tiny~within_class_machine|bare}} vs {{R5_evaluation.by_arm.afp~convnext_tiny~within_class_machine|bare}}
  - tf_efficientnetv2_s: {{R5_evaluation.by_arm.b0~tf_efficientnetv2_s~within_class_machine|bare}} vs {{R5_evaluation.by_arm.afp~tf_efficientnetv2_s~within_class_machine|bare}}
- **Rejection rate, pooled external** (share predicted outside the set's classes), B0 vs AFP:
  - convnext_tiny: {{R5_evaluation.errors_by_arm.b0~convnext_tiny.pooled.rejection_rate|pct,bare}} vs {{R5_evaluation.errors_by_arm.afp~convnext_tiny.pooled.rejection_rate|pct,bare}}
  - tf_efficientnetv2_s: {{R5_evaluation.errors_by_arm.b0~tf_efficientnetv2_s.pooled.rejection_rate|pct,bare}} vs {{R5_evaluation.errors_by_arm.afp~tf_efficientnetv2_s.pooled.rejection_rate|pct,bare}}
- **P3b on R5 models** (confirmatory for AFP; claim holds only if OR > 1 with a CI excluding 1 for both backbones).
  - Claim holds: {{R5_evaluation.p3b_confirmatory.confirmed}}.
  - OR per +0.1 EPG − area, AFP convnext_tiny: {{R5_evaluation.p3b_confirmatory.cells.afp~convnext_tiny.landmark.glm.or|dp=2}} [{{R5_evaluation.p3b_confirmatory.cells.afp~convnext_tiny.landmark.glm.or_lo|dp=2}}, {{R5_evaluation.p3b_confirmatory.cells.afp~convnext_tiny.landmark.glm.or_hi|dp=2}}].
  - AFP tf_efficientnetv2_s: {{R5_evaluation.p3b_confirmatory.cells.afp~tf_efficientnetv2_s.landmark.glm.or|dp=2}} [{{R5_evaluation.p3b_confirmatory.cells.afp~tf_efficientnetv2_s.landmark.glm.or_lo|dp=2}}, {{R5_evaluation.p3b_confirmatory.cells.afp~tf_efficientnetv2_s.landmark.glm.or_hi|dp=2}}].
- **Foundation-model probe** (frozen FetalCLIP + linear probe; pooled external balanced accuracy):
  - resize input: {{R5_fm_probe.by_mode.resize.pooled|bare}}
  - sector-letterbox-inpaint input: {{R5_fm_probe.by_mode.sector_letterbox_inpaint.pooled|bare}}
  - paired Δ: {{R5_fm_probe.paired_letterbox_minus_resize.pooled}}

Full tables are in `results/tables/R5_primary.md`, `R5_evaluation.md` and `R5_fm_probe.md`.

## Post-hoc exploratory: where does the external gain come from? (R7: E9, E10)

*Planned after the R5 results (`plans/R7_POSTHOC_FOLLOWUP.md`); not pre-registered. Seed-averaged paired patient-cluster bootstrap on the R5 primary endpoint.*

- **Input-cleaning ladder** (pooled external balanced accuracy):

  | backbone | B1 − B0 (sector letterbox) | B2 − B1 (marker inpainting + synthetic calipers) |
  |---|---|---|
  | convnext_tiny | {{E9.per_backbone.convnext_tiny.ladder.b1-b0}} | {{E9.per_backbone.convnext_tiny.ladder.b2-b1}} |
  | tf_efficientnetv2_s | {{E9.per_backbone.tf_efficientnetv2_s.ladder.b1-b0}} | {{E9.per_backbone.tf_efficientnetv2_s.ladder.b2-b1}} |

- **Marker stratification.** These are external images where the R1 detector finds a measurement marker ({{E9.per_backbone.convnext_tiny.marker_strata.with_marker.n_images|int}} images) and images where it finds none ({{E9.per_backbone.convnext_tiny.marker_strata.without_marker.n_images|int}}).

  | backbone | subset | B1 − B0 | B2 − B1 |
  |---|---|---|---|
  | convnext_tiny | with marker | {{E9.per_backbone.convnext_tiny.marker_strata.with_marker.b1-b0|bare}} | {{E9.per_backbone.convnext_tiny.marker_strata.with_marker.b2-b1|bare}} |
  | convnext_tiny | without marker | {{E9.per_backbone.convnext_tiny.marker_strata.without_marker.b1-b0|bare}} | {{E9.per_backbone.convnext_tiny.marker_strata.without_marker.b2-b1|bare}} |
  | tf_efficientnetv2_s | with marker | {{E9.per_backbone.tf_efficientnetv2_s.marker_strata.with_marker.b1-b0|bare}} | {{E9.per_backbone.tf_efficientnetv2_s.marker_strata.with_marker.b2-b1|bare}} |
  | tf_efficientnetv2_s | without marker | {{E9.per_backbone.tf_efficientnetv2_s.marker_strata.without_marker.b1-b0|bare}} | {{E9.per_backbone.tf_efficientnetv2_s.marker_strata.without_marker.b2-b1|bare}} |

- **AFP − B2** (cost of the extra AFP components): convnext_tiny {{E9.per_backbone.convnext_tiny.components_vs_b2.afp-b2}}; tf_efficientnetv2_s {{E9.per_backbone.tf_efficientnetv2_s.components_vs_b2.afp-b2}}.
- **Shared failures (E10, seed {{E10.primary_seed|raw}}).** This is the share of pooled-external images that both backbones get wrong:
  - B0: {{E10.by_seed.11.b0.both_wrong|pct,bare}}
  - B2: {{E10.by_seed.11.b2.both_wrong|pct,bare}}
  - Δ (percentage points): {{E10.by_seed.11.b2.vs_b0.both_wrong_delta|pct,bare}}
  - Of B0's shared failures, B2 rescues {{E10.by_seed.11.b2.vs_b0.rescued_share_of_b0_shared|pct,bare}}.
  - Seeds 22 and 33 are in `results/tables/E10.md`.

Full tables: `results/tables/E9.md`, `E10.md`.

## Deviations

{{@deviations}}

## Limitations

- **Public data only.** FPDB comes from one centre and four device groups, and the external sets are small (see the Data table). Nothing here is a clinical claim.
- **Overlay detection.** The caliper detector failed validation (Gate G1), so the overlay analyses are limited to UI and text outside the sector.
- **Saliency.** Saliency metrics depend on method resolution and mask area. EPG is reported with its area baseline, and "deployable" EPG variants still need expert masks.
- **Prior knowledge.** The legacy-model results (R1–R4) were known before the R5 protocol was registered.
