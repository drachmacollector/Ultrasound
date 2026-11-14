# Key findings: anatomy or acquisition in fetal-plane classifiers

*A short brief for researchers building or evaluating ultrasound AI. Every number below is rendered from `results/json` (95% patient-clustered bootstrap CIs, B = 2000); full detail is in `RESULTS.md`. Public data only: FETAL_PLANES_DB for training; HC18, UCL and a five-centre African set for external testing. No clinical claims.*

Each finding is labelled by the strength of its evidence:
- **[pre-registered]**: OSF registration before training, unmodified analysis code;
- **[pre-specified]**: in the study plan before analysis;
- **[post hoc]**: planned after earlier results were known.

## 1. The acquisition is in the signal, but direct reliance on off-anatomy content is small

- **[pre-specified] The scanner model can be read from the features.** A linear probe recovers it *within a single true class* with balanced accuracy:
  - generic ImageNet features: {{E3.models.convnext_tiny.imagenet.within_class_mean|bare}};
  - a fine-tuned classifier's features: {{E3.models.convnext_tiny.finetuned.within_class_mean|bare}}.

  So the device signature sits in the pixels themselves, and fine-tuning for the plane label does not remove it.
- **[pre-specified] The label is learnable with the anatomy removed.** With the anatomy blanked out, a model still reaches 8-way balanced accuracy {{E4.modes.nonanat_only.summary.bal_acc_8|bare}} (chance 0.125). The shortcut is *available*.
- **[pre-specified] Trained models rely little on content outside the sector.** Zeroing everything outside the imaging sector changes only {{E5.models.convnext_tiny.variants.perturb-b_zero.flip_fine|pct,bare}} of the 8-way predictions. Availability is not the same as use.
- **[post hoc] External failures are a property of the image.** Six architectures fail on largely the same external images: Fleiss' κ {{E8.by_set.pooled_external.fleiss_kappa|bare}}. Of the images all six get wrong, {{E8.by_set.pooled_external.all_wrong_majority_other|pct,bare}} are rejected as "Other" by most models.

## 2. Saliency maps do not tell you when the model is right

- **[pre-registered] On-anatomy saliency does not reliably indicate a correct prediction.** The confirmatory test asked whether more Grad-CAM energy on the expert-annotated anatomy predicts external correctness, in both backbones. It failed: claim holds = {{R5_evaluation.p3b_confirmatory.confirmed}}. The direction differs between architectures:
  - odds ratio per +0.1 EPG − area, AFP convnext_tiny: {{R5_evaluation.p3b_confirmatory.cells.afp~convnext_tiny.landmark.glm.or|dp=2}};
  - AFP tf_efficientnetv2_s: {{R5_evaluation.p3b_confirmatory.cells.afp~tf_efficientnetv2_s.landmark.glm.or|dp=2}}.
- **[pre-specified] Saliency-based conclusions depend on the method.** The "femur gets the least on-anatomy saliency" pattern reverses under occlusion for most backbones.

## 3. Cleaning the input helps more than training the model to ignore the device

The pre-registered primary test (AFP vs B0, pooled external balanced accuracy, Holm over two backbones):
- **[pre-registered]** convnext_tiny {{R5_primary.per_backbone.convnext_tiny.afp_vs_b0}} (success: {{R5_primary.per_backbone.convnext_tiny.success}});
- tf_efficientnetv2_s {{R5_primary.per_backbone.tf_efficientnetv2_s.afp_vs_b0}} (success: {{R5_primary.per_backbone.tf_efficientnetv2_s.success}}).

AFP combines input cleaning with training-time device invariance. The two parts were then separated:
- **[pre-registered secondary] Input cleaning alone (B2) gives the larger gain** over plain resizing. B2 is sector crop + measurement-marker inpainting + synthetic-caliper augmentation:
  - convnext_tiny {{R5_primary.per_backbone.convnext_tiny.b2_vs_b0|bare}};
  - tf_efficientnetv2_s {{R5_primary.per_backbone.tf_efficientnetv2_s.b2_vs_b0|bare}}.
- **[pre-registered secondary] The same cleaning helps a frozen fetal-ultrasound foundation model** (FetalCLIP + linear probe): {{R5_fm_probe.paired_letterbox_minus_resize.pooled|bare}}.
- **[post hoc] Each cleaning step fixes a different failure.** On external images carrying measurement markers ({{E9.per_backbone.tf_efficientnetv2_s.marker_strata.with_marker.n_images|int}} images):
  - marker removal adds {{E9.per_backbone.tf_efficientnetv2_s.marker_strata.with_marker.b2-b1|bare}} (tf_efficientnetv2_s);
  - on marker-free images, sector cropping does the work ({{E9.per_backbone.tf_efficientnetv2_s.marker_strata.without_marker.b1-b0|bare}}).
  - Measurement markers are frequent in external clinical sets and rare in the training set.
- **[post hoc] Training-time device-invariance costs accuracy on top of cleaning.** Each component was added to B2 separately:

  | contrast vs B2 | convnext_tiny | tf_efficientnetv2_s |
  |---|---|---|
  | + acquisition-balanced sampler | {{E9.per_backbone.convnext_tiny.components_vs_b2.b2smp-b2|bare}} | {{E9.per_backbone.tf_efficientnetv2_s.components_vs_b2.b2smp-b2|bare}} |
  | + device-style augmentation | {{E9.per_backbone.convnext_tiny.components_vs_b2.b2sty-b2|bare}} | {{E9.per_backbone.tf_efficientnetv2_s.components_vs_b2.b2sty-b2|bare}} |
  | + both (AFP) | {{E9.per_backbone.convnext_tiny.components_vs_b2.afp-b2|bare}} | {{E9.per_backbone.tf_efficientnetv2_s.components_vs_b2.afp-b2|bare}} |

  - The sampler is the main cost. It raises pooled-external "Other" rejections:
    - convnext_tiny: {{E9.per_backbone.convnext_tiny.per_arm.b2smp.rejection_rate|pct,bare}} vs {{E9.per_backbone.convnext_tiny.per_arm.b2.rejection_rate|pct,bare}} for B2;
    - tf_efficientnetv2_s: {{E9.per_backbone.tf_efficientnetv2_s.per_arm.b2smp.rejection_rate|pct,bare}} vs {{E9.per_backbone.tf_efficientnetv2_s.per_arm.b2.rejection_rate|pct,bare}}.

    Both recipes target the same class balance: class weights in B2, a uniform class mix in the sampler. So the difference lies in how balance is reached (resampling with replacement vs loss weighting) and in balancing by device. This ablation does not separate the two.
  - **Device information in the features does not track external accuracy.** Within-class machine decodability (convnext_tiny):
    - B2: {{E9.per_backbone.convnext_tiny.per_arm.b2.within_class_machine|bare}};
    - with the sampler: {{E9.per_backbone.convnext_tiny.per_arm.b2smp.within_class_machine|bare}};
    - AFP: {{E9.per_backbone.convnext_tiny.per_arm.afp.within_class_machine|bare}}.

    The arm with the most device information removed (AFP) still generalises worse than input cleaning alone.

## 4. What is not solved

- **[post hoc] A shared core of hard external images remains after cleaning.** Under every arm, the two backbones fail on the same images far more often than independence would predict (B2, seed 11: about {{E10.by_seed.11.b2.ratio_vs_independence.point|dp=1}}×; all arms and seeds in `E10.md`). Cleaning rescues part of the shared failure set but adds new shared failures, and the net change varies by seed.
- **External calibration stays poor**, worst on the African set (`R5_evaluation.md`).

## Practical recommendations

These are for reporting and preprocessing; they are not clinical.
1. **Report external, per-device results with patient-clustered CIs.** In-distribution accuracy did not predict external behaviour here.
2. **Split external errors into rejections and confusions.** Most external errors here were "Other" rejections, not anatomical confusions.
3. **Clean the input before training and testing**: sector crop, marker inpainting, synthetic-caliper augmentation. The tool is `scripts/clean_ultrasound.py` (`INPUT_CLEANING.md`). Check its QC overlays; it does not remove dotted measurement traces.
4. **Do not use saliency maps as evidence that a model "looks at the anatomy"** without a sanity check (Gate G3) and an association test against correctness.
5. **Test device-invariance methods against an input-cleaning baseline**, not only against plain resizing.
6. **Treat device-balanced resampling with caution.** It is a common default for multi-device data, and here it cost external accuracy on both architectures without any in-distribution benefit.
