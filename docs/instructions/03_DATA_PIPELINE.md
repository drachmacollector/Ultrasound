# 03 — Data Pipeline (Splitting, Preprocessing, Verification, Synthetic Augmentation)

This phase is almost entirely `[AGENT]` work — scripts to write, given the raw data landed in Phase 2. Human involvement is mostly reviewing outputs (visualizations, split stats) for sanity.

---

## Step 1 — Build the master manifest

`scripts/build_manifest.py` should read `data/raw/fetal_planes_db/<labels csv>` and produce `data/processed/manifest.csv` with at minimum:

```
image_path, patient_id, plane_label, brain_subplane_label (if applicable), source_machine, original_split_flag
```

Map raw plane labels to our canonical 8-class taxonomy exactly as defined in [00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md) §1. Write a small unit test / assertion that every row's label lands in exactly one of the 8 classes and none are null.

## Step 2 — Patient-level train/val/test split (do not skip or shortcut this)

The single most common failure mode in this literature is image-level splitting causing patient leakage across splits, which silently inflates reported accuracy. Rules:

1. Split **by `patient_id`**, never by image, so no patient's frames appear in more than one split.
2. Use the dataset's original train/test flag as the outer boundary if present (respect it — don't reshuffle across it), then carve validation out of the training pool at the patient level.
3. Target an **85/15** train/val split within the training pool (matching the rationale in the reference README: the smallest class needs enough patients represented in validation — recompute this threshold yourself against the actual per-class patient counts in your data rather than assuming 85/15 is universally right; if a class has very few patients, adjust until every class has at least a few patients in each split).
4. Write `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv` (test = the in-distribution FETAL_PLANES_DB held-out test — kept completely separate from the cross-device test set below; see step 5 for the distinction).
5. **Verification script** `scripts/verify_no_leakage.py`: assert the intersection of `patient_id` sets across train/val/test is empty. This should be a hard failing test, not a warning.

## Step 3 — Two distinct test sets, don't conflate them, and mind where they come from

- **In-distribution test set**: held-out patients from FETAL_PLANES_DB itself (same machines as training). Reports the model's ceiling performance.
- **Cross-device test set**: built **only** from `data/raw/ucl_hc18/images/HC18/` and `data/raw/ucl_hc18/images/UCL/` (see [02_DATASETS.md](02_DATASETS.md) §2). **Never build this manifest from `images/FP/` or `images/MULTICENTRE/`** — those are confirmed (via the source paper) to be sourced from the same Burgos-Artizzu/FETAL_PLANES_DB collection as our training data, and including them would leak training data into a report meant to measure generalization to genuinely unseen devices/sites.
- Keep these as two separate evaluation reports; never average them into one headline number, since blending them would misrepresent both.
- `scripts/build_cross_device_manifest.py` needs its own logic distinct from `scripts/build_manifest.py`: HC18/UCL annotation CSVs carry landmark coordinates, not a categorical plane column — the label for each image is **implicit in its anatomy subfolder** (`Head/` → map to a single collapsed "Head" bucket across all 3 of our brain sub-planes; `Abdomen/` → `Fetal_abdomen`; `Femur/` → `Fetal_femur`). There is no "Other"/non-standard class in this data by construction (it's a pure landmark-biometry set), and no "Thorax" or "Maternal cervix" coverage at all — the cross-device manifest and any script consuming it should assert this explicitly rather than silently producing a manifest that looks like it covers all 8 classes.
- Add a `source_subset` column (`HC18` or `UCL`) to the cross-device manifest — useful for reporting per-subset generalization gaps separately in [06_EVALUATION_VALIDATION.md](06_EVALUATION_VALIDATION.md), since the two subsets differ meaningfully in site count, device, and per-anatomy coverage (HC18 = Head only; UCL = Head+Abdomen+Femur).

## Step 4 — Preprocessing

- Resize to a consistent input size (start at 224×224 to match ImageNet-pretrained backbones; revisit if a chosen backbone wants a different native resolution, e.g., EfficientNet variants have their own recommended resolutions — check `timm`'s default config per model).
- Normalize with ImageNet mean/std initially; if a domain-specific pretrained checkpoint is used instead (see [04_MODEL_TRAINING.md](04_MODEL_TRAINING.md)), use whatever normalization that checkpoint was trained with.
- Convert to a consistent format (RGB even though ultrasound is greyscale — 3-channel replication or true grayscale-aware backbone modification is a design choice; simplest and most compatible with pretrained ImageNet backbones is greyscale→3-channel replication, do this unless a specific backbone variant handles 1-channel input natively).
- Cache preprocessed tensors or just preprocess on-the-fly in the Dataset class with a documented, versioned transform pipeline (on-the-fly is simpler and fine at this dataset scale — no need for a preprocessing cache given ~12k images).

## Step 5 — Class imbalance handling

Per the original class distribution (Other: 4,356 down to Trans-ventricular: 597), use one or both of:
- Weighted random sampler at the DataLoader level, or
- Class-weighted loss (weight inversely proportional to class frequency, or focal loss if the smoothed classes remain hard to separate — recall from the literature review that brain sub-planes, especially Trans-ventricular, are the hardest class in this task; expect it to need the most attention)

Document which choice was made and why in `configs/` (a comment in the YAML config is sufficient).

## Step 6 — Synthetic ego-motion clip generator (for temporal jitter robustness)

Build `src/data/synthetic_video.py`:
- Input: a single static frame + its label.
- Output: an N-frame (start with N=16) synthetic clip generated by applying a smoothly-varying sequence of small affine transforms (pan ≤ ~3% of frame width per step, zoom ≤ ~2%, rotation ≤ ~2°) plus mild speckle noise and Gaussian blur jitter, interpolated smoothly frame-to-frame (not independently randomized per frame — use e.g. a slow random walk or sinusoidal perturbation so consecutive frames are continuous, mimicking real hand tremor/probe wobble rather than jittery noise).
- **Explicitly do not** attempt to synthesize transitions *between* different anatomical classes this way — that's the documented limitation from [00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md) §5; this generator only produces within-class motion.
- Save a handful of these as actual `.mp4` files in `data/processed/synthetic_clips/` for visual inspection — `[MANUAL]` step: watch a few of these and confirm they look like plausible small probe movement, not obviously artificial jitter, before using them for training/validation of the smoothing layer.

## Step 7 — Exploratory sanity checks (produce these, review manually)

- Class distribution bar chart, train/val/test, both FETAL_PLANES_DB and the HC18/UCL cross-device sets (charted separately, per `source_subset`, not combined).
- Grid of sample images per class (visual sanity check against mislabeling).
- Patient count per class per split (confirm no class has near-zero patients in validation).
- Image resolution/aspect-ratio distribution (confirms whether resizing strategy is reasonable or whether padding is needed instead of naive resize).

`[MANUAL]`: actually look at all of the above before proceeding to Phase 4 — this is the cheapest point in the whole project to catch a labeling or leakage bug, and the most expensive point to discover it later (after burning GPU-hours training on bad splits).

## Deliverables checklist

- [ ] `data/processed/manifest.csv` (FETAL_PLANES_DB)
- [ ] `data/splits/{train,val,test}.csv`, leakage-verified
- [ ] `data/processed/cross_device_manifest.csv` built **only** from `images/HC18/` + `images/UCL/`, with a `source_subset` column; explicitly excludes `images/FP/` and `images/MULTICENTRE/`
- [ ] Cross-device manifest documented as covering Head/Abdomen/Femur only, with no "Other" class represented — not a full-pipeline validation set
- [ ] Preprocessing transform pipeline implemented and documented
- [ ] Class-imbalance strategy chosen and documented
- [ ] Synthetic ego-motion clip generator implemented, sample clips manually reviewed
- [ ] All sanity-check visualizations produced and reviewed, split by source subset where relevant
