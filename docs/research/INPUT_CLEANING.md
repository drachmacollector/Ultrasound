# Input cleaning for 2-D fetal ultrasound frames (R7 E11)

This is a label-free preprocessing step that removes two acquisition artefacts before a classifier sees the image:
- the screen around the imaging sector;
- the measurement markers inside it.

It is the preprocessing of the study's B2 arm. In this study it was the single change that most improved performance on external devices.

**Status:**
- a research tool, not a validated clinical device;
- all evidence below is from public datasets;
- the decomposition analyses are post hoc (`plans/R7_POSTHOC_FOLLOWUP.md`).

## What it does

`src/preprocess/clean.py::clean_image` (CLI: `scripts/clean_ultrasound.py`):

1. **Sector segmentation.** The imaging fan, or linear/full-frame field, is separated from the dark screen background. The threshold is the background level, not Otsu, because Otsu cut away dark fluid inside the fan (R1).
2. **Overlay detection**, with the frozen R1.5 parameters:
   - UI text and labels anywhere on the frame;
   - device-label glyphs inside the fan's hull, which are removed from the sector;
   - cross-shaped caliper markers inside the sector (normalised cross-correlation templates).
3. **Marker removal.** The marker mask is dilated by 2 px and filled by Telea inpainting.
4. **Crop and letterbox.** Everything outside the cleaned sector is set to 0 and the frame is cropped to the sector's bounding box. The crop is then letterboxed to a square: the longest side is scaled to the target size and the rest is zero-padded, so the aspect ratio is kept.

In the study, B2 *training* also drew synthetic calipers on 30% of training images (`src/preprocess/caliper_aug.py`). So the model learned to ignore markers the detector misses. The tool is the deterministic part only; add caliper augmentation when you train.

```
python scripts/clean_ultrasound.py --in my_frames/ --out cleaned/ --size 224 --save-masks
```

The output has three parts:
- the cleaned image for each input;
- with `--save-masks`, a QC overlay: sector outline in green, UI in blue, markers in red;
- `clean_log.csv` with one row per image: sector shape, sector area, number of crosses and dotted lines, UI area, and a QC flag.

The QC flag is raised when the sector covers less than 5% of the frame or no sector boundary is found ("full").

**Exactness.** On HC18, UCL and the African set, the tool's native-scale output is pixel-identical to the cached B2 input used by the study's models (`tests/test_clean.py`). FPDB images in the study also used a train-only device-geometry prior. A new user has no such prior, so the tool uses the per-image path that the external sets used.

## Why: the evidence from this study

All numbers are rendered from the results JSON in `RESULTS.md`; see these sections:

| Question | Where | Kind |
|---|---|---|
| Does B2 improve pooled external balanced accuracy over plain resizing (B0)? | RESULTS.md → RQ4, "Incremental ladder" | R5 secondary (pre-registered, uncorrected) |
| Does the same cleaning help a frozen foundation model (FetalCLIP + linear probe)? | RESULTS.md → RQ4, "Foundation-model probe" | R5 secondary (pre-registered) |
| Which step helps which images? (sector crop vs marker removal, split by detected markers) | RESULTS.md → R7, "Marker stratification"; `results/tables/E9.md` | post hoc |
| Do training-time device-invariance tricks (balanced sampling, style augmentation) add to it? | RESULTS.md → R7, "AFP − B2"; `results/tables/E9.md` | post hoc |

Short reading. This is exploratory and not a clinical claim.
- The two steps fix different failure sources:
  - **cropping** to the sector helps on frames without markers;
  - **marker removal**, with synthetic-caliper training, helps on frames that carry measurement markers. These are common in external clinical sets and rare in the training set.
- Training-time device balancing and style augmentation on top of cleaning did *not* help, and cost accuracy.

## Known limitations

- **Caliper recall is low on faint markers.** On the FPDB train audit, the cross detector reached precision 1.00 but recall 0.20, because FPDB calipers are faint, small or dotted (R1). It fires far more often on UCL and Ghana frames, whose markers are bright crosses.
- **Dotted measurement traces are not removed.** Dotted-line scoring is disabled: every dotted-run detection on the train audit was speckle. Ellipse traces around a head or abdomen therefore stay in the image (see the QC overlay). Synthetic-caliper augmentation during training is the study's mitigation.
- **Sector segmentation assumes a dark screen background.** On a first pass it was judged correct for 88–97% of frames per FPDB machine (93% for the worst machine after a device prior). It can fail on:
  - zoomed frames that fill the screen;
  - split or dual screens;
  - colour Doppler;
  - frames with a bright background.

  Check the QC overlays and `qc_flag`.
- **Greyscale B-mode only.** Colour frames are converted to grey.
- **This is not a substitute for external validation.** Cleaning reduced, but did not remove, the image-level failures that all architectures share (E8, E10). Validate on your own devices before any use.

## Licence and citation

The tool is part of this repository's code. If you use it, cite the study (manuscript in preparation) and the R1/R5 walkthroughs for the detector parameters.
