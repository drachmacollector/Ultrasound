# 02 — Dataset Acquisition

This phase is **mostly `[MANUAL]`** — downloading, licensing checks, and initial inspection are on you; the agent can write the scripts that consume the data once it's on disk in `data/raw/`.

---

## 1. FETAL_PLANES_DB (primary dataset — required)

**What it is:** ~12,400 fetal ultrasound images from 1,792 patients, labeled across 6 standard planes + "Other," collected at BCNatal on Voluson E6/S10 and Aloka machines. This is what the original reference repo trained on. Citation: Burgos-Artizzu et al., *Sci Rep* 10, 10200 (2020).

**`[MANUAL]` Steps:**
1. Locate the dataset. It's commonly mirrored on Kaggle (search "FETAL_PLANES_DB" or "Fetal Planes"), and the original release is via Zenodo (search "FETAL_PLANES_DB Burgos-Artizzu Zenodo" — the citation in the original README points here). **Verify you're getting the version with the official patient-ID and plane-label CSV**, not a re-shuffled Kaggle mirror that's dropped the patient identifiers — patient-level splitting (see [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md)) is not optional and requires the original patient IDs.
2. Download and extract into `data/raw/fetal_planes_db/`. Expected contents (confirmed against the dataset's own README): an `Images/` folder and `FETAL_PLANES_DB_data.csv` (semi-colon separated), columns `Image_name;Patient_num;Plane;Brain_plane;Operator;US_Machine;Train`. `Plane` ∈ {`Fetal brain`, `Fetal abdomen`, `Fetal femur`, `Fetal thorax`, `Maternal cervix`, `Other`}; `Brain_plane` ∈ {`Trans-thalamic`, `Trans-cerebellum`, `Trans-ventricular`, `Other`, `Not A Brain`}; `Train` ∈ {1=train, 0=test}. Filenames follow `Patient[ID]_Plane[N]_[M]_of_[K].png`.
3. Confirmed: splits are patient-disjoint by design (`Patient_num`) — this matches our own patient-level splitting requirement in Phase 3, no extra reconciliation needed.
4. Spot check: open 10–15 images per class manually and confirm the label matches what you see (sanity check against mislabeled/corrupted files before trusting the CSV blindly).

## 2. Cross-device generalization test set — **use only the `HC18` and `UCL` subsets, NOT `FP` or `MULTICENTRE`**

**What it is:** The publicly released benchmark from arXiv 2512.16710 / *Sci Rep* s41598-026-47854-3 ("A multi-centre, multi-device benchmark dataset for landmark-based comprehensive fetal biometry"), released as an archive (e.g. `FetalBiometry-Multicentre-Landmarks-2026/`) with the structure:

```
FetalBiometry-Multicentre-Landmarks-2026/
├── images/
│   ├── FP/            # Fetal Planes — DO NOT USE, see warning below
│   ├── HC18/           # Head Circumference 2018 — use this
│   ├── UCL/            # UCLH cohort — use this
│   └── MULTICENTRE/    # FP+HC18+UCL merged — DO NOT USE, contains FP
└── annotations/
    ├── FP/
    ├── HC18/
    ├── UCL/
    └── MULTICENTRE/
```

⚠️ **`[CRITICAL — confirmed against source data, this overrides earlier guidance]`**
This bundle is **not** a clean external test set as originally assumed. The paper itself states the dataset "combines three sources: the Fetal Plane (FP) dataset [burgos2020zdataset], the HC18 head dataset, and an expanded cohort from UCLH." `burgos2020zdataset` is the **same Burgos-Artizzu et al. 2020 paper that FETAL_PLANES_DB (§1, our primary training set) comes from** — confirmed independently: `FP`'s own README lists the identical two Barcelona sites (Vall d'Hebron, Sant Joan de Déu), the identical device set (Voluson E6/S8/S10, Aloka), and the identical filename convention (`Patient[ID]_Plane[N]_[M]_of_[K].png`) as FETAL_PLANES_DB.

**`FP` is a landmark-annotated re-release of a subset of FETAL_PLANES_DB, not an independent dataset.** Its 1,047 patients almost certainly overlap heavily with our own 1,792-patient FETAL_PLANES_DB training pool. Using `FP` (or `MULTICENTRE`, which contains `FP`) as part of a "held-out cross-device" test would silently leak training data into what's supposed to be a generalization report — the exact failure mode [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md) warns about for patient-level leakage, just via a different route than originally anticipated.

`HC18` (Netherlands, Voluson E8/730, van den Heuvel et al. 2018) and `UCL` (UCLH, single site, Bano et al./AutoFB) are, by contrast, genuinely independent of FETAL_PLANES_DB — different countries, different acquisition programs, no shared patient pool.

**`[MANUAL]` Steps:**
1. Download the full bundle (the UCL-hosted page at `rdr.ucl.ac.uk` may block automated/bot access — confirmed during planning — **download manually through your own browser**, not via a script). Also check the paper's "Data availability" section (arXiv 2512.16710 full text, or the *Sci Rep* version) for any alternate/mirror hosting.
2. Extract into `data/raw/ucl_hc18/` mirroring the `images/{FP,HC18,UCL,MULTICENTRE}` and `annotations/{FP,HC18,UCL,MULTICENTRE}` structure above (this replaces any earlier flat `ucl_multicentre/images/...` layout assumption — the real archive nests one level deeper, by source subset).
3. **For the cross-device test manifest (Phase 3), only ever read from `images/HC18/` and `images/UCL/` (+ matching `annotations/HC18/`, `annotations/UCL/`). Treat `images/FP/` and `images/MULTICENTRE/` as off-limits for evaluation purposes** — you may still explore them out of curiosity, but they must never enter `data/splits/test.csv`.
4. Note the schema difference from FETAL_PLANES_DB: these annotation CSVs carry **landmark coordinates only** (`bpd_1_x`, `ofd_2_y`, `tad_1_x`, `fl_2_y`, etc.), not a categorical `Plane` column. The class label for these images is **implicit in which anatomy subfolder they live in** (`Head/`, `Abdomen/`, `Femur/`). The Phase 3 manifest-builder will need a second, dedicated function for this — it cannot reuse the FETAL_PLANES_DB CSV-parsing logic.
5. Note the coverage: this gives us **999 HC18 images (Head only) + 424 UCL images (159 Head / 130 Abdomen / 135 Femur) = 1,423 images across 857 unseen subjects** — covering 3 of our 7 classes (map all 3 of our brain sub-planes to a single "Head" bucket for this comparison only, since neither HC18 nor UCL distinguishes trans-thalamic/cerebellum/ventricular). "Thorax" and "Maternal cervix" are not validated cross-device by any available dataset — document this plainly in the eval report, don't imply otherwise.
6. Note also: because HC18/UCL are pure landmark-biometry datasets, **every image in them is a valid standard plane by construction** — there is no "Other"/non-standard class represented. This test set validates *plane identity* cross-device, not Stage 1's standard-vs-other decision. Don't report it as validating the whole pipeline.
7. HC18's own `Head_Train.csv`/`Head_Test.csv` (737/262) is an internal split created by the multi-centre paper's authors for their own experiments — it is **not** the original HC18 grand-challenge split, and it's irrelevant to us since we never train on this data. Since we use HC18 purely as a held-out external set, treat the full 999 images as available for evaluation regardless of that internal flag.

## 3. IUGC / Zenodo intrapartum video dataset (temporal-methodology sandbox only — optional but recommended)

**What it is:** 774 real ultrasound videos (68,106 frames, 24fps) of intrapartum (labor) scans — genuinely a different exam from our target task, used **only** to tune/validate the temporal-smoothing logic against real probe-motion dynamics. Confirmed against the published *Scientific Data* paper (Bai et al., doi:10.5281/zenodo.17655183): 774 videos from 3 institutions — JNU (560 videos/61,924 images), SYSU (121/3,494), SMU (93/2,688) — released under **CC BY 4.0** (the earlier challenge-only release was more restrictively licensed; the Zenodo `17655183` release we use is the open one — record this license explicitly to satisfy the checklist item below).

**`[MANUAL]` Steps:**
1. Zenodo record: `zenodo.org/records/17655183` — download `DatasetV3.zip` (~1.1GB) directly.
2. Alternatively/additionally, the Kaggle mirrors (`aspirexxx/iugc-ultrasound-dataset-miccai-2025` and `...-video-dataset-miccai-2024`) can be pulled via the Kaggle CLI if you have an API token set up:
   ```bash
   kaggle datasets download -d aspirexxx/iugc-ultrasound-video-dataset-miccai-2024
   ```
3. Extract into `data/raw/iugc_video/`. Expected top-level structure per the dataset's own README: `train/`, `val/`, `test/` folders, each with `videos/`, `seg/` (masks + `landmark.json`), `cls/` (classification CSVs), and a top-level `*_info.csv`.
4. **Do not** attempt to map any of its labels (PS/FH visibility, AoP, HSD, or the per-frame `pos`/`neg` standard-plane flag) onto our 8-class plane taxonomy — they are unrelated tasks. We only use the raw video frames' *temporal characteristics* (motion, blur, transition speed) to tune smoothing hyperparameters.

## 4. Synthetic ego-motion clips (generated, not downloaded)

Not a download — built in Phase 3 ([03_DATA_PIPELINE.md](03_DATA_PIPELINE.md)) from FETAL_PLANES_DB frames via small affine/elastic perturbations. Documented here only so the folder convention is clear: outputs land in `data/processed/synthetic_clips/`, never in `data/raw/`.

## 5. `[MANUAL]` Final folder-state checklist before Phase 3

```
data/raw/
├── fetal_planes_db/
│   ├── Images/...
│   ├── FETAL_PLANES_DB_data.csv
│   ├── README.md
│   └── LICENSE_NOTE.txt
├── ucl_hc18/
│   ├── images/
│   │   ├── FP/            # downloaded but NOT used — do not read from this in Phase 3
│   │   ├── HC18/          # used
│   │   ├── UCL/           # used
│   │   └── MULTICENTRE/   # downloaded but NOT used — contains FP, do not read from this in Phase 3
│   └── annotations/ (same subfolder layout)
└── iugc_video/
    └── DatasetV3/...
```

- [ ] FETAL_PLANES_DB downloaded, CSV schema confirmed and documented in the readme`
- [ ] Spot-checked 10–15 images per class against their labels
- [ ] Cross-device bundle downloaded; confirmed folder layout matches `images/{FP,HC18,UCL,MULTICENTRE}`
- [ ] **Explicitly confirmed and documented that only `HC18/` and `UCL/` will be used for evaluation — `FP/` and `MULTICENTRE/` are excluded due to FETAL_PLANES_DB overlap (see §2 warning)**
- [ ] IUGC video downloaded (or explicitly deferred — Tier-1 smoothing can initially be tuned on synthetic data only if you want to skip this download for now, see [05_TEMPORAL_SMOOTHING_AND_REALTIME.md](05_TEMPORAL_SMOOTHING_AND_REALTIME.md) §2 for the fallback plan)
- [ ] License notes recorded for all sources (IUGC confirmed CC BY 4.0 via the Zenodo `17655183` release; check the multi-centre bundle's own `LICENSE` file and FETAL_PLANES_DB's accompanying paper for theirs)
