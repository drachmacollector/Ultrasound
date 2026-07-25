# 02 — Dataset Acquisition

This phase is **mostly `[MANUAL]`** — downloading, licensing checks, and initial inspection are on you; the agent can write the scripts that consume the data once it's on disk in `data/raw/`.

---

## 1. FETAL_PLANES_DB (primary dataset — required)

**What it is:** ~12,400 fetal ultrasound images from 1,792 patients, labeled across 6 standard planes + "Other," collected at BCNatal on Voluson E6/S10 and Aloka machines. This is what the original reference repo trained on.

**`[MANUAL]` Steps:**
1. Locate the dataset. It's commonly mirrored on Kaggle (search "FETAL_PLANES_DB" or "Fetal Planes"), and the original release is via Zenodo (search "FETAL_PLANES_DB Burgos-Artizzu Zenodo" — the citation in the original README points here). **Verify you're getting the version with the official patient-ID and plane-label CSV**, not a re-shuffled Kaggle mirror that's dropped the patient identifiers — patient-level splitting (see `03_DATA_PIPELINE.md`) is not optional and requires the original patient IDs.
2. Download and extract into `data/raw/fetal_planes_db/`. Expected contents: an images folder and a CSV (commonly `FETAL_PLANES_DB_data.csv` or similar) with columns for filename, patient ID (or number), and plane label (and often a "brain plane" sub-label, US machine, and train/test flag).
3. **Read the accompanying README/data descriptor for the exact column names** — do not assume the schema, confirm it against the actual file before writing the manifest-builder script in Phase 3.
4. Confirm the license permits your intended use (research/portfolio, non-commercial demonstration) — note the license text in `data/raw/fetal_planes_db/LICENSE_NOTE.txt` for your own records.
5. Spot check: open 10–15 images per class manually and confirm the label matches what you see (sanity check against mislabeled/corrupted files before trusting the CSV blindly).

## 2. UCL / HC18 multi-centre landmark dataset (held-out cross-device test set — required for the generalization claim)

**What it is:** 4,513 images from 1,904 subjects across 3 clinical sites and 7 different US machines, combining FETAL_PLANES_DB + HC18 + a new UCLH cohort, with expert landmark annotations for head/abdomen/femur biometry. Source paper: arXiv 2512.16710 / *Scientific Reports* s41598-026-47854-3.

**`[MANUAL]` Steps:**
1. The dataset page at `rdr.ucl.ac.uk` may block automated/bot access (confirmed during planning) — **download this one manually through your own browser**, not via a script.
2. Also check the paper's "Data availability" section (arXiv 2512.16710 full text, or the Sci Rep version) for any alternate/mirror hosting.
3. Extract into `data/raw/ucl_multicentre/`.
4. **Critical: exclude any UCLH-cohort images that overlap with images already present in your FETAL_PLANES_DB training set.** The paper explicitly combines FP + HC18 + UCLH — if we're using this as a *held-out* test set, we must only use the portion that is genuinely unseen by our trained model (i.e., the UCLH-only cohort, and ideally HC18 if it wasn't part of your FETAL_PLANES_DB training split either). Read the dataset's own train/test split documentation carefully before using it — do not just dump the whole thing in as "test data" without checking for train/test contamination against our own training set.
5. This dataset only covers **head, abdomen, femur** — it validates 3 of our 7 classes for cross-device generalization. Document this limitation in the eval report; don't imply it validates all 7.

## 3. IUGC / Zenodo intrapartum video dataset (temporal-methodology sandbox only — optional but recommended)

**What it is:** 774 real ultrasound videos (68,106 frames, 24fps) of intrapartum (labor) scans — genuinely a different exam from our target task, used **only** to tune/validate the temporal-smoothing logic against real probe-motion dynamics.

**`[MANUAL]` Steps:**
1. Zenodo record: `zenodo.org/records/17655183` — download `DatasetV3.zip` (~1.1GB) directly.
2. Alternatively/additionally, the Kaggle mirrors (`aspirexxx/iugc-ultrasound-dataset-miccai-2025` and `...-video-dataset-miccai-2024`) can be pulled via the Kaggle CLI if you have an API token set up:
   ```bash
   kaggle datasets download -d aspirexxx/iugc-ultrasound-video-dataset-miccai-2024
   ```
3. Extract into `data/raw/iugc_video/`.
4. **Do not** attempt to map any of its labels (PS/FH visibility, AoP, HSD) onto our 8-class plane taxonomy — they are unrelated. We only use the raw video frames' *temporal characteristics* (motion, blur, transition speed) to tune smoothing hyperparameters.

## 4. Synthetic ego-motion clips (generated, not downloaded)

Not a download — built in Phase 3 (`03_DATA_PIPELINE.md`) from FETAL_PLANES_DB frames via small affine/elastic perturbations. Documented here only so the folder convention is clear: outputs land in `data/processed/synthetic_clips/`, never in `data/raw/`.

## 5. `[MANUAL]` Final folder-state checklist before Phase 3

```
data/raw/
├── fetal_planes_db/
│   ├── images/...
│   ├── <labels csv>
│   └── LICENSE_NOTE.txt
├── ucl_multicentre/
│   ├── images/...
│   ├── <landmark annotations>
│   └── <train/test split docs — read before use>
└── iugc_video/
    └── DatasetV3/...
```

- [ ] FETAL_PLANES_DB downloaded, CSV schema confirmed and documented in a short `data/raw/fetal_planes_db/SCHEMA_NOTES.md`
- [ ] Spot-checked 10–15 images per class against their labels
- [ ] UCL multicentre dataset downloaded, overlap-with-training-set risk read and understood
- [ ] IUGC video downloaded (or explicitly deferred — Tier-1 smoothing can initially be tuned on synthetic data only if you want to skip this download for now, see `05_TEMPORAL_SMOOTHING_AND_REALTIME.md` §2 for the fallback plan)
- [ ] License notes recorded for all three sources
