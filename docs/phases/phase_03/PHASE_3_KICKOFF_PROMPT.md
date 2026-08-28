# Phase 3 Kickoff — Data Pipeline Build

Follow this instruction & implementation plan & read the following attached files for reference [00_PROJECT_OVERVIEW.md](../instructions/00_PROJECT_OVERVIEW.md), [02_DATASETS.md](../instructions/02_DATASETS.md), [03_DATA_PIPELINE.md](../instructions/03_DATA_PIPELINE.md), and [09_MASTER_CHECKLIST.md](../instructions/09_MASTER_CHECKLIST.md)

---

## Context

Phases 0–2 are complete: environment is set up, and all raw data is downloaded and sitting in `data/raw/` per the structure documented in [02_DATASETS.md](../instructions/02_DATASETS.md) §5:

```
data/raw/
├── fetal_planes_db/
│   ├── Images/*.png
│   └── FETAL_PLANES_DB_data.csv
├── ucl_hc18/
│   ├── images/{FP,HC18,UCL,MULTICENTRE}/...
│   └── annotations/{FP,HC18,UCL,MULTICENTRE}/...
└── iugc_video/
    └── DatasetV3/...
```

You are now executing **Phase 3** exactly as scoped in [03_DATA_PIPELINE.md](../instructions/03_DATA_PIPELINE.md). Read that file (and [02_DATASETS.md](../instructions/02_DATASETS.md) §2 for the critical dataset-contamination context) before writing anything.

## Non-negotiable constraints — read before writing any code

1. **`images/FP/` and `images/MULTICENTRE/` under `ucl_hc18/` must never be read by any script you write in this phase.** They are confirmed to be sourced from the same Burgos-Artizzu collection as `fetal_planes_db/`, and using them anywhere in the cross-device manifest would leak training data into what's supposed to be a held-out generalization test. Only `images/HC18/` and `images/UCL/` are used.
2. **Split by `patient_id`, never by image.** No patient's images may appear in more than one of train/val/test.
3. **Every assertion in the leakage-verification script must be a hard `assert` that raises, not a printed warning.**
4. Do not touch anything in `src/train/`, `src/models/`, or start any training. This phase produces manifests, split files, transforms, class weights, the synthetic clip generator, and sanity-check visualizations only.
5. Use the exact canonical class names below everywhere (manifests, configs, filenames) so Phase 4/5/6 scripts can rely on a single naming convention without re-mapping:
   ```
   Brain_Trans_cerebellum, Brain_Trans_thalamic, Brain_Trans_ventricular,
   Fetal_abdomen, Fetal_femur, Fetal_thorax, Maternal_cervix, Other
   ```
   (These match `IDX_TO_CLASS` in the original repo's `inference.py`, kept for continuity.)

---

## Task 1 — FETAL_PLANES_DB manifest builder

Create `scripts/build_manifest.py`. The tricky part is the `Plane` + `Brain_plane` → canonical-8-class mapping — use this exact logic (note the `Plane == 'Fetal brain'` + `Brain_plane == 'Other'` case, which means a *non-standard brain image* and must map to `Other`, not get dropped):

```python
"""
scripts/build_manifest.py

Reads FETAL_PLANES_DB_data.csv and produces data/processed/manifest.csv with
columns: image_path, patient_id, plane_label, brain_subplane_raw, source_machine,
operator, original_split_flag.
"""
import pandas as pd
from pathlib import Path

RAW_CSV = Path("data/raw/fetal_planes_db/FETAL_PLANES_DB_data.csv")
IMAGES_DIR = Path("data/raw/fetal_planes_db/Images")
OUT_PATH = Path("data/processed/manifest.csv")

CANONICAL_CLASSES = [
    "Brain_Trans_cerebellum", "Brain_Trans_thalamic", "Brain_Trans_ventricular",
    "Fetal_abdomen", "Fetal_femur", "Fetal_thorax", "Maternal_cervix", "Other",
]

def map_to_canonical(plane: str, brain_plane) -> str:
    plane = str(plane).strip()
    brain_plane = str(brain_plane).strip() if pd.notna(brain_plane) else "Not A Brain"

    if plane == "Fetal brain":
        if brain_plane == "Trans-thalamic":
            return "Brain_Trans_thalamic"
        elif brain_plane == "Trans-cerebellum":
            return "Brain_Trans_cerebellum"
        elif brain_plane == "Trans-ventricular":
            return "Brain_Trans_ventricular"
        else:
            # brain_plane == 'Other' -> non-standard brain image, NOT a standard
            # brain sub-plane. This is a common source of silent mislabeling bugs
            # if not handled explicitly -- it belongs in the 'Other' class.
            return "Other"
    elif plane == "Fetal abdomen":
        return "Fetal_abdomen"
    elif plane == "Fetal femur":
        return "Fetal_femur"
    elif plane == "Fetal thorax":
        return "Fetal_thorax"
    elif plane == "Maternal cervix":
        return "Maternal_cervix"
    elif plane == "Other":
        return "Other"
    else:
        raise ValueError(f"Unrecognized Plane value: {plane!r}")


def build_manifest():
    df = pd.read_csv(RAW_CSV, sep=";")
    df.columns = [c.strip() for c in df.columns]

    df["plane_label"] = df.apply(
        lambda r: map_to_canonical(r["Plane"], r.get("Brain_plane")), axis=1
    )

    assert df["plane_label"].isin(CANONICAL_CLASSES).all(), \
        "Found a row that did not map to one of the 8 canonical classes."
    assert df["plane_label"].notna().all(), "Found null canonical labels."

    df["image_path"] = df["Image_name"].apply(lambda name: str(IMAGES_DIR / f"{name}.png"))
    df["patient_id"] = df["Patient_num"]
    df["brain_subplane_raw"] = df.get("Brain_plane")
    df["source_machine"] = df["US_Machine"]
    df["operator"] = df["Operator"]
    df["original_split_flag"] = df["Train"].astype(int)  # 1=train pool, 0=in-distribution test

    out_cols = [
        "image_path", "patient_id", "plane_label", "brain_subplane_raw",
        "source_machine", "operator", "original_split_flag",
    ]
    out = df[out_cols]

    # Sanity: every image file referenced actually exists on disk.
    missing = [p for p in out["image_path"].sample(min(50, len(out))) if not Path(p).exists()]
    assert not missing, f"Sampled manifest rows point to missing files: {missing[:5]}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(out["plane_label"].value_counts())


if __name__ == "__main__":
    build_manifest()
```

Adapt the `RAW_CSV` path / column-stripping if your actual downloaded CSV column names differ slightly from what's documented (check with `df.columns.tolist()` first) — but the mapping function and assertions above should not be changed.

---

## Task 2 — Patient-level split with per-class coverage guarantee

Create `scripts/patient_split.py`. Do not do a naive `train_test_split` on patient IDs — the smallest class (`Brain_Trans_ventricular`, ~597 images total) can end up with near-zero patients in validation if you get unlucky. Use `StratifiedGroupKFold` (patient = group, canonical label = stratify target) and **search across the K candidate folds for the one with the best minimum per-class patient coverage**, rather than always taking fold 0:

```python
"""
scripts/patient_split.py

Splits the Train==1 pool of manifest.csv into train/val at the patient level,
using StratifiedGroupKFold so per-class patient representation in val is
checked and maximized rather than left to chance. The Train==0 pool becomes
the in-distribution test set, untouched.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

MANIFEST_PATH = Path("data/processed/manifest.csv")
SPLITS_DIR = Path("data/splits")
TARGET_VAL_FRACTION = 0.15       # starting point per 03_DATA_PIPELINE.md -- adjust if step 3 below fails
MIN_PATIENTS_PER_CLASS_IN_VAL = 3  # raise/lower per actual per-class patient counts you observe
RANDOM_STATE = 42


def patient_counts_per_class(df: pd.DataFrame) -> pd.Series:
    return df.groupby("plane_label")["patient_id"].nunique()


def find_best_val_fold(train_pool: pd.DataFrame, n_splits: int):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    X = train_pool.index.values
    y = train_pool["plane_label"].values
    groups = train_pool["patient_id"].values

    best_fold_idx, best_min_coverage, best_val_idx = None, -1, None
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(X, y, groups)):
        val_df = train_pool.iloc[val_idx]
        counts = patient_counts_per_class(val_df)
        # every one of the 8 canonical classes must be represented
        min_coverage = counts.reindex(
            [c for c in train_pool["plane_label"].unique()]
        ).fillna(0).min()
        if min_coverage > best_min_coverage:
            best_fold_idx, best_min_coverage, best_val_idx = fold_idx, min_coverage, val_idx

    return best_fold_idx, best_min_coverage, best_val_idx


def build_splits():
    df = pd.read_csv(MANIFEST_PATH)

    test_df = df[df["original_split_flag"] == 0].copy()          # untouched in-distribution test
    train_pool = df[df["original_split_flag"] == 1].copy().reset_index(drop=True)

    n_splits = round(1 / TARGET_VAL_FRACTION)  # e.g. 0.15 -> 7
    best_fold_idx, best_min_coverage, best_val_idx = find_best_val_fold(train_pool, n_splits)

    if best_min_coverage < MIN_PATIENTS_PER_CLASS_IN_VAL:
        raise RuntimeError(
            f"Best achievable minimum per-class patient coverage in val is "
            f"{best_min_coverage}, below the required {MIN_PATIENTS_PER_CLASS_IN_VAL}. "
            f"Recompute per-class patient counts (see printout below) and either lower "
            f"MIN_PATIENTS_PER_CLASS_IN_VAL with justification, or adjust TARGET_VAL_FRACTION."
        )

    val_df = train_pool.iloc[best_val_idx].copy()
    train_df = train_pool.drop(index=best_val_idx).copy()

    print(f"Chosen fold {best_fold_idx}/{n_splits}, min per-class patient coverage in val = {best_min_coverage}")
    print("Per-class patient counts (train pool, for reference):")
    print(patient_counts_per_class(train_pool))

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    print(f"train: {len(train_df)} images / {train_df['patient_id'].nunique()} patients")
    print(f"val:   {len(val_df)} images / {val_df['patient_id'].nunique()} patients")
    print(f"test:  {len(test_df)} images / {test_df['patient_id'].nunique()} patients")


if __name__ == "__main__":
    build_splits()
```

If `MIN_PATIENTS_PER_CLASS_IN_VAL` can't be hit at any fraction you try, that's a legitimate finding to report (per [03_DATA_PIPELINE.md](../instructions/03_DATA_PIPELINE.md) Step 2.3) — don't silently lower the bar without printing the actual per-class patient counts so it's a documented, deliberate decision.

---

## Task 3 — Leakage verification (hard assert, run it and show me the output)

Create `scripts/verify_no_leakage.py`:

```python
"""
scripts/verify_no_leakage.py
Hard-fails (raises) if any patient_id appears in more than one of train/val/test.
"""
import pandas as pd
from pathlib import Path

SPLITS_DIR = Path("data/splits")

def verify():
    train = set(pd.read_csv(SPLITS_DIR / "train.csv")["patient_id"])
    val = set(pd.read_csv(SPLITS_DIR / "val.csv")["patient_id"])
    test = set(pd.read_csv(SPLITS_DIR / "test.csv")["patient_id"])

    assert train.isdisjoint(val), f"Leakage: {len(train & val)} patients in both train and val"
    assert train.isdisjoint(test), f"Leakage: {len(train & test)} patients in both train and test"
    assert val.isdisjoint(test), f"Leakage: {len(val & test)} patients in both val and test"

    print("PASS: train/val/test patient sets are fully disjoint.")
    print(f"train patients: {len(train)}, val patients: {len(val)}, test patients: {len(test)}")

if __name__ == "__main__":
    verify()
```

---

## Task 4 — Cross-device manifest (HC18 + UCL only)

Create `scripts/build_cross_device_manifest.py`. Remember: **no `FP/`, no `MULTICENTRE/`**. Label comes from the anatomy subfolder name, not a CSV column, since these are landmark-annotation CSVs, not plane-label CSVs.

```python
"""
scripts/build_cross_device_manifest.py

Builds data/processed/cross_device_manifest.csv from ONLY:
  data/raw/ucl_hc18/images/HC18/Head/
  data/raw/ucl_hc18/images/UCL/{Head,Abdomen,Femur}/

Label is implicit in the subfolder name. 'Head' is a COLLAPSED label -- at
evaluation time (Phase 6), a prediction is counted correct for a 'Head' row
if the model's argmax is ANY of {Brain_Trans_cerebellum, Brain_Trans_thalamic,
Brain_Trans_ventricular}, since none of these external datasets distinguish
fetal brain sub-planes. This manifest does NOT include an 'Other' class --
every image here is a valid standard plane by construction.

DO NOT read from images/FP/ or images/MULTICENTRE/ -- see 02_DATASETS.md §2.
"""
from pathlib import Path
import pandas as pd

BASE = Path("data/raw/ucl_hc18/images")
OUT_PATH = Path("data/processed/cross_device_manifest.csv")
VALID_EXTS = {".png", ".jpg", ".jpeg"}

# (source_subset, subfolder_name, output_label)
SOURCES = [
    ("HC18", "Head", "Head"),
    ("UCL", "Head", "Head"),
    ("UCL", "Abdomen", "Fetal_abdomen"),
    ("UCL", "Femur", "Fetal_femur"),
]

def build_cross_device_manifest():
    rows = []
    for source_subset, subfolder, label in SOURCES:
        folder = BASE / source_subset / subfolder
        assert folder.exists(), f"Expected folder not found: {folder}"
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() in VALID_EXTS:
                rows.append({
                    "image_path": str(img_path),
                    "plane_label": label,
                    "source_subset": source_subset,
                    "is_collapsed_label": label == "Head",
                })

    df = pd.DataFrame(rows)
    assert len(df) > 0, "No images found -- check BASE path and folder names."

    # Guardrail: make sure nothing from FP/MULTICENTRE snuck in via a wrong BASE path
    forbidden = [p for p in df["image_path"] if "/FP/" in p or "/MULTICENTRE/" in p]
    assert not forbidden, f"FP/MULTICENTRE paths leaked into cross-device manifest: {forbidden[:5]}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(df.groupby(["source_subset", "plane_label"]).size())


if __name__ == "__main__":
    build_cross_device_manifest()
```

---

## Task 5 — Preprocessing transform pipeline

Create `src/data/transforms.py` using `albumentations`. Use `A.MultiplicativeNoise` for speckle (this is the ultrasound-appropriate multiplicative model the doc calls for, not additive Gaussian noise). Build both a `train_transform` (with augmentation) and an `eval_transform` (resize/normalize only), and make sure grayscale→RGB replication happens before normalization:

```python
"""
src/data/transforms.py
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),  # verified safe: none of our 8 classes are laterality-defined
        A.Affine(rotate=(-10, 10), scale=(0.9, 1.1), translate_percent=(0.0, 0.1), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=True, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transform(img_size: int = IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_and_prep_grayscale_to_rgb(image_path: str):
    """Ultrasound images are effectively single-channel; replicate to 3ch for
    ImageNet-pretrained backbones."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img_rgb
```

Wire this into a standard `torch.utils.data.Dataset` reading from `manifest.csv`/`train.csv`/`val.csv`/`test.csv` — this part is straightforward, write it yourself following standard PyTorch Dataset conventions (`__len__`, `__getitem__` returning `(tensor, label_idx)`), using the `CANONICAL_CLASSES` list from Task 1 for the label→index mapping.

---

## Task 6 — Class weights

Create `src/data/class_weights.py`. Per [04_MODEL_TRAINING.md](../instructions/04_MODEL_TRAINING.md) Step 3, class-weighted cross-entropy is the committed default — compute and persist the weights now so Phase 4 just loads a JSON rather than recomputing:

```python
"""
src/data/class_weights.py
"""
import json
import pandas as pd
from pathlib import Path

CANONICAL_CLASSES = [
    "Brain_Trans_cerebellum", "Brain_Trans_thalamic", "Brain_Trans_ventricular",
    "Fetal_abdomen", "Fetal_femur", "Fetal_thorax", "Maternal_cervix", "Other",
]

def compute_class_weights(train_csv: str = "data/splits/train.csv",
                           out_path: str = "data/processed/class_weights.json"):
    df = pd.read_csv(train_csv)
    counts = df["plane_label"].value_counts().reindex(CANONICAL_CLASSES).fillna(0)
    assert (counts > 0).all(), f"A class has zero training images: {counts[counts == 0]}"

    total = counts.sum()
    n_classes = len(CANONICAL_CLASSES)
    # inverse-frequency weighting, normalized so weights average to 1.0
    raw_weights = total / (n_classes * counts)
    weights = (raw_weights / raw_weights.mean()).to_dict()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(weights, f, indent=2)

    print("Class counts:\n", counts)
    print("Computed weights:\n", weights)
    return weights

if __name__ == "__main__":
    compute_class_weights()
```

---

## Task 7 — Synthetic ego-motion clip generator

Create `src/data/synthetic_video.py`. **The critical requirement is continuity**: parameters must drift smoothly frame-to-frame (a random walk with momentum), not be independently randomized per frame — independent randomization produces jittery noise that looks nothing like real hand tremor and would make Tier-1 smoothing validation in Phase 5 meaningless.

```python
"""
src/data/synthetic_video.py

Generates an N-frame synthetic clip from a single static frame by applying a
SMOOTHLY DRIFTING sequence of small affine transforms (never independently
randomized per frame) plus mild speckle noise and blur jitter. Mimics probe
wobble/hand tremor for temporal-smoothing validation (05_TEMPORAL_SMOOTHING...).

Does NOT synthesize class transitions -- only within-class motion.
"""
import cv2
import numpy as np


class _SmoothRandomWalk:
    """1D bounded random walk with momentum, for continuous parameter drift."""
    def __init__(self, low: float, high: float, start=None, momentum=0.85, step_std=None, rng=None):
        self.low, self.high = low, high
        self.momentum = momentum
        self.step_std = step_std if step_std is not None else (high - low) * 0.08
        self.rng = rng or np.random.default_rng()
        self.value = start if start is not None else (low + high) / 2
        self.velocity = 0.0

    def step(self) -> float:
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * self.rng.normal(0, self.step_std)
        self.value = float(np.clip(self.value + self.velocity, self.low, self.high))
        return self.value


def generate_ego_motion_clip(image: np.ndarray, n_frames: int = 16, seed: int = None) -> list:
    """
    image: HxWx3 uint8 RGB frame.
    Returns: list of n_frames HxWx3 uint8 RGB frames.
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    pan_x_walk = _SmoothRandomWalk(-0.03 * w, 0.03 * w, start=0.0, rng=rng)   # <= ~3% frame width
    pan_y_walk = _SmoothRandomWalk(-0.03 * h, 0.03 * h, start=0.0, rng=rng)
    zoom_walk = _SmoothRandomWalk(-0.02, 0.02, start=0.0, rng=rng)             # <= ~2% zoom
    rot_walk = _SmoothRandomWalk(-2.0, 2.0, start=0.0, rng=rng)                # <= ~2 degrees

    frames = []
    for _ in range(n_frames):
        dx, dy = pan_x_walk.step(), pan_y_walk.step()
        zoom, rot = zoom_walk.step(), rot_walk.step()

        M = cv2.getRotationMatrix2D((cx, cy), angle=rot, scale=1.0 + zoom)
        M[0, 2] += dx
        M[1, 2] += dy
        warped = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # mild speckle (multiplicative) noise, ultrasound-appropriate
        speckle_sigma = 0.04
        noise = rng.normal(1.0, speckle_sigma, warped.shape).astype(np.float32)
        speckled = np.clip(warped.astype(np.float32) * noise, 0, 255).astype(np.uint8)

        # slow blur jitter -- occasionally soften slightly, never sharpen
        if rng.random() < 0.3:
            k = rng.choice([3, 5])
            speckled = cv2.GaussianBlur(speckled, (k, k), 0)

        frames.append(speckled)

    return frames


def save_clip_as_mp4(frames: list, out_path: str, fps: int = 24):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
```

Write a small driver script `scripts/generate_sample_clips.py` that picks ~10 random images (a few per class) from `train.csv`, generates a clip for each via `generate_ego_motion_clip`, and saves them to `data/processed/synthetic_clips/` via `save_clip_as_mp4` — this is what I'll manually review (checkpoint #2 below).

---

## Task 8 — Sanity-check visualizations

Per [03_DATA_PIPELINE.md](../instructions/03_DATA_PIPELINE.md) Step 7, produce and save to `data/processed/sanity_checks/`:
- Class distribution bar chart for train/val/test (FETAL_PLANES_DB) **and** a separate one for the cross-device manifest, broken down by `source_subset`.
- A grid of ~5 sample images per class (use matplotlib subplots), for both datasets.
- A table/printout of patient count per class per split.
- A histogram of image resolution/aspect ratio across the FETAL_PLANES_DB images (to confirm plain resizing is reasonable, not distorting content).

Standard matplotlib/seaborn code, write this yourself — no unusual logic here, just make sure every chart gets saved as a `.png` under `data/processed/sanity_checks/` so I can review them without re-running anything.

---

## Deliverables checklist for this session

- [ ] `scripts/build_manifest.py` run successfully, `data/processed/manifest.csv` produced
- [ ] `scripts/patient_split.py` run successfully, `data/splits/{train,val,test}.csv` produced, printed per-class patient coverage shown
- [ ] `scripts/verify_no_leakage.py` run and shows `PASS`
- [ ] `scripts/build_cross_device_manifest.py` run successfully, `data/processed/cross_device_manifest.csv` produced, confirmed via printed groupby that it contains only HC18/UCL rows
- [ ] `src/data/transforms.py` and a `Dataset` class wired to the split CSVs
- [ ] `src/data/class_weights.py` run, `data/processed/class_weights.json` produced
- [ ] `src/data/synthetic_video.py` + `scripts/generate_sample_clips.py` run, ~10 sample `.mp4` clips in `data/processed/synthetic_clips/`
- [ ] All sanity-check visualizations generated and saved to `data/processed/sanity_checks/`

When done, give me a short summary: final row counts for each manifest/split, the per-class patient-coverage numbers from the split script, and confirmation the leakage check passed — don't just say "done," show the actual numbers so I can review before Phase 4.

---

## What I will check manually once you're done (don't skip producing the artifacts these depend on)

1. Open `data/processed/cross_device_manifest.csv` and confirm every `image_path` contains `/HC18/` or `/UCL/` and never `/FP/` or `/MULTICENTRE/`.
2. Watch a handful of the generated `.mp4` clips in `data/processed/synthetic_clips/` and confirm the motion looks like plausible small probe wobble.
3. Review every chart in `data/processed/sanity_checks/` — class balance, sample image grids, patient-per-split counts, resolution histogram — before Phase 4 starts.
