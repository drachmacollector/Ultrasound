"""
scripts/build_multitask_manifest.py

Builds the combined train/val CSVs for multi-task training by unioning:
  1. FETAL_PLANES_DB images (image_path + plane_label, NO bbox — classification only)
  2. UCL/HC18 annotated images (image_path + plane_label + has_bbox=True)

The resulting CSVs share the same schema as the Phase-3 splits so that
FocalPlanesDataset can load them unchanged. Two new columns are added:
  - is_annotated_subset: True if row comes from HC18/UCL (detection supervision)
  - has_valid_bbox: True if a bbox was actually derived and stored in bboxes.json

Design:
  - The train/val split is done PER PATIENT (group-based), not per image,
    to prevent a single patient's multiple structures from landing on both
    sides of the split.
  - Patient identifier columns, by file:
      HC18/Head.csv       → SubjectID
      UCL/Head.csv        → SubjectID
      UCL/Abdomen.csv     → SubjectID
      UCL/Femur.csv       → Patient      (different column name!)
  - An 80/20 train/val split is derived by hashing the patient ID, ensuring
    determinism. All images for a patient land in the same split.
  - FETAL_PLANES_DB rows come from the existing Phase-3 train.csv / val.csv
    and are appended as-is (no bbox).
  - The combined CSVs are written to:
      data/splits/multitask_train.csv
      data/splits/multitask_val.csv

Data constraints (hard):
  - FP and MULTICENTRE are EXCLUDED (patient overlap with FETAL_PLANES_DB).
  - The multitask checkpoints must NEVER be evaluated against
    cross_device_manifest.csv as a genuine cross-device held-out set,
    because HC18/UCL are now part of the training pool.

Usage:
    conda run -n fetalplane python -m scripts.build_multitask_manifest
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

# The only genuinely independent annotated datasets.
# We explicitly EXCLUDE "FP" and "MULTICENTRE" to avoid patient leakage
# with the FETAL_PLANES_DB subsets (which they heavily overlap with).
ANNOTATED_SUBSETS = ["UCL", "HC18"]

# Canonical label mapping for the detection head (0-indexed, no background)
DET_CLASS_MAP = {
    "Head":    0,
    "Abdomen": 1,
    "Femur":   2,
}

# Mask detection-only annotations from the classification head by using -100
# (the PyTorch ignore_index default for CrossEntropyLoss).
# This prevents fabricating fine-grained brain subclasses from coarse "Head" labels.
CLS_LABEL_MAP = {
    "Head":    -100,
    "Abdomen": -100,
    "Femur":   -100,
}

# Patient identifier situation (verified by inspecting raw CSV data):
# The SubjectID / Patient columns EXIST in the CSV headers but are NaN
# for all rows — the dataset was released with those fields blank.
#
# However, the image filename encodes the patient as a leading numeric
# prefix:  002_HC.jpeg, 002_3HC.jpeg, 002_AC.jpg, 002_FL.jpeg are all
# from patient "002".  We extract this prefix and use it as the patient
# group key for splitting — analogous to Phase 3's StratifiedGroupKFold.
#
# This is documented here rather than hidden so that anyone re-reading
# this code knows exactly why we aren't using the DB column.

IMAGES_BASE = Path("data/raw/ucl_hc18/images")
ANNOTATIONS_BASE = Path("data/raw/ucl_hc18/annotations")
PHASE3_TRAIN = Path("data/splits/train.csv")
PHASE3_VAL = Path("data/splits/val.csv")
BBOXES_JSON = Path("data/processed/bboxes.json")
OUT_TRAIN = Path("data/splits/multitask_train.csv")
OUT_VAL = Path("data/splits/multitask_val.csv")



def _is_train_split(patient_id: str | int) -> bool:
    """Deterministic 80/20 train/val assignment based on patient ID.

    Uses hashlib.md5 (NOT Python's built-in hash()) for stability.  Python's
    hash() is randomised by PYTHONHASHSEED and will produce a different
    train/val partition on every new interpreter session — in the worst case,
    sending all rows to the same side.  md5 is purely deterministic and
    reproducible regardless of environment.

    We hash the PATIENT ID (not the image name) so that all images belonging
    to the same patient land in the same split — mirroring the
    StratifiedGroupKFold logic used in Phase 3 for FETAL_PLANES_DB.
    """
    digest = hashlib.md5(str(patient_id).encode()).hexdigest()
    return int(digest, 16) % 5 != 0  # 4/5 → train, 1/5 → val


def _patient_id_from_filename(image_name: str) -> str:
    """Extract patient ID from filename leading numeric prefix.

    All UCL and HC18 image filenames follow the pattern:
        <patient_id>[_<sequence>]<structure_suffix>.<ext>
    e.g.  002_HC.jpeg  |  002_3HC.jpeg  |  002_AC.jpg  |  001_FL.jpg

    The leading numeric string is the patient identifier used across
    structures (Head / Abdomen / Femur), so it is safe to hash for
    group-based splitting.

    Falls back to the full image_name if no leading digits are found
    (documented limitation — leakage risk in that case).
    """
    import re
    m = re.match(r'^(\d+)', image_name)
    if m:
        return m.group(1)
    return image_name  # fallback; logged as warning by caller


def _find_image(subset: str, class_name: str, image_name: str) -> Path | None:
    """Locate an image file in the given subset/class directory.

    The annotation CSVs store just the base filename (e.g. '002_HC.jpeg').
    Images live under data/raw/ucl_hc18/images/{subset}/{class_name}/.
    """
    stem, *_ = image_name.rsplit(".", 1)  # strip extension if present
    candidates_dir = IMAGES_BASE / subset / class_name
    if not candidates_dir.exists():
        return None
    for fname in [image_name, f"{stem}.png", f"{stem}.jpeg", f"{stem}.jpg"]:
        p = candidates_dir / fname
        if p.exists():
            return p
    return None


def build_annotated_rows() -> tuple[list[dict], list[dict], dict]:
    """
    Scan all annotated subsets and return:
      - rows_train: list of row dicts destined for multitask_train.csv
      - rows_val:   list of row dicts destined for multitask_val.csv
      - bboxes: updated bboxes dict {image_path_posix: {bbox, class_id}}

    Split key: the leading NUMERIC PREFIX in the image filename (e.g. "002"
    from "002_HC.jpeg"), used as a proxy patient ID.  This groups all images
    for the same patient together across structures (Head/Abdomen/Femur),
    preventing cross-structure leakage within UCL.

    Note: SubjectID / Patient columns in the raw CSVs are NaN throughout
    (released blank).  The filename-prefix approach is the only reliable
    patient grouping available in these datasets.
    """
    rows_train: list[dict] = []
    rows_val: list[dict] = []
    bboxes: dict = {}

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.derive_bboxes import derive_bbox  # type: ignore

    for subset in ANNOTATED_SUBSETS:
        ann_dir = ANNOTATIONS_BASE / subset
        img_dir = IMAGES_BASE / subset
        if not ann_dir.exists() or not img_dir.exists():
            print(f"  [SKIP] {subset}: annotation or image dir missing")
            continue

        for class_name in ["Head", "Abdomen", "Femur"]:
            # Use the main (non-split) CSV for full coverage
            csv_path = ann_dir / f"{class_name}.csv"
            if not csv_path.exists():
                print(f"  [SKIP] {subset}/{class_name}: no CSV")
                continue

            df = pd.read_csv(csv_path)
            det_class_id = DET_CLASS_MAP[class_name]
            cls_label = CLS_LABEL_MAP[class_name]


            n_found = 0
            n_missing = 0
            n_no_patient = 0
            for _, row in df.iterrows():
                image_name = str(row["image_name"])

                # --- Patient-level deterministic split -----------------------
                # SubjectID/Patient columns exist in the CSV schema but are
                # NaN for all rows (released blank).  We derive the patient
                # group from the leading numeric prefix in the image filename,
                # which is consistent across structures for UCL patients.
                patient_id = _patient_id_from_filename(image_name)
                if patient_id == image_name:
                    # Fallback used — prefix extraction failed
                    n_no_patient += 1
                is_train = _is_train_split(patient_id)

                img_path = _find_image(subset, class_name, image_name)
                if img_path is None:
                    n_missing += 1
                    continue
                n_found += 1

                img_path_str = img_path.as_posix()

                # Get actual image dimensions to correctly clip the boxes at generation
                with Image.open(img_path) as im:
                    img_w, img_h = im.size

                bbox = derive_bbox(
                    row, class_name, margin_ratio=0.20,
                    img_width=img_w, img_height=img_h
                )

                record = {
                    "image_path": img_path_str,
                    "patient_id": patient_id,   # numeric prefix from filename
                    "plane_label": cls_label,
                    "brain_subplane_raw": "Not A Brain",
                    "source_machine": subset,
                    "operator": "annotation",
                    "original_split_flag": 1 if is_train else 0,
                    "is_annotated_subset": True,
                    "has_valid_bbox": bbox is not None,
                }

                # Key bboxes.json by the image's POSIX path for exact match in dataset
                if bbox is not None:
                    bboxes[img_path_str] = {
                        "bbox": bbox,
                        "class_id": det_class_id,
                    }

                if is_train:
                    rows_train.append(record)
                else:
                    rows_val.append(record)

            if n_no_patient > 0:
                print(
                    f"  [WARN] {subset}/{class_name}: {n_no_patient}/{n_found} images had "
                    f"no extractable patient prefix — fell back to image-name hash "
                    f"(cross-structure leakage risk for those images)."
                )
            print(f"  {subset}/{class_name}: {n_found} found, {n_missing} missing images")

    return rows_train, rows_val, bboxes


def main() -> None:
    print("=== Building multi-task manifest ===")

    # --- 1. Build annotated-subset rows and updated bboxes dict -----------------
    print("\n[1/3] Scanning annotated subsets …")
    ann_train_rows, ann_val_rows, bboxes = build_annotated_rows()
    print(f"  Annotated train rows: {len(ann_train_rows)}")
    print(f"  Annotated val rows:   {len(ann_val_rows)}")

    # --- 2. Write updated bboxes.json keyed by full posix path ------------------
    print("\n[2/3] Writing bboxes.json …")
    BBOXES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(BBOXES_JSON, "w", encoding="utf-8") as f:
        json.dump(bboxes, f, indent=2)
    print(f"  {len(bboxes)} bboxes written to {BBOXES_JSON}")

    # --- 3. Load Phase-3 FP splits and union with annotated rows ----------------
    print("\n[3/3] Building combined CSVs …")
    required_cols = [
        "image_path", "patient_id", "plane_label", "brain_subplane_raw",
        "source_machine", "operator", "original_split_flag",
    ]
    fp_train = pd.read_csv(PHASE3_TRAIN)[required_cols].copy()
    fp_val   = pd.read_csv(PHASE3_VAL)[required_cols].copy()
    fp_train["is_annotated_subset"] = False
    fp_val["is_annotated_subset"]   = False
    fp_train["has_valid_bbox"] = False
    fp_val["has_valid_bbox"]   = False

    combined_train = pd.concat(
        [fp_train, pd.DataFrame(ann_train_rows)], ignore_index=True
    )
    combined_val = pd.concat(
        [fp_val, pd.DataFrame(ann_val_rows)], ignore_index=True
    )

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    combined_train.to_csv(OUT_TRAIN, index=False)
    combined_val.to_csv(OUT_VAL, index=False)

    n_train_annot = combined_train["is_annotated_subset"].sum()
    n_train_valid = combined_train["has_valid_bbox"].sum()
    n_val_annot   = combined_val["is_annotated_subset"].sum()
    n_val_valid   = combined_val["has_valid_bbox"].sum()
    print(f"  Combined train: {len(combined_train)} rows  ({n_train_annot} annotated subset, "
          f"{n_train_valid} with valid bbox)")
    print(f"  Combined val:   {len(combined_val)} rows  ({n_val_annot} annotated subset, "
          f"{n_val_valid} with valid bbox)")
    print(f"\nWritten to:\n  {OUT_TRAIN}\n  {OUT_VAL}")
    print("\nDone.")


if __name__ == "__main__":
    main()
