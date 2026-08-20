"""
scripts/build_multitask_manifest.py

Builds the combined train/val CSVs for multi-task training by unioning:
  1. FETAL_PLANES_DB images (image_path + plane_label, NO bbox — classification only)
  2. UCL/HC18/FP/MULTICENTRE annotated images (image_path + plane_label + has_bbox=True)

The resulting CSVs share the same schema as the Phase-3 splits so that
FocalPlanesDataset can load them unchanged. A new 'bbox_path' column (or
absence thereof — indicated by has_bbox=True/False) signals whether a bbox
entry exists in bboxes.json for that image.

Design:
  - The train/val split is derived from the annotation CSVs' own Train/Test
    'Split' column (or the equivalent for HC18 which uses Head_Train.csv etc.).
  - FETAL_PLANES_DB rows come from the existing Phase-3 train.csv / val.csv
    and are appended as-is (no bbox).
  - The combined CSVs are written to:
      data/splits/multitask_train.csv
      data/splits/multitask_val.csv

Usage:
    conda run -n fetalplane python -m scripts.build_multitask_manifest
"""
from __future__ import annotations

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

IMAGES_BASE = Path("data/raw/ucl_hc18/images")
ANNOTATIONS_BASE = Path("data/raw/ucl_hc18/annotations")
PHASE3_TRAIN = Path("data/splits/train.csv")
PHASE3_VAL = Path("data/splits/val.csv")
BBOXES_JSON = Path("data/processed/bboxes.json")
OUT_TRAIN = Path("data/splits/multitask_train.csv")
OUT_VAL = Path("data/splits/multitask_val.csv")

# Subsets with annotated images (all have Head/Abdomen/Femur sub-dirs)
ANNOTATED_SUBSETS = ["UCL", "HC18"]


def _find_image(subset: str, class_name: str, image_name: str) -> Path | None:
    """Locate an image file in the given subset/class directory.

    The annotation CSVs store just the base filename (e.g. '002_HC.jpeg').
    Images live under data/raw/ucl_hc18/images/{subset}/{class_name}/.
    """
    stem, *_ = image_name.rsplit(".", 1)  # strip extension if present
    # Try the image_name as-is, then common extensions
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
      - rows: list of dicts with columns image_path, plane_label,
              patient_id, brain_subplane_raw, source_machine, operator,
              original_split_flag, has_bbox
      - bboxes: updated bboxes dict {image_name: {bbox, class_id}}
    """
    rows_train: list[dict] = []
    rows_val: list[dict] = []
    bboxes: dict = {}

    # Import bbox derivation logic directly from derive_bboxes
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
            for _, row in df.iterrows():
                image_name = str(row["image_name"])
                
                # Use a deterministic hash of the image name for an 80/20 Train/Val split
                # This guarantees that we have a val set for HC18/UCL and don't rely
                # on an unverified "Split" column.
                img_hash = hash(image_name) % 5
                is_train = img_hash != 0

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
                    "patient_id": row.get("SubjectID", -1),
                    "plane_label": cls_label,
                    "brain_subplane_raw": "Not A Brain",  # We don't have this, and it's ignored anyway
                    "source_machine": subset,
                    "operator": "annotation",
                    "original_split_flag": 1 if is_train else 0,
                    "has_bbox": True,
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
    fp_train["has_bbox"] = False
    fp_val["has_bbox"]   = False

    combined_train = pd.concat(
        [fp_train, pd.DataFrame(ann_train_rows)], ignore_index=True
    )
    combined_val = pd.concat(
        [fp_val, pd.DataFrame(ann_val_rows)], ignore_index=True
    )

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    combined_train.to_csv(OUT_TRAIN, index=False)
    combined_val.to_csv(OUT_VAL, index=False)

    n_train_box = combined_train["has_bbox"].sum()
    n_val_box   = combined_val["has_bbox"].sum()
    print(f"  Combined train: {len(combined_train)} rows  ({n_train_box} with bbox, "
          f"{len(combined_train) - n_train_box} without)")
    print(f"  Combined val:   {len(combined_val)} rows  ({n_val_box} with bbox, "
          f"{len(combined_val) - n_val_box} without)")
    print(f"\nWritten to:\n  {OUT_TRAIN}\n  {OUT_VAL}")
    print("\nDone.")


if __name__ == "__main__":
    main()
