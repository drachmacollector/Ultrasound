"""
src/data/dataset.py

PyTorch Dataset classes for the fetal plane classification project.

FocalPlanesDataset: reads any split CSV (train.csv, val.csv, test.csv)
  produced by patient_split.py. Returns (image_tensor, label_idx) pairs.

CrossDeviceDataset: reads cross_device_manifest.csv; returns the same tuple
  format. Useful for Phase 6 generalization evaluation.

Both use the same CANONICAL_CLASSES list for consistent label → index mapping
across the whole pipeline (training, eval, inference).
"""
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pandas as pd
from torch.utils.data import Dataset

import json
from src.data.transforms import get_eval_transform, load_and_prep_grayscale_to_rgb

# Single source of truth for label ordering — must match IDX_TO_CLASS in inference.py
CANONICAL_CLASSES = [
    "Brain_Trans_cerebellum",   # 0
    "Brain_Trans_thalamic",     # 1
    "Brain_Trans_ventricular",  # 2
    "Fetal_abdomen",            # 3
    "Fetal_femur",              # 4
    "Fetal_thorax",             # 5
    "Maternal_cervix",          # 6
    "Other",                    # 7
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CANONICAL_CLASSES)}
NUM_CLASSES = len(CANONICAL_CLASSES)


class FocalPlanesDataset(Dataset):
    """Dataset backed by any of train.csv / val.csv / test.csv.

    Args:
        csv_path: Path to the split CSV file.
        transform: Callable albumentations pipeline. If None, uses get_eval_transform().
        label_col: Column name holding the canonical class label string.
        bboxes_path: Path to bboxes.json to load bounding box annotations.
    """

    def __init__(
        self,
        csv_path: str,
        transform: Callable | None = None,
        label_col: str = "plane_label",
        bboxes_path: str | None = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform if transform is not None else get_eval_transform(with_bboxes=(bboxes_path is not None))
        self.label_col = label_col
        self.bboxes_dict = None
        
        if bboxes_path:
            with open(bboxes_path, 'r', encoding='utf-8') as f:
                self.bboxes_dict = json.load(f)

        # Validate that every label in the CSV is known
        unique_labels = set(str(x) for x in self.df[label_col].unique())
        allowed_labels = set(CANONICAL_CLASSES) | {"-100"}
        unknown = unique_labels - allowed_labels
        if unknown:
            raise ValueError(
                f"CSV contains unknown labels not in CANONICAL_CLASSES or -100: {unknown}\n"
                f"Expected: {CANONICAL_CLASSES}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = row["image_path"]
        label_str = str(row[self.label_col])

        img = load_and_prep_grayscale_to_rgb(image_path)  # HxWx3 uint8
        label_idx = -100 if label_str == "-100" else CLASS_TO_IDX[label_str]

        if self.bboxes_dict is not None:
            import torch
            bboxes: list = []
            class_labels: list = []

            # Determine if this row is expected to have a bbox.
            # The 'has_valid_bbox' column is written by build_multitask_manifest.py;
            # if the column is absent (legacy CSV), fall back to always trying.
            row_has_bbox: bool = bool(row.get("has_valid_bbox", True))

            if row_has_bbox:
                # Primary key: full posix path (set by build_multitask_manifest.py)
                posix_path = Path(image_path).as_posix()
                item = self.bboxes_dict.get(posix_path) or self.bboxes_dict.get(Path(image_path).name)

                if item is not None:
                    bbox = item["bbox"]
                    h, w = img.shape[:2]
                    xmin = max(0.0, min(float(bbox[0]), w - 1))
                    ymin = max(0.0, min(float(bbox[1]), h - 1))
                    xmax = max(0.0, min(float(bbox[2]), w - 1))
                    ymax = max(0.0, min(float(bbox[3]), h - 1))

                    if xmax > xmin and ymax > ymin:
                        bboxes.append([xmin, ymin, xmax, ymax])
                        class_labels.append(item["class_id"])

            augmented = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
            tensor = augmented["image"]
            aug_bboxes = torch.tensor(augmented["bboxes"], dtype=torch.float32)
            aug_labels = torch.tensor(augmented["class_labels"], dtype=torch.long)

            return tensor, label_idx, aug_bboxes, aug_labels
        else:
            augmented = self.transform(image=img)
            tensor = augmented["image"]  # [3, H, W] float32, normalized
            return tensor, label_idx


class CrossDeviceDataset(Dataset):
    """Dataset backed by cross_device_manifest.csv for Phase 6 generalization eval.

    Note: 'Head' is a collapsed label — at eval time, any of
    Brain_Trans_{cerebellum,thalamic,ventricular} is considered correct.
    This class handles raw loading only; the collapsed-label evaluation logic
    belongs in the Phase 6 evaluation script.

    Args:
        csv_path: Path to cross_device_manifest.csv.
        transform: Callable albumentations pipeline. If None, uses get_eval_transform().
    """

    CROSS_DEVICE_LABELS: ClassVar[set[str]] = {"Head", "Fetal_abdomen", "Fetal_femur"}

    def __init__(
        self,
        csv_path: str,
        transform: Callable | None = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform if transform is not None else get_eval_transform()

        # Assert: no FP or MULTICENTRE paths in this manifest
        forbidden = self.df["image_path"].apply(
            lambda p: any(part in ("FP", "MULTICENTRE") for part in Path(p).parts)
        )
        assert not forbidden.any(), (
            "CrossDeviceDataset: FP/MULTICENTRE paths found in manifest — this should "
            "never happen. Re-run build_cross_device_manifest.py."
        )

        # Assert only known cross-device labels
        unknown = set(self.df["plane_label"].unique()) - self.CROSS_DEVICE_LABELS
        assert not unknown, f"Unexpected labels in cross-device manifest: {unknown}"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = row["image_path"]
        label_str = row["plane_label"]
        is_collapsed = bool(row["is_collapsed_label"])
        source_subset = row["source_subset"]

        img = load_and_prep_grayscale_to_rgb(image_path)
        augmented = self.transform(image=img)
        tensor = augmented["image"]

        return tensor, label_str, is_collapsed, source_subset


def multitask_collate_fn(batch):
    """
    Custom collate_fn for heterogeneous batching with bounding boxes.
    Expects batch elements to be (tensor, label_idx, bboxes, class_labels)
    or just (tensor, label_idx).
    """
    import torch
    images = []
    labels = []
    bboxes = []
    class_labels = []
    
    has_bboxes = len(batch[0]) == 4
    
    for item in batch:
        images.append(item[0])
        labels.append(item[1])
        if has_bboxes:
            bboxes.append(item[2])
            class_labels.append(item[3])
        
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    if has_bboxes:
        return images, labels, bboxes, class_labels
    return images, labels
