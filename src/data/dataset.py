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
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.transforms import load_and_prep_grayscale_to_rgb, get_eval_transform

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
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        label_col: str = "plane_label",
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform if transform is not None else get_eval_transform()
        self.label_col = label_col

        # Validate that every label in the CSV is known
        unknown = set(self.df[label_col].unique()) - set(CANONICAL_CLASSES)
        if unknown:
            raise ValueError(
                f"CSV contains unknown labels not in CANONICAL_CLASSES: {unknown}\n"
                f"Expected: {CANONICAL_CLASSES}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label_str = row[self.label_col]

        img = load_and_prep_grayscale_to_rgb(image_path)  # HxWx3 uint8
        augmented = self.transform(image=img)
        tensor = augmented["image"]  # [3, H, W] float32, normalized

        label_idx = CLASS_TO_IDX[label_str]
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

    CROSS_DEVICE_LABELS = {"Head", "Fetal_abdomen", "Fetal_femur"}

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
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

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label_str = row["plane_label"]
        is_collapsed = bool(row["is_collapsed_label"])
        source_subset = row["source_subset"]

        img = load_and_prep_grayscale_to_rgb(image_path)
        augmented = self.transform(image=img)
        tensor = augmented["image"]

        return tensor, label_str, is_collapsed, source_subset
