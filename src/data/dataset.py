"""
src/data/dataset.py

PyTorch Dataset classes for the fetal plane classification project.

FocalPlanesDataset: reads a frozen FPDB split CSV (train/val/test). Returns
  (image_tensor, label_idx) for training, or a metadata dict with return_meta=True.

ExternalDataset: reads an external manifest (data/processed/external/*.csv, built
  by scripts/build_external_manifests.py) and returns a metadata dict per image.

Both accept an optional ``prepare(img_rgb_uint8, row) -> img`` hook applied before the
albumentations transform (used for sector masking / perturbations in R3-R5).

CANONICAL_CLASSES is the single source of truth for the label -> index mapping
across training, evaluation and analysis.
"""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.data.paths import fpdb_image_id, fpdb_patient_id, norm_path
from src.data.transforms import get_eval_transform, load_and_prep_grayscale_to_rgb

# Single source of truth for label ordering (index = model output position).
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


Prepare = Callable[[np.ndarray, pd.Series], np.ndarray]


class FocalPlanesDataset(Dataset):
    """Dataset backed by a frozen FPDB split CSV (train.csv / val.csv / test.csv).

    Args:
        csv_path: Path to the split CSV file.
        transform: albumentations pipeline. If None, uses get_eval_transform().
        label_col: Column holding the canonical class label string.
        return_meta: If True, __getitem__ returns a dict with image, label, image_id,
            patient_id (``fpdb_<n>``), source_machine and operator.
        prepare: Optional hook applied to the HxWx3 uint8 image before ``transform``.
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Callable | None = None,
        label_col: str = "plane_label",
        return_meta: bool = False,
        prepare: Prepare | None = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform if transform is not None else get_eval_transform()
        self.label_col = label_col
        self.return_meta = return_meta
        self.prepare = prepare

        unknown = set(str(x) for x in self.df[label_col].unique()) - set(CANONICAL_CLASSES)
        if unknown:
            raise ValueError(
                f"CSV contains unknown labels not in CANONICAL_CLASSES: {unknown}. "
                f"Expected: {CANONICAL_CLASSES}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        img = load_and_prep_grayscale_to_rgb(norm_path(row["image_path"]))  # HxWx3 uint8
        if self.prepare is not None:
            img = self.prepare(img, row)
        label_idx = CLASS_TO_IDX[str(row[self.label_col])]
        tensor = self.transform(image=img)["image"]  # [3, H, W] float32, normalized
        if not self.return_meta:
            return tensor, label_idx
        return {
            "image": tensor,
            "label": label_idx,
            "image_id": fpdb_image_id(row["image_path"]),
            "patient_id": fpdb_patient_id(row["patient_id"]),
            "source_machine": str(row["source_machine"]),
            "operator": str(row["operator"]),
        }


EXTERNAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "image_path", "image_id", "dataset", "patient_id", "centre", "device", "plane_label_collapsed",
)
EXTERNAL_ALLOWED_LABELS: frozenset[str] = frozenset({"Head", "Abdomen", "Femur", "Thorax"})


def validate_external_manifest(df: pd.DataFrame, name: str = "manifest",
                               required: tuple[str, ...] = EXTERNAL_REQUIRED_COLUMNS) -> None:
    """Raise ValueError unless ``df`` is an admissible external test manifest.

    FP/MULTICENTRE landmark images are FPDB images (or come from the FPDB centre) and must
    never be used as an external set.
    """
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    forbidden = df["image_path"].map(
        lambda p: any(part in ("FP", "MULTICENTRE") for part in norm_path(p).parts)
    )
    if forbidden.any():
        raise ValueError(f"{name}: FP/MULTICENTRE images must never be used as an external set")
    unknown = set(df["plane_label_collapsed"]) - EXTERNAL_ALLOWED_LABELS
    if unknown:
        raise ValueError(f"{name}: unexpected labels {unknown}")


class ExternalDataset(Dataset):
    """External test set from data/processed/external/<name>.csv.

    Labels are in the collapsed taxonomy (Head/Abdomen/Femur/Thorax). Returns a dict:
    image, image_id, patient_id, dataset, centre, device, label_collapsed.
    """

    REQUIRED_COLUMNS = EXTERNAL_REQUIRED_COLUMNS
    ALLOWED_LABELS = EXTERNAL_ALLOWED_LABELS

    def __init__(
        self,
        csv_path: str | Path,
        transform: Callable | None = None,
        prepare: Prepare | None = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform if transform is not None else get_eval_transform()
        self.prepare = prepare

        validate_external_manifest(self.df, str(csv_path))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        img = load_and_prep_grayscale_to_rgb(norm_path(row["image_path"]))
        if self.prepare is not None:
            img = self.prepare(img, row)
        return {
            "image": self.transform(image=img)["image"],
            "image_id": str(row["image_id"]),
            "patient_id": str(row["patient_id"]),
            "dataset": str(row["dataset"]),
            "centre": str(row["centre"]),
            "device": str(row["device"]),
            "label_collapsed": str(row["plane_label_collapsed"]),
        }
