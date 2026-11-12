"""
src/eval/datasets.py

Registry of evaluation datasets. Every dataset resolves to one *standard frame*:

    image_path, image_id, patient_id, dataset, source_machine, operator, centre,
    true_label, true_collapsed

* FPDB splits: ``true_label`` is the fine label; ``source_machine``/``operator`` from the split CSV.
* Externals (HC18, UCL, African, or any manifest with ``image_path, image_id, patient_id,
  plane_label_collapsed``): ``true_label == true_collapsed``; ``source_machine`` = device.

``present_classes`` gives the collapsed classes a dataset can contain. It drives the restricted
argmax and the rejection rate. For named datasets it is fixed (and checked against the data);
for ad-hoc manifests it is the set of labels observed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.dataset import CANONICAL_CLASSES, validate_external_manifest
from src.data.labels import COLLAPSED, collapse_label
from src.data.paths import fpdb_image_id, fpdb_patient_id, norm_path

FPDB_SPLITS = {"fpdb_train": "train", "fpdb_val": "val", "fpdb_test": "test"}
EXTERNAL_DIR = Path("data/processed/external")
EXTERNALS = ("hc18", "ucl", "african")

#: Collapsed classes each named dataset contains (order follows ``COLLAPSED``).
PRESENT_CLASSES: dict[str, tuple[str, ...]] = {
    **{k: tuple(COLLAPSED) for k in FPDB_SPLITS},
    "hc18": ("Head",),
    "ucl": ("Head", "Abdomen", "Femur"),
    "african": ("Head", "Abdomen", "Femur", "Thorax"),
}

FRAME_COLUMNS = ["image_path", "image_id", "patient_id", "dataset", "source_machine", "operator",
                 "centre", "true_label", "true_collapsed"]
GENERIC_REQUIRED = ("image_path", "image_id", "patient_id", "plane_label_collapsed")


@dataclass(frozen=True)
class EvalSet:
    name: str
    frame: pd.DataFrame
    present: tuple[str, ...]
    is_fpdb: bool
    source: str


def _fpdb(name: str) -> EvalSet:
    src = Path(f"data/splits/{FPDB_SPLITS[name]}.csv")
    d = pd.read_csv(src)
    unknown = set(d["plane_label"]) - set(CANONICAL_CLASSES)
    if unknown:
        raise ValueError(f"{src}: unknown labels {unknown}")
    f = pd.DataFrame({
        "image_path": d["image_path"].map(lambda p: norm_path(p).as_posix()),
        "image_id": d["image_path"].map(fpdb_image_id),
        "patient_id": d["patient_id"].map(fpdb_patient_id),
        "dataset": "fpdb",
        "source_machine": d["source_machine"].astype(str),
        "operator": d["operator"].astype(str),
        "centre": "BCNatal (ES)",
        "true_label": d["plane_label"].astype(str),
    })
    f["true_collapsed"] = f["true_label"].map(collapse_label)
    return EvalSet(name, f[FRAME_COLUMNS], PRESENT_CLASSES[name], True, src.as_posix())


def _external(path: Path, name: str, present: tuple[str, ...] | None) -> EvalSet:
    d = pd.read_csv(path)
    validate_external_manifest(d, str(path), required=GENERIC_REQUIRED)
    f = pd.DataFrame({
        "image_path": d["image_path"].map(lambda p: norm_path(p).as_posix()),
        "image_id": d["image_id"].astype(str),
        "patient_id": d["patient_id"].astype(str),
        "dataset": d["dataset"].astype(str) if "dataset" in d else name,
        "source_machine": d["device"].astype(str) if "device" in d else "NA",
        "operator": "NA",
        "centre": d["centre"].astype(str) if "centre" in d else "NA",
        "true_label": d["plane_label_collapsed"].astype(str),
    })
    f["true_collapsed"] = f["true_label"]
    observed = tuple(c for c in COLLAPSED if c in set(f["true_collapsed"]))
    if present is None:
        present = observed
    elif observed != present:
        raise ValueError(f"{name}: labels {observed} differ from expected present classes {present}")
    return EvalSet(name, f[FRAME_COLUMNS], present, False, path.as_posix())


def load_eval_set(name: str, manifest: str | Path | None = None) -> EvalSet:
    """Resolve a registered dataset name, or an ad-hoc external manifest (``manifest=...``)."""
    if manifest is not None:
        if name in PRESENT_CLASSES:
            raise ValueError(f"--dataset-name {name!r} collides with a registered dataset")
        es = _external(Path(manifest), name, None)
    elif name in FPDB_SPLITS:
        es = _fpdb(name)
    elif name in EXTERNALS:
        es = _external(EXTERNAL_DIR / f"{name}.csv", name, PRESENT_CLASSES[name])
    else:
        raise ValueError(f"unknown dataset {name!r}; registered: {sorted(PRESENT_CLASSES)}")
    if es.frame["image_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate image_id values")
    return es
