"""External manifests (R1.2): schema, dataset-qualified patient IDs, verified counts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.dataset import ExternalDataset

EXT = Path("data/processed/external")
pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not (EXT / "external_all.csv").exists(),
                                 reason="run scripts/build_external_manifests.py first")]


@pytest.fixture(scope="module")
def ext() -> pd.DataFrame:
    return pd.read_csv(EXT / "external_all.csv")


def test_counts(ext: pd.DataFrame) -> None:
    g = ext.groupby("dataset").agg(n=("image_id", "size"), p=("patient_id", "nunique"))
    assert g.loc["hc18"].tolist() == [999, 806]
    assert g.loc["ucl"].tolist() == [424, 51]
    assert g.loc["african"].tolist() == [450, 127]


def test_ids_unique_and_qualified(ext: pd.DataFrame) -> None:
    assert ext["image_id"].is_unique
    prefix = ext["patient_id"].str.split("_").str[0]
    assert (prefix == ext["dataset"].map({"hc18": "hc18", "ucl": "ucl", "african": "afr"})).all()


def test_african_thorax_and_split(ext: pd.DataFrame) -> None:
    afr = ext[ext["dataset"] == "african"]
    thorax = afr[afr["plane_label_collapsed"] == "Thorax"]
    assert len(thorax) == 75 and set(thorax["centre"]) == {"Algeria", "Egypt", "Malawi"}
    assert afr.groupby("patient_id")["orig_split"].nunique().max() == 1
    # Patient_num repeats across centres; qualification keeps them apart
    assert afr.loc[afr["image_id"] == "african/patient001_DZA_plane0", "patient_id"].item() == "afr_DZA_1"
    assert afr.loc[afr["image_id"] == "african/patient001_EGY_plane0", "patient_id"].item() == "afr_EGY_1"


def test_no_fp_images(ext: pd.DataFrame) -> None:
    assert not ext["image_path"].str.contains("/FP/|/MULTICENTRE/", regex=True).any()


def test_external_dataset_item() -> None:
    ds = ExternalDataset(EXT / "african.csv")
    item = ds[0]
    assert tuple(item["image"].shape) == (3, 224, 224)
    assert item["patient_id"].startswith("afr_") and item["label_collapsed"] in ExternalDataset.ALLOWED_LABELS


def test_external_dataset_rejects_fp(tmp_path: Path) -> None:
    bad = pd.read_csv(EXT / "ucl.csv").head(2)
    bad.loc[:, "image_path"] = "data/raw/ucl_hc18/images/FP/Head/x.png"
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="FP/MULTICENTRE"):
        ExternalDataset(tmp_path / "bad.csv")
