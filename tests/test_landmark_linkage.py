"""Landmark ground truth (R1.3): counts, no train rows, identity, geometry QC."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

LM = Path("data/processed/landmarks")
pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not (LM / "fpdb_landmarks.parquet").exists(),
                                 reason="run scripts/link_fp_landmarks.py first")]


@pytest.fixture(scope="module")
def fp() -> pd.DataFrame:
    return pd.read_parquet(LM / "fpdb_landmarks.parquet")


def test_fp_counts_and_splits(fp: pd.DataFrame) -> None:
    assert set(fp["split"]) == {"val", "test"}
    test = fp[fp["split"] == "test"].groupby("anatomy").size().to_dict()
    assert test == {"Head": 765, "Abdomen": 350, "Femur": 371}
    assert fp["image_id"].is_unique


def test_fp_identity_and_labels(fp: pd.DataFrame) -> None:
    assert fp["reg_ok"].all()
    assert (fp.loc[fp["anatomy"] == "Head", "fpdb_label"] == "Brain_Trans_thalamic").all()
    assert (fp.loc[fp["anatomy"] == "Abdomen", "fpdb_label"] == "Fetal_abdomen").all()
    assert (fp.loc[fp["anatomy"] == "Femur", "fpdb_label"] == "Fetal_femur").all()


def test_fp_rows_belong_to_test_or_val_split(fp: pd.DataFrame) -> None:
    test_ids = set(pd.read_csv("data/splits/test.csv")["image_path"].map(lambda p: Path(p.replace("\\", "/")).stem))
    assert set(fp.loc[fp["split"] == "test", "image_id"]) <= test_ids


def test_landmarks_inside_image(fp: pd.DataFrame) -> None:
    xs = [c for c in fp.columns if c.endswith("_x")]
    ys = [c for c in fp.columns if c.endswith("_y")]
    for c in xs:
        v = fp[c].dropna()
        assert (v >= 0).all() and (v <= fp.loc[v.index, "width"] - 1).all()
    for c in ys:
        v = fp[c].dropna()
        assert (v >= 0).all() and (v <= fp.loc[v.index, "height"] - 1).all()


def test_head_axes_semantics(fp: pd.DataFrame) -> None:
    head = fp[fp["anatomy"] == "Head"]
    assert (head["bpd_len"] < head["ofd_len"]).mean() > 0.95
    assert head["axis_angle_deg"].between(70, 110).mean() > 0.95


def test_external_landmarks() -> None:
    ext = pd.read_parquet(LM / "external_landmarks.parquet")
    assert ext.groupby(["dataset", "anatomy"]).size().to_dict() == {
        ("hc18", "Head"): 999, ("ucl", "Abdomen"): 130, ("ucl", "Femur"): 135, ("ucl", "Head"): 159}
    manifest_ids = set(pd.read_csv("data/processed/external/external_all.csv")["image_id"])
    assert set(ext["image_id"]) <= manifest_ids
