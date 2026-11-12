"""Ground rule 1: training configs never reference test/external data; FPDB splits are patient-disjoint."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.models.loading import forbidden_token_in
from src.train.train import assert_no_leakage

CONFIGS = sorted(Path("configs").rglob("*.yaml"))


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=[str(p) for p in CONFIGS])
def test_config_train_val_paths_clean(cfg_path: Path) -> None:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for key in ("train_csv", "val_csv"):
        if key in cfg:
            assert forbidden_token_in(cfg[key]) is None, f"{cfg_path}: {key}={cfg[key]}"


def test_splits_patient_disjoint() -> None:
    sets = {s: set(pd.read_csv(f"data/splits/{s}.csv", usecols=["patient_id"])["patient_id"])
            for s in ("train", "val", "test")}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]


def test_guard_rejects_test_csv() -> None:
    with pytest.raises(AssertionError, match="forbidden token"):
        assert_no_leakage({"train_csv": "data/splits/test.csv", "val_csv": "data/splits/val.csv"})


def test_guard_rejects_external_manifest() -> None:
    with pytest.raises(AssertionError, match="forbidden token"):
        assert_no_leakage({"train_csv": "data/splits/train.csv",
                           "val_csv": "data/processed/external/african.csv"})


def test_guard_rejects_patient_overlap(tmp_path: Path) -> None:
    tr = tmp_path / "tr.csv"
    va = tmp_path / "va.csv"
    pd.DataFrame({"patient_id": [1, 2, 3]}).to_csv(tr, index=False)
    pd.DataFrame({"patient_id": [3, 4]}).to_csv(va, index=False)
    with pytest.raises(AssertionError, match="patient overlap"):
        assert_no_leakage({"train_csv": str(tr), "val_csv": str(va)}, test_csv=str(tmp_path / "missing.csv"))


def test_token_check_ignores_ancestor_folders(tmp_path: Path) -> None:
    # outside the repo: only the file name counts
    assert forbidden_token_in(tmp_path / "test_run0" / "train.csv") is None
    assert forbidden_token_in(tmp_path / "hc18.csv") == "hc18"
    # inside the repo: the repo-relative path counts
    assert forbidden_token_in("data/splits/test.csv") == "test"
    assert forbidden_token_in("data/processed/external/x.csv") == "external"
    assert forbidden_token_in("data/splits/train.csv") is None


def test_guard_accepts_real_splits() -> None:
    assert_no_leakage({"train_csv": "data/splits/train.csv", "val_csv": "data/splits/val.csv"})
