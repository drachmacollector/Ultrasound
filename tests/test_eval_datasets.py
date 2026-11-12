"""src/eval/datasets.py: registry resolves every evaluation set to the standard frame."""
from __future__ import annotations

import pandas as pd
import pytest

from src.eval.datasets import FRAME_COLUMNS, load_eval_set

EXPECTED = {  # name -> (images, patients, present classes)
    "fpdb_val": (1023, 124, 6),
    "fpdb_test": (5271, 896, 6),
    "hc18": (999, 806, 1),
    "ucl": (424, 51, 3),
    "african": (450, 127, 4),
}


@pytest.mark.data
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_registry_counts(name: str) -> None:
    es = load_eval_set(name)
    n, k, c = EXPECTED[name]
    assert list(es.frame.columns) == FRAME_COLUMNS
    assert len(es.frame) == n and es.frame["patient_id"].nunique() == k and len(es.present) == c
    prefix = "fpdb_" if es.is_fpdb else {"hc18": "hc18_", "ucl": "ucl_", "african": "afr_"}[name]
    assert es.frame["patient_id"].str.startswith(prefix).all()
    assert es.frame["image_id"].is_unique


def test_unknown_and_colliding_names(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        load_eval_set("nope")
    man = tmp_path / "m.csv"
    pd.DataFrame({"image_path": ["a.png"], "image_id": ["a"], "patient_id": ["x_1"],
                  "plane_label_collapsed": ["Head"]}).to_csv(man, index=False)
    with pytest.raises(ValueError, match="collides"):
        load_eval_set("hc18", manifest=man)
    assert load_eval_set("mysite", manifest=man).present == ("Head",)


def test_fp_landmark_images_refused(tmp_path) -> None:
    man = tmp_path / "m.csv"
    pd.DataFrame({"image_path": ["data/raw/ucl_hc18/images/FP/x.png"], "image_id": ["a"],
                  "patient_id": ["x_1"], "plane_label_collapsed": ["Head"]}).to_csv(man, index=False)
    with pytest.raises(ValueError, match="FP/MULTICENTRE"):
        load_eval_set("mysite", manifest=man)
