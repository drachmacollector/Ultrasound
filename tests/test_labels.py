from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import CANONICAL_CLASSES
from src.data.labels import COLLAPSE_MATRIX, COLLAPSED, COLLAPSED_MEMBERS, collapse_label, collapse_probs
from src.data.paths import fpdb_image_id, norm_path, numeric_prefix, parse_fpdb_stem


def test_collapse_matrix_rows_one_hot() -> None:
    assert COLLAPSE_MATRIX.shape == (8, 6)
    np.testing.assert_array_equal(COLLAPSE_MATRIX.sum(axis=1), np.ones(8))
    assert COLLAPSED_MEMBERS["Head"] == [0, 1, 2]
    assert COLLAPSED_MEMBERS["Other"] == [7]


def test_collapse_probs_preserves_mass() -> None:
    p = np.random.default_rng(0).dirichlet(np.ones(8), size=50)
    c = collapse_probs(p)
    np.testing.assert_allclose(c.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(c[:, 0], p[:, :3].sum(axis=1))


def test_collapse_probs_shape_check() -> None:
    with pytest.raises(ValueError):
        collapse_probs(np.ones((3, 6)))


def test_collapse_label() -> None:
    assert [collapse_label(c) for c in CANONICAL_CLASSES] == [
        "Head", "Head", "Head", "Abdomen", "Femur", "Thorax", "Cervix", "Other"]
    assert all(collapse_label(c) == c for c in COLLAPSED)
    with pytest.raises(ValueError):
        collapse_label("Brain")


def test_paths() -> None:
    assert norm_path(r"data\raw\x\y.png").as_posix() == "data/raw/x/y.png"
    assert fpdb_image_id(r"data\raw\Images\Patient00168_Plane2_1_of_2.png") == "Patient00168_Plane2_1_of_2"
    assert parse_fpdb_stem("Patient00168_Plane2_1_of_2") == {"patient_num": 168, "plane_n": 2, "idx": 1, "of_k": 2}
    with pytest.raises(ValueError):
        parse_fpdb_stem("009_HC")
    assert numeric_prefix("783_4HC.png") == 783 and numeric_prefix("002_AC.jpg") == 2
