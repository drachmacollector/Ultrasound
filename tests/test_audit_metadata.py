"""Mask storage round-trip and label-free feature extraction (R1.6)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.audit.metadata import MASK_KEYS, image_features, load_masks, mask_path, save_masks


def test_mask_roundtrip_odd_shape(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    masks = {k: (rng.random((37, 53)) > 0.5).astype(np.uint8) for k in MASK_KEYS}
    p = tmp_path / "x" / "y.npz"
    save_masks(p, masks)
    back = load_masks(p)
    for k in MASK_KEYS:
        assert back[k].dtype == np.uint8 and np.array_equal(back[k], masks[k])


def test_mask_path_subfolders() -> None:
    assert mask_path("ucl/head/002_HC", "ucl", Path("m")).as_posix() == "m/ucl/ucl/head/002_HC.npz"
    assert mask_path("Patient00001_Plane1_1_of_15", "fpdb", Path("m")).as_posix() == \
        "m/fpdb/Patient00001_Plane1_1_of_15.npz"


def test_image_features_keys_and_ranges() -> None:
    h, w = 300, 400
    img = np.zeros((h, w), np.uint8)
    cv2.ellipse(img, (w // 2, -40), (330, 330), 90, -35, 35, 120, -1)
    cv2.putText(img, "E6", (330, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 220, 1)
    feats, masks = image_features(img)
    assert feats["height"] == h and feats["width"] == w
    assert 0 < feats["sector_area_frac"] < 1 and feats["sector_shape_class"] == "fan"
    assert feats["ui_area_frac"] > 0 and feats["caliper_score"] == 0
    assert set(masks) == set(MASK_KEYS) and all(m.shape == (h, w) for m in masks.values())


def test_e2_feature_builder_on_synthetic_frame() -> None:
    """10-row synthetic metadata frame: feature sets, categorical levels, no forbidden columns."""
    import pandas as pd

    from src.audit.meta_features import FEATURE_SETS, FORBIDDEN, build_features, category_levels

    n = 10
    meta = pd.DataFrame({
        "source_machine": ["Aloka", "Voluson E6"] * 5, "operator": ["Op. 3", "Op. 1"] * 5,
        "width": range(600, 610), "height": [400] * n, "aspect": [1.5] * n, "sector_area_frac": [0.5] * n,
        "sector_x0": [0.1] * n, "sector_y0": [0.0] * n, "sector_x1": [0.9] * n, "sector_y1": [1.0] * n,
        "sector_width_ratio": [0.7] * n, "sector_shape_class": ["fan", "linear"] * 5,
        "sector_touch_top": [True] * n, "sector_touch_right": [False] * n, "sector_touch_bottom": [True] * n,
        "sector_touch_left": [False] * n, "mean_intensity": [30.0] * n, "std_intensity": [20.0] * n,
        "frac_nonzero": [0.6] * n, "ui_area_frac": [0.02] * n, "n_ui_components": [40] * n,
        "plane_label": ["Other"] * n, "caliper_score": [0] * n,
    })
    X = build_features(meta, "all")
    assert list(X.columns) == list(FEATURE_SETS["all"]) and len(X) == n
    assert not FORBIDDEN & set(X.columns)
    assert isinstance(X["source_machine"].dtype, pd.CategoricalDtype) and X["sector_touch_top"].dtype.kind == "i"
    # test rows with an unseen level get NaN codes instead of silently new categories
    test = meta.head(2).assign(source_machine=["Voluson S10", "Aloka"])
    Xt = build_features(test, "prior", category_levels(meta))
    assert Xt["source_machine"].isna().tolist() == [True, False]
    assert list(build_features(meta, "prior").columns) == ["source_machine", "operator"]
