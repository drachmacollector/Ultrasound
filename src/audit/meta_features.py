"""
src/audit/meta_features.py

Feature sets for the no-pixel metadata model (R3 E2). All features are label-free and come from
``data/processed/metadata/acquisition_metadata.parquet`` (R1.6).

    prior      machine, operator
    geometry   frame size/aspect and sector geometry (nearly collinear with machine)
    intensity  whole-image mean / std / non-zero share
    overlay    UI/text detector outputs only. Gate G1 validated the UI detector (recall 0.97,
               precision 1.00) but not the caliper detector (recall 0.20), so caliper features
               are excluded.
    all        union
"""
from __future__ import annotations

import pandas as pd

CATEGORICAL = ("source_machine", "operator", "sector_shape_class")
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "prior": ("source_machine", "operator"),
    "geometry": ("width", "height", "aspect", "sector_area_frac", "sector_x0", "sector_y0", "sector_x1",
                 "sector_y1", "sector_width_ratio", "sector_shape_class", "sector_touch_top",
                 "sector_touch_right", "sector_touch_bottom", "sector_touch_left"),
    "intensity": ("mean_intensity", "std_intensity", "frac_nonzero"),
    "overlay": ("ui_area_frac", "n_ui_components"),
}
FEATURE_SETS["all"] = tuple(dict.fromkeys(c for k in ("prior", "geometry", "intensity", "overlay")
                                          for c in FEATURE_SETS[k]))
#: columns that must never enter a metadata model (labels, ids, in-sector content, failed detectors)
FORBIDDEN = {"plane_label", "plane_collapsed", "brain_subplane_raw", "patient_id", "image_id", "image_path",
             "md5", "sector_mean_intensity", "caliper_score", "n_crosses", "n_dotted_lines", "marker_area_frac"}
assert not FORBIDDEN & set(FEATURE_SETS["all"])


def build_features(meta: pd.DataFrame, feature_set: str,
                   categories: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Feature matrix for ``feature_set``: categoricals as pandas ``category`` with fixed levels
    (pass the training levels via ``categories`` so train and test codes agree), booleans as 0/1."""
    cols = list(FEATURE_SETS[feature_set])
    missing = set(cols) - set(meta.columns)
    if missing:
        raise KeyError(f"metadata lacks {sorted(missing)}")
    X = meta[cols].copy()
    for c in cols:
        if c in CATEGORICAL:
            levels = (categories or {}).get(c, sorted(X[c].astype(str).unique()))
            X[c] = pd.Categorical(X[c].astype(str), categories=levels)
        elif X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X


def category_levels(meta: pd.DataFrame) -> dict[str, list[str]]:
    """Training-set levels of every categorical column present in ``meta``."""
    return {c: sorted(meta[c].astype(str).unique()) for c in CATEGORICAL if c in meta}
