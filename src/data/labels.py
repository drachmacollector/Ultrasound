"""
src/data/labels.py

Label taxonomies shared by every analysis.

* Fine taxonomy: the 8 model outputs, ``CANONICAL_CLASSES`` (order = logit index;
  single source of truth stays in ``src.data.dataset``).
* Collapsed taxonomy: ``COLLAPSED`` = Head / Abdomen / Femur / Thorax / Cervix / Other.
  The three trans-axial brain planes collapse to Head; external datasets are
  labelled in this taxonomy.

Collapsed probabilities are sums of member-class probabilities
(``probs @ COLLAPSE_MATRIX``).
"""
from __future__ import annotations

import numpy as np

from src.data.dataset import CANONICAL_CLASSES

COLLAPSED: list[str] = ["Head", "Abdomen", "Femur", "Thorax", "Cervix", "Other"]
COLLAPSED_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(COLLAPSED)}

FINE_TO_COLLAPSED: dict[str, str] = {
    "Brain_Trans_cerebellum": "Head",
    "Brain_Trans_thalamic": "Head",
    "Brain_Trans_ventricular": "Head",
    "Fetal_abdomen": "Abdomen",
    "Fetal_femur": "Femur",
    "Fetal_thorax": "Thorax",
    "Maternal_cervix": "Cervix",
    "Other": "Other",
}
assert set(FINE_TO_COLLAPSED) == set(CANONICAL_CLASSES)

#: (8, 6) 0/1 matrix; row i = fine class i, column j = collapsed class j.
COLLAPSE_MATRIX: np.ndarray = np.zeros((len(CANONICAL_CLASSES), len(COLLAPSED)), dtype=np.float32)
for _i, _fine in enumerate(CANONICAL_CLASSES):
    COLLAPSE_MATRIX[_i, COLLAPSED_TO_IDX[FINE_TO_COLLAPSED[_fine]]] = 1.0

#: Fine-class indices belonging to each collapsed class.
COLLAPSED_MEMBERS: dict[str, list[int]] = {
    c: [i for i, f in enumerate(CANONICAL_CLASSES) if FINE_TO_COLLAPSED[f] == c] for c in COLLAPSED
}


def collapse_label(label: str) -> str:
    """Map a fine label to the collapsed taxonomy (collapsed labels pass through)."""
    if label in COLLAPSED_TO_IDX:
        return label
    try:
        return FINE_TO_COLLAPSED[label]
    except KeyError as exc:
        raise ValueError(f"unknown label {label!r}") from exc


def collapse_probs(p8: np.ndarray) -> np.ndarray:
    """(N, 8) fine probabilities -> (N, 6) collapsed probabilities (row sums preserved)."""
    p8 = np.asarray(p8, dtype=np.float64)
    if p8.ndim != 2 or p8.shape[1] != len(CANONICAL_CLASSES):
        raise ValueError(f"expected (N, {len(CANONICAL_CLASSES)}), got {p8.shape}")
    return p8 @ COLLAPSE_MATRIX.astype(np.float64)
