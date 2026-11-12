"""src/audit/probes.py: patient-grouped nested-CV probe."""
from __future__ import annotations

import numpy as np

from src.audit.probes import C_GRID, OUTER_FOLDS, probe_oof


def data(n: int = 300, d: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.arange(n) // 3                                  # 100 patients, 3 images each
    y = np.array(["Aloka", "Voluson E6"])[(groups % 2)]          # label constant within patient
    X = rng.normal(size=(n, d)) + 3.0 * (y == "Aloka")[:, None] * np.eye(1, d)
    return X, y, groups


def test_oof_shapes_labels_and_accuracy() -> None:
    X, y, g = data()
    oof, Cs = probe_oof(X, y, g)
    assert oof.shape == y.shape and set(oof) <= set(y)
    assert len(Cs) == OUTER_FOLDS and set(Cs) <= set(C_GRID)
    assert (oof == y).mean() > 0.9


def test_deterministic_and_chance_on_noise() -> None:
    X, y, g = data()
    assert np.array_equal(probe_oof(X, y, g)[0], probe_oof(X, y, g)[0])
    noise = np.random.default_rng(1).normal(size=X.shape)       # no signal -> about chance out of fold
    oof, _ = probe_oof(noise, y, g)
    assert 0.3 < (oof == y).mean() < 0.7
