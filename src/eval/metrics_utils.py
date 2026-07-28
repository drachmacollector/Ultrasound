"""
src/eval/metrics_utils.py

Shared evaluation utilities used by both src/train/train.py and
src/eval/evaluate_test.py.  Factored here to avoid circular imports and
code duplication.

Functions:
    save_confusion_matrix()  — plots and saves a normalized confusion matrix PNG.
    format_classification_report()  — returns a compact single-line-per-class
        string for the aggregate summary file.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from src.data.dataset import CANONICAL_CLASSES, NUM_CLASSES

log = logging.getLogger(__name__)


def save_confusion_matrix(
    all_targets: list[int],
    all_preds: list[int],
    save_path: Path,
    epoch: int,
) -> None:
    """Save a row-normalised confusion matrix PNG using matplotlib/seaborn.

    Args:
        all_targets: Ground-truth label indices.
        all_preds:   Predicted label indices.
        save_path:   Destination .png path (parent dirs are created).
        epoch:       Epoch number used in the title string.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        short_labels = [
            c.replace("Brain_Trans_", "BT_")
             .replace("Fetal_", "F_")
             .replace("Maternal_", "M_")
            for c in CANONICAL_CLASSES
        ]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            xticklabels=short_labels,
            yticklabels=short_labels,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(f"Normalised Confusion Matrix — epoch {epoch}")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        log.info("Confusion matrix saved → %s", save_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not save confusion matrix: %s", exc)
