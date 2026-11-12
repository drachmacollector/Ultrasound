"""
src/audit/probes.py

Linear probes for feature decodability (R3 E3, P1b).

* ``extract_imagenet_features``: penultimate features of the ImageNet-pretrained backbone
  (timm ``pretrained=True, num_classes=0``) with the *same* eval transform as the fine-tuned
  checkpoint, written next to the fine-tuned features:
  ``results/preds/imagenet_<short>/fpdb_test.features.npy`` (+ ``.features_ids.txt``).
* ``probe_oof``: patient-grouped nested CV. Outer ``StratifiedGroupKFold(5, shuffle, seed 0)``;
  inner ``GroupKFold(3)`` picks C from ``C_GRID`` by balanced accuracy; the model is standardisation
  + multinomial logistic regression with balanced class weights. Returns out-of-fold predictions,
  so every image is predicted by a probe that never saw its patient.
* ``within_class_machine_targets`` / ``within_class_mean``: the P1b/P4b shortcut index, i.e. the
  machine decoded within one true class (classes in ``WITHIN_CLASSES``, machines with >= ``MIN_CELL``
  images in that class), averaged over classes. Shared by R3 E3 and R5 evaluation.
* Speed: run probes with single-threaded BLAS (``OMP_NUM_THREADS=1`` etc.) and parallelise over
  probes; multi-threaded BLAS oversubscribes the CPU and makes each fit 10-25x slower.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (0.01, 0.1, 1.0, 10.0)
WITHIN_CLASSES = ("Fetal_abdomen", "Fetal_femur", "Brain_Trans_thalamic", "Other")
MIN_CELL = 30
OUTER_FOLDS = 5
INNER_FOLDS = 3


def _probe(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def _select_C(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    best, best_c = -1.0, C_GRID[0]
    for C in C_GRID:
        scores = []
        for tr, va in GroupKFold(INNER_FOLDS).split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            m = _probe(C).fit(X[tr], y[tr])
            scores.append(balanced_accuracy_score(y[va], m.predict(X[va])))
        s = float(np.mean(scores)) if scores else -1.0
        if s > best:
            best, best_c = s, C
    return best_c


def probe_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold predictions of a nested-CV linear probe; also returns the C chosen per outer fold."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    oof = np.empty(len(y), dtype=y.dtype)
    chosen = []
    outer = StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y, groups):
        C = _select_C(X[tr], y[tr], groups[tr])
        chosen.append(C)
        oof[te] = _probe(C).fit(X[tr], y[tr]).predict(X[te])
    return oof, chosen


def load_features(run_dir: Path, stem: str = "fpdb_test") -> pd.DataFrame:
    """Features as a frame indexed by image_id (row order of the .features_ids.txt file)."""
    X = np.load(run_dir / f"{stem}.features.npy").astype(np.float32)
    ids = (run_dir / f"{stem}.features_ids.txt").read_text(encoding="utf-8").split()
    if len(ids) != len(X):
        raise ValueError(f"{run_dir}: {len(ids)} ids vs {len(X)} feature rows")
    return pd.DataFrame(X, index=pd.Index(ids, name="image_id"))


def extract_imagenet_features(ckpt: Path, out_dir: Path, dataset: str = "fpdb_test", batch_size: int = 64,
                              num_workers: int = 4) -> Path:
    """Penultimate features of the ImageNet-pretrained version of ``ckpt``'s backbone."""
    import timm
    import torch
    from torch.utils.data import DataLoader

    from src.eval.datasets import load_eval_set
    from src.eval.predict import FrameDataset
    from src.models.loading import load_model

    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # weights are in the local HF cache; no network calls
    lm = load_model(ckpt)                       # only for the backbone name and the eval transform
    device = lm.device
    model = timm.create_model(lm.backbone_name, pretrained=True, num_classes=0).to(device).eval()
    es = load_eval_set(dataset)
    loader = DataLoader(FrameDataset(es.frame, lm.eval_transform()), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    feats = []
    with torch.no_grad():
        for x in loader:
            feats.append(model(x.to(device)).float().cpu().numpy().astype(np.float16))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{dataset}.features.npy", np.concatenate(feats))
    (out_dir / f"{dataset}.features_ids.txt").write_text("\n".join(es.frame["image_id"]) + "\n", encoding="utf-8")
    return out_dir / f"{dataset}.features.npy"


def within_class_machine_targets(meta: pd.DataFrame, classes=WITHIN_CLASSES,
                                 min_cell: int = MIN_CELL) -> dict[str, pd.Series]:
    """class -> machine labels of that class's images (machines with >= min_cell images kept).

    ``meta`` is indexed by image_id with columns ``true_label`` and ``source_machine``.
    """
    out = {}
    for c in classes:
        g = meta[meta["true_label"] == c]
        keep = g["source_machine"].value_counts()[lambda v: v >= min_cell].index
        out[c] = g.loc[g["source_machine"].isin(keep), "source_machine"]
    return out


def within_class_mean(df: pd.DataFrame) -> float:
    """Mean over classes (column ``cls``) of the out-of-fold balanced accuracy of ``oof`` vs ``y``."""
    from src.eval import metrics as M

    vals = []
    for _, g in df.groupby("cls"):
        labels = sorted(set(g["y"]) | set(g["oof"]))
        idx = {c: i for i, c in enumerate(labels)}
        vals.append(M.balanced_accuracy(M.confusion(g["y"].map(idx), g["oof"].map(idx), len(labels))))
    return float(np.mean(vals))
