"""src/eval/predict.py: parquet schema on a tiny fake checkpoint and a synthetic manifest."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import CANONICAL_CLASSES
from src.data.labels import COLLAPSED, FINE_TO_COLLAPSED
from src.eval.datasets import EvalSet
from src.eval.predict import build_frame, predict
from src.models.backbone import build_model

CFG = {"backbone": "resnet10t", "num_classes": 8, "image_size": 64,
       "normalize_mean": [0.5, 0.5, 0.5], "normalize_std": [0.5, 0.5, 0.5],
       "train_csv": "data/splits/train.csv", "val_csv": "data/splits/val.csv"}
LABELS = ["Head", "Head", "Abdomen", "Femur", "Head", "Femur"]


@pytest.fixture()
def fake(tmp_path: Path) -> tuple[Path, Path]:
    torch.manual_seed(0)
    model = build_model("resnet10t", num_classes=8, pretrained=False)
    ckpt = tmp_path / "best.pt"
    torch.save({"config": CFG, "model_state_dict": model.state_dict()}, ckpt)
    rng = np.random.default_rng(0)
    rows = []
    for i, lab in enumerate(LABELS):
        p = tmp_path / "img" / f"im{i}.png"
        p.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(p), rng.integers(0, 255, (80 + 5 * i, 100), dtype=np.uint8))
        rows.append({"image_path": p.as_posix(), "image_id": f"toy/im{i}", "patient_id": f"toy_{i // 2}",
                     "plane_label_collapsed": lab})
    man = tmp_path / "toy_manifest.csv"
    pd.DataFrame(rows).to_csv(man, index=False)
    return ckpt, man


def test_parquet_schema(fake: tuple[Path, Path], tmp_path: Path) -> None:
    ckpt, man = fake
    pq = predict(ckpt, "toy", tmp_path / "preds" / "run0", manifest=man, num_workers=0,
                 batch_size=4, save_features=True)
    f = pd.read_parquet(pq)
    assert len(f) == 6 and f["image_id"].is_unique
    for i in range(8):
        assert f[f"prob_{i}"].dtype == np.float32 and f[f"logit_{i}"].dtype == np.float32
    pc = f[[f"pc_{c}" for c in COLLAPSED]].to_numpy(np.float64)
    assert np.allclose(pc.sum(axis=1), 1.0, atol=1e-5)
    assert set(f["pred_restricted"].astype(str)) <= {"Head", "Abdomen", "Femur"}
    assert (f["pred_full"].astype(str) == f["pred_fine"].astype(str).map(FINE_TO_COLLAPSED)).all()
    for c in ("dataset", "pred_fine", "pred_full", "true_collapsed"):
        assert isinstance(f[c].dtype, pd.CategoricalDtype)
    assert f["correct_fine"].isna().all()                       # externals have no fine label
    assert f["correct_full"].dtype == bool
    assert (f["run_id"] == "run0").all() and f["ckpt_sha256"].str.len().eq(64).all()
    feats = np.load(pq.with_name("toy.features.npy"))
    assert feats.shape == (6, 512) and feats.dtype == np.float16
    ids = pq.with_name("toy.features_ids.txt").read_text(encoding="utf-8").split()
    assert ids == f["image_id"].tolist()
    meta = json.loads(pq.with_name("toy.meta.json").read_text(encoding="utf-8"))
    assert meta["n_images"] == 6 and meta["present_classes"] == ["Head", "Abdomen", "Femur"]


def test_features_pass_gives_same_logits(fake: tuple[Path, Path], tmp_path: Path) -> None:
    ckpt, man = fake
    a = pd.read_parquet(predict(ckpt, "toy", tmp_path / "a", manifest=man, num_workers=0))
    b = pd.read_parquet(predict(ckpt, "toy", tmp_path / "b", manifest=man, num_workers=0, save_features=True))
    cols = [f"logit_{i}" for i in range(8)]
    assert np.allclose(a[cols].to_numpy(), b[cols].to_numpy(), atol=1e-4)


def test_perturb_needs_variant_and_is_applied(fake: tuple[Path, Path], tmp_path: Path) -> None:
    ckpt, man = fake
    with pytest.raises(ValueError, match="variant"):
        predict(ckpt, "toy", tmp_path / "x", manifest=man, perturb=lambda g, m: g)
    seen = []

    def blackout(gray: np.ndarray, meta: dict) -> np.ndarray:
        seen.append(meta["image_id"])
        return np.zeros_like(gray)

    pq = predict(ckpt, "toy", tmp_path / "x", manifest=man, perturb=blackout, variant="perturb-black",
                 num_workers=0)
    assert pq.name == "toy__perturb-black.parquet"
    f = pd.read_parquet(pq)
    assert sorted(seen) == sorted(f["image_id"]) and (f["variant"] == "perturb-black").all()
    logits = f[[f"logit_{i}" for i in range(8)]].to_numpy()
    assert np.allclose(logits, logits[0], atol=1e-4)            # identical black inputs


def test_fpdb_train_requires_features(fake: tuple[Path, Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fpdb_train"):
        predict(fake[0], "fpdb_train", tmp_path / "y")


def test_build_frame_fpdb_columns() -> None:
    frame = pd.DataFrame({
        "image_path": ["a.png", "b.png"], "image_id": ["a", "b"], "patient_id": ["fpdb_1", "fpdb_2"],
        "dataset": "fpdb", "source_machine": "Aloka", "operator": "Op. 3", "centre": "x",
        "true_label": ["Other", "Fetal_femur"], "true_collapsed": ["Other", "Femur"]})
    es = EvalSet("fpdb_test", frame, tuple(COLLAPSED), True, "x")
    logits = np.full((2, 8), -5.0)
    logits[0, CANONICAL_CLASSES.index("Other")] = 5.0           # correct
    logits[1, CANONICAL_CLASSES.index("Brain_Trans_thalamic")] = 5.0   # wrong
    f = build_frame(es, logits)
    assert f["correct_fine"].tolist() == [True, False]
    assert f["pred_full"].astype(str).tolist() == ["Other", "Head"]
    assert f["correct_full"].tolist() == [True, False]
    assert "image_path" not in f
