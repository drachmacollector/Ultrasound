"""
src/eval/predict.py

Generic predictor: one checkpoint x one dataset -> one per-image parquet file.

    results/preds/<run_id>/<dataset>[__<variant>].parquet   (+ .meta.json, optional .features.npy)

Model, input size, normalisation and preprocess mode come from the checkpoint
(``src.models.loading``). ``perturb(gray_uint8, meta) -> gray_uint8`` is applied to the
native-resolution greyscale image before the geometric transform; ``meta`` carries
``image_id``, ``dataset`` and the R1 masks (``masks`` is None if no mask file exists).
Perturbed or re-preprocessed predictions must name a ``variant`` so files never collide.

Scoring columns (see plans/R2_EVALUATION_CORE.md §R2.2):
  pred_fine          8-way argmax
  pred_full          collapse(pred_fine)                       <- primary, matches legacy scoring
  pred_full_summed   argmax over the 6 summed collapsed probabilities
  pred_restricted    argmax of collapsed probabilities over the dataset's present classes

CLI:
  python -m src.eval.predict --ckpt checkpoints/convnext_tiny/best.pt --dataset fpdb_test --run-id legacy_convnext_s42
  python -m src.eval.predict --ckpt ... --manifest my.csv --dataset-name mysite --run-id ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader, Dataset

from src.audit.metadata import load_masks, mask_path
from src.data.dataset import CANONICAL_CLASSES
from src.data.labels import COLLAPSED, FINE_TO_COLLAPSED, collapse_probs
from src.data.transforms import gray_to_rgb, load_grayscale
from src.eval.datasets import EvalSet, load_eval_set
from src.models.loading import REPO_ROOT, load_model

PREDS_ROOT = Path("results/preds")
Perturb = Callable[[np.ndarray, dict], np.ndarray]

CATEGORY_COLS = ["dataset", "source_machine", "operator", "centre", "true_label", "true_collapsed",
                 "pred_fine", "pred_full", "pred_full_summed", "pred_restricted"]
PROVENANCE_COLS = ["run_id", "ckpt_sha256", "git_hash", "preprocess_mode", "variant"]


class _FrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: A.Compose, perturb: Perturb | None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.perturb = perturb

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> torch.Tensor:
        row = self.frame.iloc[i]
        gray = load_grayscale(row["image_path"])
        if self.perturb is not None:
            mp = mask_path(row["image_id"], row["dataset"])
            meta = {"image_id": row["image_id"], "dataset": row["dataset"],
                    "masks": load_masks(mp) if mp.exists() else None}
            out = self.perturb(gray, meta)
            if out.shape != gray.shape or out.dtype != np.uint8:
                raise ValueError(f"perturb must return {gray.shape} uint8, got {out.shape} {out.dtype}")
            gray = out
        return self.transform(image=gray_to_rgb(gray))["image"]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_hash() -> str:
    """HEAD commit, suffixed ``-dirty`` when tracked files under src/ have uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{head}-dirty" if dirty else head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def output_stem(dataset: str, variant: str | None) -> str:
    return f"{dataset}__{variant}" if variant else dataset


def build_frame(es: EvalSet, logits: np.ndarray) -> pd.DataFrame:
    """Per-image prediction table from raw logits (N, 8). Pure function: no model, no I/O."""
    logits = logits.astype(np.float64)
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=1, keepdims=True)
    pc = collapse_probs(probs)

    fine_names = np.array(CANONICAL_CLASSES)
    coll_names = np.array(COLLAPSED)
    pred_fine = fine_names[probs.argmax(axis=1)]
    present_idx = np.array([COLLAPSED.index(c) for c in es.present])
    pred_restricted = coll_names[present_idx[pc[:, present_idx].argmax(axis=1)]]

    f = es.frame.reset_index(drop=True).drop(columns=["image_path"]).copy()
    for i in range(probs.shape[1]):
        f[f"prob_{i}"] = probs[:, i].astype(np.float32)
    for i in range(logits.shape[1]):
        f[f"logit_{i}"] = logits[:, i].astype(np.float32)
    for j, c in enumerate(COLLAPSED):
        f[f"pc_{c}"] = pc[:, j].astype(np.float32)
    f["pred_fine"] = pred_fine
    f["pred_full"] = [FINE_TO_COLLAPSED[p] for p in pred_fine]
    f["pred_full_summed"] = coll_names[pc.argmax(axis=1)]
    f["pred_restricted"] = pred_restricted
    if es.is_fpdb:
        f["correct_fine"] = pd.array(f["pred_fine"] == f["true_label"], dtype="boolean")
    else:
        f["correct_fine"] = pd.array([pd.NA] * len(f), dtype="boolean")
    f["correct_full"] = f["pred_full"] == f["true_collapsed"]
    f["correct_restricted"] = f["pred_restricted"] == f["true_collapsed"]
    f["conf_max8"] = probs.max(axis=1).astype(np.float32)
    f["conf_full_summed"] = pc.max(axis=1).astype(np.float32)
    for c in CATEGORY_COLS:
        f[c] = f[c].astype("category")
    for c in ("image_id", "patient_id"):
        f[c] = f[c].astype("string")
    return f


@torch.no_grad()
def _infer(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool,
           save_features: bool) -> tuple[np.ndarray, np.ndarray | None]:
    logits_all, feats_all = [], []
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            if save_features:
                fmap = model.forward_features(x)
                pre = model.forward_head(fmap, pre_logits=True)
                logits = model.forward_head(fmap)
                feats_all.append(pre.float().cpu().numpy().astype(np.float16))
            else:
                logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
    feats = np.concatenate(feats_all) if save_features else None
    return np.concatenate(logits_all), feats


def predict(ckpt: str | Path, dataset: str, out_dir: str | Path, *, run_id: str | None = None,
            manifest: str | Path | None = None, preprocess_override: str | None = None,
            perturb: Perturb | None = None, variant: str | None = None, batch_size: int = 64,
            num_workers: int = 4, save_features: bool = False, amp: bool = False) -> Path:
    """Run one checkpoint on one dataset; return the parquet path."""
    if dataset == "fpdb_train" and not save_features:
        raise ValueError("fpdb_train may only be predicted with save_features=True (R5 probes)")
    if (perturb is not None or preprocess_override is not None) and not variant:
        raise ValueError("perturbed / re-preprocessed predictions need a variant suffix")
    ckpt, out_dir = Path(ckpt), Path(out_dir)
    run_id = run_id or out_dir.name
    torch.backends.cudnn.benchmark = False

    t0 = time.perf_counter()
    lm = load_model(ckpt)
    es = load_eval_set(dataset, manifest)
    mode = preprocess_override or lm.preprocess_mode
    ds = _FrameDataset(es.frame, lm.eval_transform(mode), perturb)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=lm.device.type == "cuda", persistent_workers=False)
    t1 = time.perf_counter()
    logits, feats = _infer(lm.model, loader, lm.device, amp, save_features)
    t2 = time.perf_counter()

    f = build_frame(es, logits)
    prov = {"run_id": run_id, "ckpt_sha256": sha256_file(ckpt), "git_hash": git_hash(),
            "preprocess_mode": mode, "variant": variant or ""}
    for k, v in prov.items():
        f[k] = pd.Series([v] * len(f), dtype="string")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(es.name, variant)
    pq = out_dir / f"{stem}.parquet"
    f.to_parquet(pq, index=False)
    if feats is not None:
        np.save(out_dir / f"{stem}.features.npy", feats)
        (out_dir / f"{stem}.features_ids.txt").write_text("\n".join(f["image_id"].astype(str)) + "\n",
                                                          encoding="utf-8")
    meta = {
        **prov, "ckpt": ckpt.as_posix(), "dataset": es.name, "source": es.source,
        "present_classes": list(es.present), "n_images": len(f), "n_patients": int(f["patient_id"].nunique()),
        "acc_full": float(f["correct_full"].mean()), "amp": amp, "batch_size": batch_size,
        "features_dim": None if feats is None else int(feats.shape[1]),
        "timings_s": {"load": round(t1 - t0, 2), "inference": round(t2 - t1, 2)},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in lm.config.items()},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
                     "albumentations": A.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "cuda": torch.version.cuda,
                     "device": torch.cuda.get_device_name(0) if lm.device.type == "cuda" else "cpu"},
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return pq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", help="fpdb_val | fpdb_test | hc18 | ucl | african (| fpdb_train with --features)")
    ap.add_argument("--manifest", help="ad-hoc external CSV (image_path, image_id, patient_id, plane_label_collapsed)")
    ap.add_argument("--dataset-name", help="name for --manifest outputs")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-root", default=str(PREDS_ROOT))
    ap.add_argument("--preprocess", default=None, help="override the checkpoint's preprocess mode")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    a = ap.parse_args()
    if bool(a.manifest) == bool(a.dataset):
        ap.error("give exactly one of --dataset or --manifest")
    if a.manifest and not a.dataset_name:
        ap.error("--manifest needs --dataset-name")
    pq = predict(a.ckpt, a.dataset_name if a.manifest else a.dataset, Path(a.out_root) / a.run_id,
                 run_id=a.run_id, manifest=a.manifest, preprocess_override=a.preprocess, variant=a.variant,
                 batch_size=a.batch_size, num_workers=a.num_workers, save_features=a.features, amp=a.amp)
    print(pq)


if __name__ == "__main__":
    main()
