"""
scripts/research/r5_fm_linear_probe.py  (R5.8: foundation-model linear probe; secondary, exploratory)

A frozen FetalCLIP image encoder (licence CC BY-NC 4.0; see docs/research/DATA_CARDS.md) is probed
with a multinomial logistic regression on fpdb_train features. The "B0 vs AFP input" contrast feeds the
two arms' prepared images into the same encoder:

  resize                    full frame, squashed to 224 x 224 (the B0 geometry)
  sector_letterbox_inpaint  sector crop, markers inpainted, letterboxed to 224 x 224 (the AFP geometry)

Each prepared 224 x 224 image then goes through FetalCLIP's own eval transform (its resize to 224 is a
no-op on a 224 x 224 input; its normalisation applies). C is chosen on fpdb_val from {0.01, 0.1, 1, 10}
by 8-way macro-F1 (val only); the probe is fitted on fpdb_train only.

  --extract  features -> results/preds/fm_fetalclip_<mode>/<dataset>.features.npy (+ .features_ids.txt)
  --probe    C selection on val; per-image predictions in the standard predict format
             -> results/preds/fm_fetalclip_<mode>/<dataset>.parquet (val, test, externals) + probe.meta.json
  --score    -> results/json/R5_fm_probe.json, results/tables/R5_fm_probe.md

Model files (not in git; download after Gate G4, see docs/research/REPRODUCE.md):
  data/external_models/fetalclip/FetalCLIP_config.json   (github.com/BioMedIA-MBZUAI/FetalCLIP)
  data/external_models/fetalclip/FetalCLIP_weights.pt    (huggingface.co/numansaeed/fetalclip-model)
Needs ``open_clip_torch``. Refuses to run before Gate G4 (same check as r5_run_matrix.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data.dataset import CANONICAL_CLASSES  # noqa: E402
from src.eval.datasets import EvalSet, load_eval_set  # noqa: E402
from src.preprocess.pipeline import load_prepared, to_input_geometry  # noqa: E402

MODEL_DIR = Path("data/external_models/fetalclip")
CONFIG = MODEL_DIR / "FetalCLIP_config.json"
WEIGHTS = MODEL_DIR / "FetalCLIP_weights.pt"
MODES = ("resize", "sector_letterbox_inpaint")
DATASETS = ("fpdb_train", "fpdb_val", "fpdb_test", "hc18", "ucl", "african")
SCORED = ("fpdb_val", "fpdb_test", "hc18", "ucl", "african")
EXTERNAL = ("hc18", "ucl", "african")
POOL_CLASSES = ("Head", "Abdomen", "Femur", "Thorax")
IMG_SIZE = 224
C_GRID = (0.01, 0.1, 1.0, 10.0)
REFERENCE_RUN = "legacy_convnext_s42"
PREDS = Path("results/preds")
OUT_JSON = Path("results/json/R5_fm_probe.json")
OUT_MD = Path("results/tables/R5_fm_probe.md")


def run_id(mode: str) -> str:
    return f"fm_fetalclip_{mode}"


# ----------------------------------------------------------------------------- inputs
def fm_input(gray: np.ndarray, mode: str, img_size: int = IMG_SIZE) -> np.ndarray:
    """Prepared greyscale image -> the arm's 224 x 224 geometry -> 3-channel uint8 (H, W, 3)."""
    sq = to_input_geometry(gray, img_size, mode)
    if sq.shape[:2] != (img_size, img_size):
        raise ValueError(f"expected {img_size}x{img_size}, got {sq.shape}")
    return np.repeat(sq[..., None], 3, axis=2)


def load_fetalclip(device):
    """(model, eval transform) following the FetalCLIP README (open_clip config registration)."""
    import open_clip

    for p in (CONFIG, WEIGHTS):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; see docs/research/REPRODUCE.md (R5.8)")
    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = json.loads(CONFIG.read_text(encoding="utf-8"))
    model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained=str(WEIGHTS))
    return model.eval().to(device), preprocess


def extract(mode: str, dataset: str, model, preprocess, device, batch_size: int = 32,
            workers: int = 8, out_root: Path = PREDS) -> Path:
    """Frozen image embeddings (fp16 autocast) for every image of ``dataset``; row order = eval-set frame."""
    import torch
    from PIL import Image

    es = load_eval_set(dataset)
    f = es.frame.reset_index(drop=True)

    def one(i: int):
        row = f.iloc[i]
        gray, _ = load_prepared(mode, IMG_SIZE, row["dataset"], row["image_id"], row["image_path"])
        return preprocess(Image.fromarray(fm_input(gray, mode)))

    feats = []
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex, torch.no_grad():
        for s in range(0, len(f), batch_size):
            x = torch.stack(list(ex.map(one, range(s, min(s + batch_size, len(f)))))).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                z = model.encode_image(x)
            feats.append(z.float().cpu().numpy().astype(np.float16))
    out = out_root / run_id(mode)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{dataset}.features.npy", np.concatenate(feats))
    (out / f"{dataset}.features_ids.txt").write_text("\n".join(f["image_id"].astype(str)) + "\n", encoding="utf-8")
    print(f"  {run_id(mode)}/{dataset}: {len(f)} images in {time.time() - t0:.0f}s", flush=True)
    return out / f"{dataset}.features.npy"


# ----------------------------------------------------------------------------- probe
def probe_pipeline(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def select_and_fit(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
                   grid=C_GRID, n_classes: int = 8):
    """Fit on train for each C; pick the best 8-way val macro-F1 (ties -> smaller C). Returns (model, C, scores)."""
    from sklearn.metrics import f1_score

    if set(np.unique(ytr)) != set(range(n_classes)):
        raise ValueError("every class must be present in the training features")
    scores, models = {}, {}
    for C in grid:
        m = probe_pipeline(C).fit(Xtr, ytr)
        scores[C] = float(f1_score(yva, m.predict(Xva), average="macro", labels=list(range(n_classes)),
                                   zero_division=0))
        models[C] = m
    best = max(grid, key=lambda c: (scores[c], -c))
    return models[best], best, scores


def _features(mode: str, dataset: str, out_root: Path = PREDS) -> tuple[EvalSet, np.ndarray]:
    from src.audit.probes import load_features

    es = load_eval_set(dataset)
    feats = load_features(out_root / run_id(mode), dataset)
    ids = es.frame["image_id"].astype(str).to_numpy()
    if not np.array_equal(feats.index.to_numpy(), ids):
        raise ValueError(f"{run_id(mode)}/{dataset}: feature ids do not match the eval-set frame order")
    return es, feats.to_numpy(np.float64)


def probe(mode: str, out_root: Path = PREDS) -> dict:
    from src.eval.predict import build_frame, git_hash

    es_tr, Xtr = _features(mode, "fpdb_train", out_root)
    es_va, Xva = _features(mode, "fpdb_val", out_root)
    cls = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
    ytr = es_tr.frame["true_label"].astype(str).map(cls).to_numpy()
    yva = es_va.frame["true_label"].astype(str).map(cls).to_numpy()
    model, C, scores = select_and_fit(Xtr, ytr, Xva, yva)
    rid, gh = run_id(mode), git_hash()
    for ds in SCORED:
        es, X = (es_va, Xva) if ds == "fpdb_val" else _features(mode, ds, out_root)
        f = build_frame(es, model.decision_function(X))
        f["run_id"], f["git_hash"], f["preprocess_mode"] = rid, gh, mode
        f.to_parquet(out_root / rid / f"{ds}.parquet", index=False)
    meta = {"run_id": rid, "mode": mode, "C": C, "val_macro_f1_by_C": {str(k): v for k, v in scores.items()},
            "n_train": len(Xtr), "n_val": len(Xva), "git_hash": gh,
            "weights": str(WEIGHTS), "encoder": "FetalCLIP image encoder (frozen)"}
    (out_root / rid / "probe.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"  {rid}: C={C} val macro-F1 {scores[C]:.3f}", flush=True)
    return meta


# ----------------------------------------------------------------------------- scoring
def load(rid: str, dataset: str, root: Path = PREDS) -> pd.DataFrame:
    d = pd.read_parquet(root / rid / f"{dataset}.parquet")
    for c in d.select_dtypes("category").columns:
        d[c] = d[c].astype(str)
    return d


def score(root: Path = PREDS, reference: str | None = REFERENCE_RUN) -> dict:
    """Per mode: 8-way macro-F1 (FPDB test), full-argmax balanced accuracy per external set and pooled;
    paired letterbox-inpaint − resize; the legacy convnext as an unpaired reference row."""
    from src.eval.scoring import bal_acc_collapsed, estimate_json, macro_f1_8
    from src.eval.stats import cluster_bootstrap, paired_cluster_bootstrap

    def metrics_for(rid: str) -> tuple[dict, dict]:
        frames = {ds: load(rid, ds, root) for ds in ("fpdb_test", *EXTERNAL)}
        frames["pooled"] = pd.concat([frames[d] for d in EXTERNAL], ignore_index=True)
        ms = {"fpdb_test": macro_f1_8(), "pooled": bal_acc_collapsed("pred_full", POOL_CLASSES)}
        for ds in EXTERNAL:
            ms[ds] = bal_acc_collapsed("pred_full", [c for c in POOL_CLASSES if c in set(frames[ds]["true_collapsed"])])
        return frames, ms

    res: dict = {"metric": {"fpdb_test": "8-way macro-F1",
                            "externals": "full-argmax balanced accuracy over present classes of {Head, Abdomen, Femur, Thorax}"},
                 "by_mode": {}, "paired_letterbox_minus_resize": {}, "reference": {}}
    frames, ms = {}, {}
    for mode in MODES:
        rid = run_id(mode)
        frames[mode], ms = metrics_for(rid)          # same images, so the same metric definitions per mode
        res["by_mode"][mode] = {k: estimate_json(cluster_bootstrap(frames[mode][k], ms[k])) for k in ms}
        res["by_mode"][mode]["probe"] = json.loads((root / rid / "probe.meta.json").read_text(encoding="utf-8"))
    for k, m in ms.items():
        res["paired_letterbox_minus_resize"][k] = estimate_json(
            paired_cluster_bootstrap(frames["resize"][k], frames["sector_letterbox_inpaint"][k], m))
    if reference and (root / reference / "fpdb_test.parquet").exists():
        fr, ms = metrics_for(reference)
        res["reference"] = {"run_id": reference,
                            **{k: estimate_json(cluster_bootstrap(fr[k], ms[k])) for k in ms}}
    return res


def _ci(e: dict) -> str:
    if "delta" in e:
        return f"{e['delta']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]"
    return f"{e['point']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]"


def markdown(res: dict) -> str:
    cols = ("fpdb_test", "hc18", "ucl", "african", "pooled")
    L = ["# R5.8 Foundation-model linear probe (FetalCLIP, frozen; secondary, exploratory)", "",
         f"FPDB test: {res['metric']['fpdb_test']}. Externals: {res['metric']['externals']}. "
         "95% patient-clustered bootstrap CIs (B = 2000).", "",
         "| input | C (val) | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for mode, r in res["by_mode"].items():
        L.append(f"| FetalCLIP + {mode} | {r['probe']['C']} | " + " | ".join(_ci(r[c]) for c in cols) + " |")
    L.append("| Δ letterbox-inpaint − resize (paired) | – | "
             + " | ".join(_ci(res["paired_letterbox_minus_resize"][c]) for c in cols) + " |")
    if res.get("reference"):
        L.append(f"| reference: `{res['reference']['run_id']}` (fine-tuned, unpaired) | – | "
                 + " | ".join(_ci(res["reference"][c]) for c in cols) + " |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    from r5_run_matrix import check_g4  # same directory; same gate

    check_g4()
    if a.extract:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, preprocess = load_fetalclip(device)
        for mode in MODES:
            for ds in DATASETS:
                if not (PREDS / run_id(mode) / f"{ds}.features.npy").exists():
                    extract(mode, ds, model, preprocess, device, a.batch_size)
    if a.probe:
        for mode in MODES:
            probe(mode)
    if a.score:
        res = score()
        OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
        OUT_MD.write_text(markdown(res), encoding="utf-8")
        print(f"wrote {OUT_JSON}, {OUT_MD}")


if __name__ == "__main__":
    main()
