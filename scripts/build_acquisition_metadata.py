"""
scripts/build_acquisition_metadata.py  (R1.6)

Label-free acquisition metadata + masks for every FPDB image (all splits) and every
external image.

Outputs
  data/processed/metadata/acquisition_metadata.parquet   (FPDB, 12,400 rows)
  data/processed/metadata/external_metadata.parquet      (HC18 + UCL + African, 1,873 rows)
  data/processed/masks/<dataset>/<image_id>.npz          (sector, sector_clean, ui, marker)
  results/json/R1_metadata_summary.json

Two passes (Gate G1 enabled the consensus prior):
  1. --no-consensus   per-image sectors only  -> scripts/build_sector_consensus.py (train-only cores)
  2. default          FPDB sectors refined with the device-geometry core; externals unchanged.
Resumable: per-chunk results cached in data/processed/metadata/_parts/ (separate per pass).

Usage:  python scripts/build_acquisition_metadata.py [--workers 6] [--no-consensus]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.audit.metadata import (  # noqa: E402
    consensus_key,
    image_features,
    load_consensus_index,
    mask_path,
    save_masks,
)
from src.data.labels import collapse_label  # noqa: E402
from src.data.paths import fpdb_image_id, fpdb_patient_id, norm_path, parse_fpdb_stem  # noqa: E402
from src.preprocess.overlays import OverlayParams  # noqa: E402

OUT = Path("data/processed/metadata")
PARTS = OUT / "_parts"
PARAMS_JSON = Path("data/processed/audit/overlay_params.json")
CHUNK = 250

EXPECTED_SPLITS = {"train": 6106, "val": 1023, "test": 5271}
EXPECTED_OTHER_SHARE = {"Aloka": 0.651, "Voluson E6": 0.136}     # train+val, 00_INDEX §3.1
EXPECTED_S10_TEST_OTHER = 0.516


def fpdb_rows() -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        d = pd.read_csv(f"data/splits/{split}.csv")
        d["split"] = split
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["image_id"] = d["image_path"].map(fpdb_image_id)
    parsed = pd.DataFrame([parse_fpdb_stem(s) for s in d["image_id"]])
    d = pd.concat([d, parsed[["plane_n", "idx", "of_k"]]], axis=1)
    d["patient_id"] = d["patient_id"].map(fpdb_patient_id)
    d["plane_collapsed"] = d["plane_label"].map(collapse_label)
    d["dataset"] = "fpdb"
    d["image_path"] = d["image_path"].map(lambda p: norm_path(p).as_posix())
    return d


def external_rows() -> pd.DataFrame:
    d = pd.read_csv("data/processed/external/external_all.csv")
    # geometry is recomputed from the decoded image; drop the manifest copies to avoid merge suffixes
    d = d.drop(columns=["width", "height", "channels"])
    return d.rename(columns={"plane_label_collapsed": "plane_collapsed", "device": "source_machine"})


def _process(args: tuple[list[dict], str, bool]) -> pd.DataFrame:
    rows, params_json, use_consensus = args
    params = OverlayParams.from_json(Path(params_json))
    consensus = load_consensus_index() if use_consensus else {}
    out = []
    for r in rows:
        raw = Path(r["image_path"]).read_bytes()
        gray = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(r["image_path"])
        cons = None
        if r["dataset"] == "fpdb":
            cons = consensus.get(consensus_key(r["source_machine"], gray.shape[1], gray.shape[0]))
        feats, masks = image_features(gray, params, cons)
        save_masks(mask_path(r["image_id"], r["dataset"]), masks)
        out.append({"image_id": r["image_id"], "md5": hashlib.md5(raw).hexdigest(), **feats})
    return pd.DataFrame(out)


def run(rows: pd.DataFrame, tag: str, workers: int, use_consensus: bool) -> pd.DataFrame:
    PARTS.mkdir(parents=True, exist_ok=True)
    tag = f"{tag}_cons" if use_consensus else tag
    recs = rows[["image_id", "image_path", "dataset", "source_machine"]].to_dict("records")
    chunks = [(i, recs[i:i + CHUNK]) for i in range(0, len(recs), CHUNK)]
    todo = [(i, c) for i, c in chunks if not (PARTS / f"{tag}_{i:06d}.parquet").exists()]
    print(f"{tag}: {len(chunks)} chunks, {len(todo)} to do")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {i: ex.submit(_process, (c, str(PARAMS_JSON), use_consensus)) for i, c in todo}
        for n, (i, f) in enumerate(futs.items(), 1):
            f.result().to_parquet(PARTS / f"{tag}_{i:06d}.parquet", index=False)
            print(f"  {tag}: {n}/{len(todo)} chunks done", flush=True)
    feats = pd.concat([pd.read_parquet(PARTS / f"{tag}_{i:06d}.parquet") for i, _ in chunks], ignore_index=True)
    return rows.merge(feats, on="image_id", how="left", validate="one_to_one")


def validate_fpdb(meta: pd.DataFrame) -> dict:
    counts = meta["split"].value_counts().to_dict()
    assert counts == EXPECTED_SPLITS, counts
    assert meta["md5"].notna().all()
    tv = meta[meta["split"] != "test"]
    share = tv.assign(o=tv["plane_label"].eq("Other")).groupby("source_machine")["o"].mean()
    for m, v in EXPECTED_OTHER_SHARE.items():
        assert round(share[m], 3) == v, (m, share[m])
    te = meta[meta["split"] == "test"]
    s10 = te.loc[te["source_machine"] == "Voluson S10", "plane_label"].eq("Other").mean()
    assert round(s10, 3) == EXPECTED_S10_TEST_OTHER, s10
    return {"other_share_trainval": share.round(4).to_dict(), "s10_test_other": round(float(s10), 4)}


def summary(meta: pd.DataFrame, ext: pd.DataFrame, checks: dict) -> dict:
    def by(df: pd.DataFrame, col: str) -> dict:
        out: dict = {}
        for key, g in df.groupby(col):
            size = (g["width"].astype(str) + "x" + g["height"].astype(str)).value_counts()
            out[str(key)] = {
                "n": len(g),
                "modal_size": size.index[0], "modal_size_share": round(float(size.iloc[0] / len(g)), 3),
                "sector_area_frac_median": round(float(g["sector_area_frac"].median()), 3),
                "shape_class": g["sector_shape_class"].value_counts().to_dict(),
                "ui_area_frac_median": round(float(g["ui_area_frac"].median()), 4),
                "caliper_detected_rate": round(float((g["caliper_score"] > 0).mean()), 4),
            }
        return out
    return {"checks": checks, "fpdb_by_machine": by(meta, "source_machine"),
            "external_by_centre": by(ext, "centre"),
            "duplicate_md5_fpdb": int(meta["md5"].duplicated().sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-consensus", action="store_true",
                    help="first pass (needed once to build the consensus sectors)")
    args = ap.parse_args()
    use_consensus = not args.no_consensus
    if use_consensus and not load_consensus_index():
        raise SystemExit("no consensus sectors: run with --no-consensus, then build_sector_consensus.py")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = run(fpdb_rows(), "fpdb", args.workers, use_consensus)
    checks = validate_fpdb(meta)
    meta.to_parquet(OUT / "acquisition_metadata.parquet", index=False)
    ext = run(external_rows(), "external", args.workers, use_consensus)
    ext.to_parquet(OUT / "external_metadata.parquet", index=False)
    s = summary(meta, ext, checks)
    Path("results/json").mkdir(parents=True, exist_ok=True)
    Path("results/json/R1_metadata_summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    print(json.dumps(s["checks"], indent=2))


if __name__ == "__main__":
    main()
