"""
scripts/link_fp_landmarks.py  (R1.3)

Expert biometry landmarks -> localisation ground truth.

* FP subset (= FETAL_PLANES_DB images): joined to our frozen splits by filename stem.
  The FP images are pixel-identical to the FPDB originals (verified 2026-11-12), so no
  registration is needed; an identity check (shape + exact array equality) replaces it.
  Only rows whose image is in our **val** (development) or **test** (reporting) split are
  kept. Train-split landmarks are dropped so they can never reach a training loader.
* HC18 and UCL: native image space; image_id/patient_id as in the external manifests.

Outputs
  data/processed/landmarks/fpdb_landmarks.parquet       (val + test FPDB images)
  data/processed/landmarks/external_landmarks.parquet   (HC18 + UCL)
  results/tables/R1_landmark_linkage.md
  results/json/R1_landmark_linkage.json

This is the only code allowed to read data/raw/ucl_hc18/*/FP/.
Usage:  python scripts/link_fp_landmarks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data.paths import fpdb_image_id, fpdb_patient_id, norm_path, numeric_prefix  # noqa: E402

ROOT = Path("data/raw/ucl_hc18")
FPDB_IMAGES = Path("data/raw/fetal_planes_db/Images")
OUT_DIR = Path("data/processed/landmarks")
TABLE = Path("results/tables/R1_landmark_linkage.md")
JSON_OUT = Path("results/json/R1_landmark_linkage.json")

#: anatomy -> (major-axis prefix, minor-axis prefix or None)
AXES = {"Head": ("ofd", "bpd"), "Abdomen": ("tad", "apad"), "Femur": ("fl", None)}
#: Max |mid(axis1) - mid(axis2)| / (major/2). Head axes (OFD/BPD) bisect each other closely;
#: abdominal TAD/APAD are skin-edge caliper pairs that typically do not (median 0.08, p95 0.19 of
#: the semi-axis, measured on all FP+UCL rows), so a looser bound flags only gross inconsistency.
MAX_MID_OFFSET = {"Head": 0.15, "Abdomen": 0.30}
EXPECTED_FP_TEST = {"Head": 765, "Abdomen": 350, "Femur": 371}
QC_MAX_FLAG_RATE_HEAD = 0.05


def _pts(row: pd.Series, prefix: str) -> np.ndarray:
    return np.array([[row[f"{prefix}_1_x"], row[f"{prefix}_1_y"]],
                     [row[f"{prefix}_2_x"], row[f"{prefix}_2_y"]]], dtype=np.float64)


def geometry_qc(row: pd.Series, anatomy: str) -> dict:
    """Axis lengths, angle between axes and centre offset; QC flag."""
    a, b = AXES[anatomy]
    pa = _pts(row, a)
    la = float(np.linalg.norm(pa[1] - pa[0]))
    out: dict = {f"{a}_len": la}
    flag = la < 10.0
    if b is not None:
        pb = _pts(row, b)
        lb = float(np.linalg.norm(pb[1] - pb[0]))
        va, vb = pa[1] - pa[0], pb[1] - pb[0]
        cosang = abs(float(va @ vb)) / max(la * lb, 1e-9)
        angle = float(np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0))))
        major_half = max(la, lb) / 2.0
        offset = float(np.linalg.norm(pa.mean(0) - pb.mean(0))) / max(major_half, 1e-9)
        out.update({f"{b}_len": lb, "axis_angle_deg": angle, "mid_offset_rel": offset})
        flag = flag or not (70.0 <= angle <= 110.0) or offset > MAX_MID_OFFSET[anatomy]
        if anatomy == "Head":
            flag = flag or lb >= la          # BPD must be shorter than OFD
    out["qc_flag"] = bool(flag)
    return out


def clip_to_image(row: pd.Series, anatomy: str, w: int, h: int) -> tuple[pd.Series, bool]:
    row = row.copy()
    clipped = False
    for prefix in [p for p in AXES[anatomy] if p]:
        for i in (1, 2):
            x, y = float(row[f"{prefix}_{i}_x"]), float(row[f"{prefix}_{i}_y"])
            cx, cy = min(max(x, 0.0), w - 1.0), min(max(y, 0.0), h - 1.0)
            clipped |= (cx != x) or (cy != y)
            row[f"{prefix}_{i}_x"], row[f"{prefix}_{i}_y"] = cx, cy
    return row, clipped


def landmark_cols(anatomy: str) -> list[str]:
    return [f"{p}_{i}_{c}" for p in AXES[anatomy] if p for i in (1, 2) for c in ("x", "y")]


def load_splits() -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(f"data/splits/{split}.csv")
        df["split"] = split
        frames.append(df)
    s = pd.concat(frames, ignore_index=True)
    s["image_id"] = s["image_path"].map(fpdb_image_id)
    return s.set_index("image_id")


def build_fp(splits: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, report = [], {}
    for anatomy in AXES:
        ann = pd.read_csv(ROOT / f"annotations/FP/{anatomy}.csv")
        ann["image_id"] = ann["image_name"].str.replace(r"\.png$", "", regex=True)
        matched = ann["image_id"].isin(splits.index)
        assert matched.all(), f"FP {anatomy}: {(~matched).sum()} images not in FPDB splits"
        ann = ann.join(splits[["split", "patient_id", "source_machine", "plane_label", "image_path"]], on="image_id")
        rep = {"total": int(len(ann)), "by_split": ann["split"].value_counts().to_dict()}
        keep = ann[ann["split"].isin(["val", "test"])]
        n_identity_fail = 0
        for _, r in keep.iterrows():
            fp_img = cv2.imread(str(ROOT / f"images/FP/{anatomy}" / r["image_name"]), cv2.IMREAD_GRAYSCALE)
            db_img = cv2.imread(str(norm_path(r["image_path"])), cv2.IMREAD_GRAYSCALE)
            if db_img is None:
                raise FileNotFoundError(f"FPDB image unreadable: {r['image_path']}")
            reg_ok = fp_img is not None and fp_img.shape == db_img.shape and np.array_equal(fp_img, db_img)
            n_identity_fail += int(not reg_ok)
            h, w = db_img.shape[:2]
            r2, clipped = clip_to_image(r, anatomy, w, h)
            rec = {
                "image_id": r["image_id"], "dataset": "fpdb", "split": r["split"], "anatomy": anatomy,
                "patient_id": fpdb_patient_id(r["patient_id"]), "source_machine": r["source_machine"],
                "fpdb_label": r["plane_label"], "algo": (r.get("Algo") if isinstance(r.get("Algo"), str) else "GE"),
                "px_to_mm": float(r["px_to_mm_rate"]) if pd.notna(r["px_to_mm_rate"]) else np.nan,
                "width": w, "height": h, "reg_ok": bool(reg_ok), "lm_clipped": bool(clipped),
            }
            rec.update({c: float(r2[c]) for c in landmark_cols(anatomy)})
            rec.update(geometry_qc(r2, anatomy))
            rows.append(rec)
        rep["kept_val_test"] = int(len(keep))
        rep["identity_failures"] = n_identity_fail
        report[anatomy] = rep
    df = pd.DataFrame(rows)
    return df, report


def build_external() -> pd.DataFrame:
    rows = []
    sources = [("hc18", "HC18", "Head"), ("ucl", "UCL", "Head"), ("ucl", "UCL", "Abdomen"), ("ucl", "UCL", "Femur")]
    for ds, folder, anatomy in sources:
        ann = pd.read_csv(ROOT / f"annotations/{folder}/{anatomy}.csv")
        for _, r in ann.iterrows():
            stem = Path(r["image_name"]).stem
            img = cv2.imread(str(ROOT / f"images/{folder}/{anatomy}" / r["image_name"]), cv2.IMREAD_GRAYSCALE)
            assert img is not None, r["image_name"]
            h, w = img.shape[:2]
            r2, clipped = clip_to_image(r, anatomy, w, h)
            image_id = f"hc18/{stem}" if ds == "hc18" else f"ucl/{anatomy.lower()}/{stem}"
            rec = {
                "image_id": image_id, "dataset": ds, "split": "external", "anatomy": anatomy,
                "patient_id": f"{ds}_{numeric_prefix(r['image_name'])}",
                "source_machine": "GE Voluson E8/730" if ds == "hc18" else "GE Voluson (model unspecified)",
                "fpdb_label": None, "algo": "GE",
                "px_to_mm": float(r["px_to_mm_rate"]) if pd.notna(r["px_to_mm_rate"]) else np.nan,
                "width": w, "height": h, "reg_ok": True, "lm_clipped": bool(clipped),
            }
            rec.update({c: float(r2[c]) for c in landmark_cols(anatomy)})
            rec.update(geometry_qc(r2, anatomy))
            rows.append(rec)
    return pd.DataFrame(rows)


def summarise(fp: pd.DataFrame, ext: pd.DataFrame, report: dict) -> dict:
    allr = pd.concat([fp, ext], ignore_index=True)
    grp = allr.groupby(["dataset", "split", "anatomy"])
    summary = {
        "fp_report": report,
        "counts": {f"{d}/{s}/{a}": int(n) for (d, s, a), n in grp.size().items()},
        "qc_flag_rate": {f"{d}/{s}/{a}": float(v) for (d, s, a), v in grp["qc_flag"].mean().items()},
        "clipped_rate": {f"{d}/{s}/{a}": float(v) for (d, s, a), v in grp["lm_clipped"].mean().items()},
        "fp_test_by_machine": {f"{a}/{m}": int(n) for (a, m), n in
                               fp[fp["split"] == "test"].groupby(["anatomy", "source_machine"]).size().items()},
    }
    return summary


def write_table(fp: pd.DataFrame, ext: pd.DataFrame, summary: dict) -> None:
    allr = pd.concat([fp, ext], ignore_index=True)
    t = allr.groupby(["dataset", "split", "anatomy"]).agg(
        images=("image_id", "size"), patients=("patient_id", "nunique"),
        qc_flag_rate=("qc_flag", "mean"), clipped_rate=("lm_clipped", "mean"),
        identity_ok=("reg_ok", "mean"))
    m = pd.crosstab(fp.loc[fp["split"] == "test", "anatomy"], fp.loc[fp["split"] == "test", "source_machine"],
                    margins=True, margins_name="Total")
    lines = [
        "# R1 — Landmark ground truth linkage (auto-generated by scripts/link_fp_landmarks.py)", "",
        "FP landmark images are pixel-identical to FPDB originals (identity check per row: `identity_ok`).",
        "Only FPDB val (development) and test (reporting) rows are kept.", "",
        t.round(4).to_markdown(), "", "## FPDB-test landmark images by machine", "", m.to_markdown(), "",
        "## FP rows by our split (before filtering)", "",
        pd.DataFrame({a: r["by_split"] for a, r in summary["fp_report"].items()}).fillna(0).astype(int).to_markdown(),
        "",
    ]
    TABLE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    fp, report = build_fp(splits)
    ext = build_external()

    # --- assertions (00_INDEX §3.2) ---
    test_counts = fp[fp["split"] == "test"].groupby("anatomy").size().to_dict()
    assert test_counts == EXPECTED_FP_TEST, f"FP test counts {test_counts} != {EXPECTED_FP_TEST}"
    assert (fp.loc[fp["anatomy"] == "Head", "fpdb_label"] == "Brain_Trans_thalamic").all()
    assert sum(r["identity_failures"] for r in report.values()) == 0, "FP/FPDB identity check failed"
    assert "train" not in set(fp["split"])
    head_flag = fp.loc[fp["anatomy"] == "Head", "qc_flag"].mean()
    assert head_flag < QC_MAX_FLAG_RATE_HEAD, f"Head QC flag rate {head_flag:.3f} >= 5%: column semantics?"

    fp.to_parquet(OUT_DIR / "fpdb_landmarks.parquet", index=False)
    ext.to_parquet(OUT_DIR / "external_landmarks.parquet", index=False)
    summary = summarise(fp, ext, report)
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_table(fp, ext, summary)
    print(json.dumps({k: summary[k] for k in ("counts", "qc_flag_rate", "clipped_rate")}, indent=1))


if __name__ == "__main__":
    main()
