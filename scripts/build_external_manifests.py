"""
scripts/build_external_manifests.py  (R1.2; supersedes build_cross_device_manifest.py)

Builds the external test-set manifests with dataset-qualified patient IDs:

    data/processed/external/{hc18,ucl,african}.csv  and  external_all.csv
    results/tables/R1_external_counts.md

Patient IDs:
  * HC18 / UCL: the annotation CSVs' SubjectID/Patient columns are empty, so the
    patient is the numeric filename prefix (``783_4HC.png`` -> ``hc18_783``). UCL
    prefixes are treated as one cluster across anatomies (conservative).
  * African: ``Patient_num`` repeats across centres -> ``afr_<ISO3>_<num>``.

The FP and MULTICENTRE subsets are never listed (FP = FPDB images; localisation GT only).

Usage:  python scripts/build_external_manifests.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, ".")
from src.data.paths import numeric_prefix  # noqa: E402

LANDMARK_ROOT = Path("data/raw/ucl_hc18")
AFRICAN_ROOT = Path("data/raw/african_planes")
OUT_DIR = Path("data/processed/external")
TABLE_PATH = Path("results/tables/R1_external_counts.md")

COLUMNS = ["image_path", "image_id", "dataset", "patient_id", "centre", "device",
           "plane_label_collapsed", "orig_split", "width", "height", "channels"]

UCL_ANATOMY = {"Head": "Head", "Abdomen": "Abdomen", "Femur": "Femur"}
AFRICAN_PLANE = {"Fetal brain": "Head", "Fetal abdomen": "Abdomen",
                 "Fetal femur": "Femur", "Fetal thorax": "Thorax"}
#: Sendra-Balcells et al., Sci Rep 2023, Table 1.
AFRICAN_CENTRES = {
    "Algeria": ("DZA", "GE Voluson S8"),
    "Egypt": ("EGY", "GE Voluson P8"),
    "Ghana": ("GHA", "EDAN DUS 60"),
    "Malawi": ("MWI", "Mindray DC-N2"),
    "Uganda": ("UGA", "Siemens ACUSON X600"),
}

EXPECTED = {
    "hc18": {"images": 999, "patients": 806},
    "ucl": {"images": 424},
    "african": {"images": 450, "patients": 127, "thorax": 75},
}


def _image_shape(path: Path) -> tuple[int, int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read {path}")
    channels = 1 if img.ndim == 2 else img.shape[2]
    return img.shape[1], img.shape[0], channels


def _add_shapes(df: pd.DataFrame) -> pd.DataFrame:
    shapes = [_image_shape(Path(p)) for p in df["image_path"]]
    df["width"], df["height"], df["channels"] = zip(*shapes)
    return df


def build_hc18() -> pd.DataFrame:
    ann = pd.read_csv(LANDMARK_ROOT / "annotations/HC18/Head.csv")
    rows = []
    for _, r in ann.iterrows():
        name = r["image_name"]
        rows.append({
            "image_path": (LANDMARK_ROOT / "images/HC18/Head" / name).as_posix(),
            "image_id": f"hc18/{Path(name).stem}",
            "dataset": "hc18",
            "patient_id": f"hc18_{numeric_prefix(name)}",
            "centre": "Radboud UMC (NL)",
            "device": "GE Voluson E8/730",
            "plane_label_collapsed": "Head",
            "orig_split": r["Split"],
        })
    return _add_shapes(pd.DataFrame(rows))


def build_ucl() -> pd.DataFrame:
    rows = []
    for anatomy, label in UCL_ANATOMY.items():
        ann = pd.read_csv(LANDMARK_ROOT / f"annotations/UCL/{anatomy}.csv")
        for _, r in ann.iterrows():
            name = r["image_name"]
            rows.append({
                "image_path": (LANDMARK_ROOT / f"images/UCL/{anatomy}" / name).as_posix(),
                "image_id": f"ucl/{anatomy.lower()}/{Path(name).stem}",
                "dataset": "ucl",
                "patient_id": f"ucl_{numeric_prefix(name)}",
                "centre": "UCLH (UK)",
                "device": "GE Voluson (model unspecified)",
                "plane_label_collapsed": label,
                "orig_split": r["Split"],
            })
    return _add_shapes(pd.DataFrame(rows))


def build_african() -> pd.DataFrame:
    ann = pd.read_csv(AFRICAN_ROOT / "African_planes_database.csv")
    rows = []
    for _, r in ann.iterrows():
        iso3, device = AFRICAN_CENTRES[r["Center"]]
        assert f"_{iso3}_" in r["Filename"], f"centre/filename mismatch: {r['Filename']}"
        rows.append({
            "image_path": (AFRICAN_ROOT / r["Center"] / f"{r['Filename']}.png").as_posix(),
            "image_id": f"african/{r['Filename']}",
            "dataset": "african",
            "patient_id": f"afr_{iso3}_{int(r['Patient_num'])}",
            "centre": r["Center"],
            "device": device,
            "plane_label_collapsed": AFRICAN_PLANE[r["Plane"]],
            "orig_split": "Train" if int(r["Train"]) == 1 else "Test",
        })
    return _add_shapes(pd.DataFrame(rows))


def validate(name: str, df: pd.DataFrame) -> None:
    exp = EXPECTED[name]
    assert len(df) == exp["images"], f"{name}: {len(df)} images, expected {exp['images']}"
    if "patients" in exp:
        n = df["patient_id"].nunique()
        assert n == exp["patients"], f"{name}: {n} patients, expected {exp['patients']}"
    assert df["image_id"].is_unique, f"{name}: duplicate image_id"
    forbidden = df["image_path"].map(lambda p: any(part in ("FP", "MULTICENTRE") for part in Path(p).parts))
    assert not forbidden.any(), f"{name}: FP/MULTICENTRE path listed"
    if name == "african":
        thorax = df[df["plane_label_collapsed"] == "Thorax"]
        assert len(thorax) == exp["thorax"]
        assert set(thorax["centre"]) == {"Algeria", "Egypt", "Malawi"}, "thorax centres changed"
        # authors' split must be patient-disjoint
        assert df.groupby("patient_id")["orig_split"].nunique().max() == 1


def counts_table(all_df: pd.DataFrame) -> str:
    lines = ["# R1 — External test sets (auto-generated by scripts/build_external_manifests.py)", ""]
    ct = pd.crosstab([all_df["dataset"], all_df["centre"], all_df["device"]], all_df["plane_label_collapsed"],
                     margins=True, margins_name="Total")
    lines += ["## Images by dataset / centre / device × class", "", ct.to_markdown(), ""]
    pt = all_df.groupby("dataset").agg(images=("image_id", "size"), patients=("patient_id", "nunique"))
    lines += ["## Images and patients (clusters) per dataset", "", pt.to_markdown(), ""]
    geo = (all_df.groupby(["dataset", "centre"])
           .apply(lambda g: g[["width", "height", "channels"]].astype(str).agg("x".join, axis=1)
                  .value_counts().head(3).to_dict(), include_groups=False))
    lines += ["## Most common image geometry (W x H x channels)", "", geo.to_frame("top sizes").to_markdown(), ""]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {"hc18": build_hc18(), "ucl": build_ucl(), "african": build_african()}
    for name, df in frames.items():
        validate(name, df)
        df[COLUMNS].to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} images, {df['patient_id'].nunique()} patients")
    all_df = pd.concat(frames.values(), ignore_index=True)[COLUMNS]
    assert all_df["image_id"].is_unique and all_df["patient_id"].map(lambda s: s.split("_")[0]).isin(
        ["hc18", "ucl", "afr"]).all()
    all_df.to_csv(OUT_DIR / "external_all.csv", index=False)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(counts_table(all_df), encoding="utf-8")
    print(f"wrote {OUT_DIR}/*.csv and {TABLE_PATH}")


if __name__ == "__main__":
    main()
