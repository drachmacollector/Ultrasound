"""
scripts/audit_sheets.py  (R1.7: binary overlay audit, contact-sheet workflow)

The audit answers two image-level yes/no questions per image:
    calipers : any measurement marks inside the image (calipers '+'/'x', dotted measurement
               lines, traced measurement ellipses)?
    text     : any on-screen text / annotation anywhere (device name, settings, labels)?

Workflow (no drawing, boxing or measuring anywhere):
  1. ``--build-sample``  stratified sample: 150 train (detector tuning) + 150 test (reporting),
     machine x collapsed-class strata, seed 20261109. The sample CSV is committed.
  2. ``--render``        numbered 2x2 contact sheets of the RAW images (no detector output, so the
     rater is blind to it) -> data/processed/audit/sheets/<split>_NNN.png
  3. A rater fills ``data/processed/audit/overlay_audit_<split>.csv`` (image_id, calipers, text,
     rater, timestamp). In this study the rater is a vision model ("claude-vision") viewing the
     sheets, as agreed with the PI; a human can re-rate any sheet for agreement checks.
  4. ``--render-pairs`` / ``--render-sector`` render sheets for the caliper-pair confirmation and
     the sector-outline check (used after the detector exists).

Usage:
  python scripts/audit_sheets.py --build-sample
  python scripts/audit_sheets.py --render --split train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data.labels import collapse_label  # noqa: E402
from src.data.paths import fpdb_image_id, fpdb_patient_id, norm_path  # noqa: E402

AUDIT_DIR = Path("data/processed/audit")
SAMPLE_CSV = AUDIT_DIR / "overlay_audit_sample.csv"
SHEET_DIR = AUDIT_DIR / "sheets"
SEED = 20261109
N_PER_SPLIT = 150
MIN_PER_CELL = 4
TILE_W, TILE_H = 760, 520


def build_sample() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out = []
    for split in ("train", "test"):
        df = pd.read_csv(f"data/splits/{split}.csv")
        df["image_id"] = df["image_path"].map(fpdb_image_id)
        df["plane_collapsed"] = df["plane_label"].map(collapse_label)
        cells = df.groupby(["source_machine", "plane_collapsed"])
        sizes = cells.size()
        alloc = sizes.clip(upper=MIN_PER_CELL)
        remaining = N_PER_SPLIT - int(alloc.sum())
        spare = (sizes - alloc).astype(float)
        weights = np.sqrt(sizes.astype(float)) * (spare > 0)
        extra = np.floor(remaining * weights / weights.sum()).astype(int)
        extra = np.minimum(extra, spare.astype(int))
        alloc = alloc + extra
        # hand out rounding leftovers to the largest-weight cells with spare capacity
        order = (weights / (extra + 1)).sort_values(ascending=False).index
        i = 0
        while alloc.sum() < N_PER_SPLIT:
            key = order[i % len(order)]
            if alloc[key] < sizes[key]:
                alloc[key] += 1
            i += 1
        for key, n in alloc.items():
            cell = cells.get_group(key)
            # spread over patients: one image per patient first, then fill
            first = cell.groupby("patient_id").sample(n=1, random_state=int(rng.integers(1 << 31)))
            first = first.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
            pick = first.head(n)
            if len(pick) < n:
                rest = cell.drop(pick.index).sample(n=n - len(pick), random_state=int(rng.integers(1 << 31)))
                pick = pd.concat([pick, rest])
            out.append(pick.assign(split=split))
    sample = pd.concat(out, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)   # interleave strata
    sample["audit_idx"] = sample.groupby("split").cumcount()
    sample["patient_id"] = sample["patient_id"].map(fpdb_patient_id)
    cols = ["split", "audit_idx", "image_id", "image_path", "patient_id", "source_machine", "operator",
            "plane_label", "plane_collapsed"]
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    sample[cols].to_csv(SAMPLE_CSV, index=False)
    print(pd.crosstab([sample["split"], sample["source_machine"]], sample["plane_collapsed"], margins=True))
    return sample[cols]


def _read_color(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(norm_path(path)), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _tile(img: np.ndarray, label: str) -> np.ndarray:
    """Fit an image into TILE_W x TILE_H (aspect preserved), grey padding, number banner."""
    h, w = img.shape[:2]
    s = min(TILE_W / w, (TILE_H - 40) / h)
    r = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((TILE_H, TILE_W, 3), 60, np.uint8)
    y0, x0 = 40 + (TILE_H - 40 - r.shape[0]) // 2, (TILE_W - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r if r.ndim == 3 else cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas, (0, 0), (TILE_W, 38), (0, 140, 255), -1)
    cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def _sheet(tiles: list[np.ndarray]) -> np.ndarray:
    while len(tiles) < 4:
        tiles.append(np.full((TILE_H, TILE_W, 3), 30, np.uint8))
    top = np.hstack([tiles[0], np.full((TILE_H, 8, 3), 255, np.uint8), tiles[1]])
    bot = np.hstack([tiles[2], np.full((TILE_H, 8, 3), 255, np.uint8), tiles[3]])
    return np.vstack([top, np.full((8, top.shape[1], 3), 255, np.uint8), bot])


def render(split: str) -> None:
    sample = pd.read_csv(SAMPLE_CSV)
    sample = sample[sample["split"] == split].sort_values("audit_idx")
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(sample.itertuples())
    for k in range(0, len(rows), 4):
        tiles = []
        for r in rows[k:k + 4]:
            img = _read_color(str(r.image_path))
            tiles.append(_tile(img, f"#{r.audit_idx}"))
        cv2.imwrite(str(SHEET_DIR / f"{split}_{k // 4:03d}.png"), _sheet(tiles))
    print(f"rendered {(len(rows) + 3) // 4} sheets for {split} -> {SHEET_DIR}")


EXTERNAL_SAMPLE_CSV = AUDIT_DIR / "external_audit_sample.csv"
N_PER_CENTRE = 6


def build_external_sample() -> pd.DataFrame:
    """N_PER_CENTRE random images per external centre (calipers y/n audit; quantifies external overlays)."""
    ext = pd.read_csv("data/processed/external/external_all.csv")
    pick = pd.concat([g.sample(N_PER_CENTRE, random_state=SEED) for _, g in ext.groupby("centre")])
    pick = pick.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    pick["audit_idx"] = np.arange(len(pick))
    pick[["audit_idx", "image_id", "image_path", "dataset", "centre", "plane_label_collapsed"]].to_csv(
        EXTERNAL_SAMPLE_CSV, index=False)
    return pick


def render_external() -> None:
    sample = pd.read_csv(EXTERNAL_SAMPLE_CSV)
    rows = list(sample.itertuples())
    for k in range(0, len(rows), 4):
        tiles = [_tile(_read_color(str(r.image_path)), f"#{r.audit_idx}") for r in rows[k:k + 4]]
        cv2.imwrite(str(SHEET_DIR / f"external_{k // 4:03d}.png"), _sheet(tiles))
    print(f"rendered {(len(rows) + 3) // 4} external sheets")


RECHECK_CSV = AUDIT_DIR / "sector_recheck_sample.csv"
RECHECK = {"Voluson E6": 30, "Aloka": 10}     # G1: E6 failed (<90%), Aloka borderline


def render_sector_recheck() -> None:
    """Sector-outline sheets for NEW test images (not in the overlay audit) after the consensus fix."""
    from src.audit.metadata import load_masks, mask_path

    meta = pd.read_parquet("data/processed/metadata/acquisition_metadata.parquet")
    audited = set(pd.read_csv(SAMPLE_CSV)["image_id"])
    pool = meta[(meta["split"] == "test") & ~meta["image_id"].isin(audited)]
    pick = pd.concat([pool[pool["source_machine"] == m].sample(n, random_state=SEED + 1) for m, n in RECHECK.items()])
    pick = pick.reset_index(drop=True)
    pick["recheck_idx"] = np.arange(len(pick))
    pick[["recheck_idx", "image_id", "source_machine", "plane_label"]].to_csv(RECHECK_CSV, index=False)
    out_dir = SHEET_DIR / "sector_recheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for r in pick.itertuples():
        img = _read_color(str(r.image_path))
        m = load_masks(mask_path(str(r.image_id), "fpdb"))
        img[m["ui"] > 0] = (0, 0, 255)
        cnts, _ = cv2.findContours(m["sector_clean"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 0), max(2, img.shape[1] // 250))
        tiles.append(cv2.resize(_tile(img, f"#{r.recheck_idx}"), (TILE_W // 2 + 120, TILE_H // 2 + 80)))
    for k in range(0, len(tiles), 9):
        chunk = tiles[k:k + 9]
        while len(chunk) < 9:
            chunk.append(np.full_like(tiles[0], 30))
        cv2.imwrite(str(out_dir / f"recheck_{k // 9:02d}.png"), np.vstack([np.hstack(chunk[i:i + 3]) for i in (0, 3, 6)]))
    print(f"rendered {len(pick)} recheck images -> {out_dir}")


def render_sector_check(split: str) -> None:
    """3x3 sheets of audit images with the cleaned sector outline (green) and UI mask (red)."""
    from src.audit.metadata import load_masks, mask_path

    sample = pd.read_csv(SAMPLE_CSV)
    sample = sample[sample["split"] == split].sort_values("audit_idx")
    out_dir = SHEET_DIR / "sector"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(sample.itertuples())
    for k in range(0, len(rows), 9):
        tiles = []
        for r in rows[k:k + 9]:
            img = _read_color(str(r.image_path))
            m = load_masks(mask_path(str(r.image_id), "fpdb"))
            img[m["ui"] > 0] = (0, 0, 255)
            cnts, _ = cv2.findContours(m["sector_clean"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), max(2, img.shape[1] // 250))
            t = _tile(img, f"#{r.audit_idx}")
            tiles.append(cv2.resize(t, (TILE_W // 2 + 120, TILE_H // 2 + 80)))
        while len(tiles) < 9:
            tiles.append(np.full_like(tiles[0], 30))
        grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)])
        cv2.imwrite(str(out_dir / f"{split}_sector_{k // 9:03d}.png"), grid)
    print(f"rendered {(len(rows) + 8) // 9} sector sheets -> {out_dir}")


def render_landmark_glance(n_per_anatomy: int = 5) -> pd.DataFrame:
    """15 FPDB-val landmark images with landmark dots + axis lines, for a y/n glance."""
    lm = pd.read_parquet("data/processed/landmarks/fpdb_landmarks.parquet")
    lm = lm[lm["split"] == "val"]
    pick = pd.concat([g.sample(n_per_anatomy, random_state=SEED) for _, g in lm.groupby("anatomy")])
    pick = pick.reset_index(drop=True)
    pick["glance_idx"] = np.arange(len(pick))
    out_dir = SHEET_DIR / "landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = pd.read_parquet("data/processed/metadata/acquisition_metadata.parquet").set_index("image_id")["image_path"]
    tiles = []
    for r in pick.itertuples():
        img = _read_color(str(paths[str(r.image_id)]))
        axes = {"Head": ("ofd", "bpd"), "Abdomen": ("tad", "apad"), "Femur": ("fl",)}[r.anatomy]
        for prefix, colour in zip(axes, ((0, 255, 255), (255, 0, 255))):
            p1 = (int(getattr(r, f"{prefix}_1_x")), int(getattr(r, f"{prefix}_1_y")))
            p2 = (int(getattr(r, f"{prefix}_2_x")), int(getattr(r, f"{prefix}_2_y")))
            cv2.line(img, p1, p2, colour, 2)
            for p in (p1, p2):
                cv2.circle(img, p, 7, (0, 0, 255), -1)
        tiles.append(cv2.resize(_tile(img, f"#{r.glance_idx} {r.anatomy}"), (TILE_W // 2 + 120, TILE_H // 2 + 80)))
    for k in range(0, len(tiles), 6):
        chunk = tiles[k:k + 6]
        while len(chunk) < 6:
            chunk.append(np.full_like(tiles[0], 30))
        cv2.imwrite(str(out_dir / f"landmarks_{k // 6:02d}.png"),
                    np.vstack([np.hstack(chunk[i:i + 3]) for i in (0, 3)]))
    pick[["glance_idx", "image_id", "anatomy"]].to_csv(AUDIT_DIR / "landmark_glance_sample.csv", index=False)
    print(f"rendered {len(pick)} landmark glance images -> {out_dir}")
    return pick


def import_ratings(split: str, ratings_txt: Path, rater: str) -> pd.DataFrame:
    """Turn a plain ``idx calipers text`` (0/1) rating file into overlay_audit_<split>.csv."""
    sample = pd.read_csv(SAMPLE_CSV)
    sample = sample[sample["split"] == split].set_index("audit_idx")
    rows = []
    for line in ratings_txt.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        idx, cal, txt = (int(v) for v in line.split())
        if cal not in (0, 1) or txt not in (0, 1):
            raise ValueError(f"non-binary rating in line: {line!r}")
        rows.append({"audit_idx": idx, "calipers": cal, "text": txt})
    ratings = pd.DataFrame(rows)
    if ratings["audit_idx"].duplicated().any():
        raise ValueError("duplicate audit_idx in ratings")
    missing = set(sample.index) - set(ratings["audit_idx"])
    extra = set(ratings["audit_idx"]) - set(sample.index)
    if missing or extra:
        raise ValueError(f"ratings do not cover the {split} sample exactly (missing={sorted(missing)}, "
                         f"extra={sorted(extra)})")
    out = ratings.join(sample[["image_id"]], on="audit_idx")
    out["rater"] = rater
    out["timestamp"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    out = out[["audit_idx", "image_id", "calipers", "text", "rater", "timestamp"]].sort_values("audit_idx")
    path = AUDIT_DIR / f"overlay_audit_{split}.csv"
    out.to_csv(path, index=False)
    print(f"{path}: {len(out)} rows, calipers={out['calipers'].sum()}, text={out['text'].sum()}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-sample", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--split", choices=["train", "test"])
    ap.add_argument("--import-ratings", type=Path, help="plain 'idx calipers text' file for --split")
    ap.add_argument("--rater", default="claude-vision")
    ap.add_argument("--render-sector", action="store_true", help="sector-outline check sheets for --split")
    ap.add_argument("--render-landmarks", action="store_true", help="15-image landmark glance sheets (val)")
    ap.add_argument("--external", action="store_true", help="build + render the external caliper audit sample")
    ap.add_argument("--render-recheck", action="store_true", help="sector re-check sheets after the consensus fix")
    args = ap.parse_args()
    if args.render_recheck:
        render_sector_recheck()
    if args.external:
        if not EXTERNAL_SAMPLE_CSV.exists():
            build_external_sample()
        render_external()
    if args.render_sector:
        if args.split is None:
            raise SystemExit("--render-sector needs --split")
        render_sector_check(args.split)
    if args.render_landmarks:
        render_landmark_glance()
    if args.import_ratings is not None:
        if args.split is None:
            raise SystemExit("--import-ratings needs --split")
        import_ratings(args.split, args.import_ratings, args.rater)
    if args.build_sample:
        if SAMPLE_CSV.exists():
            raise SystemExit(f"{SAMPLE_CSV} exists; the audit sample is frozen once built.")
        build_sample()
    if args.render:
        if args.split is None:
            raise SystemExit("--render needs --split")
        render(args.split)


if __name__ == "__main__":
    main()
