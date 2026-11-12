"""
scripts/research/check_evidence_table.py  (R6 C.2)

Validates docs/research/review_evidence_table.csv:
  * exact column schema (02 §8.1);
  * every row has a DOI or URL;
  * no duplicate DOI/URL or citation;
  * year is an integer 1990-2030; y/n fields are 'y' or 'n';
and prints the counts of rows, verified rows and preprints (the review keeps preprints <= 15% of
cited references and cites only verified rows). Exit code 1 on any schema error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TABLE = Path("docs/research/review_evidence_table.csv")
COLUMNS = ["citation", "year", "venue", "peer_reviewed", "doi_or_url", "dataset(s)", "n_images", "n_sites", "devices",
           "external_test", "per_device_results", "overlay_handling", "xai_method", "xai_quantitative",
           "calibration_reported", "india_data", "notes", "verified"]
YN = ["peer_reviewed", "verified"]
YN_OPTIONAL = ["external_test", "per_device_results", "xai_quantitative", "calibration_reported", "india_data"]


def check(df: pd.DataFrame) -> list[str]:
    errs = []
    if list(df.columns) != COLUMNS:
        errs.append(f"columns differ from the schema: {list(df.columns)}")
        return errs
    if df["doi_or_url"].isna().any():
        errs.append(f"rows without DOI/URL: {df.index[df['doi_or_url'].isna()].tolist()}")
    for col in ("doi_or_url", "citation"):
        dup = df[col].dropna()[df[col].dropna().duplicated()]
        if len(dup):
            errs.append(f"duplicate {col}: {dup.tolist()}")
    years = pd.to_numeric(df["year"], errors="coerce")
    if years.isna().any() or not years.between(1990, 2030).all():
        errs.append("year must be an integer in 1990-2030")
    for col in YN:
        bad = ~df[col].isin(["y", "n"])
        if bad.any():
            errs.append(f"{col} must be y/n (rows {df.index[bad].tolist()})")
    for col in YN_OPTIONAL:
        bad = df[col].notna() & ~df[col].isin(["y", "n", "partly"])
        if bad.any():
            errs.append(f"{col} must be y/n/partly or empty (rows {df.index[bad].tolist()})")
    return errs


def main() -> None:
    df = pd.read_csv(TABLE)
    errs = check(df)
    for e in errs:
        print(f"ERROR: {e}")
    n, ver = len(df), int((df["verified"] == "y").sum())
    pre = int((df["peer_reviewed"] == "n").sum())
    print(f"{n} rows, {ver} verified (target >= 30), {pre} not peer-reviewed ({100 * pre / max(n, 1):.0f}%)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
