"""
scripts/research/g1_report.py  (R1.9: Gate G1)

Builds docs/research/GATE_G1.md and results/json/G1.json from:
  data/processed/audit/overlay_audit_{train,test}.csv       binary caliper/text audit
  data/processed/audit/ratings_raw/sector_test.txt          'idx ok(0/1)' sector-outline check
  data/processed/audit/ratings_raw/landmark_glance.txt      'idx ok(0/1)' landmark glance
  data/processed/metadata/acquisition_metadata.parquet      detector outputs
  results/json/R1_caliper_pairs.json, R1_landmark_linkage.json
and applies the pre-specified decision rules (plans/R1_DATA_FOUNDATION.md §R1.9).

Usage:  python scripts/research/g1_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

sys.path.insert(0, ".")

AUDIT = Path("data/processed/audit")
RAW = AUDIT / "ratings_raw"
META = Path("data/processed/metadata/acquisition_metadata.parquet")
OUT_MD = Path("docs/research/GATE_G1.md")
OUT_JSON = Path("results/json/G1.json")

RULE_PREVALENCE_DIFF = 0.10
RULE_RECALL = 0.80
RULE_MIN_PAIRS = 20
RULE_SECTOR_OK = 0.90
RULE_GLANCE_MIN = 14


def wilson(k: int, n: int) -> tuple[float, float]:
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson") if n else (float("nan"),) * 2
    return float(lo), float(hi)


def fmt_p(k: int, n: int) -> str:
    if n == 0:
        return "–"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]"


def read_binary(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            idx, ok = (int(v) for v in line.split()[:2])
            rows.append({"idx": idx, "ok": ok})
    return pd.DataFrame(rows)


def load_audit() -> pd.DataFrame:
    sample = pd.read_csv(AUDIT / "overlay_audit_sample.csv")
    meta = pd.read_parquet(META)[["image_id", "caliper_score", "n_ui_components"]]
    frames = []
    for split in ("train", "test"):
        a = pd.read_csv(AUDIT / f"overlay_audit_{split}.csv")
        s = sample[sample["split"] == split][["audit_idx", "source_machine", "plane_collapsed"]]
        frames.append(a.merge(s, on="audit_idx").assign(split=split))
    return pd.concat(frames, ignore_index=True).merge(meta, on="image_id", how="left")


def detector_metrics(df: pd.DataFrame, truth: str, pred: pd.Series) -> dict:
    t = df[truth] == 1
    tp, fp, fn = int((pred & t).sum()), int((pred & ~t).sum()), int((~pred & t).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": int((~pred & ~t).sum()),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "recall_ci": wilson(tp, tp + fn)}


def main() -> None:
    a = load_audit()
    test = a[a["split"] == "test"]
    std = test["plane_collapsed"] != "Other"

    # 1. prevalence
    k1, n1 = int(test.loc[std, "calipers"].sum()), int(std.sum())
    k2, n2 = int(test.loc[~std, "calipers"].sum()), int((~std).sum())
    diff = k1 / n1 - k2 / n2
    dlo, dhi = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb", compare="diff")
    by_cell = (test.groupby(["source_machine", "plane_collapsed"])
               .agg(n=("calipers", "size"), calipers=("calipers", "sum"), text=("text", "sum")).reset_index())
    pooled = {"calipers": fmt_p(int(a["calipers"].sum()), len(a)), "text": fmt_p(int(a["text"].sum()), len(a))}

    # 2. detector vs audit
    det = {}
    for split, d in a.groupby("split"):
        det[split] = {
            "calipers": detector_metrics(d, "calipers", d["caliper_score"] >= 1),
            "text": detector_metrics(d, "text", d["n_ui_components"] >= 1),
        }

    # 3. sector check, 4. pairs, 5. landmarks
    sector = read_binary(RAW / "sector_test.txt") if (RAW / "sector_test.txt").exists() else pd.DataFrame()
    sector_by_m = pd.DataFrame(columns=["sum", "size"])
    if len(sector):
        sector = sector.merge(test[["audit_idx", "source_machine"]], left_on="idx", right_on="audit_idx")
        sector_by_m = sector.groupby("source_machine")["ok"].agg(["sum", "size"])
    recheck = pd.DataFrame()
    recheck_by_m = pd.DataFrame(columns=["sum", "size"])
    if (RAW / "sector_recheck.txt").exists():
        recheck = read_binary(RAW / "sector_recheck.txt").merge(
            pd.read_csv(AUDIT / "sector_recheck_sample.csv"), left_on="idx", right_on="recheck_idx")
        recheck_by_m = recheck.groupby("source_machine")["ok"].agg(["sum", "size"])
    pairs = json.loads(Path("results/json/R1_caliper_pairs.json").read_text(encoding="utf-8"))
    link = json.loads(Path("results/json/R1_landmark_linkage.json").read_text(encoding="utf-8"))
    glance = read_binary(RAW / "landmark_glance.txt") if (RAW / "landmark_glance.txt").exists() else pd.DataFrame()

    ext_audit = pd.read_csv(AUDIT / "external_audit.csv") if (AUDIT / "external_audit.csv").exists() else None
    ext_meta = pd.read_parquet("data/processed/metadata/external_metadata.parquet")
    ext_tab = pd.DataFrame()
    if ext_audit is not None:
        ext_tab = ext_audit.groupby("centre").agg(n=("calipers", "size"), calipers=("calipers", "sum"),
                                                   text=("text", "sum"))
        ext_tab["detector_caliper_rate_all_images"] = ext_meta.groupby("centre")["caliper_score"].apply(
            lambda s: round(float((s > 0).mean()), 3))
        ext_tab["calipers"] = [fmt_p(int(k), int(n)) for k, n in zip(ext_tab["calipers"], ext_tab["n"])]
        ext_tab["text"] = [fmt_p(int(k), int(n)) for k, n in zip(ext_tab["text"], ext_tab["n"])]

    # --- decisions ---
    rec = det["train"]["calipers"]["recall"]
    sector_ok_min = float((sector_by_m["sum"] / sector_by_m["size"]).min()) if len(sector) else float("nan")
    recheck_ok_min = (float((recheck_by_m["sum"] / recheck_by_m["size"]).min()) if len(recheck)
                      else float("nan"))
    head_qc = link["qc_flag_rate"].get("fpdb/test/Head", 0.0)
    decisions = {
        "overlays_primary_axis": bool(abs(diff) >= RULE_PREVALENCE_DIFF and not (dlo <= 0 <= dhi)),
        "caliper_detector_recall_train": rec,
        "caliper_detector_passes": bool(rec >= RULE_RECALL),
        "E5a_restricted_to_agreed_images": bool(rec < RULE_RECALL),
        "E5d_feasible": bool(pairs["caliper_pair_candidates"] >= RULE_MIN_PAIRS),
        "sector_consensus_prior_needed": bool(sector_ok_min < RULE_SECTOR_OK) if len(sector) else None,
        "sector_recheck_passes": bool(recheck_ok_min >= RULE_SECTOR_OK) if len(recheck) else None,
        "landmark_glance_ok": bool(len(glance) and glance["ok"].sum() >= RULE_GLANCE_MIN),
        "head_qc_ok": bool(head_qc < 0.05),
    }

    result = {
        "prevalence": {"calipers_standard": [k1, n1], "calipers_other": [k2, n2], "diff": diff,
                       "diff_ci_newcombe": [float(dlo), float(dhi)], "pooled_300": pooled},
        "detector": det, "pairs": pairs, "landmark_qc": link["qc_flag_rate"],
        "sector_check": ({m: {"ok": int(r["sum"]), "n": int(r["size"])} for m, r in sector_by_m.iterrows()}
                         if len(sector) else None),
        "sector_recheck": ({m: {"ok": int(r["sum"]), "n": int(r["size"])} for m, r in recheck_by_m.iterrows()}
                           if len(recheck) else None),
        "external_audit": ({c: {"n": len(g), "calipers": int(g["calipers"].sum()), "text": int(g["text"].sum())}
                            for c, g in ext_audit.groupby("centre")} if ext_audit is not None else None),
        "landmark_glance": {"ok": int(glance["ok"].sum()), "n": len(glance)} if len(glance) else None,
        "decisions": decisions,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    def m(d: dict) -> str:
        return (f"TP {d['tp']} · FP {d['fp']} · FN {d['fn']} · TN {d['tn']} · precision "
                f"{d['precision']:.2f} · recall {d['recall']:.2f} [{d['recall_ci'][0]:.2f}, {d['recall_ci'][1]:.2f}]")

    cell_md = by_cell.assign(
        calipers=lambda x: x.apply(lambda r: fmt_p(int(r.calipers), int(r.n)), axis=1),
        text=lambda x: x.apply(lambda r: fmt_p(int(r.text), int(r.n)), axis=1)).to_markdown(index=False)
    lines = [
        "# Gate G1: Overlay prevalence, detector validation, data linkage",
        "",
        "Auto-generated by `scripts/research/g1_report.py` (numbers from `results/json/G1.json`).",
        "**Rater:** the binary audit (calipers y/n, text y/n, sector outline ok y/n, landmark glance y/n) was",
        "performed by a vision-capable model (`claude-vision`) viewing rendered contact sheets of raw images,",
        "as agreed with the PI, instead of a human rater. The rater did not see detector output when rating",
        "calipers/text. A human may re-rate any sheet (`data/processed/audit/sheets/`) for an agreement check.",
        "",
        "## 1. Overlay prevalence (binary audit)",
        f"* Pooled over all 300 audited FPDB images: calipers {pooled['calipers']}; on-screen text {pooled['text']}.",
        f"* Test audit, calipers: standard planes {fmt_p(k1, n1)} vs Other {fmt_p(k2, n2)}; "
        f"difference {diff:+.1%} (Newcombe 95% CI {dlo:+.1%} to {dhi:+.1%}).",
        "",
        "Test audit by machine × collapsed class (Wilson 95% CIs):",
        "",
        cell_md,
        "",
        "## 2. Detector vs audit (image level)",
        f"* Calipers, train audit (tuning; one round): {m(det['train']['calipers'])}",
        f"* Calipers, test audit (reporting): {m(det['test']['calipers'])}",
        f"* Text/UI, train: {m(det['train']['text'])}",
        f"* Text/UI, test: {m(det['test']['text'])}",
        "",
        "## 2b. External test sets: overlay prevalence (6 random images per centre, same binary audit)",
        "Detector rate = share of *all* images of that centre with ≥1 detected cross (high-precision detector;",
        "a lower bound, since dotted/dark calipers are missed).",
        "",
        (ext_tab.to_markdown() if ext_audit is not None else "_not run_"),
        "",
        "## 3. Sector segmentation check (test audit images, outline y/n; first-pass per-image masks)",
        (sector_by_m.assign(rate=lambda x: x["sum"] / x["size"]).to_markdown() if len(sector) else "_not run_"),
        "",
        "**Re-check after enabling the consensus prior** (train-only device-geometry core sector, union; "
        "new test images not in the audit sample):",
        "",
        (recheck_by_m.assign(rate=lambda x: x["sum"] / x["size"]).to_markdown() if len(recheck) else "_not run_"),
        "",
        "## 4. Near-duplicates and caliper pairs (FPDB test)",
        "```", json.dumps(pairs, indent=2), "```",
        "",
        "## 5. Landmarks",
        f"* FP↔FPDB identity check failures: {sum(v['identity_failures'] for v in link['fp_report'].values())}.",
        f"* QC flag rates: {json.dumps({k: round(v, 4) for k, v in link['qc_flag_rate'].items()})}",
        f"* Landmark glance (15 val images): "
        f"{(str(int(glance['ok'].sum())) + '/' + str(len(glance))) if len(glance) else 'not run'} judged correct.",
        "",
        "## 6. Pre-specified decision rules → outcome",
        "| rule | value | outcome |",
        "|---|---|---|",
        f"| overlays primary iff caliper prevalence diff ≥ 10 pp (CI excludes 0) | {diff:+.1%} | "
        f"{'PRIMARY' if decisions['overlays_primary_axis'] else 'SECONDARY: emphasise E2/E3/E6/A3'} |",
        f"| caliper detector recall (train) ≥ 0.80 after one re-tune | {rec:.2f} | "
        f"{'pass' if decisions['caliper_detector_passes'] else 'FAIL → E5a restricted to detector∧audit-positive test images'} |",
        f"| confirmed caliper pairs ≥ 20 | {pairs['caliper_pair_candidates']} candidates | "
        f"{'E5d kept' if decisions['E5d_feasible'] else 'E5d DROPPED'} |",
        f"| sector outline ok ≥ 90% on every machine | min {sector_ok_min:.0%} | "
        f"{'no consensus prior needed' if decisions['sector_consensus_prior_needed'] is False else 'enable consensus prior'} |",
        f"| re-check after consensus ≥ 90% per re-checked machine | min {recheck_ok_min:.0%} | "
        f"{'PASS' if decisions['sector_recheck_passes'] else 'FAIL'} |",
        f"| landmark glance ≥ 14/15 and Head QC < 5% | {result['landmark_glance']} / {head_qc:.1%} | "
        f"{'R4 may proceed' if decisions['landmark_glance_ok'] and decisions['head_qc_ok'] else 'R4 BLOCKED'} |",
        "",
        "## 7. Consequences for later phases",
        f"* **Markers in FPDB are rare** ({pooled['calipers']} of audited images) and not enriched in standard planes, so the",
        "  'calipers ⇒ standard plane' shortcut cannot be a major driver *within* FPDB. E5a (marker inpainting) has "
        f"{det['test']['calipers']['tp']} detector∧audit-positive test images → **E5a infeasible**; E5d dropped (no pairs).",
        "* **On-screen text/UI is universal in FPDB** (100% of audited images) and the UI detector is accurate "
        f"(test recall {det['test']['text']['recall']:.2f}, precision {det['test']['text']['precision']:.2f}). The overlay axis"
        " that remains testable is therefore **UI/out-of-sector content** (E4 non-anatomical / overlay-only models,"
        " E5 `a_text` and `b_*` perturbations), not calipers.",
        "* **External sets differ in the opposite direction:** calipers/measurement graphics are frequent in Ghana, Uganda and"
        " UCL but rare in FPDB training, and the African images carry little or no UI. This is a train→test overlay shift to"
        " report in R2/R3. Synthetic caliper augmentation (A2) keeps its rationale for R5; test-time marker inpainting is"
        " limited by detector recall (it works for Ghana's '+' markers, detector rate ≈0.97).",
        f"* **Exact duplicates** (see results/json/R1_exact_duplicates.json): byte-identical FPDB images exist within"
        " patients (never across splits); 8 test-split groups carry contradictory labels for the identical image. Splits"
        " stay frozen; patient-clustered CIs (R2) account for the within-patient redundancy.",
        "",
        "## 8. Sign-off",
        "Emphasis decision recorded by the implementer; **PI to confirm** (append name/date below).",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
