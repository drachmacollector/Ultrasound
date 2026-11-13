"""
scripts/research/make_figures.py  (R6 B.1: the eight study figures)

Figures 1, 2, 4, 5 and 6 are drawn by the scripts that produce their data (one implementation per
panel). This script draws the three figures that combine phases, and writes an index of all eight:

  F3_availability_reliance   E4 availability (balanced accuracy from non-anatomical input only) + E5 reliance
  F7_rejection_vs_confusion  external outcome shares: legacy convnext/effv2s and, when present, R5 arms
                             (seed-averaged)
  F8_risk_coverage           selective risk vs coverage, FPDB test vs pooled external; legacy and, when
                             present, R5 B0/AFP (seed-averaged)
  STUDY_FIGURES.md           figure number -> file(s), sources, producing script, present/pending

Inputs: results/json/{E4,E5}.json and results/preds/<run_id>/<dataset>.parquet. Never runs inference.
Every PDF names its sources in the Subject metadata field. Run after the producing scripts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, ".")
from src.audit.plotting import FIG_DIR, INK, NEUTRAL, SERIES, plt, save  # noqa: E402
from src.eval import metrics as M  # noqa: E402

PREDS = Path("results/preds")
JSON = Path("results/json")
EXTERNAL = ("hc18", "ucl", "african")
BACKBONES = {"convnext_tiny": "convnext", "tf_efficientnetv2_s": "effv2s"}
LEGACY = {"convnext_tiny": "legacy_convnext_s42", "tf_efficientnetv2_s": "legacy_effv2s_s42"}
ARMS = ("b0", "b1", "b2", "afp")
SEEDS = (11, 22, 33)
OUTCOMES = (("correct", "Correct", "#d9d8d4"), ("other", "Predicted Other", "#2a78d6"),
            ("absent", "Absent class", "#eb6834"), ("confusion", "Wrong present class", "#1baf7a"))

STUDY_FIGURES = [
    (1, "Confounding mosaic: class share per machine, train+val vs test", ["E1_confound_mosaic"], "E1.json",
     "e1_confounding_tables.py"),
    (2, "Metadata-only ROC (no pixels)", ["E2_metadata_roc"], "E2.json", "e2_metadata_only_model.py"),
    (3, "Availability (E4) and reliance (E5)", ["F3_availability_reliance"], "E4.json, E5.json", "make_figures.py"),
    (4, "Saliency grid with landmark outlines", ["R4_saliency_grid"], "R4 saliency parquets/maps",
     "r4_summarise.py"),
    (5, "EPG − area by class × backbone; association with external correctness",
     ["R4_epg_by_class_backbone", "R4_association"], "R4_localisation.json, R4_association.json",
     "r4_summarise.py, r4_association.py"),
    (6, "Forest plot: B1/B2/AFP − B0 per backbone, pooled and per dataset", ["R5_forest"], "R5_primary.json",
     "r5_primary_endpoint.py"),
    (7, "Rejection vs confusion per external dataset (legacy and R5 arms)", ["F7_rejection_vs_confusion"],
     "R2 + R5 prediction parquets", "make_figures.py"),
    (8, "Risk–coverage, in-distribution vs external", ["F8_risk_coverage"], "R2 + R5 prediction parquets",
     "make_figures.py"),
]


def _load(rid: str, ds: str) -> pd.DataFrame | None:
    p = PREDS / rid / f"{ds}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p, columns=["image_id", "true_collapsed", "pred_full", "correct_full", "conf_full_summed"])
    for c in ("true_collapsed", "pred_full"):
        d[c] = d[c].astype(str)
    return d


def r5_runs(arm: str, bb: str) -> list[str]:
    """Seed run ids of an arm, only if all three seeds have test and external predictions."""
    ids = [f"{arm}_{BACKBONES[bb]}_s{s}" for s in SEEDS]
    ok = all((PREDS / i / f"{ds}.parquet").exists() for i in ids for ds in ("fpdb_test", *EXTERNAL))
    return ids if ok else []


# ----------------------------------------------------------------------------- outcome shares
def outcome_shares(d: pd.DataFrame, present: set[str]) -> dict[str, float]:
    """Share of images: correct / predicted Other / predicted a class absent from the set / wrong present class."""
    err = d["pred_full"] != d["true_collapsed"]
    other = err & (d["pred_full"] == "Other")
    absent = err & ~other & ~d["pred_full"].isin(present)
    conf = err & ~other & d["pred_full"].isin(present)
    n = len(d)
    return {"correct": float((~err).sum() / n), "other": float(other.sum() / n),
            "absent": float(absent.sum() / n), "confusion": float(conf.sum() / n)}


def shares_for(run_ids: list[str]) -> dict[str, dict[str, float]]:
    """Seed-averaged outcome shares per external dataset and pooled (pooled = image-weighted over the three)."""
    out: dict[str, dict[str, float]] = {}
    per_seed: list[dict[str, dict[str, float]]] = []
    for rid in run_ids:
        frames = {ds: _load(rid, ds) for ds in EXTERNAL}
        r = {ds: outcome_shares(f, set(f["true_collapsed"])) for ds, f in frames.items()}
        n = {ds: len(f) for ds, f in frames.items()}
        r["pooled"] = {k: sum(r[ds][k] * n[ds] for ds in EXTERNAL) / sum(n.values()) for k, _, _ in OUTCOMES}
        per_seed.append(r)
    for ds in (*EXTERNAL, "pooled"):
        out[ds] = {k: float(np.mean([s[ds][k] for s in per_seed])) for k, _, _ in OUTCOMES}
    return out


def fig7() -> list[str]:
    rows: list[tuple[str, dict]] = []
    used = []
    for bb, short in BACKBONES.items():
        rows.append((f"{short} legacy (seed 42)", shares_for([LEGACY[bb]])))
        used.append(LEGACY[bb])
        for arm in ARMS:
            ids = r5_runs(arm, bb)
            if ids:
                rows.append((f"{short} {arm.upper()} (3 seeds)", shares_for(ids)))
                used += ids
    panels = [("hc18", "HC18"), ("ucl", "UCL"), ("african", "African"), ("pooled", "Pooled")]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 0.28 * len(rows) + 1.0), sharey=True)
    for ax, (ds, title) in zip(axes, panels):
        for i, (_, sh) in enumerate(rows):
            left = 0.0
            for key, _, col in OUTCOMES:
                ax.barh(i, sh[ds][key], left=left, color=col, edgecolor="white", linewidth=1.2, height=0.72)
                left += sh[ds][key]
        ax.set_title(title, loc="left")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0], ["0", "50%", "100%"])
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in OUTCOMES]
    fig.legend(handles, [lab for _, lab, _ in OUTCOMES], loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("External sets: share of images by outcome (full 8-way argmax)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "F7_rejection_vs_confusion", sources="results/preds/{" + ",".join(used) + "}/{hc18,ucl,african}.parquet",
         dpi=300)
    return used


# ----------------------------------------------------------------------------- risk-coverage
def curve(run_ids: list[str], kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Seed-averaged selective risk on fpdb_test or the pooled externals."""
    risks = []
    for rid in run_ids:
        d = (_load(rid, "fpdb_test") if kind == "fpdb_test"
             else pd.concat([_load(rid, ds) for ds in EXTERNAL], ignore_index=True))
        grid, risk = M.risk_coverage(d["conf_full_summed"].to_numpy(np.float64), d["correct_full"].to_numpy(bool))
        risks.append(risk)
    return grid, np.mean(risks, axis=0)


def fig8() -> list[str]:
    used = []
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0), sharex=True)
    styles = {"legacy": (NEUTRAL, "--", "s"), "b0": (SERIES[0], "-", "o"), "afp": (SERIES[1], "-", "^")}
    for i, (bb, short) in enumerate(BACKBONES.items()):
        series = [("legacy", f"legacy (seed 42)", [LEGACY[bb]])]
        series += [(arm, f"{arm.upper()} (3 seeds)", r5_runs(arm, bb)) for arm in ("b0", "afp")]
        for j, (kind, title) in enumerate((("fpdb_test", "FPDB test (in-distribution)"),
                                           ("pooled", "Pooled external (HC18 + UCL + African)"))):
            ax = axes[i, j]
            for key, lab, ids in series:
                if not ids:
                    continue
                grid, risk = curve(ids, kind)
                col, ls, mk = styles[key]
                ax.plot(grid, risk, color=col, ls=ls, lw=1.6, marker=mk, ms=3.5, markevery=4, label=lab)
                if j == 0:
                    used += ids
            ax.set_title(f"{short}: {title}", loc="left")
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=None))
            ax.grid(axis="y", color="#ececea", lw=0.6)
            if j == 0:
                ax.set_ylabel("Selective risk (1 − accuracy)")
            if i == 1:
                ax.set_xlabel("Coverage (share of most-confident images kept)")
        axes[i, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "F8_risk_coverage", sources="results/preds/{" + ",".join(used) + "}/{fpdb_test,hc18,ucl,african}.parquet",
         dpi=300)
    return used


# ----------------------------------------------------------------------------- figure 3
def _err(ax, x, y, e, col, mk="o", label=None):
    ax.errorbar(e["point"], y, xerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], fmt=mk, ms=4,
                color=col, elinewidth=1.1, capsize=1.5, label=label)


def fig3() -> None:
    e4 = json.loads((JSON / "E4.json").read_text(encoding="utf-8"))
    e5 = json.loads((JSON / "E5.json").read_text(encoding="utf-8"))["models"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    rows = [("Full image (legacy convnext, seed 42)", e4["b0_full_image"]["bal_acc_8"], INK, "D")]
    for mode, lab in (("overlay_only", "Overlays only"), ("nonanat_only", "Everything but the sector"),
                      ("sector_shape_only", "Sector shape only")):
        m = e4["modes"][mode]
        rows.append((f"{lab} ({len(m['seeds'])} seed{'s' if len(m['seeds']) > 1 else ''})",
                     m["summary"]["bal_acc_8"], SERIES[0], "o"))
    rows += [("Metadata prior (E2, no pixels)", e4["e2_metadata"]["prior"]["bal_acc_8"], NEUTRAL, "s"),
             ("Metadata + geometry (E2, no pixels)", e4["e2_metadata"]["all"]["bal_acc_8"], NEUTRAL, "s")]
    for i, (lab, e, col, mk) in enumerate(rows):
        _err(a, None, i, e, col, mk)
    a.axvline(e4["chance"]["bal_acc_8"], color=NEUTRAL, ls=":", lw=0.9)
    a.text(e4["chance"]["bal_acc_8"] + 0.01, len(rows) - 1.45, "chance", color=NEUTRAL, fontsize=7, va="center")
    a.set_yticks(range(len(rows)), [r[0] for r in rows])
    a.invert_yaxis()
    a.set_xlim(0, 1)
    a.set_xlabel("8-way balanced accuracy, FPDB test")
    a.set_title("a  Availability (E4)", loc="left")
    variants = [("perturb-a_text", "text masked"), ("perturb-b_zero", "out-of-sector zeroed"),
                ("perturb-b_mean", "out-of-sector mean"), ("perturb-b_blur", "out-of-sector blurred")]
    marks = ("o", "s", "^", "D")
    models = list(e5)
    for j, (v, lab) in enumerate(variants):
        col = SERIES[j] if j < 3 else NEUTRAL
        for i, bb in enumerate(models):
            _err(b, None, i + (j - 1.5) * 0.18, e5[bb]["variants"][v]["flip_fine"], col, marks[j],
                 label=lab if i == 0 else None)
    b.set_yticks(range(len(models)), models)
    b.invert_yaxis()
    b.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    b.set_xlabel("Flip rate of the 8-way prediction, FPDB test")
    b.set_title("b  Reliance (E5): frozen legacy models", loc="left")
    b.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    fig.tight_layout()
    save(fig, "F3_availability_reliance", sources="results/json/E4.json, results/json/E5.json", dpi=300)


# ----------------------------------------------------------------------------- index
def index() -> str:
    L = ["# Study figures (R6 B.1)", "",
         "Generated by `scripts/research/make_figures.py`; PDFs are tracked, PNGs are regenerated.", "",
         "| # | figure | file(s) | sources | drawn by | status |", "|---|---|---|---|---|---|"]
    for n, title, stems, src, script in STUDY_FIGURES:
        present = all((FIG_DIR / f"{s}.pdf").exists() for s in stems)
        files = ", ".join(f"`{s}.pdf`" for s in stems)
        L.append(f"| {n} | {title} | {files} | {src} | `{script}` | {'present' if present else 'pending'} |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    fig3()
    fig7()
    fig8()
    md = index()
    (FIG_DIR / "STUDY_FIGURES.md").write_text(md, encoding="utf-8")
    print(f"wrote F3, F7, F8 and {FIG_DIR / 'STUDY_FIGURES.md'}: {md.count('| present |')} present, "
          f"{md.count('| pending |')} pending")


if __name__ == "__main__":
    main()
