"""
scripts/research/make_review_figures.py  (R6 C.4: figures for the JAPI review article, at most 4)

  review_fig1_schematic     "Acquisition signals vs anatomy": vector drawing only (no images)
  review_fig2_box1          Box 1: machine x class confound (E1) + no-pixel ROC (E2), seen machines
  review_fig3_saliency      2 x 3 saliency examples with landmark outlines; reuses the R4 grid
                            (scripts/research/r4_summarise.py::fig_grid), public data, licence in caption
  review_fig4_checklist     pre-adoption checklist rendered as a table

Every figure is PDF (tracked) + 300-dpi PNG; a greyscale copy (``_grey.png``) is written for the
greyscale-safety check (series are separated by marker/line style and labels, not colour alone).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyBboxPatch, Wedge
from sklearn.metrics import roc_curve

sys.path.insert(0, ".")
from src.audit.plotting import CLASS_COLORS, FIG_DIR, INK, NEUTRAL, SERIES, plt  # noqa: E402

MACHINES = ("Aloka", "Voluson E6", "Voluson S10")
CHECKLIST = [
    ("Per-device results", "Are accuracy and rejection rates reported separately for each ultrasound machine?"),
    ("External sites", "Was the model tested at centres and on devices not seen in training?"),
    ("Rejection rate", "How often are true standard planes rejected as 'Other' on new devices?"),
    ("Overlay robustness", "Does performance hold when calipers, text and UI differ from training?"),
    ("Confound audit", "Can metadata or the frame alone (no anatomy) predict the label in the training data?"),
    ("Saliency sanity check", "Do explanation maps pass a model-randomisation test before being shown?"),
    ("Calibration", "Are confidence scores calibrated on the target device, with a stated abstention rule?"),
    ("Local pilot", "Is there a silent local pilot on the site's own machines before clinical use?"),
    ("Data governance", "Are data handled under PCPNDT and institutional rules (no sex determination, consent)?"),
]


def save_all(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    from PIL import Image

    Image.open(FIG_DIR / f"{stem}.png").convert("L").save(FIG_DIR / f"{stem}_grey.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- fig 1
def schematic() -> None:
    """Numbered call-outs on a drawn frame; the key on the right says which are anatomy vs acquisition."""
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 4.6, 3.8, boxstyle="round,pad=0.02", fc="#1f1f1e", ec=NEUTRAL))
    ax.add_patch(Wedge((2.5, 4.0), 3.4, 235, 305, fc="#5b5a55", ec="none"))           # fan sector
    ax.add_patch(Ellipse((2.5, 1.9), 1.5, 1.15, angle=10, fc="#9c9a92", ec=CLASS_COLORS["Head"], lw=2))
    for x, y in ((1.85, 1.75), (3.15, 2.05)):                                              # calipers
        ax.plot([x - 0.1, x + 0.1], [y, y], color="white", lw=1.2)
        ax.plot([x, x], [y - 0.1, y + 0.1], color="white", lw=1.2)
    ax.plot([1.85, 3.15], [1.75, 2.05], color="white", lw=0.8, ls=(0, (1, 2)))
    ax.text(0.35, 3.85, "Voluson E6", color="white", fontsize=6.5, family="monospace")      # device badge
    ax.text(3.6, 3.85, "TIs 0.4\nMI 1.1\n17 Hz", color="#d8d6cf", fontsize=5, family="monospace", va="top")
    for k in range(6):                                                                      # depth ruler
        ax.plot([0.3, 0.42], [0.8 + 0.45 * k] * 2, color="#d8d6cf", lw=0.8)
    items = [((2.5, 1.45), "fetal anatomy (the intended signal)", True),
             ((1.5, 3.55), "burned-in device name", False),
             ((4.35, 3.0), "acquisition-settings text", False),
             ((0.62, 2.15), "depth ruler and orientation marks", False),
             ((3.4, 2.35), "calipers and measurement lines", False),
             ((3.9, 0.75), "sector shape, zoom and frame size", False)]
    for k, ((x, y), _, anat) in enumerate(items, start=1):
        ax.add_patch(plt.Circle((x, y), 0.17, fc=CLASS_COLORS["Head"] if anat else SERIES[1], ec="white", lw=0.8))
        ax.text(x, y, str(k), ha="center", va="center", fontsize=6.5, color="white", weight="bold")
    for k, (_, txt, anat) in enumerate(items, start=1):
        y = 3.9 - 0.42 * k
        ax.add_patch(plt.Circle((5.25, y), 0.14, fc=CLASS_COLORS["Head"] if anat else SERIES[1], ec="none"))
        ax.text(5.25, y, str(k), ha="center", va="center", fontsize=6, color="white", weight="bold")
        ax.text(5.5, y, txt, va="center", fontsize=6.5, color=INK, weight="bold" if anat else "normal")
    ax.text(5.1, 0.55, "1 = anatomy;  2-6 = acquisition signals that can\npredict the label without anatomy (shortcuts)",
            fontsize=5.8, color=INK, va="center")
    save_all(fig, "review_fig1_schematic")


# ----------------------------------------------------------------------------- fig 2
def box1() -> None:
    e1 = json.loads(Path("results/json/E1.json").read_text(encoding="utf-8"))
    e2 = json.loads(Path("results/json/E2.json").read_text(encoding="utf-8"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(MACHINES))
    for k, (split, hatch) in enumerate((("trainval", ""), ("test", "///"))):
        vals, lo, hi = [], [], []
        for m in MACHINES:
            e = e1["other_share"].get(split, {}).get(m)
            if e is None or e.get("point") is None:
                vals.append(np.nan); lo.append(0); hi.append(0)   # noqa: E702
                continue
            vals.append(e["point"])
            lo.append(e["point"] - (e["lo"] if e["lo"] is not None else e["point"]))
            hi.append((e["hi"] if e["hi"] is not None else e["point"]) - e["point"])
        a.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=CLASS_COLORS["Other"], alpha=1 if k == 0 else 0.55,
              hatch=hatch, edgecolor="white", label={"trainval": "train + val", "test": "test"}[split])
        a.errorbar(x + (k - 0.5) * 0.36, vals, yerr=[lo, hi], fmt="none", ecolor=INK, lw=0.8, capsize=2)
    a.set_xticks(x, ["Aloka", "Voluson E6", "Voluson S10\n(11 train+val images)"], fontsize=6.5)
    a.set_ylabel("share of images labelled 'Other'")
    a.set_ylim(0, 0.8)
    a.legend(fontsize=6.5, loc="upper right")
    a.set_title("A  The label depends on the machine (FPDB)", fontsize=8, loc="left")
    preds = {fs: pd.read_parquet(f"results/preds/e2_metadata/fpdb_test__{fs}.parquet") for fs in ("prior", "geometry")}
    for (fs, d), col, ls in zip(preds.items(), SERIES, ("-", "--")):
        d = d[d["source_machine"] != "Voluson S10"]
        fpr, tpr, _ = roc_curve(d["plane_label"] != "Other", d["score_standard"])
        auc = e2["results"][fs]["seen_machines"]["auroc_standard_vs_other"]
        name = {"prior": "machine + operator", "geometry": "frame geometry"}[fs]
        b.plot(fpr, tpr, color=col, ls=ls, lw=1.5, label=f"{name}: AUROC {auc['point']:.2f} [{auc['lo']:.2f}, {auc['hi']:.2f}]")
    b.plot([0, 1], [0, 1], color="#cfcfcb", lw=0.8, ls=":")
    b.set_xlabel("false-positive rate")
    b.set_ylabel("true-positive rate")
    b.set_aspect("equal")
    b.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    b.set_title("B  'Standard vs Other' without pixels", fontsize=8, loc="left")
    fig.text(0.01, -0.04, "Illustrative re-analysis of publicly available data (FETAL_PLANES_DB, Burgos-Artizzu et al. 2020); "
             "not a clinical study. 95% patient-cluster bootstrap CIs.", fontsize=5.5, color=INK)
    fig.tight_layout()
    save_all(fig, "review_fig2_box1")


# ----------------------------------------------------------------------------- fig 3
def saliency() -> None:
    spec = importlib.util.spec_from_file_location("r4s", Path("scripts/research/r4_summarise.py"))
    r4s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r4s)  # type: ignore[union-attr]
    r4s.save = lambda fig, stem: save_all(fig, "review_fig3_saliency")   # same drawing, review file name
    r4s.fig_grid("gradcam")


# ----------------------------------------------------------------------------- fig 4
def checklist() -> None:
    import textwrap

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.axis("off")
    rows = [[str(i + 1), k, textwrap.fill(q, 72)] for i, (k, q) in enumerate(CHECKLIST)]
    tab = ax.table(cellText=rows, colLabels=["#", "Item", "Question to ask before adopting a plane classifier"],
                   colWidths=[0.04, 0.22, 0.74], loc="upper left", cellLoc="left", colLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.3)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#d8d6cf")
        cell.set_height(0.118 if r and "\n" in rows[r - 1][2] else 0.075)
        cell.PAD = 0.03
        if r == 0:
            cell.set_text_props(weight="bold", color=INK)
            cell.set_facecolor("#efeee9")
    ax.set_title("Pre-adoption checklist for fetal-ultrasound plane classifiers", fontsize=8, loc="left")
    save_all(fig, "review_fig4_checklist")


def main() -> None:
    schematic()
    box1()
    saliency()
    checklist()
    print("wrote results/figures/review_fig{1..4}_*.pdf/.png")


if __name__ == "__main__":
    main()
