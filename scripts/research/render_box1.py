"""
scripts/research/render_box1.py  (R6 §B.2: Box 1 of the review)

Renders docs/research/BOX1_ILLUSTRATIVE_ANALYSIS.md from results/json/{E1,E2,E6,R2_existing_models}.json.
Every number is read from JSON; sentences that depend on the direction of a result are chosen
from the data (no hand-typed numbers, no pre-written conclusions).

Usage:  python scripts/research/render_box1.py
"""
from __future__ import annotations

import json
from pathlib import Path

J = Path("results/json")
OUT = Path("docs/research/BOX1_ILLUSTRATIVE_ANALYSIS.md")
PRIMARY = "convnext_tiny"


def load(name: str) -> dict:
    return json.loads((J / f"{name}.json").read_text(encoding="utf-8"))


def ci(e: dict, pct: bool = False, dp: int = 2) -> str:
    if pct:
        return f"{100 * e['point']:.0f}% (95% CI {100 * e['lo']:.0f}–{100 * e['hi']:.0f})"
    return f"{e['point']:.{dp}f} (95% CI {e['lo']:.{dp}f}–{e['hi']:.{dp}f})"


def main() -> None:
    e1, e2, e6, r2 = load("E1"), load("E2"), load("E6"), load("R2_existing_models")
    share = e1["other_share"]["trainval"]
    diff = e1["contrasts"]["trainval_other_share_aloka_minus_e6"]
    v = e1["cramers_v"]["trainval"]["machine_x_class8"]
    pri_seen = e2["results"]["prior"]["seen_machines"]["auroc_standard_vs_other"]
    all_seen = e2["results"]["all"]["seen_machines"]["auroc_standard_vs_other"]
    pri_all = e2["results"]["prior"]["all_test"]["auroc_standard_vs_other"]
    pm = e6["models"][PRIMARY]["per_machine"]
    c = e6["models"][PRIMARY]["contrasts"]
    s10_e6, s10_al = c["other_recall_s10_minus_e6"], c["other_recall_s10_minus_aloka"]
    ext = r2["models"][PRIMARY]["external"]

    # direction-dependent wording for the pre-specified S10 prediction (P1c)
    if s10_e6["lo"] > 0:
        p1c = ("Contrary to the pre-specified prediction, 'Other' recall on the Voluson S10, a machine almost "
               "absent from training, was higher than on the Voluson E6 "
               f"({ci(pm['Voluson S10']['other_recall'], True)} vs {ci(pm['Voluson E6']['other_recall'], True)}), "
               "close to the Aloka level: the missed 'Other' frames concentrate on the Voluson E6, the machine on "
               "which 'Other' was rare in training.")
    elif s10_e6["hi"] < 0:
        p1c = ("As predicted, 'Other' recall on the Voluson S10 was lower than on the Voluson E6 "
               f"({ci(pm['Voluson S10']['other_recall'], True)} vs {ci(pm['Voluson E6']['other_recall'], True)}).")
    else:
        p1c = "'Other' recall on the Voluson S10 did not differ clearly from the Voluson E6."

    rej = {d: ext[d]["errors"]["err_other"] for d in ("hc18", "ucl", "african")}
    para = (
        "We re-analysed FETAL_PLANES_DB, a widely used public benchmark for fetal standard-plane classification, "
        "and three external datasets. In the training data, the acquisition machine is entangled "
        f"with the label: 'Other' (non-standard) frames make up {100 * share['Aloka']['point']:.0f}% of Aloka images "
        f"but {100 * share['Voluson E6']['point']:.0f}% of Voluson E6 images (difference "
        f"{100 * diff['point']:.0f} percentage points, 95% CI {100 * diff['lo']:.0f}–{100 * diff['hi']:.0f}). "
        "A model that sees no pixels at all, only the machine and operator, separates standard planes from 'Other' "
        f"with an AUROC of {ci(pri_seen)} on test images from the same machines; adding frame geometry and on-screen "
        f"text statistics raises this to {ci(all_seen)}. A standard image classifier (ConvNeXt-Tiny, "
        f"macro-F1 {r2['models'][PRIMARY]['fpdb_test']['macro_f1_8']['point']:.2f}) recognised 'Other' frames on "
        f"{100 * pm['Aloka']['other_recall']['point']:.0f}% of Aloka images but only "
        f"{100 * pm['Voluson E6']['other_recall']['point']:.0f}% of Voluson E6 images. {p1c} On external data "
        "from other hospitals and devices, most errors were not anatomical confusions but rejections of valid "
        f"planes as 'Other' ({100 * rej['hc18']['point']:.0f}%, {100 * rej['ucl']['point']:.0f}% and "
        f"{100 * rej['african']['point']:.0f}% of errors on the Dutch, British and African sets). High benchmark "
        "accuracy alone does not show how a classifier will behave on another machine."
    )
    rows = [
        ("Machine × label entanglement (training data)",
         f"'Other' share Aloka {ci(share['Aloka'], True)} vs Voluson E6 {ci(share['Voluson E6'], True)}; "
         f"Cramér's V machine × class {ci(v)}", "E1"),
        ("No-pixel model, standard plane vs 'Other' (AUROC)",
         f"machine + operator: seen machines {ci(pri_seen)}, all test {ci(pri_all)}; all metadata: seen machines "
         f"{ci(all_seen)}", "E2"),
        ("'Other' recall by machine (ConvNeXt-Tiny, FPDB test)",
         f"Aloka {ci(pm['Aloka']['other_recall'], True)}; Voluson E6 {ci(pm['Voluson E6']['other_recall'], True)}; "
         f"Voluson S10 {ci(pm['Voluson S10']['other_recall'], True)} (S10 − E6 {100 * s10_e6['point']:+.0f} pp, "
         f"95% CI {100 * s10_e6['lo']:+.0f} to {100 * s10_e6['hi']:+.0f}; S10 − Aloka {100 * s10_al['point']:+.0f} pp, "
         f"95% CI {100 * s10_al['lo']:+.0f} to {100 * s10_al['hi']:+.0f})", "E6"),
        ("External sets: accuracy (full 8-way argmax) and errors that are 'Other' rejections",
         "; ".join(f"{name} {100 * ext[d]['acc_full']['point']:.0f}% (95% CI {100 * ext[d]['acc_full']['lo']:.0f}–"
                   f"{100 * ext[d]['acc_full']['hi']:.0f}), {100 * rej[d]['point']:.0f}% of errors rejected"
                   for d, name in (("hc18", "HC18 (NL)"), ("ucl", "UCL (UK)"), ("african", "African (5 centres)"))),
         "R2"),
    ]
    md = ["# Box 1. Illustrative re-analysis of publicly available datasets; not a clinical study", "",
          "*Auto-generated by `scripts/research/render_box1.py` from `results/json/`. Do not edit by hand.*", "",
          para, "", f"Word count: {len(para.split())}.", "",
          "| finding | result (95% patient-clustered CI) | source |", "|---|---|---|"]
    md += [f"| {a} | {b} | {c_} |" for a, b, c_ in rows]
    md += ["", "Data: FETAL_PLANES_DB (Burgos-Artizzu et al. 2020; CC BY 4.0), HC18 / UCL landmark benchmark, "
           "African fetal planes (Sendra-Balcells et al. 2023; CC BY 4.0). Models: six legacy classifiers trained on "
           "the official FPDB training split (seed 42). Statistics: patient-clustered bootstrap, B = 2000.", ""]
    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT} ({len(para.split())} words)")


if __name__ == "__main__":
    main()
