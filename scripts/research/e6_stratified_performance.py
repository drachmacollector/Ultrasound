"""
scripts/research/e6_stratified_performance.py  (R3 E6: stratified in-distribution performance, P1c)

Per machine (Aloka, Voluson E6, Voluson S10, Other machine), for all six frozen checkpoints:
macro-F1, balanced accuracy, Other recall, standard-plane false-rejection rate (standard plane
predicted Other) and confusion matrices. P1c contrasts (Other recall and false rejection, S10 − E6,
S10 − Aloka, Aloka − E6) use one *joint* patient-cluster bootstrap over the whole test set, so
patients with images from several machines are handled. Supplementary: GEE logit
correct_fine ~ machine + true class, patient clusters (exchangeable), odds ratios for machine.

Overlay strata are not reported: Gate G1 did not validate the caliper detector, and UI text is
present on 100% of FPDB images (no contrast to stratify on).

Outputs: results/json/E6.json, results/tables/E6.md,
         results/figures/E6_other_recall_by_machine.{pdf,png}, E6_confusions_convnext.{pdf,png}
Usage:   python scripts/research/e6_stratified_performance.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.audit.plotting import SERIES, plt, save  # noqa: E402
from src.data.dataset import CANONICAL_CLASSES  # noqa: E402
from src.eval.predict import PREDS_ROOT  # noqa: E402
from src.eval.scoring import bal_acc_8, by_stratum, estimate_json, macro_f1_8  # noqa: E402
from src.eval.stats import SumsMetric, cluster_bootstrap, safe_div  # noqa: E402

BACKBONES = {"convnext_tiny": "convnext", "tf_efficientnetv2_s": "effv2s", "repvgg_a1": "repvgga1",
             "repvgg_a2": "repvgga2", "mobilenetv3_large_100": "mnv3l", "efficientnet_lite0": "efflite0"}
MACHINES = ["Aloka", "Voluson E6", "Voluson S10", "Other"]
KEY = {"Aloka": "aloka", "Voluson E6": "e6", "Voluson S10": "s10", "Other": "othm"}
CONTRASTS = [("Voluson S10", "Voluson E6"), ("Voluson S10", "Aloka"), ("Aloka", "Voluson E6")]
OUT_JSON = Path("results/json/E6.json")
OUT_MD = Path("results/tables/E6.md")


def load(bb: str) -> pd.DataFrame:
    f = pd.read_parquet(PREDS_ROOT / f"legacy_{BACKBONES[bb]}_s42" / "fpdb_test.parquet")
    for c in f.select_dtypes("category").columns:
        f[c] = f[c].astype(str)
    is_o = f["true_label"] == "Other"
    pred_o = f["pred_fine"] == "Other"
    for m, k in KEY.items():
        on = f["source_machine"] == m
        f[f"o_{k}"] = (on & is_o).astype(int)                 # Other images on machine
        f[f"oh_{k}"] = (on & is_o & pred_o).astype(int)       # ... recognised as Other
        f[f"s_{k}"] = (on & ~is_o).astype(int)                # standard-plane images on machine
        f[f"sr_{k}"] = (on & ~is_o & pred_o).astype(int)      # ... falsely rejected as Other
    return f


def rate_metric(name: str, num: str, den: str) -> SumsMetric:
    return SumsMetric(name, [num, den], lambda s: safe_div(s[num], s[den]))


def diff_metric(name: str, prefix_num: str, prefix_den: str, a: str, b: str) -> SumsMetric:
    na, da, nb, db = f"{prefix_num}_{KEY[a]}", f"{prefix_den}_{KEY[a]}", f"{prefix_num}_{KEY[b]}", f"{prefix_den}_{KEY[b]}"
    return SumsMetric(name, [na, da, nb, db], lambda s: safe_div(s[na], s[da]) - safe_div(s[nb], s[db]))


def per_machine(df: pd.DataFrame) -> dict:
    out = {}
    for m, k in KEY.items():
        out[m] = {
            "other_recall": cluster_bootstrap(df, rate_metric("other_recall", f"oh_{k}", f"o_{k}"), max_nan_frac=1.0),
            "false_rejection": cluster_bootstrap(df, rate_metric("false_rejection", f"sr_{k}", f"s_{k}"),
                                                 max_nan_frac=1.0),
            "n_other": int(df[f"o_{k}"].sum()), "n_standard": int(df[f"s_{k}"].sum()),
        }
    return out


def strata_scores(df: pd.DataFrame) -> dict:
    def sc(g: pd.DataFrame) -> dict:
        labels = [c for c in CANONICAL_CLASSES if c in set(g["true_label"])]
        cm = macro_f1_8(labels).cm(g)
        return {"macro_f1_8": cluster_bootstrap(g, macro_f1_8(labels), max_nan_frac=1.0),
                "bal_acc_8": cluster_bootstrap(g, bal_acc_8(labels), max_nan_frac=1.0),
                "n_images": len(g), "n_patients": int(g["patient_id"].nunique()),
                "confusion": cm.tolist()}
    return by_stratum(df, "source_machine", sc)


def contrasts(df: pd.DataFrame) -> dict:
    out = {}
    for a, b in CONTRASTS:
        tag = f"{KEY[a]}_minus_{KEY[b]}"
        out[f"other_recall_{tag}"] = cluster_bootstrap(df, diff_metric(tag, "oh", "o", a, b), max_nan_frac=0.01)
        out[f"false_rejection_{tag}"] = cluster_bootstrap(df, diff_metric(tag, "sr", "s", a, b), max_nan_frac=0.01)
    return out


def gee(df: pd.DataFrame) -> dict:
    """Odds ratios of being correct (8-way) by machine, adjusted for true class; reference = Voluson E6."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = df.assign(correct=df["correct_fine"].astype(int),
                  machine=df["source_machine"].replace({"Other": "Other machine"}), cls=df["true_label"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.gee("correct ~ C(machine, Treatment('Voluson E6')) + C(cls)", groups="patient_id", data=d,
                      family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = fit.conf_int()
    out = {}
    for name in fit.params.index:
        if name.startswith("C(machine"):
            lvl = name.split("[T.")[1].rstrip("]")
            out[lvl] = {"odds_ratio": float(np.exp(fit.params[name])), "lo": float(np.exp(ci.loc[name, 0])),
                        "hi": float(np.exp(ci.loc[name, 1])), "p": float(fit.pvalues[name])}
    return {"reference": "Voluson E6", "adjusted_for": "true class (8)", "cov_struct": "exchangeable",
            "odds_ratios": out}


def main() -> None:
    res: dict = {"models": {}}
    for bb in BACKBONES:
        df = load(bb)
        multi = df.groupby("patient_id")["source_machine"].nunique()
        res["n_patients_multi_machine"] = int((multi > 1).sum())
        res["models"][bb] = {"per_machine": per_machine(df), "strata": strata_scores(df),
                             "contrasts": contrasts(df), "gee": gee(df)}
        print(f"  {bb}", flush=True)
    OUT_JSON.write_text(json.dumps(estimate_json(res), indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(res), encoding="utf-8")
    figures(res)
    print(f"wrote {OUT_JSON}, {OUT_MD}")


def pct(e) -> str:
    return e.fmt(pct=True)


def markdown(res: dict) -> str:
    M = res["models"]
    s = ["# E6: stratified in-distribution performance (FPDB test)", "",
         "Six frozen legacy checkpoints (seed 42). 95% patient-clustered CIs (B = 2000). Contrasts use one joint "
         f"bootstrap over all test patients ({res['n_patients_multi_machine']} patients have images from more than one "
         "machine). Training prior (E1): Other share 65% on Aloka, 14% on Voluson E6, S10 almost absent from training.",
         "", "## 1. Other recall by machine (Other images correctly called Other)", "",
         "| model | " + " | ".join(f"{m} (n={M['convnext_tiny']['per_machine'][m]['n_other']})" for m in MACHINES) + " |",
         "|---|" + "---|" * len(MACHINES)]
    for bb, r in M.items():
        s.append(f"| `{bb}` | " + " | ".join(pct(r["per_machine"][m]["other_recall"]) for m in MACHINES) + " |")
    s += ["", "## 2. False rejection by machine (standard planes predicted Other)", "",
          "| model | " + " | ".join(f"{m} (n={M['convnext_tiny']['per_machine'][m]['n_standard']})" for m in MACHINES)
          + " |", "|---|" + "---|" * len(MACHINES)]
    for bb, r in M.items():
        s.append(f"| `{bb}` | " + " | ".join(pct(r["per_machine"][m]["false_rejection"]) for m in MACHINES) + " |")
    s += ["", "## 3. P1c contrasts (percentage points, joint cluster bootstrap)", "",
          "| model | " + " | ".join(f"Other recall {a.split()[-1]} − {b.split()[-1]}" for a, b in CONTRASTS) + " | "
          + " | ".join(f"false rejection {a.split()[-1]} − {b.split()[-1]}" for a, b in CONTRASTS) + " |",
          "|---|" + "---|" * (2 * len(CONTRASTS))]
    for bb, r in M.items():
        c = r["contrasts"]
        cells = [pct(c[f"other_recall_{KEY[a]}_minus_{KEY[b]}"]) for a, b in CONTRASTS]
        cells += [pct(c[f"false_rejection_{KEY[a]}_minus_{KEY[b]}"]) for a, b in CONTRASTS]
        s.append(f"| `{bb}` | " + " | ".join(cells) + " |")
    s += ["", "## 4. Macro-F1 / balanced accuracy by machine (classes present on the machine)", "",
          "| model | " + " | ".join(MACHINES) + " |", "|---|" + "---|" * len(MACHINES)]
    for bb, r in M.items():
        s.append(f"| `{bb}` | " + " | ".join(f"F1 {r['strata'][m]['macro_f1_8'].fmt()}<br>BA {r['strata'][m]['bal_acc_8'].fmt()}"
                                              for m in MACHINES) + " |")
    s += ["", "## 5. Adjusted odds of a correct 8-way prediction by machine (GEE logit, patient clusters; "
          "reference Voluson E6; adjusted for true class)", "",
          "| model | " + " | ".join(lv for lv in M["convnext_tiny"]["gee"]["odds_ratios"]) + " |",
          "|---|" + "---|" * len(M["convnext_tiny"]["gee"]["odds_ratios"])]
    for bb, r in M.items():
        o = r["gee"]["odds_ratios"]
        s.append(f"| `{bb}` | " + " | ".join(f"{v['odds_ratio']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]" for v in o.values())
                 + " |")
    s += ["", "Overlay strata (plan, conditional on G1): not run; the caliper detector failed validation and "
          "UI text is universal in FPDB.", ""]
    return "\n".join(s)


def figures(res: dict) -> None:
    M = res["models"]
    machines = ["Aloka", "Voluson E6", "Voluson S10"]
    models = list(M)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for j, (m, col) in enumerate(zip(machines, SERIES)):
        for i, bb in enumerate(models):
            e = M[bb]["per_machine"][m]["other_recall"]
            y = i + (j - 1) * 0.22
            ax.errorbar(e.point, y, xerr=[[e.point - e.lo], [e.hi - e.point]], fmt="o", color=col, ms=4,
                        elinewidth=1.2, capsize=0, label=m if i == 0 else None)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Other recall on FPDB test (95% patient-clustered CI)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7,
              title="test machine (share of 'Other' in training: Aloka 65%, Voluson E6 14%; S10: 11 training images)",
              title_fontsize=6.5)
    ax.set_title("'Other' is recognised on Aloka, missed on Voluson E6", loc="left")
    save(fig, "E6_other_recall_by_machine")

    short = [c.replace("Brain_Trans_", "BT-").replace("Fetal_", "").replace("Maternal_", "") for c in CANONICAL_CLASSES]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    for ax, m in zip(axes, MACHINES):
        cm = np.array(M["convnext_tiny"]["strata"][m]["confusion"], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(8):
            for j in range(8):
                if cm[i, j] > 0:
                    ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=5,
                            color="white" if norm[i, j] > 0.6 else "#2b2b2a")
        ax.set_xticks(range(8))
        ax.set_xticklabels(short, rotation=90, fontsize=6)
        ax.set_yticks(range(8))
        ax.set_yticklabels(short if m == "Aloka" else [], fontsize=6)
        ax.set_title(f"{m} (n={int(cm.sum())})", loc="left", fontsize=8)
        ax.set_xlabel("predicted", fontsize=7)
    axes[0].set_ylabel("true", fontsize=7)
    fig.suptitle("convnext_tiny confusion matrices by machine (row-normalised colour, counts printed)",
                 x=0.01, ha="left", fontsize=9)
    save(fig, "E6_confusions_convnext")


if __name__ == "__main__":
    main()
