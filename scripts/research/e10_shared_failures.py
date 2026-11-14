"""
scripts/research/e10_shared_failures.py  (R7 E10, POST-HOC: does input cleaning shrink the shared failure set?)

E8 found that the six legacy architectures fail on the same external images. Here, per R5/R7 arm, the "raters"
are the two backbones (convnext_tiny, tf_efficientnetv2_s) trained with the same arm and seed:

  * both_wrong        share of pooled-external images (HC18 + UCL + African) that both backbones get wrong
  * ratio             all-wrong ratio vs independence within dataset x true-class strata (E8 metric, m = 2)
  * B2 - B0 contrasts (paired, same images): both_wrong; 'rescued' = share of images both B0 models get wrong
    that both B2 models get right; 'new' = share of images both B2 models get wrong that both B0 models got right
"Wrong" = correct_full == False (collapsed full 8-way argmax), as in R2/E8.

Primary seed: 11 (R7 plan); seeds 22 and 33 are reported as a robustness check. 95% patient-cluster bootstrap CIs
(B = 2000, seed 20261109). Outputs: results/json/E10.json, results/tables/E10.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.eval import metrics as M  # noqa: E402
from src.eval.scoring import estimate_json  # noqa: E402
from src.eval.stats import cluster_bootstrap  # noqa: E402

PREDS = Path("results/preds")
ARMS = ("b0", "b1", "b2", "afp", "b2smp", "b2sty")
SHORT = ("convnext", "effv2s")
EXTERNAL = ("hc18", "ucl", "african")
SEEDS = (11, 22, 33)
PRIMARY_SEED = 11
OUT_JSON = Path("results/json/E10.json")
OUT_MD = Path("results/tables/E10.md")


def available(arm: str, seed: int) -> bool:
    return all((PREDS / f"{arm}_{s}_s{seed}" / f"{ds}.parquet").exists() for s in SHORT for ds in EXTERNAL)


def wrong_frame(arm: str, seed: int) -> pd.DataFrame:
    """One row per pooled-external image: w_convnext, w_effv2s (int8), patient_id, stratum."""
    out = None
    for s in SHORT:
        d = pd.concat([pd.read_parquet(PREDS / f"{arm}_{s}_s{seed}" / f"{ds}.parquet",
                                       columns=["image_id", "patient_id", "dataset", "true_collapsed", "correct_full"])
                       for ds in EXTERNAL], ignore_index=True)
        for c in ("image_id", "patient_id", "dataset", "true_collapsed"):
            d[c] = d[c].astype(str)
        d[f"w_{s}"] = (~d.pop("correct_full").astype(bool)).astype(np.int8)
        out = d if out is None else out.merge(d[["image_id", f"w_{s}"]], on="image_id", validate="1:1")
    out["stratum"] = out["dataset"] + "|" + out["true_collapsed"]
    out["both"] = (out["w_convnext"] & out["w_effv2s"]).astype(np.int8)
    return out.sort_values("image_id").reset_index(drop=True)


def both_wrong(d: pd.DataFrame) -> float:
    return float(d["both"].mean())


def ratio(d: pd.DataFrame) -> float:
    return M.all_wrong_ratio(d[["w_convnext", "w_effv2s"]].to_numpy(bool), d["stratum"].to_numpy())


def contrasts(b0: pd.DataFrame, arm: pd.DataFrame) -> dict:
    j = b0[["image_id", "patient_id", "both"]].merge(arm[["image_id", "both"]], on="image_id",
                                                     suffixes=("_b0", "_arm"), validate="1:1")
    diff = lambda d: float(d["both_arm"].mean() - d["both_b0"].mean())                       # noqa: E731
    rescued = lambda d: float((d["both_b0"].astype(bool) & ~d["both_arm"].astype(bool)).sum()   # noqa: E731
                              / max(int(d["both_b0"].sum()), 1))
    new = lambda d: float((d["both_arm"].astype(bool) & ~d["both_b0"].astype(bool)).sum()       # noqa: E731
                          / max(int(d["both_arm"].sum()), 1))
    return {"both_wrong_delta": estimate_json(cluster_bootstrap(j, diff)),
            "rescued_share_of_b0_shared": estimate_json(cluster_bootstrap(j, rescued)),
            "new_share_of_arm_shared": estimate_json(cluster_bootstrap(j, new))}


def analyse() -> dict:
    res: dict = {"status": "post-hoc exploratory (R7 E10; not pre-registered)", "primary_seed": PRIMARY_SEED,
                 "raters": ["convnext_tiny", "tf_efficientnetv2_s"], "by_seed": {}}
    for seed in SEEDS:
        if not available("b0", seed):
            continue
        b0 = wrong_frame("b0", seed)
        per = {}
        for arm in ARMS:
            if not available(arm, seed):
                continue
            d = b0 if arm == "b0" else wrong_frame(arm, seed)
            r = {"n_images": len(d), "both_wrong": estimate_json(cluster_bootstrap(d, both_wrong)),
                 "ratio_vs_independence": estimate_json(cluster_bootstrap(d, ratio, max_nan_frac=0.05))}
            if arm != "b0":
                r["vs_b0"] = contrasts(b0, d)
            per[arm] = r
        res["by_seed"][str(seed)] = per
    return res


def fmt(e: dict, dp: int = 3, pct: bool = False) -> str:
    k = 100 if pct else 1
    return f"{k * e['point']:.{dp}f} [{k * e['lo']:.{dp}f}, {k * e['hi']:.{dp}f}]"


def markdown(res: dict) -> str:
    L = ["# E10: shared external failures across the two backbones, per arm (post-hoc, exploratory)", "",
         "Pooled external (HC18 + UCL + African). 'Both wrong' = both backbones (same arm, same seed) wrong. "
         "Rescued = share of images both B0 models get wrong that both arm models get right. "
         "New = share of the arm's shared failures that both B0 models got right. 95% patient-cluster bootstrap CIs.", ""]
    for seed, per in res["by_seed"].items():
        tag = " (primary)" if int(seed) == res["primary_seed"] else " (robustness)"
        L += [f"## Seed {seed}{tag}", "",
              "| arm | both wrong (%) | ratio vs independence | Δ both wrong vs B0 (pp) | rescued (%) | new (%) |",
              "|---|---|---|---|---|---|"]
        for arm, r in per.items():
            c = r.get("vs_b0")
            extra = (f"{fmt(c['both_wrong_delta'], 1, True)} | {fmt(c['rescued_share_of_b0_shared'], 0, True)} | "
                     f"{fmt(c['new_share_of_arm_shared'], 0, True)}") if c else "– | – | –"
            L.append(f"| {arm} | {fmt(r['both_wrong'], 1, True)} | {fmt(r['ratio_vs_independence'], 1)} | {extra} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    res = analyse()
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(res), encoding="utf-8")
    print(markdown(res))


if __name__ == "__main__":
    main()
