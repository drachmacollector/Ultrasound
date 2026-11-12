"""
scripts/research/r4_sanity_randomisation.py  (R4.5, Gate G3)

Cascading model-parameter randomisation test (Adebayo et al., NeurIPS 2018) for every saliency
method, on convnext_tiny and tf_efficientnetv2_s:

    stage 0 trained -> 1 head re-initialised -> 2 + last stage -> 3 + second-to-last stage
    convnext_tiny:        head | stages[3] | stages[2]
    tf_efficientnetv2_s:  conv_head + bn2 + classifier | blocks[5] | blocks[4]

Re-initialisation: each sub-module's own ``reset_parameters()`` (BatchNorm also resets its running
statistics), seed 0. ConvNeXt layer-scale ``gamma`` vectors are parameters of the block itself,
not of a sub-module, and keep their trained values (documented in GATE_G3.md).

Images: 200 FPDB-test images, 25 per fine class (seed 20261109); target = true class (Head = set).
Similarity of each randomised map to the trained map (positive part, each map scaled to [0, 1]
by its own maximum): SSIM (skimage, data_range 1 = the trained map's range), Spearman rank
correlation, and Pearson correlation of HOG features. A map that is all zero has no ranking;
its Spearman/HOG correlation is counted as 0 and the count is reported.

Pass rule (pre-declared, plan R4.5): median SSIM < 0.5 AND median Spearman < 0.5 at stage 2,
on both backbones. Running this on test images is allowed: it measures the method, not the model.

Outputs: results/json/G3.json (+ code_freeze_commit), results/tables/G3.md,
results/figures/G3_randomisation_cascade.pdf, docs/research/GATE_G3.md.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from skimage.feature import hog
from skimage.metrics import structural_similarity

sys.path.insert(0, ".")
from src.audit.plotting import INK, plt, save  # noqa: E402
from src.eval.predict import PREDS_ROOT, git_hash  # noqa: E402
from src.localisation import runner as R  # noqa: E402
from src.models.loading import load_model  # noqa: E402
from src.models.saliency import METHODS, SaliencyEngine, normalised_fill  # noqa: E402

BACKBONES = ("convnext_tiny", "tf_efficientnetv2_s")
STAGES = {
    "convnext_tiny": [("head", ["head"]), ("+stages[3]", ["stages.3"]), ("+stages[2]", ["stages.2"])],
    "tf_efficientnetv2_s": [("head", ["conv_head", "bn2", "classifier"]), ("+blocks[5]", ["blocks.5"]),
                            ("+blocks[4]", ["blocks.4"])],
}
PER_CLASS = 25
PASS_STAGE = 2
THRESH = 0.5
OUT_JSON = Path("results/json/G3.json")
OUT_MD = Path("results/tables/G3.md")
GATE_MD = Path("docs/research/GATE_G3.md")


def reinit(model: torch.nn.Module, names: list[str], seed: int = 0) -> None:
    torch.manual_seed(seed)
    for name in names:
        sub = model.get_submodule(name)
        for m in sub.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.reset_running_stats()
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


def sample(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(R.SEL_SEED)
    parts = [g.loc[rng.choice(g.index, PER_CLASS, replace=False)] for _, g in frame.groupby("true_label")]
    return pd.concat(parts).sort_values("image_id").reset_index(drop=True)


def load_batches(lm, frame: pd.DataFrame, num_workers: int) -> list[dict]:
    """The 200 inputs, loaded once per backbone and reused for every method and stage."""
    return list(R.threaded_batches(R.SaliencyInputs(frame, lm, R.load_landmarks()), 32, num_workers))


def maps_for(lm, model, method: str, frame: pd.DataFrame, batches: list[dict]) -> np.ndarray:
    true_t, _ = R.target_sets(frame)
    eng = SaliencyEngine(model, lm.backbone_name, method)
    out = []
    try:
        for b in batches:
            t = [true_t[i] for i in b["idx"].numpy()]
            fill = normalised_fill(b["fill_grey"].numpy(), lm.normalize_mean, lm.normalize_std)
            out.append(eng(b["x"], t, fill))
    finally:
        eng.close()
    return np.concatenate(out)


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.maximum(m.astype(np.float64), 0)
    mx = m.max()
    return m / mx if mx > 0 else m


def similarity(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = _unit(a), _unit(b)
    ssim = float(structural_similarity(a, b, data_range=1.0))
    flat = a.max() == 0 or b.max() == 0
    sp = 0.0 if flat else float(spearmanr(a.ravel(), b.ravel()).statistic)
    ha, hb = (hog(x, pixels_per_cell=(16, 16), cells_per_block=(2, 2)) for x in (a, b))
    hg = 0.0 if flat or ha.std() == 0 or hb.std() == 0 else float(pearsonr(ha, hb).statistic)
    return {"ssim": ssim, "spearman": sp, "hog": hg, "flat": flat}


def summarise(vals: list[dict]) -> dict:
    d = pd.DataFrame(vals)
    out = {"n": len(d), "n_flat": int(d["flat"].sum())}
    for k in ("ssim", "spearman", "hog"):
        q = np.percentile(d[k], [25, 50, 75])
        out[k] = {"median": float(q[1]), "q25": float(q[0]), "q75": float(q[2])}
    return out


def clean_head() -> str:
    h = git_hash()
    if h.endswith("-dirty") or h == "unknown":
        raise SystemExit(f"src/ has uncommitted changes ({h}); commit the frozen R4 code first")
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def run(num_workers: int, allow_dirty: bool) -> dict:
    commit = git_hash() if allow_dirty else clean_head()
    res: dict = {"code_freeze_commit": commit, "per_backbone": {}, "examples": {}}
    for bb in BACKBONES:
        lm = load_model(R.legacy_ckpt(bb))
        frame = sample(R.load_frame("fpdb_test", PREDS_ROOT / R.legacy_run_id(bb)))
        res["image_ids"] = frame["image_id"].tolist()
        batches = load_batches(lm, frame, num_workers)
        res["per_backbone"][bb] = {}
        for method in METHODS:
            t0 = time.perf_counter()
            model = copy.deepcopy(lm.model)
            ref = maps_for(lm, model, method, frame, batches)
            stages = [{"stage": 0, "name": "trained"}]
            ex = {"trained": ref[:3].astype(np.float16)}
            for s, (name, mods) in enumerate(STAGES[bb], start=1):
                reinit(model, mods, seed=0)
                cur = maps_for(lm, model, method, frame, batches)
                st = summarise([similarity(a, b) for a, b in zip(ref, cur)])
                stages.append({"stage": s, "name": name, **st})
                ex[name] = cur[:3].astype(np.float16)
            res["per_backbone"][bb][method] = stages
            np.savez_compressed(PREDS_ROOT / R.legacy_run_id(bb) / f"g3_examples_{method}.npz", **ex)
            del model
            torch.cuda.empty_cache()
            print(f"  {bb} {method}: {time.perf_counter() - t0:.0f}s  stage{PASS_STAGE} "
                  f"SSIM {stages[PASS_STAGE]['ssim']['median']:.3f} Spearman {stages[PASS_STAGE]['spearman']['median']:.3f}",
                  flush=True)
        del lm
    res["decision"] = {}
    for m in METHODS:
        per = {bb: res["per_backbone"][bb][m][PASS_STAGE] for bb in BACKBONES}
        ok = {bb: v["ssim"]["median"] < THRESH and v["spearman"]["median"] < THRESH for bb, v in per.items()}
        res["decision"][m] = {"pass": all(ok.values()), "per_backbone": ok}
    res["rule"] = (f"median SSIM < {THRESH} and median Spearman < {THRESH} at stage {PASS_STAGE} "
                   "(head + last stage re-initialised), on both backbones")
    return res


# ----------------------------------------------------------------------------- reporting
def table(res: dict) -> str:
    lines = ["| backbone | method | stage | SSIM median [IQR] | Spearman median [IQR] | HOG r median [IQR] | flat maps |",
             "|---|---|---|---|---|---|---|"]
    for bb, per in res["per_backbone"].items():
        for m, stages in per.items():
            for st in stages[1:]:
                f = lambda k: f"{st[k]['median']:.3f} [{st[k]['q25']:.3f}, {st[k]['q75']:.3f}]"  # noqa: E731
                lines.append(f"| `{bb}` | {m} | {st['stage']} {st['name']} | {f('ssim')} | {f('spearman')} | "
                             f"{f('hog')} | {st['n_flat']}/{st['n']} |")
    return "\n".join(lines)


def figure(res: dict) -> None:
    names = ["trained"] + [n for n, _ in STAGES["convnext_tiny"]]
    fig, axes = plt.subplots(len(METHODS), len(names), figsize=(1.5 * len(names), 1.6 * len(METHODS)))
    for r, m in enumerate(METHODS):
        with np.load(PREDS_ROOT / R.legacy_run_id("convnext_tiny") / f"g3_examples_{m}.npz") as z:
            for c, n in enumerate(names):
                ax = axes[r, c]
                ax.imshow(_unit(z[n][0].astype(np.float32)), cmap="inferno", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(n, fontsize=7, color=INK)
                if c == 0:
                    ax.set_ylabel(m, fontsize=7, color=INK)
    fig.suptitle("Cascading randomisation, convnext_tiny, one FPDB-test image (FPDB, Burgos-Artizzu et al. 2020)",
                 fontsize=7)
    fig.tight_layout()
    save(fig, "G3_randomisation_cascade")


def stage_plot(res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), sharey=True)
    colors = {"gradcam": "#2a78d6", "gradcampp": "#eb6834", "occlusion": "#1baf7a"}
    for ax, bb in zip(axes, BACKBONES):
        for m in METHODS:
            st = res["per_backbone"][bb][m]
            x = [s["stage"] for s in st[1:]]
            for k, ls in (("ssim", "-"), ("spearman", "--")):
                ax.plot(x, [s[k]["median"] for s in st[1:]], ls, color=colors[m], marker="o", ms=3,
                        label=f"{m} {k}" if bb == BACKBONES[0] else None, lw=1.2)
        ax.axhline(THRESH, color="#8a8984", lw=0.8, ls=":")
        ax.set_xticks([1, 2, 3], [n for n, _ in STAGES[bb]], fontsize=6)
        ax.set_title(bb, fontsize=8)
        ax.set_ylim(-0.1, 1.0)
    axes[0].set_ylabel("median similarity to trained map")
    fig.legend(loc="lower center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "G3_similarity_by_stage")


def report(res: dict) -> None:
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    t = table(res)
    OUT_MD.write_text("# G3: cascading model-parameter randomisation\n\n" + t + "\n", encoding="utf-8")
    figure(res)
    stage_plot(res)
    dec = "\n".join(f"| {m} | {'PASS' if d['pass'] else 'FAIL'} | "
                    + ", ".join(f"{bb}: {'pass' if v else 'fail'}" for bb, v in d["per_backbone"].items()) + " |"
                    for m, d in res["decision"].items())
    passed = [m for m, d in res["decision"].items() if d["pass"]]
    GATE_MD.write_text(f"""# Gate G3: saliency sanity check (model-parameter randomisation)

Auto-generated by `scripts/research/r4_sanity_randomisation.py` (numbers from `results/json/G3.json`).

**Code frozen at commit:** `{res['code_freeze_commit']}`. The R4 saliency, mask and metric code was developed
on the 238 FPDB-val landmark images only (`--dev`); no test-image saliency was computed before this commit.

**Test (Adebayo et al., 2018), cascading:** 200 FPDB-test images (25 per fine class), true-class target
(Head = set of the three brain sub-planes). Stages: 1 = classifier head re-initialised; 2 = + last stage;
3 = + second-to-last stage (convnext_tiny: `stages[3]`, `stages[2]`; tf_efficientnetv2_s: `blocks[5]`, `blocks[4]`).
Re-initialisation with each sub-module's `reset_parameters()` (BatchNorm running statistics reset too), seed 0.
ConvNeXt layer-scale `gamma` vectors keep their trained values (they belong to the block, not a sub-module).
Similarity of each map to the trained map: SSIM, Spearman rank correlation, HOG-feature Pearson; maps are
compared after taking the positive part and scaling each to [0, 1] by its maximum. Running the check on test
images measures the method, not the model, and is not used for any selection.

**Pre-declared pass rule:** {res['rule']}.

## Decision

| method | G3 | per backbone |
|---|---|---|
{dec}

Methods used for R4.6 claims and the R4.7 association: **{', '.join(passed) if passed else 'none'}**.

## Similarity to the trained map (median [IQR] over 200 images)

{t}

Figures: `results/figures/G3_similarity_by_stage.pdf`, `results/figures/G3_randomisation_cascade.pdf`.

**Notes.** HOG correlation is reported but is not part of the pass rule. For occlusion it stays high
(about 0.5–0.65) at every stage, because every occlusion map is built from the same 16 px (20 px) patch grid,
and HOG features pick up those block edges whatever the model. "Flat maps" are maps with no positive
energy; these randomised occlusion maps had no positive probability drop anywhere.
""", encoding="utf-8")
    print(f"wrote {OUT_JSON}, {GATE_MD}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--allow-dirty", action="store_true", help="development only: skip the clean-tree check")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    res = json.loads(OUT_JSON.read_text(encoding="utf-8")) if a.report_only else run(a.num_workers, a.allow_dirty)
    report(res)


if __name__ == "__main__":
    main()
