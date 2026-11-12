"""
src/train/sampler.py  (R5.3, arm A3: acquisition-balanced sampling)

Target sampled joint distribution of machine group m and class y:

    q(m, y) = p(m) * 1 / K_m          (K_m = number of classes present in machine group m)

so machine and class are independent in what the model sees, while class balance matches the
B0 recipe's effective balance (B0 uses inverse-frequency class weights; AFP turns those off).

    w_i = p(m_i) * (1 / K_m) / p_hat(m_i, y_i)     p_hat with Laplace alpha = 1 on the cell counts
    w_i <- min(w_i, cap * median(w))                cap = 10

``WeightedRandomSampler(weights, num_samples=N, replacement=True, generator=seeded)``.

Machine groups: Aloka, Voluson E6, Other_machine (= Other + Voluson S10). A group with fewer
than ``MIN_GROUP`` images in total is pooled into the largest group (none in FPDB train). The
plan's alternative reading ("< 30 images in any class") would pool Other_machine (254 images,
11 'Other') and contradict the plan's own expected multipliers (Other_machine-Other 11 -> ~32 per
epoch), so the total-size rule is used; recorded in the pre-registration.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

MACHINE_GROUPS = {"Aloka": "Aloka", "Voluson E6": "Voluson E6", "Voluson S10": "Other_machine",
                  "Other": "Other_machine"}
MIN_GROUP = 30
ALPHA = 1.0
CAP = 10.0
TOLERANCE_PP = 5.0


def machine_group(machines: Sequence[str]) -> np.ndarray:
    unknown = set(machines) - set(MACHINE_GROUPS)
    if unknown:
        raise ValueError(f"unknown machines {unknown}")
    return np.array([MACHINE_GROUPS[m] for m in machines])


@dataclass
class SamplerPlan:
    weights: np.ndarray                 # per training row
    groups: np.ndarray                  # machine group per row
    labels: np.ndarray                  # class per row
    target: pd.DataFrame                # group x class target share of all draws
    expected: pd.DataFrame              # group x class expected share given the (capped) weights
    clipped_share: float
    pooled: dict = field(default_factory=dict)

    def report(self) -> dict:
        within_t = self.target.div(self.target.sum(axis=1), axis=0)
        within_e = self.expected.div(self.expected.sum(axis=1), axis=0)
        return {"target_joint": self.target.round(5).to_dict(), "expected_joint": self.expected.round(5).to_dict(),
                "max_abs_within_group_gap_pp": float(100 * (within_e - within_t).abs().max().max()),
                "clipped_share": self.clipped_share, "pooled_groups": self.pooled, "cap": CAP, "alpha": ALPHA,
                "group_sizes": pd.Series(self.groups).value_counts().to_dict()}


def plan_weights(machines: Sequence[str], labels: Sequence[str], alpha: float = ALPHA, cap: float = CAP,
                 min_group: int = MIN_GROUP) -> SamplerPlan:
    g = machine_group(machines)
    y = np.asarray(labels, dtype=object)
    sizes = pd.Series(g).value_counts()
    pooled = {}
    for grp, n in sizes.items():
        if n < min_group:
            pooled[grp] = sizes.idxmax()
    g = np.array([pooled.get(v, v) for v in g])
    n = len(g)
    counts = pd.crosstab(pd.Series(g, name="group"), pd.Series(y, name="class"))
    p_m = counts.sum(axis=1) / n
    k_m = (counts > 0).sum(axis=1)
    n_cells = int((counts > 0).sum().sum())
    target = pd.DataFrame(0.0, index=counts.index, columns=counts.columns)
    for m in counts.index:
        present = counts.columns[counts.loc[m] > 0]
        target.loc[m, present] = p_m[m] / k_m[m]
    p_hat = (counts + alpha) / (n + alpha * n_cells)
    w = np.array([target.at[gi, yi] / p_hat.at[gi, yi] for gi, yi in zip(g, y)])
    limit = cap * np.median(w)
    clipped = w > limit
    w = np.minimum(w, limit)
    df = pd.DataFrame({"group": g, "class": y, "w": w})
    expected = df.pivot_table(index="group", columns="class", values="w", aggfunc="sum", fill_value=0.0)
    expected = expected.reindex(index=target.index, columns=target.columns, fill_value=0.0) / w.sum()
    return SamplerPlan(w, g, y, target, expected, float(clipped.mean()), pooled)


def build_sampler(plan: SamplerPlan, seed: int, num_samples: int | None = None) -> WeightedRandomSampler:
    gen = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(plan.weights, dtype=torch.double),
                                 num_samples=num_samples or len(plan.weights), replacement=True, generator=gen)


def realised_joint(plan: SamplerPlan, indices: Sequence[int]) -> pd.DataFrame:
    idx = np.asarray(indices)
    t = pd.crosstab(pd.Series(plan.groups[idx], name="group"), pd.Series(plan.labels[idx], name="class"))
    return t.reindex(index=plan.target.index, columns=plan.target.columns, fill_value=0) / len(idx)


def realised_gap(plan: SamplerPlan, realised: pd.DataFrame) -> float:
    """Max |realised − target| share within machine group, in percentage points."""
    within_t = plan.target.div(plan.target.sum(axis=1), axis=0)
    within_r = realised.div(realised.sum(axis=1), axis=0)
    return float(100 * (within_r - within_t).abs().max().max())


def check_realised(plan: SamplerPlan, realised: pd.DataFrame, tol_pp: float = TOLERANCE_PP) -> float:
    """``realised_gap``; raises if above ``tol_pp``.

    Training checks the draws of epochs 1 and 2 pooled: the smallest group (Other_machine, ~254
    draws per epoch, ~32 per class) has a per-cell sampling SD of ~2 pp in one epoch, so a
    single-epoch check would fail by chance in roughly 1 run in 8; pooled over two epochs the
    false-alarm rate is < 1%.
    """
    gap = realised_gap(plan, realised)
    if gap > tol_pp:
        raise AssertionError(f"sampler realised joint deviates from target by {gap:.1f} pp (> {tol_pp})")
    return gap
