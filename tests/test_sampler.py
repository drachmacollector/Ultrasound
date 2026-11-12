"""src/train/sampler.py: acquisition-balanced sampling hits its target joint distribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.train import sampler as S


def confounded(seed: int = 0) -> tuple[list[str], list[str]]:
    """Aloka mostly 'Other', E6 mostly 'Head', a small third group lacking one class."""
    rng = np.random.default_rng(seed)
    m, y = [], []
    for machine, n, probs in (("Aloka", 3000, [0.7, 0.1, 0.1, 0.1]), ("Voluson E6", 3000, [0.1, 0.6, 0.2, 0.1]),
                              ("Voluson S10", 150, [0.0, 0.4, 0.3, 0.3])):
        m += [machine] * n
        y += list(rng.choice(["Other", "Head", "Femur", "Abdomen"], size=n, p=probs))
    return m, y


def test_target_is_group_prior_times_uniform_class() -> None:
    m, y = confounded()
    plan = S.plan_weights(m, y)
    assert plan.target.sum().sum() == pytest.approx(1.0)
    om = plan.target.loc["Other_machine"]
    assert om["Other"] == 0.0                                    # empty cell gets no mass
    assert om["Head"] == pytest.approx((150 / 6150) / 3)          # K_m = 3 present classes
    assert plan.target.loc["Aloka", "Other"] == pytest.approx((3000 / 6150) / 4)


def test_realised_joint_within_1pp_of_target() -> None:
    m, y = confounded()
    plan = S.plan_weights(m, y)
    idx = list(iter(S.build_sampler(plan, seed=11, num_samples=200_000)))
    real = S.realised_joint(plan, idx)
    assert (real - plan.target).abs().max().max() < 0.01
    assert real.loc["Other_machine", "Other"] == 0.0             # empty cells never sampled
    assert S.check_realised(plan, real) < 2.0


def test_cap_is_applied() -> None:
    m = ["Aloka"] * 1000 + ["Voluson E6"] * 1000
    y = ["Other"] * 999 + ["Head"] + ["Head"] * 1000              # one Aloka-Head image -> huge raw weight
    plan = S.plan_weights(m, y, cap=10.0)
    assert plan.weights.max() == pytest.approx(10 * np.median(plan.weights))
    assert plan.clipped_share == pytest.approx(1 / 2000)


def test_small_group_is_pooled_and_unknown_machine_rejected() -> None:
    plan = S.plan_weights(["Aloka"] * 100 + ["Other"] * 5, ["Head", "Other"] * 50 + ["Head"] * 5)
    assert plan.pooled == {"Other_machine": "Aloka"}
    assert set(plan.groups) == {"Aloka"}
    with pytest.raises(ValueError):
        S.machine_group(["Philips"])


def test_check_realised_raises_on_large_gap() -> None:
    m, y = confounded()
    plan = S.plan_weights(m, y)
    bad = pd.DataFrame(plan.target.copy())
    bad.loc["Aloka"] = 0.0
    bad.loc["Aloka", "Other"] = plan.target.loc["Aloka"].sum()
    with pytest.raises(AssertionError):
        S.check_realised(plan, bad)


def test_sampler_is_deterministic() -> None:
    m, y = confounded()
    plan = S.plan_weights(m, y)
    a = list(iter(S.build_sampler(plan, seed=3, num_samples=500)))
    b = list(iter(S.build_sampler(plan, seed=3, num_samples=500)))
    assert a == b


def test_realised_gap_matches_check_and_does_not_raise() -> None:
    m, y = confounded()
    plan = S.plan_weights(m, y)
    bad = pd.DataFrame(plan.target.copy())
    bad.loc["Aloka"] = 0.0
    bad.loc["Aloka", "Other"] = plan.target.loc["Aloka"].sum()
    assert S.realised_gap(plan, bad) == pytest.approx(75.0)      # Other share 100% vs target 25%
    assert S.realised_gap(plan, plan.target) == 0.0
