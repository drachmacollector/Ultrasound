"""src/eval/stats.py: patient-clustered and paired bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval import metrics as M
from src.eval.stats import (
    CMMetric,
    RatioMetric,
    _cluster_codes,
    cluster_bootstrap,
    draw_weights,
    image_bootstrap,
    multi_seed_summary,
    paired_cluster_bootstrap,
    replicates,
    wilson,
)

ACC = RatioMetric("acc", "correct")


def clustered(n_clusters: int = 20, per: int = 10) -> pd.DataFrame:
    """Correctness constant within each cluster; half the clusters all-correct."""
    rows = [{"image_id": f"c{c}_{i}", "patient_id": f"p{c}", "correct": int(c % 2 == 0)}
            for c in range(n_clusters) for i in range(per)]
    return pd.DataFrame(rows)


def test_cluster_ci_wider_than_image_ci() -> None:
    df = clustered()
    cl = cluster_bootstrap(df, ACC)
    im = image_bootstrap(df, ACC)
    assert cl.point == im.point == 0.5
    assert (cl.hi - cl.lo) >= 2 * (im.hi - im.lo)
    assert (cl.n_clusters, cl.n_images, im.n_clusters) == (20, 200, 200)
    assert cl.method == "cluster_percentile" and im.method == "image_percentile"


def test_fast_and_general_paths_identical() -> None:
    df = clustered()
    rng = np.random.default_rng(1)
    df["correct"] = (rng.random(len(df)) < 0.7).astype(int)
    codes, K = _cluster_codes(df, "patient_id")
    w = draw_weights(K, B=300, seed=5)
    fast = replicates(df, ACC, codes, w, path="fast")
    general = replicates(df, ACC, codes, w, path="general")
    assert np.allclose(fast, general)
    # the same holds for a confusion-matrix metric
    df["t"] = rng.choice(["a", "b"], len(df))
    df["p"] = np.where(df["correct"] == 1, df["t"], rng.choice(["a", "b"], len(df)))
    cmm = CMMetric("bal", "t", "p", ["a", "b"], M.balanced_accuracy)
    assert np.allclose(replicates(df, cmm, codes, w, path="fast"), replicates(df, cmm, codes, w, path="general"))


def test_same_seed_same_result() -> None:
    df = clustered()
    assert cluster_bootstrap(df, ACC) == cluster_bootstrap(df, ACC)
    assert cluster_bootstrap(df, ACC, seed=1) != cluster_bootstrap(df, ACC, seed=2)


def test_paired_self_is_zero() -> None:
    df = clustered()
    pe = paired_cluster_bootstrap(df, df.copy(), ACC)
    assert (pe.delta, pe.lo, pe.hi, pe.p_two_sided) == (0.0, 0.0, 0.0, 1.0)


def test_paired_detects_degradation() -> None:
    rng = np.random.default_rng(3)
    a = clustered(60, 10)
    a["correct"] = (rng.random(len(a)) < 0.8).astype(int)
    b = a.copy()
    flip = rng.choice(len(b), size=len(b) // 10, replace=False)
    b.loc[flip, "correct"] = 0
    pe = paired_cluster_bootstrap(a, b, ACC)
    assert pe.delta < 0 and pe.hi < 0 and pe.p_two_sided < 0.05
    assert pe.point_b < pe.point_a


def test_paired_alignment_is_checked() -> None:
    a = clustered()
    with pytest.raises(ValueError, match="image_id"):
        paired_cluster_bootstrap(a, a.iloc[1:], ACC)
    b = a.copy()
    b.loc[0, "patient_id"] = "other"
    with pytest.raises(ValueError, match="patient_id"):
        paired_cluster_bootstrap(a, b, ACC)


def test_paired_order_independent() -> None:
    a = clustered()
    b = a.sample(frac=1.0, random_state=0)
    assert paired_cluster_bootstrap(a, b, ACC).delta == 0.0


def test_multi_seed_of_identical_runs_equals_single() -> None:
    df = clustered()
    single = cluster_bootstrap(df, ACC)
    multi = multi_seed_summary([df, df.copy(), df.copy()], ACC)
    assert (multi.point, multi.lo, multi.hi) == pytest.approx((single.point, single.lo, single.hi))


def test_nan_replicates_raise() -> None:
    """Recall of a class present in only one of 20 clusters is undefined in ~36% of resamples."""
    df = clustered()
    df["is_x"] = (df["patient_id"] == "p0").astype(int)
    df["hit_x"] = df["is_x"]
    with pytest.raises(ValueError, match="NaN"):
        cluster_bootstrap(df, RatioMetric("recall_x", "hit_x", "is_x"))
    est = cluster_bootstrap(df, RatioMetric("recall_x", "hit_x", "is_x"), max_nan_frac=1.0)
    assert est.n_nan > 0 and est.point == 1.0


def test_wilson_closed_form() -> None:
    """k = 0: lower bound 0, upper bound z^2 / (n + z^2)."""
    z = 1.959963984540054
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(z * z / (10 + z * z))


def test_estimate_format() -> None:
    est = cluster_bootstrap(clustered(), ACC)
    assert est.fmt().startswith("0.500 [")
    assert est.fmt(pct=True).startswith("50.0% [")
    assert est.to_dict()["B"] == 2000 and est.to_dict()["seed"] == 20261109


def test_sums_metric_fast_equals_general() -> None:
    """Difference of two group rates computed in the same resample."""
    from src.eval.stats import SumsMetric, safe_div

    df = clustered(40, 5)
    rng = np.random.default_rng(7)
    df["g"] = (rng.random(len(df)) < 0.5).astype(int)
    df["hit_a"] = df["correct"] * df["g"]
    df["hit_b"] = df["correct"] * (1 - df["g"])
    df["n_b"] = 1 - df["g"]
    met = SumsMetric("diff", ["hit_a", "g", "hit_b", "n_b"],
                     lambda s: safe_div(s["hit_a"], s["g"]) - safe_div(s["hit_b"], s["n_b"]))
    codes, K = _cluster_codes(df, "patient_id")
    w = draw_weights(K, B=200, seed=1)
    assert np.allclose(replicates(df, met, codes, w, path="fast"), replicates(df, met, codes, w, path="general"))
    a = df[df.g == 1].correct.mean() - df[df.g == 0].correct.mean()
    assert met(df) == pytest.approx(a)


def test_contingency_metric_with_different_axes() -> None:
    df = pd.DataFrame({"patient_id": [f"p{i}" for i in range(6)], "m": ["x", "x", "y", "y", "y", "x"],
                       "c": ["A", "B", "C", "C", "A", "A"]})
    met = CMMetric("tab", "m", "c", ["x", "y"], lambda t: t.sum(), pred_classes=["A", "B", "C"])
    assert met.cm(df).tolist() == [[2, 1, 0], [1, 0, 2]]
    with pytest.raises(ValueError, match="label outside"):
        CMMetric("tab", "m", "c", ["x"], lambda t: 0.0, pred_classes=["A", "B", "C"]).cm(df)
