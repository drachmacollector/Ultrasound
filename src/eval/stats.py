"""
src/eval/stats.py

Patient-clustered bootstrap: every reported number goes through this module.

* Resampling unit = cluster (``patient_id``, dataset-qualified). Each replicate draws K
  clusters with replacement (``rng.integers(0, K, K)``). All draws for a call are generated
  up front from ``np.random.default_rng(seed)``, so every metric evaluated with the same
  ``seed`` and cluster set sees the same resamples.
* Fast paths: ``CMMetric`` (confusion-matrix metrics) and ``RatioMetric`` (sum/sum ratios)
  aggregate per-cluster statistics and weight them by cluster draw counts. Any other callable
  ``metric(df) -> float`` uses the general path, which materialises each resample by
  repeating rows according to their cluster's draw count. The two paths give the same
  resampled multisets.
* CI = percentile interval. Paired p-value = ``2 * min(P(Δ* <= 0), P(Δ* >= 0))``, floored
  at 1/(B+1) and capped at 1.
* NaN replicates (a metric undefined in a resample) are counted and excluded from the
  percentiles; more than ``max_nan_frac`` (default 1%) raises.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.eval.metrics import confusion

B_DEFAULT = 2000
SEED_DEFAULT = 20261109
ALPHA_DEFAULT = 0.05
MAX_NAN_FRAC = 0.01


# --------------------------------------------------------------------------- results
@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float
    n_images: int
    n_clusters: int
    method: str = "cluster_percentile"
    B: int = B_DEFAULT
    seed: int = SEED_DEFAULT
    n_nan: int = 0

    def fmt(self, pct: bool = False, dp: int = 3) -> str:
        """``0.893 [0.881, 0.904]`` (or ``89.3% [88.1, 90.4]`` with ``pct``)."""
        if np.isnan(self.point):
            return "n/a"
        if pct:
            d = max(dp - 2, 0)
            return f"{100 * self.point:.{d}f}% [{100 * self.lo:.{d}f}, {100 * self.hi:.{d}f}]"
        return f"{self.point:.{dp}f} [{self.lo:.{dp}f}, {self.hi:.{dp}f}]"

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class PairedEstimate:
    """``delta = metric(b) - metric(a)``."""
    delta: float
    lo: float
    hi: float
    p_two_sided: float
    point_a: float
    point_b: float
    n_images: int
    n_clusters: int
    method: str = "paired_cluster_percentile"
    B: int = B_DEFAULT
    seed: int = SEED_DEFAULT
    n_nan: int = 0

    def fmt(self, pct: bool = False, dp: int = 3) -> str:
        s = 100 if pct else 1
        d = max(dp - 2, 0) if pct else dp
        unit = " pp" if pct else ""
        return (f"{s * self.delta:+.{d}f}{unit} [{s * self.lo:+.{d}f}, {s * self.hi:+.{d}f}], "
                f"p={self.p_two_sided:.3g}")

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in asdict(self).items()}


# --------------------------------------------------------------------------- metrics
class CMMetric:
    """Metric computed from a confusion matrix of two label columns.

    ``classes`` fixes the index order; labels outside it raise. ``fn(cm) -> float``.
    """

    def __init__(self, name: str, true_col: str, pred_col: str, classes: Sequence[str],
                 fn: Callable[[np.ndarray], float]):
        self.name, self.true_col, self.pred_col, self.fn = name, true_col, pred_col, fn
        self.classes = list(classes)
        self._idx = {c: i for i, c in enumerate(self.classes)}

    def codes(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        try:
            t = df[self.true_col].astype(str).map(self._idx).to_numpy()
            p = df[self.pred_col].astype(str).map(self._idx).to_numpy()
            return t.astype(np.int64), p.astype(np.int64)
        except (TypeError, ValueError) as exc:  # NaN after map -> unknown label
            raise ValueError(f"{self.name}: label outside {self.classes}") from exc

    def cm(self, df: pd.DataFrame) -> np.ndarray:
        t, p = self.codes(df)
        return confusion(t, p, len(self.classes))

    def __call__(self, df: pd.DataFrame) -> float:
        return float(self.fn(self.cm(df)))


class RatioMetric:
    """``sum(num) / sum(den)`` over rows; ``den=None`` means a plain mean of ``num``.

    Columns are boolean or numeric. Use it for accuracy-like rates and conditional shares
    (e.g. share of errors predicted Other: num = error & pred Other, den = error).
    """

    def __init__(self, name: str, num: str, den: str | None = None):
        self.name, self.num, self.den = name, num, den

    def arrays(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        num = df[self.num].to_numpy(dtype=np.float64)
        den = np.ones_like(num) if self.den is None else df[self.den].to_numpy(dtype=np.float64)
        return num, den

    def __call__(self, df: pd.DataFrame) -> float:
        num, den = self.arrays(df)
        return float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")


Metric = CMMetric | RatioMetric | Callable[[pd.DataFrame], float]


# --------------------------------------------------------------------------- core
def _cluster_codes(df: pd.DataFrame, cluster_col: str) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(df[cluster_col].astype(str), sort=True)
    return codes.astype(np.int64), len(uniques)


def draw_weights(n_clusters: int, B: int = B_DEFAULT, seed: int = SEED_DEFAULT) -> np.ndarray:
    """(B, K) cluster multiplicities; replicate b draws ``rng.integers(0, K, K)``."""
    rng = np.random.default_rng(seed)
    w = np.empty((B, n_clusters), dtype=np.int64)
    for b in range(B):
        w[b] = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
    return w


def replicates(df: pd.DataFrame, metric: Metric, codes: np.ndarray, weights: np.ndarray,
               path: str = "auto") -> np.ndarray:
    """Metric value for each replicate (rows of ``weights``). ``path``: auto | fast | general."""
    K = weights.shape[1]
    fast = path == "fast" or (path == "auto" and isinstance(metric, CMMetric | RatioMetric))
    if fast and isinstance(metric, CMMetric):
        t, p = metric.codes(df)
        c = len(metric.classes)
        per = np.zeros((K, c, c), dtype=np.int64)
        np.add.at(per, (codes, t, p), 1)
        cms = np.einsum("bk,kij->bij", weights, per)
        return np.array([metric.fn(cm) for cm in cms], dtype=np.float64)
    if fast and isinstance(metric, RatioMetric):
        num, den = metric.arrays(df)
        sn = np.bincount(codes, weights=num, minlength=K)
        sd = np.bincount(codes, weights=den, minlength=K)
        tn, td = weights @ sn, weights @ sd
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(td > 0, tn / td, np.nan)
    if fast:
        raise TypeError(f"no fast path for {type(metric).__name__}")
    out = np.empty(weights.shape[0], dtype=np.float64)
    for b, w in enumerate(weights):
        out[b] = metric(df.iloc[np.repeat(np.arange(len(df)), w[codes])])
    return out


def _check_nan(reps: np.ndarray, max_nan_frac: float, label: str) -> int:
    n_nan = int(np.isnan(reps).sum())
    if n_nan > max_nan_frac * reps.size:
        raise ValueError(f"{label}: {n_nan}/{reps.size} NaN bootstrap replicates (> {max_nan_frac:.0%})")
    return n_nan


def _percentile(reps: np.ndarray, alpha: float) -> tuple[float, float]:
    ok = reps[~np.isnan(reps)]
    if ok.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _name(metric: Metric) -> str:
    return getattr(metric, "name", getattr(metric, "__name__", type(metric).__name__))


def cluster_bootstrap(df: pd.DataFrame, metric: Metric, cluster_col: str = "patient_id",
                      B: int = B_DEFAULT, seed: int = SEED_DEFAULT, alpha: float = ALPHA_DEFAULT,
                      max_nan_frac: float = MAX_NAN_FRAC, method: str = "cluster_percentile") -> Estimate:
    """Point estimate on ``df`` plus a percentile CI from B cluster resamples."""
    codes, K = _cluster_codes(df, cluster_col)
    point = float(metric(df))
    if len(df) == 0 or np.isnan(point):
        return Estimate(float("nan"), float("nan"), float("nan"), len(df), K, method, B, seed, 0)
    reps = replicates(df, metric, codes, draw_weights(K, B, seed))
    n_nan = _check_nan(reps, max_nan_frac, _name(metric))
    lo, hi = _percentile(reps, alpha)
    return Estimate(point, lo, hi, len(df), K, method, B, seed, n_nan)


def image_bootstrap(df: pd.DataFrame, metric: Metric, B: int = B_DEFAULT, seed: int = SEED_DEFAULT,
                    alpha: float = ALPHA_DEFAULT, max_nan_frac: float = MAX_NAN_FRAC) -> Estimate:
    """Legacy image-level bootstrap. ONLY for the 'CI correction' comparison column."""
    d = df.assign(_row=np.arange(len(df)).astype(str))
    return cluster_bootstrap(d, metric, "_row", B, seed, alpha, max_nan_frac, method="image_percentile")


def _align(dfs: Sequence[pd.DataFrame], key: str, cluster_col: str) -> list[pd.DataFrame]:
    """Sort every frame by ``key``; assert identical keys and cluster per key."""
    out = []
    for d in dfs:
        if d[key].duplicated().any():
            raise ValueError(f"duplicate {key} values")
        out.append(d.sort_values(key, kind="stable").reset_index(drop=True))
    ref = out[0]
    for d in out[1:]:
        if len(d) != len(ref) or not np.array_equal(d[key].to_numpy(), ref[key].to_numpy()):
            raise ValueError(f"paired frames have different {key} sets")
        if not np.array_equal(d[cluster_col].astype(str).to_numpy(), ref[cluster_col].astype(str).to_numpy()):
            raise ValueError(f"paired frames disagree on {cluster_col} for some {key}")
    return out


def _p_two_sided(deltas: np.ndarray, B: int) -> float:
    ok = deltas[~np.isnan(deltas)]
    p = 2 * min(np.mean(ok <= 0), np.mean(ok >= 0))
    return float(min(1.0, max(p, 1.0 / (B + 1))))


def paired_multi_seed(dfs_a: Sequence[pd.DataFrame], dfs_b: Sequence[pd.DataFrame], metric: Metric,
                      key: str = "image_id", cluster_col: str = "patient_id", B: int = B_DEFAULT,
                      seed: int = SEED_DEFAULT, alpha: float = ALPHA_DEFAULT,
                      max_nan_frac: float = MAX_NAN_FRAC) -> PairedEstimate:
    """Δ = mean over seeds of metric(b) − mean over seeds of metric(a); one cluster draw per replicate."""
    aligned = _align(list(dfs_a) + list(dfs_b), key, cluster_col)
    A_, B_ = aligned[: len(dfs_a)], aligned[len(dfs_a):]
    codes, K = _cluster_codes(aligned[0], cluster_col)
    w = draw_weights(K, B, seed)
    pa = float(np.mean([metric(d) for d in A_]))
    pb = float(np.mean([metric(d) for d in B_]))
    ra = np.mean([replicates(d, metric, codes, w) for d in A_], axis=0)
    rb = np.mean([replicates(d, metric, codes, w) for d in B_], axis=0)
    deltas = rb - ra
    n_nan = _check_nan(deltas, max_nan_frac, _name(metric))
    lo, hi = _percentile(deltas, alpha)
    method = "paired_cluster_percentile" if len(A_) == len(B_) == 1 else "paired_multi_seed_cluster_percentile"
    return PairedEstimate(pb - pa, lo, hi, _p_two_sided(deltas, B), pa, pb, len(aligned[0]), K,
                          method, B, seed, n_nan)


def paired_cluster_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: Metric, key: str = "image_id",
                             cluster_col: str = "patient_id", B: int = B_DEFAULT, seed: int = SEED_DEFAULT,
                             alpha: float = ALPHA_DEFAULT, max_nan_frac: float = MAX_NAN_FRAC) -> PairedEstimate:
    """Paired comparison of two models scored on the same images (Δ = b − a)."""
    return paired_multi_seed([df_a], [df_b], metric, key, cluster_col, B, seed, alpha, max_nan_frac)


def multi_seed_summary(dfs: Sequence[pd.DataFrame], metric: Metric, key: str = "image_id",
                       cluster_col: str = "patient_id", B: int = B_DEFAULT, seed: int = SEED_DEFAULT,
                       alpha: float = ALPHA_DEFAULT, max_nan_frac: float = MAX_NAN_FRAC) -> Estimate:
    """Mean of the per-seed metric; each replicate applies one cluster draw to every seed."""
    aligned = _align(dfs, key, cluster_col)
    codes, K = _cluster_codes(aligned[0], cluster_col)
    w = draw_weights(K, B, seed)
    point = float(np.mean([metric(d) for d in aligned]))
    reps = np.mean([replicates(d, metric, codes, w) for d in aligned], axis=0)
    n_nan = _check_nan(reps, max_nan_frac, _name(metric))
    lo, hi = _percentile(reps, alpha)
    return Estimate(point, lo, hi, len(aligned[0]), K, "multi_seed_cluster_percentile", B, seed, n_nan)


def wilson(k: int, n: int, alpha: float = ALPHA_DEFAULT) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (NaN for n = 0)."""
    from statsmodels.stats.proportion import proportion_confint

    if n <= 0:
        return float("nan"), float("nan")
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)
