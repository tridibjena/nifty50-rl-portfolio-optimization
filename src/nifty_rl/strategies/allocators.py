"""Portfolio-construction baselines.

The notebook benchmarked a continuous *allocator* (PPO emitting softmax weights) only
against binary *timing* rules -- RSI, MA crossover, breakout. Those answer a different
question. The natural opponents for a weight-emitting agent are weight-emitting methods,
and their absence was the single most conspicuous gap in the evaluation.

Every allocator here takes a trailing window of returns and emits long-only weights that
sum to one. Estimation never sees beyond the rebalance date.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

AllocatorFn = Callable[[pd.DataFrame], pd.Series]


def _clean(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").fillna(0.0)


def _normalise(weights: np.ndarray, columns) -> pd.Series:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        weights = np.ones(len(columns)) / len(columns)
    else:
        weights = weights / total
    return pd.Series(weights, index=columns, name="weight")


def _covariance(returns: pd.DataFrame, shrinkage: bool = True) -> np.ndarray:
    """Sample covariance, optionally Ledoit-Wolf shrunk.

    Shrinkage matters here: with 10 assets and a 60-day window the sample covariance is
    badly conditioned, and min-variance optimisers famously concentrate into whichever
    asset the noise happened to favour.
    """
    if shrinkage and len(returns) > len(returns.columns):
        try:
            from sklearn.covariance import LedoitWolf

            return LedoitWolf().fit(returns.to_numpy()).covariance_
        except Exception:
            pass
    return np.cov(returns.to_numpy(), rowvar=False)


# ------------------------------------------------------------------- simple rules


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    cols = _clean(returns).columns
    return _normalise(np.ones(len(cols)), cols)


def inverse_volatility(returns: pd.DataFrame) -> pd.Series:
    r = _clean(returns)
    vol = r.std().replace(0, np.nan)
    inv = (1.0 / vol).fillna(0.0)
    return _normalise(inv.to_numpy(), r.columns)


def momentum_tilted(returns: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Equal weight tilted toward positive trailing momentum; negatives get zero."""
    r = _clean(returns).tail(lookback)
    cumulative = (1.0 + r).prod() - 1.0
    tilt = cumulative.clip(lower=0.0)
    if tilt.sum() <= 0:
        return equal_weight(returns)
    return _normalise(tilt.to_numpy(), r.columns)


# ------------------------------------------------------------------- optimisers


def minimum_variance(returns: pd.DataFrame, shrinkage: bool = True) -> pd.Series:
    """Long-only minimum-variance portfolio."""
    r = _clean(returns)
    n = r.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    cov = _covariance(r, shrinkage)

    def objective(w):
        return float(w @ cov @ w)

    result = minimize(
        objective,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return _normalise(result.x if result.success else np.ones(n) / n, r.columns)


def maximum_sharpe(
    returns: pd.DataFrame, risk_free_daily: float = 0.0, shrinkage: bool = True
) -> pd.Series:
    """Long-only tangency portfolio on the trailing window."""
    r = _clean(returns)
    n = r.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    cov = _covariance(r, shrinkage)
    mu = r.mean().to_numpy() - risk_free_daily

    def negative_sharpe(w):
        vol = np.sqrt(max(w @ cov @ w, 1e-18))
        return float(-(w @ mu) / vol)

    result = minimize(
        negative_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return _normalise(result.x if result.success else np.ones(n) / n, r.columns)


def risk_parity(returns: pd.DataFrame, shrinkage: bool = True, iterations: int = 500) -> pd.Series:
    """Equal risk contribution: every asset supplies the same share of portfolio variance.

    Solved by the standard fixed-point iteration ``w_i <- w_i * (target / RC_i)`` rather
    than a general optimiser -- it is faster and cannot wander into a local minimum.
    """
    r = _clean(returns)
    n = r.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    cov = _covariance(r, shrinkage)

    w = np.ones(n) / n
    for _ in range(iterations):
        # Risk contribution of asset i = w_i * (Cov @ w)_i. These sum to portfolio
        # variance, so an equal split means each asset supplies `total / n` of the risk.
        marginal = cov @ w
        contribution = w * marginal
        total = contribution.sum()
        if total <= 0 or not np.isfinite(total):
            break
        target = total / n
        # Nudge each weight toward its target: an asset contributing more risk than its
        # share gets scaled down. The square root damps the step -- because changing w_i
        # also changes every other asset's contribution, the full correction overshoots
        # and oscillates instead of converging.
        w = w * (target / np.maximum(contribution, 1e-18)) ** 0.5
        w = np.clip(w, 1e-12, None)
        w = w / w.sum()
    return _normalise(w, r.columns)


# --------------------------------------------------- hierarchical risk parity (HRP)


def _inverse_variance_weights(cov: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Weight the assets in one cluster inversely to their variance, summing to 1."""
    sub = cov[np.ix_(indices, indices)]
    ivp = 1.0 / np.maximum(np.diag(sub), 1e-18)
    return ivp / ivp.sum()


def _cluster_variance(cov: np.ndarray, indices: np.ndarray) -> float:
    """Variance a cluster would have if held at inverse-variance weights.

    This is how HRP compares two candidate halves without inverting anything: it asks
    "how risky is this bundle if allocated sensibly inside itself?" and splits the
    parent's weight against the answer.
    """
    w = _inverse_variance_weights(cov, indices)
    sub = cov[np.ix_(indices, indices)]
    return float(w @ sub @ w)


def hierarchical_risk_parity(returns: pd.DataFrame) -> pd.Series:
    """Lopez de Prado's HRP.

    Three steps: tree clustering on a correlation distance, quasi-diagonalisation of the
    covariance matrix by the resulting leaf order, and recursive bisection allocating
    between sibling clusters by inverse cluster variance.

    HRP needs no matrix inversion, which is why it stays stable where min-variance
    concentrates -- the strongest of the classical baselines and the one PPO must beat
    for the project to claim anything.
    """
    r = _clean(returns)
    n = r.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return _normalise(np.ones(1), r.columns)

    cov = np.cov(r.to_numpy(), rowvar=False)
    corr = np.corrcoef(r.to_numpy(), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    # Step 1 -- tree clustering. Turn correlation into a distance so that highly
    # correlated names sit close together: perfectly correlated -> 0, uncorrelated ->
    # 0.71, perfectly opposed -> 1. Single-linkage clustering then orders the assets so
    # that similar ones are adjacent.
    distance = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)

    # Step 2 -- quasi-diagonalisation. `order` is the leaf order of the tree, e.g. the
    # two banks land beside each other rather than at opposite ends. Bisecting this
    # ordering therefore splits the portfolio along genuine similarity lines.
    order = leaves_list(linkage(condensed, method="single"))

    # Step 3 -- recursive bisection. Start with all assets in one cluster holding 100% of
    # the weight. Repeatedly cut each cluster in half and divide its weight between the
    # two halves in inverse proportion to their variance: the calmer half gets more. Each
    # asset's final weight is the product of every split it survived.
    weights = np.ones(n)
    clusters = [order]
    while clusters:
        # Split every multi-asset cluster down the middle. Single assets have nothing
        # left to divide, so they drop out and the loop ends when all clusters are.
        clusters = [
            half
            for cluster in clusters
            if len(cluster) > 1
            for half in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
        ]
        # Halves are appended in order, so consecutive pairs are always siblings.
        for left, right in zip(clusters[::2], clusters[1::2]):
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            # alpha is the share going to the left sibling. Note the inversion: a *high*
            # left variance makes the fraction large, so alpha shrinks and the riskier
            # half is scaled down. The two shares sum to 1, conserving parent weight.
            alpha = 1.0 - var_left / (var_left + var_right + 1e-18)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha

    return _normalise(weights, r.columns)


ALLOCATORS: Dict[str, AllocatorFn] = {
    "EqualWeight": equal_weight,
    "InverseVol": inverse_volatility,
    "MomentumTilt": momentum_tilted,
    "MinVariance": minimum_variance,
    "MaxSharpe": maximum_sharpe,
    "RiskParity": risk_parity,
    "HRP": hierarchical_risk_parity,
}
