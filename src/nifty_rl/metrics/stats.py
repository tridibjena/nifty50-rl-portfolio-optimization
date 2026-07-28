"""Inferential statistics for backtests.

A backtest reports a point estimate from one sample of one history. These are the tools
that say how much of it to believe.

The search space in this project is large: SL/TP grids, PPO hyperparameter candidates,
validation-selected rule parameters, feature-set ablations and five regime backends. The
best observed Sharpe is therefore an *order statistic*, and comparing it to zero is the
wrong test. Deflated Sharpe Ratio corrects for exactly that, and it is the first thing a
quantitative reader will look for.

References
----------
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest Overfitting".
Politis & Romano (1994), "The Stationary Bootstrap".
White (2000), "A Reality Check for Data Snooping".
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


# ------------------------------------------------------------------ Sharpe tests


# See metrics.performance._DEGENERATE_VOL -- a series with no dispersion has an
# undefined Sharpe, not an enormous one.
_DEGENERATE_VOL = 1e-10


def sharpe_from_returns(
    returns: np.ndarray, trading_days: int = 252, risk_free_daily: float = 0.0
) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free_daily
    sd = np.std(excess, ddof=1)
    if not np.isfinite(sd) or sd <= _DEGENERATE_VOL:
        return float("nan")
    return float(np.mean(excess) / sd * np.sqrt(trading_days))


def probabilistic_sharpe_ratio(
    returns: Sequence[float],
    benchmark_sharpe: float = 0.0,
    trading_days: int = 252,
) -> float:
    """P(true Sharpe > benchmark), correcting for skew and kurtosis.

    Daily returns are neither normal nor independent. Negative skew and fat tails both
    inflate the naive Sharpe's apparent precision, and this adjusts for both.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3 or np.std(r, ddof=1) == 0:
        return float("nan")

    sharpe_daily = np.mean(r) / np.std(r, ddof=1)
    benchmark_daily = benchmark_sharpe / np.sqrt(trading_days)
    skew = float(stats.skew(r))
    kurtosis = float(stats.kurtosis(r, fisher=False))

    denominator = 1.0 - skew * sharpe_daily + ((kurtosis - 1.0) / 4.0) * sharpe_daily ** 2
    if denominator <= 0:
        return float("nan")

    z = (sharpe_daily - benchmark_daily) * np.sqrt(n - 1) / np.sqrt(denominator)
    return float(stats.norm.cdf(z))


def expected_maximum_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe under the null that every trial has true Sharpe zero.

    This is the bar the *best* strategy must clear. Testing the winner of 50 trials
    against zero instead of against this is the core data-snooping error.
    """
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    sd = np.sqrt(sharpe_variance)
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b))


def deflated_sharpe_ratio(
    returns: Sequence[float],
    n_trials: int,
    trial_sharpes: Optional[Sequence[float]] = None,
    trading_days: int = 252,
) -> Dict[str, float]:
    """Probability the observed Sharpe survives correction for the number of trials.

    ``trial_sharpes`` should be the annualised Sharpes of *every* configuration tried,
    including the losers -- their dispersion is what sets the deflation. Passing only the
    winners understates the correction.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return {"sharpe": float("nan"), "dsr": float("nan"), "threshold": float("nan")}

    observed = sharpe_from_returns(r, trading_days)

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        variance = float(np.nanvar(np.asarray(trial_sharpes, dtype=float), ddof=1))
    else:
        variance = 1.0 / len(r) * trading_days  # crude fallback under the null

    threshold = expected_maximum_sharpe(max(n_trials, 2), variance)
    dsr = probabilistic_sharpe_ratio(r, benchmark_sharpe=threshold, trading_days=trading_days)

    return {
        "sharpe": observed,
        "n_trials": float(n_trials),
        "trial_sharpe_variance": variance,
        "deflation_threshold": threshold,
        "dsr": dsr,
        "psr_vs_zero": probabilistic_sharpe_ratio(r, 0.0, trading_days),
    }


# ---------------------------------------------------------------------- bootstrap


def stationary_bootstrap_indices(
    n_obs: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices with geometric block lengths.

    An IID bootstrap on daily returns destroys the autocorrelation and volatility
    clustering that drive drawdowns, so its confidence intervals come out far too tight.
    """
    p = 1.0 / max(mean_block, 1.0)
    indices = np.empty(n_obs, dtype=int)
    current = rng.integers(0, n_obs)
    for t in range(n_obs):
        indices[t] = current
        if rng.random() < p:
            current = rng.integers(0, n_obs)
        else:
            current = (current + 1) % n_obs
    return indices


def bootstrap_metric_ci(
    returns: Sequence[float],
    statistic=sharpe_from_returns,
    n_boot: int = 2000,
    mean_block: float = 21.0,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Percentile confidence interval for any return-based statistic."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan")}

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = statistic(r[stationary_bootstrap_indices(len(r), mean_block, rng)])

    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan")}

    return {
        "point": float(statistic(r)),
        "lower": float(np.percentile(draws, 100 * alpha / 2)),
        "upper": float(np.percentile(draws, 100 * (1 - alpha / 2))),
        "p_positive": float((draws > 0).mean()),
        "n_boot": float(len(draws)),
    }


# ------------------------------------------------------- probability of overfitting


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    n_splits: int = 10,
    trading_days: int = 252,
) -> Dict[str, float]:
    """PBO via combinatorially symmetric cross-validation.

    Split the sample into ``n_splits`` blocks; for every balanced in-sample/out-of-sample
    partition, pick the best strategy in-sample and record its out-of-sample rank. PBO is
    the fraction of partitions where the in-sample winner lands in the bottom half
    out-of-sample.

    PBO near 0.5 means selection carries no information -- the winner is being chosen by
    noise. This is the honest way to report a leaderboard built from many candidates.
    """
    matrix = returns_matrix.dropna(how="any")
    n_obs, n_strategies = matrix.shape
    if n_strategies < 2 or n_obs < n_splits * 2:
        return {"pbo": float("nan"), "n_partitions": 0.0}

    block_size = n_obs // n_splits
    blocks = [
        matrix.iloc[i * block_size : (i + 1) * block_size] for i in range(n_splits)
    ]

    half = n_splits // 2
    logits: List[float] = []

    for in_sample_ids in combinations(range(n_splits), half):
        out_ids = [i for i in range(n_splits) if i not in in_sample_ids]
        in_sample = pd.concat([blocks[i] for i in in_sample_ids])
        out_sample = pd.concat([blocks[i] for i in out_ids])

        in_sharpes = in_sample.apply(lambda c: sharpe_from_returns(c.to_numpy(), trading_days))
        out_sharpes = out_sample.apply(lambda c: sharpe_from_returns(c.to_numpy(), trading_days))
        if in_sharpes.isna().all() or out_sharpes.isna().all():
            continue

        winner = in_sharpes.idxmax()
        ranks = out_sharpes.rank(pct=True)
        omega = float(ranks[winner])
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1 - omega))))

    if not logits:
        return {"pbo": float("nan"), "n_partitions": 0.0}

    logits_array = np.asarray(logits)
    return {
        "pbo": float((logits_array <= 0).mean()),
        "median_oos_rank": float(stats.norm.cdf(np.median(logits_array))),
        "n_partitions": float(len(logits_array)),
    }


# --------------------------------------------------------------- multiple testing


def whites_reality_check(
    strategy_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    n_boot: int = 1000,
    mean_block: float = 21.0,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap p-value that the *best* strategy beats the benchmark.

    Tests the maximum of the family rather than each member, so it does not need a
    Bonferroni correction and is not as conservative as one.
    """
    aligned = strategy_returns.join(benchmark_returns.rename("__benchmark__"), how="inner").dropna()
    if aligned.empty or aligned.shape[1] < 2:
        return {"p_value": float("nan"), "best_strategy": "", "best_statistic": float("nan")}

    benchmark = aligned.pop("__benchmark__").to_numpy()
    excess = aligned.to_numpy() - benchmark[:, None]
    n_obs = len(excess)

    observed = np.sqrt(n_obs) * excess.mean(axis=0)
    best_index = int(np.argmax(observed))
    best_statistic = float(observed[best_index])

    rng = np.random.default_rng(seed)
    centred = excess - excess.mean(axis=0)
    null_max = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n_obs, mean_block, rng)
        null_max[b] = np.max(np.sqrt(n_obs) * centred[idx].mean(axis=0))

    return {
        "p_value": float((null_max >= best_statistic).mean()),
        "best_strategy": str(aligned.columns[best_index]),
        "best_statistic": best_statistic,
        "n_strategies": float(aligned.shape[1]),
    }


def summarise_significance(
    returns_by_strategy: Dict[str, pd.Series],
    benchmark_name: str,
    n_trials: int,
    trading_days: int = 252,
    risk_free_annual: float = 0.0,
    exposure_by_strategy: Optional[Dict[str, float]] = None,
    min_exposure: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per strategy: Sharpe, bootstrap CI, PSR and DSR.

    Replaces the bare leaderboard. A Sharpe with no interval beside it invites a
    conclusion that ~220 trading days cannot support.

    Strategies whose average exposure falls below ``min_exposure`` are flagged
    ``cash_like``: they spent the window in cash, so their "return" is the risk-free rate
    and their risk-adjusted statistics describe nothing. Ranking them alongside invested
    strategies is how a do-nothing rule ends up topping a leaderboard.
    """
    rf_daily = (
        (1.0 + risk_free_annual) ** (1.0 / trading_days) - 1.0 if risk_free_annual > 0 else 0.0
    )
    exposure_by_strategy = exposure_by_strategy or {}

    def _sharpe(values: np.ndarray) -> float:
        return sharpe_from_returns(values, trading_days, rf_daily)

    trial_sharpes = [
        _sharpe(s.dropna().to_numpy()) for s in returns_by_strategy.values()
    ]
    trial_sharpes = [s for s in trial_sharpes if np.isfinite(s)]

    rows = []
    for name, series in returns_by_strategy.items():
        values = series.dropna().to_numpy()
        exposure = exposure_by_strategy.get(name, np.nan)
        cash_like = bool(np.isfinite(exposure) and exposure < min_exposure)

        # Pass RAW returns: `_sharpe` already nets out the risk-free rate. Handing it
        # pre-adjusted returns subtracts rf twice and shifts the whole interval down, so
        # the point estimate lands outside its own confidence band.
        ci = bootstrap_metric_ci(values, statistic=_sharpe, seed=seed)
        # deflated_sharpe_ratio uses rf = 0 internally, so it takes the adjusted series.
        dsr = deflated_sharpe_ratio(values - rf_daily, n_trials, trial_sharpes, trading_days)

        rows.append(
            {
                "strategy": name,
                "sharpe": dsr["sharpe"],
                "sharpe_ci_lower": ci["lower"],
                "sharpe_ci_upper": ci["upper"],
                "ci_excludes_zero": bool(
                    not cash_like
                    and np.isfinite(ci["lower"])
                    and np.isfinite(ci["upper"])
                    and (ci["lower"] > 0 or ci["upper"] < 0)
                ),
                "psr_vs_zero": dsr["psr_vs_zero"],
                "deflation_threshold": dsr["deflation_threshold"],
                "dsr": dsr["dsr"],
                "mean_exposure": exposure,
                "cash_like": cash_like,
                "is_benchmark": name == benchmark_name,
            }
        )

    frame = pd.DataFrame(rows)
    # Cash-like rows sort to the bottom rather than being silently dropped -- their
    # presence is itself a finding about the test window.
    return (
        frame.sort_values(["cash_like", "sharpe"], ascending=[True, False])
        .reset_index(drop=True)
    )
