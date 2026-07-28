"""Market-level regime feature panel.

Regimes are a property of the *market*, not of any single name, so these are computed
once per date from the whole universe rather than per ticker.

Every column is causal by construction: rolling windows look backward only, and nothing
is z-scored against full-sample statistics (a common silent leak -- standardising the
whole series bakes the future's mean and variance into every row).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

REGIME_FEATURES: List[str] = [
    "realized_vol_21",
    "realized_vol_5",
    "vix_level",
    "vix_change_5",
    "trend_21",
    "dispersion",
    "mean_correlation",
    "breadth",
]


def build_regime_features(
    panel: pd.DataFrame,
    benchmark_column: str = "benchmark_return",
    trading_days: int = 252,
) -> pd.DataFrame:
    """Build the daily market-level regime feature panel, indexed by date."""
    wide_prices = panel.pivot_table(index="Date", columns="ticker", values="price", aggfunc="last")
    wide_prices = wide_prices.sort_index()
    wide_returns = wide_prices.pct_change()

    market_return = _market_return(panel, wide_returns, benchmark_column)

    out = pd.DataFrame(index=wide_prices.index)
    out.index.name = "Date"

    # --- volatility axis
    out["realized_vol_21"] = market_return.rolling(21).std() * np.sqrt(trading_days)
    out["realized_vol_5"] = market_return.rolling(5).std() * np.sqrt(trading_days)

    # --- implied fear
    if "india_vix" in panel.columns:
        vix = panel.groupby("Date")["india_vix"].first().reindex(out.index)
        out["vix_level"] = vix
        out["vix_change_5"] = vix.pct_change(5)
    else:
        out["vix_level"] = np.nan
        out["vix_change_5"] = np.nan

    # --- trend axis
    out["trend_21"] = (1.0 + market_return).rolling(21).apply(np.prod, raw=True) - 1.0

    # --- cross-sectional structure
    out["dispersion"] = wide_returns.std(axis=1)
    out["mean_correlation"] = _rolling_mean_pairwise_correlation(wide_returns, window=21)

    # --- participation
    moving_average = wide_prices.rolling(50).mean()
    out["breadth"] = (wide_prices > moving_average).mean(axis=1)

    return out


def _market_return(
    panel: pd.DataFrame, wide_returns: pd.DataFrame, benchmark_column: str
) -> pd.Series:
    """Benchmark return when available, else the equal-weight universe return."""
    if benchmark_column in panel.columns:
        benchmark = panel.groupby("Date")[benchmark_column].first()
        if benchmark.notna().sum() > 0.5 * len(benchmark):
            return benchmark.reindex(wide_returns.index)
    return wide_returns.mean(axis=1)


def _rolling_mean_pairwise_correlation(returns: pd.DataFrame, window: int = 21) -> pd.Series:
    """Average off-diagonal correlation over a trailing window.

    Correlation spiking toward one is among the most reliable crisis markers there is:
    in a sell-off, cross-sectional structure collapses and everything moves together.
    It is free from data already loaded, and it is the feature most likely to separate a
    genuine crisis from ordinary high volatility.
    """
    values = returns.to_numpy()
    n_obs, n_assets = values.shape
    if n_assets < 2:
        return pd.Series(np.nan, index=returns.index)

    out = np.full(n_obs, np.nan)
    upper = np.triu_indices(n_assets, k=1)
    for end in range(window, n_obs + 1):
        block = values[end - window : end]
        if np.isnan(block).all():
            continue
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(np.nan_to_num(block, nan=0.0), rowvar=False)
        if corr.shape != (n_assets, n_assets):
            continue
        out[end - 1] = np.nanmean(corr[upper])
    return pd.Series(out, index=returns.index)


def standardise_causally(
    features: pd.DataFrame, train_index: Optional[pd.Index] = None
) -> pd.DataFrame:
    """Z-score using **training-window** statistics only.

    Standardising against full-sample mean and variance leaks the future into every row.
    Fitting on the training slice and applying those constants everywhere else is the
    only version compatible with the causality contract.
    """
    source = features.loc[train_index] if train_index is not None else features
    mean = source.mean()
    std = source.std().replace(0, np.nan)
    return ((features - mean) / std).fillna(0.0)
