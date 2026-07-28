"""Per-ticker technical features.

Every indicator is computed inside a per-ticker groupby, so no rolling window ever spans
a ticker boundary.

Fixes carried over from the notebook:

* **RSI no longer deletes rows** (bug #19). The notebook used
  ``loss.rolling(14).mean().replace(0, np.nan)``, so 14 consecutive up-days produced a
  NaN RSI, and the blanket ``dropna()`` at the end then dropped that row entirely --
  silently removing observations precisely during the strongest momentum runs. Wilder's
  smoothing is used here, and the zero-loss case resolves to RSI 100 (its correct value)
  rather than NaN.
* **Targeted dropna** (bug #20). The notebook's bare ``dropna()`` gated every row on
  *all* 49 columns, including ``ma100`` and ``momentum_60`` which only the ``full``
  feature set uses. That cost ~100 days of warm-up on every run regardless of the
  feature set in play -- which is why a 2020-01-01 start became 2020-05-29.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Columns required by the smallest feature set. Rows are dropped only on these unless
# the caller asks for more.
CORE_FEATURES: Sequence[str] = (
    "ret",
    "ma_ratio",
    "trend_20_50",
    "rsi",
    "macd_hist",
    "bb_position",
    "bb_width",
    "atr_pct",
    "momentum_5",
    "momentum_20",
    "volume_change",
)


def wilder_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """RSI with Wilder's smoothing.

    Boundary cases resolved explicitly instead of propagating NaN:
      * no losses over the window -> 100 (maximum strength)
      * no gains and no losses    -> 50 (neutral; a flat series has no momentum)
    """
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    rsi = pd.Series(np.nan, index=price.index, dtype="float64")
    valid = avg_gain.notna() & avg_loss.notna()

    both_zero = valid & (avg_gain <= 0) & (avg_loss <= 0)
    no_loss = valid & (avg_loss <= 0) & (avg_gain > 0)
    normal = valid & (avg_loss > 0)

    rs = avg_gain[normal] / avg_loss[normal]
    rsi.loc[normal] = 100.0 - (100.0 / (1.0 + rs))
    rsi.loc[no_loss] = 100.0
    rsi.loc[both_zero] = 50.0
    return rsi


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average true range with Wilder's smoothing.

    Requires that ``high``/``low``/``close`` share one price scale -- guaranteed by
    ``auto_adjust=True`` in the data layer (bug #6).
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def add_features_single(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators for one ticker. Input must be a single-ticker frame."""
    out = df.copy().reset_index(drop=True)
    price = out["price"]
    high = out["High"] if "High" in out.columns else price
    low = out["Low"] if "Low" in out.columns else price
    volume = out["Volume"] if "Volume" in out.columns else pd.Series(0.0, index=out.index)

    # --- returns and momentum
    out["ret"] = price.pct_change()
    out["momentum_5"] = price.pct_change(5)
    out["momentum_20"] = price.pct_change(20)
    out["momentum_60"] = price.pct_change(60)

    vol_ma20 = volume.rolling(20).mean()
    out["volume_change"] = (volume / vol_ma20.replace(0, np.nan)) - 1.0
    out["volume_change"] = out["volume_change"].replace([np.inf, -np.inf], np.nan)

    # --- moving averages
    out["ma5"] = price.rolling(5).mean()
    out["ma10"] = price.rolling(10).mean()
    out["ma20"] = price.rolling(20).mean()
    out["ma50"] = price.rolling(50).mean()
    out["ma100"] = price.rolling(100).mean()
    # NOTE: ma_ratio is ma5/ma20 - 1, not "Price / MA20" as the README claims.
    out["ma_ratio"] = out["ma5"] / out["ma20"] - 1.0
    out["trend_20_50"] = out["ma20"] / out["ma50"] - 1.0

    # --- volatility / channels
    out["vol20"] = out["ret"].rolling(20).std() * np.sqrt(252)
    out["high20"] = price.rolling(20).max()
    out["low20"] = price.rolling(20).min()
    out["high20_prev"] = out["high20"].shift(1)
    out["low10_prev"] = price.rolling(10).min().shift(1)

    # --- oscillators
    out["rsi"] = wilder_rsi(price, 14)

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    bb_mid = price.rolling(20).mean()
    bb_std = price.rolling(20).std()
    band = (4.0 * bb_std).replace(0, np.nan)
    out["bb_width"] = band / bb_mid
    out["bb_position"] = (price - (bb_mid - 2.0 * bb_std)) / band

    out["atr14"] = wilder_atr(high, low, price, 14)
    out["atr_pct"] = out["atr14"] / price

    # --- VIX-derived market-wide features
    # sentiment is -zscore(india_vix, 60d): high VIX = fear = negative. Market-wide by
    # construction, so it is identical across all tickers -- see report notes before
    # plotting it as a per-ticker heatmap (bug #27).
    if "india_vix" in out.columns:
        vix_mu = out["india_vix"].rolling(60).mean()
        vix_sd = out["india_vix"].rolling(60).std().replace(0, np.nan)
        out["sentiment"] = -((out["india_vix"] - vix_mu) / vix_sd)
        out["sent_ma3"] = out["sentiment"].rolling(3).mean()
        out["sent_ma7"] = out["sentiment"].rolling(7).mean()
        out["sent_momentum"] = out["sent_ma3"] - out["sent_ma7"]
        out["vix_change"] = out["india_vix"].pct_change()
        # NOTE: 75th percentile, not the 60th the README documents.
        out["high_vix_regime"] = (
            out["india_vix"] > out["india_vix"].rolling(60).quantile(0.75)
        ).astype(float)

    if "cs_dispersion" in out.columns:
        cs_mu = out["cs_dispersion"].rolling(20).mean()
        cs_sd = out["cs_dispersion"].rolling(20).std().replace(0, np.nan)
        out["dispersion_zscore"] = (out["cs_dispersion"] - cs_mu) / cs_sd
    else:
        out["dispersion_zscore"] = np.nan

    if "benchmark_return" in out.columns:
        out["relative_strength"] = out["ret"] - out["benchmark_return"]
        out["rolling_alpha"] = out["relative_strength"].rolling(20).mean()

    return out


def add_features(
    panel: pd.DataFrame,
    required: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute features per ticker and drop warm-up rows.

    ``required`` controls which columns gate row removal. Defaults to ``CORE_FEATURES``
    so unused long-window columns (``ma100``, ``momentum_60``) do not cost warm-up.
    """
    parts = [add_features_single(g) for _, g in panel.groupby("ticker", sort=False)]
    out = pd.concat(parts, ignore_index=True)

    subset = list(required) if required is not None else list(CORE_FEATURES)
    subset = [c for c in subset if c in out.columns]
    if subset:
        out = out.dropna(subset=subset)

    return out.sort_values(["ticker", "Date"]).reset_index(drop=True)


def align_to_common_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict the panel to dates present for **every** ticker.

    Per-ticker warm-up and holiday differences otherwise leave a ragged panel, which the
    aggregation step in the backtester turns into phantom equity holes (bug #5). Keeping
    the panel rectangular removes that failure mode at the source; the backtester also
    defends against it independently.
    """
    counts = df.groupby("Date")["ticker"].nunique()
    full = counts[counts == df["ticker"].nunique()].index
    return df[df["Date"].isin(full)].reset_index(drop=True)
