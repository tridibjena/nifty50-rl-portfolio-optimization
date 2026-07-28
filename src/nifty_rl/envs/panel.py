"""Dense NumPy panel for the RL environment.

The notebook stored the panel as ``dict[date] -> DataFrame`` and did a pandas ``.loc``
per ticker per step. At 250k steps with ten tickers that is ~2.5 million pandas lookups
per training run, and it dominated wall clock -- which is why the "training budget is
system-constrained" note existed at all.

Here the same data is a contiguous ``(n_dates, n_tickers, n_features)`` float32 array
addressed by integer index. Nothing clever, just the representation the inner loop
actually wants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class PanelArrays:
    """Rectangular market panel addressed by integer index."""

    dates: pd.DatetimeIndex
    tickers: List[str]
    features: np.ndarray  # (n_dates, n_tickers, n_features) float32
    prices: np.ndarray  # (n_dates, n_tickers) float32
    available: np.ndarray  # (n_dates, n_tickers) float32, 1.0 where price is usable
    feature_names: List[str]
    regime: Optional[np.ndarray] = None  # (n_dates,) int, optional regime label

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def n_tickers(self) -> int:
        return len(self.tickers)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def observation_size(self) -> int:
        """Per-ticker features + availability flag, plus weights and cash."""
        return self.n_tickers * (self.n_features + 1) + (self.n_tickers + 1) + 1


def build_panel_arrays(
    df: pd.DataFrame,
    tickers: Sequence[str],
    feature_names: Sequence[str],
    regime_labels: Optional[pd.Series] = None,
) -> PanelArrays:
    """Pivot a long-format frame into dense arrays.

    Missing ``(date, ticker)`` cells get zero features and ``available = 0`` rather than
    a forward-filled value, so the agent can learn to ignore a stale name instead of
    acting on a carried-over price.
    """
    tickers = list(tickers)
    feature_names = list(feature_names)

    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    dates = pd.DatetimeIndex(np.sort(frame["Date"].unique()))
    date_pos = {d: i for i, d in enumerate(dates)}
    ticker_pos = {t: i for i, t in enumerate(tickers)}

    n_dates, n_tickers, n_features = len(dates), len(tickers), len(feature_names)
    features = np.zeros((n_dates, n_tickers, n_features), dtype=np.float32)
    prices = np.zeros((n_dates, n_tickers), dtype=np.float32)
    available = np.zeros((n_dates, n_tickers), dtype=np.float32)

    subset = frame[frame["ticker"].isin(ticker_pos)]
    rows = subset["Date"].map(date_pos).to_numpy()
    cols = subset["ticker"].map(ticker_pos).to_numpy()

    values = subset[feature_names].to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    features[rows, cols] = values

    price_values = subset["price"].to_numpy(dtype=np.float32)
    prices[rows, cols] = np.nan_to_num(price_values, nan=0.0)
    available[rows, cols] = (np.isfinite(price_values) & (price_values > 0)).astype(np.float32)

    regime = None
    if regime_labels is not None:
        aligned = regime_labels.reindex(dates).ffill().bfill()
        regime = aligned.to_numpy(dtype=np.int64)

    return PanelArrays(
        dates=dates,
        tickers=tickers,
        features=features,
        prices=prices,
        available=available,
        feature_names=feature_names,
        regime=regime,
    )
