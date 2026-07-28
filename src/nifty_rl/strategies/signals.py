"""Rule-based signal generators.

Every function takes a single-ticker frame and returns a 0/1 Series aligned to it.

The central fix here is bug #1. The notebook's ``ma_crossover_signals`` and
``breakout_signals`` recomputed their rolling windows *on whatever slice they were
handed*::

    sig = (price.rolling(short).mean() > price.rolling(long).mean()).astype(int)

Since the caller passes a split (or a walk-forward window), the first ``long`` rows come
back NaN and collapse to 0. That forced **50 of 222 test days flat (23%)** and **50 of
120 walk-forward days flat (42%)** -- MA_20_50 was structurally handicapped in exactly
the evaluation that concluded it underperformed. The frame already carries ``ma20``,
``ma50``, ``high20_prev`` and ``low10_prev`` computed over full history in the feature
layer, and ``momentum_pullback_signals`` already used them; the others now do too.

Rule: **no signal function may call ``.rolling()``.** Anything needing a window belongs
in ``features.py``, where it sees the whole series.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


def _require(df: pd.DataFrame, *columns: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Signal requires precomputed feature column(s) {missing}. "
            "Run features.add_features() before backtesting -- signal functions must "
            "never recompute rolling windows on a split slice (bug #1)."
        )


def _as_signal(values, index) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=int), index=index, name="signal")


def buy_hold_signals(df: pd.DataFrame) -> pd.Series:
    return _as_signal(np.ones(len(df)), df.index)


def ma_crossover_signals(
    df: pd.DataFrame,
    short: int = 20,
    long: int = 50,
    vix_limit: Optional[float] = None,
) -> pd.Series:
    """Long while the short MA is above the long MA.

    Uses the precomputed ``ma{short}``/``ma{long}`` columns so the first ``long`` bars of
    a split are not silently forced flat.
    """
    short_col, long_col = f"ma{short}", f"ma{long}"
    _require(df, short_col, long_col)
    sig = (df[short_col] > df[long_col]).astype(int)
    if vix_limit is not None:
        _require(df, "india_vix")
        sig = sig.where(df["india_vix"] <= vix_limit, 0)
    return _as_signal(sig.fillna(0), df.index)


def rsi_mean_reversion_signals(
    df: pd.DataFrame,
    buy_below: float = 35.0,
    sell_above: float = 60.0,
    trend_filter: bool = True,
) -> pd.Series:
    """Enter when oversold, exit when the bounce matures.

    Stateful by nature (entry and exit thresholds differ), so it runs as a scan -- but
    over precomputed ``rsi``/``ma50``, not recomputed ones.
    """
    _require(df, "rsi")
    if trend_filter:
        _require(df, "ma50")
        trend_ok = (df["price"] > df["ma50"]).to_numpy()
    else:
        trend_ok = np.ones(len(df), dtype=bool)

    rsi = df["rsi"].to_numpy()
    out = np.zeros(len(df), dtype=int)
    position = 0
    for i in range(len(df)):
        if position == 0:
            if rsi[i] < buy_below and trend_ok[i]:
                position = 1
        else:
            if rsi[i] > sell_above or not trend_ok[i]:
                position = 0
        out[i] = position
    return _as_signal(out, df.index)


def breakout_signals(
    df: pd.DataFrame,
    lookback: int = 20,
    exit_lookback: int = 10,
    vix_limit: Optional[float] = None,
) -> pd.Series:
    """Long on a breakout above the prior N-day high; exit below the prior M-day low."""
    if lookback == 20 and exit_lookback == 10:
        _require(df, "high20_prev", "low10_prev")
        entry = (df["price"] > df["high20_prev"]).to_numpy()
        exit_ = (df["price"] < df["low10_prev"]).to_numpy()
    else:
        # Non-default windows are not precomputed; derive them from the full-history
        # channel columns if present, else raise rather than silently warm up on a slice.
        _require(df, f"high{lookback}_prev", f"low{exit_lookback}_prev")
        entry = (df["price"] > df[f"high{lookback}_prev"]).to_numpy()
        exit_ = (df["price"] < df[f"low{exit_lookback}_prev"]).to_numpy()

    if vix_limit is not None:
        _require(df, "india_vix")
        allowed = (df["india_vix"] <= vix_limit).to_numpy()
    else:
        allowed = np.ones(len(df), dtype=bool)

    out = np.zeros(len(df), dtype=int)
    position = 0
    for i in range(len(df)):
        if position == 0:
            if entry[i] and allowed[i]:
                position = 1
        else:
            if exit_[i] or not allowed[i]:
                position = 0
        out[i] = position
    return _as_signal(out, df.index)


def momentum_pullback_signals(
    df: pd.DataFrame,
    rsi_max: float = 55.0,
    rsi_min: float = 35.0,
    vix_limit: Optional[float] = None,
) -> pd.Series:
    """Uptrend confirmed, entered on an RSI pullback rather than at the highs."""
    _require(df, "ma20", "ma50", "rsi")
    trend = (df["price"] > df["ma50"]) & (df["ma20"] > df["ma50"])
    pullback = (df["rsi"] < rsi_max) & (df["rsi"] > rsi_min)
    sig = (trend & pullback).astype(int)
    if vix_limit is not None:
        _require(df, "india_vix")
        sig = sig.where(df["india_vix"] <= vix_limit, 0)
    return _as_signal(sig.fillna(0), df.index)


def sentiment_momentum_signals(df: pd.DataFrame) -> pd.Series:
    """VIX-proxy sentiment improving while price holds above its 20-day mean.

    Note that ``sentiment`` is derived from India VIX and is therefore market-wide --
    identical across every ticker. It carries no cross-sectional information.
    """
    _require(df, "sent_momentum", "ma20")
    sig = ((df["sent_momentum"] > 0) & (df["price"] > df["ma20"])).astype(int)
    return _as_signal(sig.fillna(0), df.index)


def vix_regime_momentum_signals(df: pd.DataFrame, quantile_col: str = "vix_q40") -> pd.Series:
    """Long only in a low-VIX regime with a confirmed uptrend."""
    _require(df, "ma20", "ma50")
    if quantile_col in df.columns:
        low_vix = df["india_vix"] < df[quantile_col]
    else:
        _require(df, "high_vix_regime")
        low_vix = df["high_vix_regime"] <= 0
    uptrend = (df["ma20"] > df["ma50"]) & (df["price"] > df["ma20"])
    sig = (low_vix & uptrend).astype(int)
    return _as_signal(sig.fillna(0), df.index)


def random_policy_signals(
    df: pd.DataFrame,
    probability: float = 0.5,
    seed: int = 42,
    ticker_offset: int = 0,
) -> pd.Series:
    """Independent random baseline.

    The notebook seeded every ticker with the same value, so all ten received an
    identical signal sequence and the "random" portfolio went all-in and all-out
    simultaneously across the whole book (bug #21). That concentration -- not
    randomness -- is what produced its -23.88% print. ``ticker_offset`` decorrelates the
    draws so this behaves like an actual random-portfolio control.
    """
    rng = np.random.default_rng(seed + ticker_offset)
    return _as_signal((rng.random(len(df)) < probability).astype(int), df.index)


SIGNAL_REGISTRY: Dict[str, Callable[..., pd.Series]] = {
    "buy_hold": buy_hold_signals,
    "ma": ma_crossover_signals,
    "rsi": rsi_mean_reversion_signals,
    "breakout": breakout_signals,
    "momentum_pullback": momentum_pullback_signals,
    "sentiment_momentum": sentiment_momentum_signals,
    "vix_regime_momentum": vix_regime_momentum_signals,
    "random": random_policy_signals,
}


def make_signal_fn(kind: str, **params) -> Callable[[pd.DataFrame], pd.Series]:
    """Bind a registry entry and its parameters into a single-argument callable.

    The notebook's ``_signal_fn`` dispatcher silently returned an all-zero Series for any
    unregistered name -- which is how ``VIX_Regime_Momentum`` ran as a no-op until it was
    noticed. Unknown kinds raise here instead.
    """
    if kind not in SIGNAL_REGISTRY:
        raise KeyError(
            f"Unknown signal kind {kind!r}. Registered: {sorted(SIGNAL_REGISTRY)}"
        )
    fn = SIGNAL_REGISTRY[kind]

    def _bound(df: pd.DataFrame) -> pd.Series:
        return fn(df, **params)

    _bound.__name__ = f"signal_{kind}"
    return _bound


def make_random_signal_fn(seed: int, probability: float = 0.5) -> Callable[[pd.DataFrame], pd.Series]:
    """Random baseline whose draws are decorrelated across tickers."""
    counter = {"n": 0}

    def _bound(df: pd.DataFrame) -> pd.Series:
        offset = counter["n"]
        counter["n"] += 1
        return random_policy_signals(df, probability=probability, seed=seed, ticker_offset=offset)

    _bound.__name__ = "signal_random"
    return _bound
