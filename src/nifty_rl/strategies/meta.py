"""Regime-conditioned meta-strategies.

Two ways to spend a regime signal, in increasing order of commitment:

* **Exposure overlay** -- keep the underlying strategy, scale gross exposure by regime.
  Applies uniformly to every strategy including the baselines, which isolates the value
  of the regime signal itself rather than confounding it with one strategy's quirks.
  Highest information per line of code in the project.
* **Strategy selection** -- learn which strategy wins in each regime on the training
  window, then route out-of-sample by the online regime estimate.

Both consume ``labels`` produced by a detector's ``label_online``, so both inherit the
causality guarantee. Neither may be handed a retrospective segmentation.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

SignalFn = Callable[[pd.DataFrame], pd.Series]


def regime_exposure_overlay(
    signal_fn: SignalFn,
    labels: pd.Series,
    exposure_by_regime: Mapping[int, float],
    threshold: float = 0.5,
) -> SignalFn:
    """Gate a binary signal by regime exposure.

    With binary per-ticker signals, "50% exposure" cannot be expressed as a half
    position, so exposure is applied as a participation gate: the signal survives in a
    regime whose exposure clears ``threshold``. For continuous allocators use
    :func:`scale_weights_by_regime` instead, which scales properly.
    """

    def _bound(df: pd.DataFrame) -> pd.Series:
        base = signal_fn(df).astype(int)
        dates = pd.to_datetime(df["Date"])
        regime = labels.reindex(dates).ffill()
        exposure = regime.map(exposure_by_regime).fillna(1.0).to_numpy()
        gated = np.where(exposure >= threshold, base.to_numpy(), 0)
        return pd.Series(gated, index=df.index, name="signal")

    _bound.__name__ = f"regime_gated_{getattr(signal_fn, '__name__', 'signal')}"
    return _bound


def scale_weights_by_regime(
    weights: pd.DataFrame,
    labels: pd.Series,
    exposure_by_regime: Mapping[int, float],
) -> pd.DataFrame:
    """Scale a weight matrix by regime exposure; the remainder sits in cash.

    Rows sum to ``exposure`` rather than 1, so the shortfall is genuine cash and earns
    the risk-free rate in the backtester -- not a silent renormalisation back to fully
    invested, which would quietly undo the de-risking.
    """
    regime = labels.reindex(weights.index).ffill()
    exposure = regime.map(exposure_by_regime).fillna(1.0)
    return weights.mul(exposure, axis=0)


def default_exposure_ladder(n_regimes: int) -> Dict[int, float]:
    """Full exposure in the calmest regime, stepping down to a defensive floor."""
    if n_regimes <= 1:
        return {0: 1.0}
    ladder = np.linspace(1.0, 0.25, n_regimes)
    return {i: float(round(v, 3)) for i, v in enumerate(ladder)}


def learn_regime_strategy_map(
    train_panel: pd.DataFrame,
    labels: pd.Series,
    strategy_fns: Mapping[str, SignalFn],
    backtest_fn: Callable[[pd.DataFrame, SignalFn, str], "object"],
    score_fn: Callable[[object], float],
    min_days: int = 40,
) -> pd.DataFrame:
    """Score every strategy within every regime on the training window.

    Regimes are sliced by *date*, so a strategy is judged only on the days it actually
    faced that regime. Regimes with fewer than ``min_days`` observations are reported but
    excluded from routing -- picking a winner from twenty days is how a meta-strategy
    overfits.
    """
    rows = []
    dates = pd.to_datetime(train_panel["Date"])

    for regime in sorted(pd.unique(labels.dropna())):
        regime_dates = labels.index[labels == regime]
        slice_ = train_panel[dates.isin(regime_dates)]
        n_days = slice_["Date"].nunique() if not slice_.empty else 0

        for name, fn in strategy_fns.items():
            if n_days < min_days:
                rows.append(
                    {"regime": int(regime), "strategy": name, "score": np.nan,
                     "n_days": n_days, "eligible": False}
                )
                continue
            try:
                result = backtest_fn(slice_, fn, name)
                score = float(score_fn(result))
            except Exception:
                score = float("nan")
            rows.append(
                {"regime": int(regime), "strategy": name, "score": score,
                 "n_days": n_days, "eligible": True}
            )

    return pd.DataFrame(rows)


def best_strategy_per_regime(scores: pd.DataFrame, fallback: str) -> Dict[int, str]:
    """Route each regime to its training-window winner, falling back when unreliable."""
    mapping: Dict[int, str] = {}
    for regime, group in scores.groupby("regime"):
        eligible = group[group["eligible"] & group["score"].notna()]
        mapping[int(regime)] = (
            str(eligible.loc[eligible["score"].idxmax(), "strategy"])
            if not eligible.empty
            else fallback
        )
    return mapping


def regime_switching_signal(
    strategy_fns: Mapping[str, SignalFn],
    labels: pd.Series,
    regime_to_strategy: Mapping[int, str],
    fallback: str,
) -> SignalFn:
    """Emit the routed strategy's signal on each date, per its online regime."""

    def _bound(df: pd.DataFrame) -> pd.Series:
        dates = pd.to_datetime(df["Date"])
        regime = labels.reindex(dates).ffill()

        # Evaluate each candidate once over the whole frame, then select per row --
        # cheaper than slicing, and keeps every strategy's internal state continuous
        # rather than restarting it at each regime boundary.
        computed = {name: fn(df).astype(int).to_numpy() for name, fn in strategy_fns.items()}
        chosen = np.zeros(len(df), dtype=int)
        for i, r in enumerate(regime.to_numpy()):
            name = regime_to_strategy.get(int(r), fallback) if np.isfinite(r) else fallback
            chosen[i] = computed.get(name, computed[fallback])[i]
        return pd.Series(chosen, index=df.index, name="signal")

    _bound.__name__ = "regime_switching"
    return _bound


def performance_by_regime(
    returns_by_strategy: Mapping[str, pd.Series],
    labels: pd.Series,
    regime_names: Optional[Sequence[str]] = None,
    trading_days: int = 252,
    risk_free_annual: float = 0.0,
) -> pd.DataFrame:
    """Long-format strategy x regime performance.

    The table the project was missing. It answers directly whether a drawdown-penalised
    agent earns its penalty when conditions are bad, instead of reporting one blended
    number over a window that happened to be a drawdown.

    Sharpe comes from the shared helper so the degenerate-cash guard applies here too. A
    strategy parked in cash has constant returns *within every regime slice*, so this
    table is if anything more exposed to the 0/0 blow-up than the headline one.
    """
    from ..metrics.stats import sharpe_from_returns

    rf_daily = (
        (1.0 + risk_free_annual) ** (1.0 / trading_days) - 1.0 if risk_free_annual > 0 else 0.0
    )

    rows = []
    for name, returns in returns_by_strategy.items():
        aligned = pd.concat(
            [returns.rename("ret"), labels.rename("regime")], axis=1, join="inner"
        ).dropna()
        for regime, group in aligned.groupby("regime"):
            series = group["ret"]
            label = (
                regime_names[int(regime)]
                if regime_names is not None and int(regime) < len(regime_names)
                else f"regime_{int(regime)}"
            )
            equity = (1.0 + series).cumprod()
            rows.append(
                {
                    "strategy": name,
                    "regime": label,
                    "regime_index": int(regime),
                    "n_days": int(len(series)),
                    "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else np.nan,
                    "annual_return": float(series.mean() * trading_days),
                    "excess_annual_return": float((series.mean() - rf_daily) * trading_days),
                    "volatility": float(series.std() * np.sqrt(trading_days)),
                    "sharpe": sharpe_from_returns(series.to_numpy(), trading_days, rf_daily),
                    "max_drawdown": float((equity / equity.cummax() - 1.0).min())
                    if len(equity)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)
