"""Performance metrics.

Fixes carried over from the notebook:

* **Information ratio is annualised exactly once** (bug #9). The notebook computed::

      te = (sr - br).std() * np.sqrt(252)
      ir = (sr - br).mean() / te * np.sqrt(252)

  The two roots cancel, leaving a *daily* mean/std ratio. Every published IR was
  therefore about 15.9x too small.

* **Signal accuracy is genuinely per-ticker** (bug #8). The notebook reindexed the
  *aggregate* 0..N position count against each individual ticker's forward return, so
  ``pos > 0`` meant "invested in anything at all". That is why every strategy landed in
  a 44-47% band -- the metric was measuring portfolio participation, not signal quality.

* **Risk-free rate is applied** (bug #10 companion). Sharpe and Sortino in the notebook
  used ``mean/std``, i.e. rf = 0. At an Indian risk-free near 6.5% that materially
  flatters fully-invested strategies relative to selective ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import MetricsConfig


def _daily_rf(cfg: MetricsConfig) -> float:
    if cfg.risk_free_annual <= 0:
        return 0.0
    return (1.0 + cfg.risk_free_annual) ** (1.0 / cfg.trading_days) - 1.0


def _clean_equity(equity: pd.Series) -> pd.Series:
    """Collapse duplicate timestamps and coerce to float."""
    return equity.groupby(level=0).last().astype(float).sort_index()


# ------------------------------------------------------------------ risk measures


# A strategy parked in cash earns the risk-free rate exactly, so its excess return is
# zero up to floating-point noise. Dividing by that noise produces Sharpe ratios in the
# trillions. Any dispersion below this floor means "no risk was taken", and the honest
# answer is undefined rather than enormous.
_DEGENERATE_VOL = 1e-10


def sharpe_ratio(returns: pd.Series, cfg: MetricsConfig) -> float:
    """Annualised Sharpe of excess-over-cash returns.

    Returns NaN for a portfolio that never left cash: with zero excess return and zero
    excess volatility the ratio is 0/0, and reporting a number there would rank a
    do-nothing strategy at the top of the leaderboard.
    """
    if len(returns) < 2:
        return float("nan")
    excess = returns - _daily_rf(cfg)
    sd = excess.std()
    if not np.isfinite(sd) or sd <= _DEGENERATE_VOL:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(cfg.trading_days))


def sortino_ratio(returns: pd.Series, cfg: MetricsConfig) -> float:
    """Annualised Sortino: downside deviation measured against the risk-free rate."""
    if len(returns) < 2:
        return float("nan")
    rf = _daily_rf(cfg)
    excess = returns - rf
    if not np.isfinite(excess.std()) or excess.std() <= _DEGENERATE_VOL:
        return float("nan")
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    dd = np.sqrt((downside ** 2).mean())
    if dd <= _DEGENERATE_VOL or not np.isfinite(dd):
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(cfg.trading_days))


def information_ratio(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, cfg: MetricsConfig
) -> float:
    """Annualised IR = mean(active) / stdev(active) * sqrt(252).

    Annualised once. The notebook multiplied by sqrt(252) after already folding it into
    the tracking-error denominator, so the factors cancelled (bug #9).
    """
    aligned = pd.concat(
        [strategy_returns.rename("s"), benchmark_returns.rename("b")], axis=1, join="inner"
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    active = aligned["s"] - aligned["b"]
    sd = active.std()
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(active.mean() / sd * np.sqrt(cfg.trading_days))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    return float((equity / equity.cummax() - 1.0).min())


def ulcer_index(equity: pd.Series) -> float:
    """RMS drawdown -- penalises depth *and* duration, unlike max drawdown."""
    if equity.empty:
        return float("nan")
    dd = equity / equity.cummax() - 1.0
    return float(np.sqrt((dd ** 2).mean()))


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Probability-weighted gains over losses relative to a threshold."""
    if returns.empty:
        return float("nan")
    gains = (returns - threshold).clip(lower=0).sum()
    losses = (threshold - returns).clip(lower=0).sum()
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def tail_ratio(returns: pd.Series, quantile: float = 0.05) -> float:
    """|95th percentile| / |5th percentile| -- upside tail versus downside tail."""
    if len(returns) < 20:
        return float("nan")
    upper = abs(returns.quantile(1 - quantile))
    lower = abs(returns.quantile(quantile))
    if lower <= 0:
        return float("nan")
    return float(upper / lower)


def longest_drawdown_days(equity: pd.Series) -> int:
    """Longest run of consecutive observations spent below a prior peak."""
    if equity.empty:
        return 0
    underwater = (equity < equity.cummax()).to_numpy()
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def calmar_ratio(cagr: float, mdd: float) -> float:
    if not np.isfinite(cagr) or not np.isfinite(mdd) or mdd >= 0:
        return float("nan")
    return float(cagr / abs(mdd))


# --------------------------------------------------------------- signal accuracy


def signal_accuracy_per_ticker(
    per_ticker_positions: Dict[str, pd.Series], panel: pd.DataFrame
) -> float:
    """Fraction of invested days on which *that ticker* rose, equal-weighted.

    Each ticker contributes one accuracy figure regardless of its price level, then those
    are averaged. The notebook instead compared the aggregate portfolio position count
    against each ticker's return (bug #8).
    """
    accuracies = []
    for ticker, positions in per_ticker_positions.items():
        ticker_df = panel[panel["ticker"] == ticker]
        if ticker_df.empty:
            continue

        prices = ticker_df.set_index(pd.to_datetime(ticker_df["Date"]))["price"]
        prices = prices.groupby(level=0).last().sort_index()
        forward = prices.pct_change().shift(-1)

        aligned = pd.concat(
            [positions.rename("pos"), forward.rename("fwd")], axis=1, join="inner"
        ).dropna()
        invested = aligned[aligned["pos"] > 0]
        if not invested.empty:
            accuracies.append(float((invested["fwd"] > 0).mean()))

    return float(np.mean(accuracies)) if accuracies else float("nan")


# ------------------------------------------------------------------- aggregation


@dataclass
class PerformanceMetrics:
    strategy: str
    final_equity: float
    total_return: float
    benchmark_excess_return: float
    CAGR: float
    alpha_annual: float
    annual_volatility: float
    Sharpe: float
    Sortino: float
    Calmar: float
    max_drawdown: float
    ulcer_index: float
    longest_drawdown_days: int
    information_ratio: float
    omega_ratio: float
    tail_ratio: float
    daily_VaR_95: float
    daily_CVaR_95: float
    trades: int
    win_rate: float
    payoff_ratio: float
    signal_accuracy: float
    exposure: float

    def to_dict(self) -> dict:
        return asdict(self)


def performance_metrics(
    result,
    benchmark,
    panel: pd.DataFrame,
    cfg: MetricsConfig,
    initial_cash: float,
) -> PerformanceMetrics:
    """Compute the full metric set for one backtest result.

    ``result`` and ``benchmark`` are :class:`~nifty_rl.backtest.engine.BacktestResult`.
    """
    equity = _clean_equity(result.equity)
    bench_equity = _clean_equity(benchmark.equity)

    returns = equity.pct_change().dropna()
    bench_returns = bench_equity.pct_change().dropna()

    years = max(len(equity) / cfg.trading_days, 1.0 / cfg.trading_days)
    total_return = equity.iloc[-1] / initial_cash - 1.0
    bench_total = bench_equity.iloc[-1] / initial_cash - 1.0
    cagr = (equity.iloc[-1] / initial_cash) ** (1.0 / years) - 1.0
    bench_cagr = (bench_equity.iloc[-1] / initial_cash) ** (1.0 / years) - 1.0

    mdd = max_drawdown(equity)
    var95 = float(returns.quantile(cfg.var_quantile)) if len(returns) else float("nan")
    tail = returns[returns <= var95]
    cvar95 = float(tail.mean()) if len(tail) else float("nan")

    # Trade count is always meaningful; win rate and payoff need round-trip P&L, which
    # only the signal engine produces. The weight engine logs individual fills, where a
    # "win" is undefined -- reporting NaN there is correct, but the count still counts.
    trades = result.trades
    n_trades = int(len(trades)) if trades is not None and not trades.empty else 0
    if n_trades and "pnl" in trades.columns:
        win_rate = float((trades["pnl"] > 0).mean())
        wins = trades[trades["pnl"] > 0]["pnl"]
        losses = trades[trades["pnl"] < 0]["pnl"]
        payoff = float(-wins.mean() / losses.mean()) if len(losses) and len(wins) else float("nan")
    else:
        win_rate, payoff = float("nan"), float("nan")

    if result.per_ticker_positions:
        accuracy = signal_accuracy_per_ticker(result.per_ticker_positions, panel)
        n_names = max(len(result.per_ticker_positions), 1)
        exposure = float(result.positions.mean() / n_names) if len(result.positions) else float("nan")
    else:
        accuracy = float("nan")
        exposure = float("nan")

    return PerformanceMetrics(
        strategy=result.strategy,
        final_equity=float(equity.iloc[-1]),
        total_return=float(total_return),
        benchmark_excess_return=float(total_return - bench_total),
        CAGR=float(cagr),
        alpha_annual=float(cagr - bench_cagr),
        annual_volatility=float(returns.std() * np.sqrt(cfg.trading_days)) if len(returns) > 1 else float("nan"),
        Sharpe=sharpe_ratio(returns, cfg),
        Sortino=sortino_ratio(returns, cfg),
        Calmar=calmar_ratio(cagr, mdd),
        max_drawdown=mdd,
        ulcer_index=ulcer_index(equity),
        longest_drawdown_days=longest_drawdown_days(equity),
        information_ratio=information_ratio(returns, bench_returns, cfg),
        omega_ratio=omega_ratio(returns),
        tail_ratio=tail_ratio(returns, cfg.var_quantile),
        daily_VaR_95=var95,
        daily_CVaR_95=cvar95,
        trades=n_trades,
        win_rate=win_rate,
        payoff_ratio=payoff,
        signal_accuracy=accuracy,
        exposure=exposure,
    )


def metrics_frame(metrics_list, sort_by: str = "Sharpe") -> pd.DataFrame:
    """Assemble a leaderboard, de-duplicating identical strategies.

    The notebook ran ``MA_20_50`` from both ``FIXED_STRATS`` and the validation selector,
    so the leaderboard and both bar charts carried two identical rows (bug #26).
    """
    frame = pd.DataFrame([m.to_dict() for m in metrics_list])
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(subset=["strategy"], keep="first")
    if sort_by in frame.columns:
        frame = frame.sort_values(sort_by, ascending=False)
    return frame.reset_index(drop=True)
