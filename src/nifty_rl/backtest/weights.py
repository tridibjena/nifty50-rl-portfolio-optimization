"""Weight-based portfolio backtester for continuous allocators.

The signal engine models ten independent per-ticker cash buckets, which suits binary
timing rules. Allocators emit a *joint* weight vector, so they need a single pooled cash
account -- and that difference is itself worth stating, because it means a timing rule
and an allocator are not automatically comparable. The original notebook compared PPO
(pooled, able to concentrate) against rule-based strategies (ten isolated buckets, unable
to reallocate) and called it apples-to-apples.

Rebalancing executes **all sells before any buys**. Doing it in one interleaved pass --
as the notebook's RL environment did -- means a sale late in the ticker list cannot fund
a purchase early in it, so target weights are frequently unreachable and whichever names
sit first in the universe tuple get systematic funding priority.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import BacktestConfig, CostConfig
from .costs import BUY, SELL, CostModel, build_cost_model
from .engine import BacktestResult

AllocatorFn = Callable[[pd.DataFrame], pd.Series]


def price_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Dates x tickers price matrix, forward filled within each column."""
    wide = panel.pivot_table(index="Date", columns="ticker", values="price", aggfunc="last")
    return wide.sort_index().ffill()


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "ME") -> List[pd.Timestamp]:
    """Last trading day of each period present in the index."""
    if len(index) == 0:
        return []
    marks = pd.Series(index, index=index).resample(frequency).last().dropna()
    return [d for d in marks.tolist() if d in set(index)]


def build_allocator_weights(
    prices: pd.DataFrame,
    allocator: AllocatorFn,
    lookback: int = 252,
    frequency: str = "ME",
    min_history: int = 60,
) -> pd.DataFrame:
    """Weights at each rebalance date, estimated from trailing returns only.

    The estimation window ends **at** the rebalance date, never after it. Fitting a
    covariance matrix on the full sample and then "rebalancing" through history is the
    classic way an allocator backtest becomes fiction.
    """
    returns = prices.pct_change()
    marks = rebalance_dates(prices.index, frequency)

    rows: Dict[pd.Timestamp, pd.Series] = {}
    for date in marks:
        window = returns.loc[:date].tail(lookback).dropna(how="all")
        if len(window) < min_history:
            continue
        usable = window.dropna(axis=1, how="any")
        if usable.shape[1] < 2:
            continue
        weights = allocator(usable)
        rows[date] = weights.reindex(prices.columns).fillna(0.0)

    if not rows:
        return pd.DataFrame(columns=prices.columns)
    return pd.DataFrame(rows).T.sort_index()


def run_weight_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    strategy_name: str,
    cfg: BacktestConfig,
    cost_cfg: CostConfig,
    cost_model: Optional[CostModel] = None,
) -> BacktestResult:
    """Backtest a schedule of target weights against a pooled cash account.

    ``weights`` carries a row only for rebalance dates (month ends by default), not for
    every trading day. On those dates the portfolio is traded back to target; on every
    other day it simply **drifts** with prices, which is what a real monthly-rebalanced
    fund does. Charging a trade every day would make turnover costs dominate everything.

    Rows of ``weights`` need not sum to one -- a shortfall is held as cash and earns the
    configured rate, which is how a regime exposure overlay expresses de-risking without
    silently renormalising back to fully invested.

    Rebalancing is two-pass, sells before buys, for the same reason as
    :meth:`envs.core.PortfolioSimulator.step`: proceeds from a sale must be available to
    fund a purchase, otherwise the achievable weights depend on column order. Both
    backtesters share this so PPO and the allocators are charged identically -- if they
    executed differently, any performance gap between them would be partly an artefact
    of the execution model rather than the strategy.
    """
    model = cost_model if cost_model is not None else build_cost_model(cost_cfg)
    tickers = list(prices.columns)
    n = len(tickers)

    daily_cash_rate = (
        (1.0 + cfg.cash_rate_annual) ** (1.0 / cfg.trading_days) - 1.0
        if cfg.cash_rate_annual > 0
        else 0.0
    )

    cash = float(cfg.initial_cash)
    shares = np.zeros(n)
    schedule = {d: weights.loc[d].to_numpy(dtype=float) for d in weights.index}

    equity_path: List[float] = []
    weight_path: List[np.ndarray] = []
    trades: List[dict] = []

    for i, date in enumerate(prices.index):
        px = prices.loc[date].to_numpy(dtype=float)
        px = np.where(np.isfinite(px) & (px > 0), px, np.nan)

        if i > 0 and daily_cash_rate:
            cash *= 1.0 + daily_cash_rate

        holdings_value = np.nansum(shares * px)
        portfolio_value = cash + holdings_value

        if date in schedule and portfolio_value > 0:
            target = np.nan_to_num(schedule[date], nan=0.0)
            target_value = target * portfolio_value
            current_value = np.nan_to_num(shares * px, nan=0.0)
            delta = target_value - current_value

            # --- pass 1: sells, which free cash for pass 2
            for k in range(n):
                if not np.isfinite(px[k]) or delta[k] >= 0:
                    continue
                quantity = min(int(abs(delta[k]) // px[k]), int(shares[k]))
                if quantity <= 0:
                    continue
                fill = model.fill_price(SELL, px[k], quantity)
                cash += quantity * fill
                shares[k] -= quantity
                trades.append(
                    {"date": date, "ticker": tickers[k], "side": "sell",
                     "shares": quantity, "price": fill, "value": quantity * fill}
                )

            # --- pass 2: buys, pro-rated if the plan exceeds available cash
            wanted = np.array(
                [max(delta[k], 0.0) if np.isfinite(px[k]) else 0.0 for k in range(n)]
            )
            total_wanted = wanted.sum()
            budget = max(cash, 0.0)
            scale = min(1.0, budget / total_wanted) if total_wanted > budget > 0 else 1.0

            for k in range(n):
                if wanted[k] <= 0 or not np.isfinite(px[k]):
                    continue
                fill = model.fill_price(BUY, px[k], 1.0)
                quantity = int((wanted[k] * scale) // fill)
                if quantity <= 0:
                    continue
                cost = quantity * model.fill_price(BUY, px[k], quantity)
                if cost > cash:
                    quantity = int(cash // fill)
                    if quantity <= 0:
                        continue
                    cost = quantity * model.fill_price(BUY, px[k], quantity)
                cash = max(cash - cost, 0.0)
                shares[k] += quantity
                trades.append(
                    {"date": date, "ticker": tickers[k], "side": "buy",
                     "shares": quantity, "price": cost / quantity, "value": cost}
                )

        holdings_value = np.nansum(shares * px)
        total = cash + holdings_value
        equity_path.append(total)
        weight_path.append(
            np.nan_to_num(shares * px, nan=0.0) / total if total > 0 else np.zeros(n)
        )

    equity = pd.Series(equity_path, index=prices.index, name="equity")
    realised_weights = pd.DataFrame(weight_path, index=prices.index, columns=tickers)
    positions = pd.Series((realised_weights > 1e-6).sum(axis=1), index=prices.index, name="position")

    per_ticker_positions = {
        ticker: (realised_weights[ticker] > 1e-6).astype(int) for ticker in tickers
    }

    return BacktestResult(
        strategy=strategy_name,
        equity=equity,
        positions=positions,
        trades=pd.DataFrame(trades),
        per_ticker_positions=per_ticker_positions,
        weights=realised_weights,
    )


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One-sided turnover per period -- the quantity a cost model actually charges."""
    return weights.diff().abs().sum(axis=1) / 2.0


def concentration_hhi(weights: pd.DataFrame) -> pd.Series:
    """Herfindahl index of realised weights: 1/n is equal-weight, 1.0 is all-in-one."""
    return (weights ** 2).sum(axis=1)
