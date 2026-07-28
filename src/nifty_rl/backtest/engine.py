"""Equal-weight multi-stock portfolio backtester.

Each ticker gets its own capital slice, cash bucket, position and trade log; the
portfolio equity curve is the sum of the per-ticker curves.

Fixes carried over from the notebook:

* **Stop-loss no longer re-enters on the next bar** (bug #4). The notebook set
  ``cur_sig = 0`` on a risk exit, which made ``prev_sig = 0``; if the underlying signal
  was still 1 the edge-triggered entry condition fired again the following bar. Stops
  therefore generated round-trip costs without providing protection -- which is why the
  4x4 SL/TP sensitivity grid came out nearly flat. Entry is now level-triggered with an
  explicit lockout that clears only when the signal returns to 0.
* **No phantom equity holes** (bug #5). The notebook aggregated with
  ``agg_equity.add(eq, fill_value=0)``. When one ticker lacked a date another had, that
  ticker contributed *zero* instead of its carried equity -- an artificial ~-10%
  portfolio drop. Series are now reindexed to the union of dates and forward filled
  before summing.
* **Per-ticker position series** (bug #8). The notebook returned only the aggregate
  0..N position count, which downstream metrics then compared against each individual
  ticker's forward return. Both are kept, and metrics use the per-ticker mapping.
* **Configurable execution lag** (bug #14). The notebook generated a signal from bar t's
  close and filled at bar t's close. ``execution="next_bar"`` shifts the signal one bar,
  which is the defensible default for a pipeline advertising lookahead-free evaluation.
* **Idle cash earns the risk-free rate** (bug #10). Cash returned 0% in the notebook. At
  an Indian risk-free near 6.5% that silently penalised every low-exposure strategy
  relative to fully-invested buy-and-hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import BacktestConfig, CostConfig
from .costs import BUY, SELL, CostModel, build_cost_model

SignalFn = Callable[[pd.DataFrame], pd.Series]


@dataclass
class BacktestResult:
    """Outcome of a portfolio backtest."""

    strategy: str
    equity: pd.Series
    positions: pd.Series  # aggregate count of tickers held, 0..N
    trades: pd.DataFrame
    per_ticker_equity: Dict[str, pd.Series] = field(default_factory=dict)
    per_ticker_positions: Dict[str, pd.Series] = field(default_factory=dict)
    weights: Optional[pd.DataFrame] = None

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def _daily_cash_rate(cfg: BacktestConfig) -> float:
    if cfg.cash_rate_annual <= 0:
        return 0.0
    return (1.0 + cfg.cash_rate_annual) ** (1.0 / cfg.trading_days) - 1.0


def _apply_execution_lag(signal: pd.Series, cfg: BacktestConfig) -> pd.Series:
    if cfg.execution == "next_bar":
        return signal.shift(1).fillna(0).astype(int)
    if cfg.execution == "same_close":
        return signal.fillna(0).astype(int)
    raise ValueError(
        f"Unknown execution mode {cfg.execution!r}; expected 'next_bar' or 'same_close'."
    )


def run_single_ticker_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    capital: float,
    cfg: BacktestConfig,
    cost_model: CostModel,
    adv: Optional[pd.Series] = None,
):
    """Backtest one ticker against its own cash bucket.

    Returns ``(equity, positions, trades)`` indexed by date.
    """
    frame = df.reset_index(drop=True)
    signal = _apply_execution_lag(
        signal.reset_index(drop=True).reindex(frame.index).fillna(0).astype(int), cfg
    )

    dates = pd.to_datetime(frame["Date"]).to_numpy()
    prices = frame["price"].astype(float).to_numpy()
    ticker = frame["ticker"].iloc[0] if "ticker" in frame.columns else "UNKNOWN"
    adv_values = adv.reindex(frame.index).to_numpy() if adv is not None else None

    daily_cash_rate = _daily_cash_rate(cfg)

    cash = float(capital)
    qty = 0
    entry_price: Optional[float] = None
    entry_date = None
    locked = False  # set by a risk exit; cleared when the signal returns to 0

    equity: List[float] = []
    positions: List[int] = []
    trades: List[dict] = []

    for i in range(len(frame)):
        price = prices[i]
        date = dates[i]
        sig = int(signal.iloc[i])
        adv_i = float(adv_values[i]) if adv_values is not None and np.isfinite(adv_values[i]) else None

        if i > 0 and daily_cash_rate:
            cash *= 1.0 + daily_cash_rate

        # A flat signal always clears the post-stop lockout.
        if sig == 0:
            locked = False

        if qty > 0 and entry_price is not None:
            trade_return = price / entry_price - 1.0
            hit_stop = cfg.stop_loss is not None and trade_return <= -cfg.stop_loss
            hit_target = cfg.take_profit is not None and trade_return >= cfg.take_profit
            risk_exit = hit_stop or hit_target

            if risk_exit or sig == 0:
                fill = cost_model.fill_price(SELL, price, qty, adv_i)
                cash += qty * fill
                trades.append(
                    {
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": fill,
                        "shares": qty,
                        "pnl": (fill - entry_price) * qty,
                        "return": fill / entry_price - 1.0,
                        "exit_reason": "stop" if hit_stop else ("target" if hit_target else "signal"),
                    }
                )
                qty = 0
                entry_price = None
                entry_date = None
                # Only a risk exit arms the lockout. A signal exit already implies
                # sig == 0, so the next entry needs a fresh 0 -> 1 transition anyway.
                locked = bool(risk_exit and cfg.reenter_lockout)

        if qty == 0 and sig == 1 and not locked and price > 0:
            fill = cost_model.fill_price(BUY, price, 1.0, adv_i)
            shares = int(cash // fill)
            if shares > 0:
                # Re-price with the actual size so market impact scales with the order.
                fill = cost_model.fill_price(BUY, price, shares, adv_i)
                shares = int(cash // fill)
            if shares > 0:
                cash -= shares * fill
                qty = shares
                entry_price = fill
                entry_date = date

        equity.append(cash + qty * price)
        positions.append(1 if qty > 0 else 0)

    # Liquidate any open position at the final close.
    if qty > 0 and entry_price is not None:
        price = prices[-1]
        fill = cost_model.fill_price(SELL, price, qty, None)
        cash += qty * fill
        trades.append(
            {
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": dates[-1],
                "entry_price": entry_price,
                "exit_price": fill,
                "shares": qty,
                "pnl": (fill - entry_price) * qty,
                "return": fill / entry_price - 1.0,
                "exit_reason": "final_liquidation",
            }
        )
        equity[-1] = cash
        positions[-1] = 0

    index = pd.DatetimeIndex(dates, name="Date")
    return (
        pd.Series(equity, index=index, name="equity"),
        pd.Series(positions, index=index, name="position"),
        trades,
    )


def _align_and_sum(series_map: Dict[str, pd.Series], fill_leading: Dict[str, float]) -> pd.Series:
    """Sum per-ticker series on the union of dates without inventing zeros.

    ``Series.add(..., fill_value=0)`` treats a missing date as *zero equity* rather than
    *carried equity*, which manufactures portfolio-wide drawdowns out of ragged
    calendars. Reindex to the union, forward fill, backfill the warm-up with the
    ticker's starting capital, then sum.
    """
    if not series_map:
        return pd.Series(dtype="float64")

    union = pd.DatetimeIndex(sorted(set().union(*[s.index for s in series_map.values()])))
    aligned = []
    for name, series in series_map.items():
        s = series.reindex(union).ffill()
        s = s.fillna(fill_leading.get(name, 0.0))
        aligned.append(s)
    total = aligned[0].copy()
    for s in aligned[1:]:
        total = total + s
    total.index.name = "Date"
    return total


def _align_and_sum_positions(series_map: Dict[str, pd.Series]) -> pd.Series:
    if not series_map:
        return pd.Series(dtype="int64")
    union = pd.DatetimeIndex(sorted(set().union(*[s.index for s in series_map.values()])))
    total = None
    for series in series_map.values():
        s = series.reindex(union).fillna(0)
        total = s if total is None else total + s
    total.index.name = "Date"
    return total.astype(int)


def run_portfolio_backtest(
    df: pd.DataFrame,
    signal_fn: SignalFn,
    strategy_name: str,
    cfg: BacktestConfig,
    cost_cfg: CostConfig,
    tickers: Optional[Sequence[str]] = None,
    cost_model: Optional[CostModel] = None,
) -> BacktestResult:
    """Run an equal-weight multi-stock backtest.

    ``signal_fn`` receives a single-ticker frame and returns a 0/1 Series. Signals are
    generated per ticker so no position state bleeds across ticker boundaries.
    """
    universe = list(tickers) if tickers is not None else list(pd.unique(df["ticker"]))
    if not universe:
        raise ValueError("No tickers to backtest.")

    model = cost_model if cost_model is not None else build_cost_model(cost_cfg)
    per_ticker_capital = cfg.initial_cash / len(universe)

    equity_map: Dict[str, pd.Series] = {}
    position_map: Dict[str, pd.Series] = {}
    all_trades: List[dict] = []

    for ticker in universe:
        ticker_df = df[df["ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if ticker_df.empty:
            continue

        signal = signal_fn(ticker_df)

        adv = None
        if cost_cfg.impact_coef > 0 and "Volume" in ticker_df.columns:
            adv = (
                (ticker_df["Volume"] * ticker_df["price"])
                .rolling(cost_cfg.impact_adv_window)
                .mean()
            )

        equity, positions, trades = run_single_ticker_backtest(
            ticker_df, signal, per_ticker_capital, cfg, model, adv
        )
        equity_map[ticker] = equity
        position_map[ticker] = positions
        all_trades.extend(trades)

    total_equity = _align_and_sum(
        equity_map, {t: per_ticker_capital for t in equity_map}
    )
    total_positions = _align_and_sum_positions(position_map)

    return BacktestResult(
        strategy=strategy_name,
        equity=total_equity,
        positions=total_positions,
        trades=pd.DataFrame(all_trades),
        per_ticker_equity=equity_map,
        per_ticker_positions=position_map,
    )
