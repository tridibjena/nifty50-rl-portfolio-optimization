"""Engine tests.

Each test named ``test_bug_NN_*`` fails against the notebook's original behaviour and
passes against the fixed engine. Where the difference is behavioural rather than a
crash, the test drives both configurations and asserts they differ.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import make_panel, make_ticker_frame

from nifty_rl.backtest.costs import FlatCostModel
from nifty_rl.backtest.engine import run_portfolio_backtest, run_single_ticker_backtest
from nifty_rl.config import BacktestConfig, CostConfig
from nifty_rl.strategies.signals import buy_hold_signals


# --------------------------------------------------------------- known-answer P&L


def test_roundtrip_pnl_is_exact(flat_cost_cfg, simple_bt_cfg):
    """Buy at 100, sell at 110, 15 bps each side, Rs10,000 of capital.

    buy fill  = 100 * 1.0015 = 100.15  -> int(10000 // 100.15) = 99 shares
    cash left = 10000 - 99 * 100.15    = 85.15
    sell fill = 110 * 0.9985 = 109.835
    final     = 85.15 + 99 * 109.835   = 10958.815
    """
    frame = make_ticker_frame([100.0, 105.0, 110.0, 108.0])
    signal = pd.Series([1, 1, 0, 0])

    equity, positions, trades = run_single_ticker_backtest(
        frame, signal, capital=10_000.0, cfg=simple_bt_cfg, cost_model=FlatCostModel(flat_cost_cfg)
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade["shares"] == 99
    assert trade["entry_price"] == pytest.approx(100.15)
    assert trade["exit_price"] == pytest.approx(109.835)
    assert trade["pnl"] == pytest.approx(958.815)
    assert trade["exit_reason"] == "signal"

    assert equity.iloc[0] == pytest.approx(85.15 + 99 * 100.0)
    assert equity.iloc[-1] == pytest.approx(10_958.815)
    assert positions.tolist() == [1, 1, 0, 0]


# ------------------------------------------------------- bug #4: stop-loss re-entry


def _stoploss_scenario(reenter_lockout: bool):
    """Stop out at bar 2 while the signal stays 1 for the whole path."""
    frame = make_ticker_frame([100.0, 100.0, 92.0, 95.0, 98.0, 101.0])
    cfg = BacktestConfig(
        initial_cash=10_000.0,
        stop_loss=0.06,
        take_profit=None,
        execution="same_close",
        cash_rate_annual=0.0,
        reenter_lockout=reenter_lockout,
    )
    cost_model = FlatCostModel(CostConfig(transaction_cost=0.001, slippage=0.0005))
    return run_single_ticker_backtest(
        frame, pd.Series([1] * 6), capital=10_000.0, cfg=cfg, cost_model=cost_model
    )


def test_bug_04_stoploss_does_not_reenter_next_bar():
    """A risk exit must not be undone by an immediate re-entry.

    The notebook set ``cur_sig = 0`` on a risk exit, so ``prev_sig`` became 0 and the
    edge-triggered entry fired again the very next bar whenever the signal was still 1.
    The stop then cost a round trip and protected nothing.
    """
    equity, positions, trades = _stoploss_scenario(reenter_lockout=True)

    assert len(trades) == 1, "lockout must prevent re-entry while the signal stays long"
    assert trades[0]["exit_reason"] == "stop"
    # Flat from the stop bar onward -- the signal never returns to 0, so never re-armed.
    assert positions.tolist() == [1, 1, 0, 0, 0, 0]
    # Equity is frozen after the stop because the position is closed.
    assert equity.iloc[2] == pytest.approx(equity.iloc[-1])


def test_bug_04_lockout_changes_behaviour_vs_notebook():
    """Guard against the fix being silently reverted."""
    _, locked_pos, locked_trades = _stoploss_scenario(reenter_lockout=True)
    _, open_pos, open_trades = _stoploss_scenario(reenter_lockout=False)

    assert len(open_trades) > len(locked_trades)
    assert open_pos.sum() > locked_pos.sum()


def test_lockout_clears_when_signal_goes_flat():
    """After a stop, a fresh 0 -> 1 transition must be tradeable again."""
    frame = make_ticker_frame([100.0, 92.0, 95.0, 98.0, 101.0, 104.0])
    signal = pd.Series([1, 1, 0, 0, 1, 1])  # goes flat at bars 2-3, re-arms at bar 4
    cfg = BacktestConfig(
        initial_cash=10_000.0,
        stop_loss=0.06,
        take_profit=None,
        execution="same_close",
        cash_rate_annual=0.0,
    )
    _, positions, trades = run_single_ticker_backtest(
        frame, signal, 10_000.0, cfg, FlatCostModel(CostConfig())
    )

    assert len(trades) == 2
    assert trades[0]["exit_reason"] == "stop"
    assert positions.tolist()[4] == 1, "signal returned to 0 then 1, so entry is allowed"


# --------------------------------------------------- bug #5: ragged-date equity holes


def test_bug_05_ragged_dates_do_not_create_equity_holes(no_cost_cfg):
    """A ticker missing a date must contribute carried equity, not zero.

    ``Series.add(other, fill_value=0)`` treats an absent date as zero equity, which
    manufactures a portfolio-wide drawdown out of a ragged calendar.
    """
    panel = make_panel({"A.NS": [100.0] * 5, "B.NS": [100.0] * 5})
    # Drop B's third date to make the panel ragged.
    missing_date = sorted(panel["Date"].unique())[2]
    panel = panel[~((panel["ticker"] == "B.NS") & (panel["Date"] == missing_date))]

    cfg = BacktestConfig(
        initial_cash=10_000.0,
        stop_loss=None,
        take_profit=None,
        execution="same_close",
        cash_rate_annual=0.0,
    )
    result = run_portfolio_backtest(
        panel, buy_hold_signals, "BuyHold", cfg, no_cost_cfg
    )

    equity = result.equity
    assert len(equity) == 5
    # Flat prices and zero costs -> equity is flat. A phantom hole would halve it.
    assert equity.std() == pytest.approx(0.0, abs=1e-6)
    assert equity.loc[missing_date] == pytest.approx(equity.iloc[0])
    assert equity.min() > 0.9 * equity.max()


# -------------------------------------------------------- bug #8: per-ticker positions


def test_bug_08_per_ticker_positions_are_exposed(no_cost_cfg, simple_bt_cfg):
    """Metrics need per-ticker positions, not just the aggregate 0..N count.

    The notebook reindexed the aggregate count against each individual ticker's forward
    return, so ``signal_accuracy`` really measured "invested in anything" -- which is why
    every strategy landed at 44-47%.
    """
    panel = make_panel({"A.NS": [100.0] * 4, "B.NS": [100.0] * 4})
    result = run_portfolio_backtest(
        panel, buy_hold_signals, "BuyHold", simple_bt_cfg, no_cost_cfg
    )

    assert set(result.per_ticker_positions) == {"A.NS", "B.NS"}
    for series in result.per_ticker_positions.values():
        assert set(series.unique()) <= {0, 1}
    # Aggregate is the sum, and is genuinely different information.
    assert result.positions.max() == 2


# ------------------------------------------------------------- bug #14: execution lag


def test_bug_14_next_bar_execution_delays_the_fill(no_cost_cfg):
    """Default execution must not fill on the same close that produced the signal."""
    frame = make_ticker_frame([100.0, 110.0, 120.0, 130.0])
    signal = pd.Series([1, 1, 1, 1])

    same_close = BacktestConfig(
        initial_cash=10_000.0, stop_loss=None, take_profit=None,
        execution="same_close", cash_rate_annual=0.0,
    )
    next_bar = BacktestConfig(
        initial_cash=10_000.0, stop_loss=None, take_profit=None,
        execution="next_bar", cash_rate_annual=0.0,
    )
    model = FlatCostModel(no_cost_cfg)

    _, pos_same, _ = run_single_ticker_backtest(frame, signal, 10_000.0, same_close, model)
    _, pos_next, _ = run_single_ticker_backtest(frame, signal, 10_000.0, next_bar, model)

    assert pos_same.iloc[0] == 1
    assert pos_next.iloc[0] == 0, "next_bar must skip the bar that generated the signal"
    assert pos_next.iloc[1] == 1


# -------------------------------------------------------------- bug #10: cash interest


def test_bug_10_idle_cash_accrues_interest(no_cost_cfg):
    """Uninvested capital must earn the risk-free rate.

    With cash at 0% the notebook penalised low-exposure strategies twice: once by not
    paying interest, and again through a Sharpe computed against rf = 0.
    """
    frame = make_ticker_frame([100.0] * 253)
    flat_signal = pd.Series([0] * 253)

    earning = BacktestConfig(
        initial_cash=10_000.0, stop_loss=None, take_profit=None,
        execution="same_close", cash_rate_annual=0.065,
    )
    idle = BacktestConfig(
        initial_cash=10_000.0, stop_loss=None, take_profit=None,
        execution="same_close", cash_rate_annual=0.0,
    )
    model = FlatCostModel(no_cost_cfg)

    eq_earning, _, _ = run_single_ticker_backtest(frame, flat_signal, 10_000.0, earning, model)
    eq_idle, _, _ = run_single_ticker_backtest(frame, flat_signal, 10_000.0, idle, model)

    assert eq_idle.iloc[-1] == pytest.approx(10_000.0)
    # 252 compounding steps over 253 bars (no accrual on the first bar).
    assert eq_earning.iloc[-1] == pytest.approx(10_000.0 * 1.065, rel=1e-3)


# ------------------------------------------------------------------- sanity guards


def test_unknown_execution_mode_raises(no_cost_cfg):
    frame = make_ticker_frame([100.0, 101.0])
    cfg = BacktestConfig(execution="teleport")
    with pytest.raises(ValueError, match="Unknown execution mode"):
        run_single_ticker_backtest(
            frame, pd.Series([1, 1]), 10_000.0, cfg, FlatCostModel(no_cost_cfg)
        )


def test_equal_weight_capital_is_split_across_universe(no_cost_cfg, simple_bt_cfg):
    panel = make_panel({f"T{i}.NS": [100.0] * 3 for i in range(4)})
    result = run_portfolio_backtest(
        panel, buy_hold_signals, "BuyHold", simple_bt_cfg, no_cost_cfg
    )
    # Zero costs, flat prices: total equity stays at initial_cash.
    assert result.equity.iloc[0] == pytest.approx(simple_bt_cfg.initial_cash)
