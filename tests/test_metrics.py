"""Metric tests, including the degenerate-cash guard.

The all-cash case is not hypothetical: in the real test window ``RSI_35_60`` never
entered a position, so its equity grew at exactly the risk-free rate. Excess return and
excess volatility were both zero, and the naive ``mean/std`` produced a Sharpe of
2.1e13 -- which ranked a do-nothing rule first on every leaderboard and heatmap in the
run. These tests pin that behaviour everywhere it is computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_rl.config import MetricsConfig
from nifty_rl.metrics.performance import (
    information_ratio,
    longest_drawdown_days,
    sharpe_ratio,
    signal_accuracy_per_ticker,
    sortino_ratio,
    ulcer_index,
)
from nifty_rl.metrics.stats import (
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    sharpe_from_returns,
)


@pytest.fixture
def cfg():
    return MetricsConfig(risk_free_annual=0.065, trading_days=252)


def _cash_returns(n=250, annual=0.065, trading_days=252):
    daily = (1.0 + annual) ** (1.0 / trading_days) - 1.0
    return pd.Series([daily] * n)


# ------------------------------------------------------- degenerate cash portfolios


def test_all_cash_sharpe_is_undefined_not_enormous(cfg):
    """Excess return and excess vol are both zero, so the ratio is 0/0."""
    assert np.isnan(sharpe_ratio(_cash_returns(), cfg))
    assert np.isnan(sortino_ratio(_cash_returns(), cfg))


def test_all_cash_sharpe_guard_survives_float_noise(cfg):
    """Compounding introduces ~1e-19 jitter; the guard must be wider than that."""
    rng = np.random.default_rng(0)
    noisy = _cash_returns() + rng.normal(0, 1e-18, 250)
    assert np.isnan(sharpe_ratio(noisy, cfg))


def test_stats_sharpe_has_the_same_guard():
    """Both Sharpe implementations must agree, or leaderboards and heatmaps diverge."""
    assert np.isnan(sharpe_from_returns(_cash_returns().to_numpy()))


def test_a_real_strategy_still_gets_a_finite_sharpe(cfg):
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0008, 0.011, 500))
    value = sharpe_ratio(returns, cfg)
    assert np.isfinite(value) and abs(value) < 10


def test_risk_free_rate_lowers_sharpe(cfg):
    """A 1.76% return over a year is a loss against 6.5% cash, not a win.

    The notebook used rf = 0, so any positive number read as success.
    """
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0001, 0.004, 252))
    with_rf = sharpe_ratio(returns, cfg)
    without_rf = sharpe_ratio(returns, MetricsConfig(risk_free_annual=0.0))
    assert with_rf < without_rf


# ------------------------------------------------------------- bug #9: IR scaling


def test_bug_09_information_ratio_is_annualised_once(cfg):
    """The notebook's two sqrt(252) factors cancelled, leaving a daily ratio."""
    rng = np.random.default_rng(11)
    strategy = pd.Series(rng.normal(0.0010, 0.010, 400))
    benchmark = pd.Series(rng.normal(0.0004, 0.009, 400))

    value = information_ratio(strategy, benchmark, cfg)

    active = strategy - benchmark
    expected = active.mean() / active.std() * np.sqrt(252)
    assert value == pytest.approx(expected)

    notebook_te = active.std() * np.sqrt(252)
    notebook_ir = active.mean() / notebook_te * np.sqrt(252)
    assert notebook_ir == pytest.approx(value / np.sqrt(252))


# ---------------------------------------------------- bug #8: per-ticker accuracy


def test_bug_08_accuracy_is_per_ticker_not_aggregate():
    """Ticker A held and rising; ticker B never held and falling.

    Per ticker: only A contributes, and it was right every day -> 1.0.
    The notebook compared the *aggregate* position count against B's returns too, which
    dragged the number toward 0.5 and made every strategy look identical.
    """
    dates = pd.bdate_range("2024-01-01", periods=6)
    panel = pd.concat(
        [
            pd.DataFrame({"Date": dates, "ticker": "A.NS", "price": np.linspace(100, 110, 6)}),
            pd.DataFrame({"Date": dates, "ticker": "B.NS", "price": np.linspace(100, 90, 6)}),
        ],
        ignore_index=True,
    )
    positions = {
        "A.NS": pd.Series(1, index=dates),
        "B.NS": pd.Series(0, index=dates),
    }
    assert signal_accuracy_per_ticker(positions, panel) == pytest.approx(1.0)


# ------------------------------------------------------------- deflated Sharpe


def test_expected_maximum_sharpe_grows_with_trials():
    """More trials means a higher bar for the winner."""
    low = expected_maximum_sharpe(5, 0.25)
    high = expected_maximum_sharpe(500, 0.25)
    assert high > low > 0


def test_deflated_sharpe_penalises_a_wide_search():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0006, 0.010, 500)
    few = deflated_sharpe_ratio(returns, n_trials=2, trial_sharpes=[0.4, 0.6])
    many = deflated_sharpe_ratio(returns, n_trials=200, trial_sharpes=list(rng.normal(0, 0.8, 200)))
    assert many["deflation_threshold"] > few["deflation_threshold"]
    assert many["dsr"] <= few["dsr"]


# ------------------------------------------------------------------ drawdown shape


def test_ulcer_index_penalises_long_drawdowns():
    quick = pd.Series([100, 90, 100, 101, 102, 103.0])
    slow = pd.Series([100, 90, 91, 92, 93, 94.0])
    assert ulcer_index(slow) > ulcer_index(quick)


def test_longest_drawdown_days_counts_the_worst_run():
    equity = pd.Series([100, 99, 98, 101, 100, 99, 98, 97, 105.0])
    assert longest_drawdown_days(equity) == 4


def test_bootstrap_ci_brackets_its_own_point_estimate():
    """The CI and the point estimate must apply the risk-free rate the same number of times.

    A double subtraction (passing rf-adjusted returns to a statistic that also nets out
    rf) shifts the whole interval downward, producing the impossible situation of a
    point estimate lying outside its own confidence band.
    """
    from nifty_rl.metrics.stats import summarise_significance

    rng = np.random.default_rng(19)
    returns = {
        "A": pd.Series(rng.normal(0.0003, 0.008, 400)),
        "B": pd.Series(rng.normal(-0.0002, 0.012, 400)),
    }
    frame = summarise_significance(
        returns, "A", n_trials=10, risk_free_annual=0.065,
        exposure_by_strategy={"A": 0.9, "B": 0.9},
    )
    for _, row in frame.iterrows():
        assert row["sharpe_ci_lower"] <= row["sharpe"] <= row["sharpe_ci_upper"], (
            f"{row['strategy']}: point {row['sharpe']:.3f} outside CI "
            f"[{row['sharpe_ci_lower']:.3f}, {row['sharpe_ci_upper']:.3f}]"
        )
