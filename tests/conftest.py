"""Shared synthetic fixtures.

Tests must never hit the network. Every frame here is constructed in-process so the
suite is deterministic and runnable in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_rl.config import BacktestConfig, CostConfig


def make_ticker_frame(prices, ticker="TEST.NS", start="2024-01-01"):
    """Minimal single-ticker frame with the columns the engine needs."""
    prices = np.asarray(prices, dtype=float)
    dates = pd.bdate_range(start=start, periods=len(prices))
    return pd.DataFrame(
        {
            "Date": dates,
            "ticker": ticker,
            "price": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": np.full(len(prices), 1_000_000.0),
        }
    )


def make_panel(price_map, start="2024-01-01"):
    """Multi-ticker panel from ``{ticker: prices}``."""
    frames = [make_ticker_frame(p, ticker=t, start=start) for t, p in price_map.items()]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def no_cost_cfg():
    """Zero-cost model, for isolating engine mechanics from cost arithmetic."""
    return CostConfig(model="flat", transaction_cost=0.0, slippage=0.0, impact_coef=0.0)


@pytest.fixture
def flat_cost_cfg():
    """Notebook-compatible 10 bps + 5 bps."""
    return CostConfig(model="flat", transaction_cost=0.0010, slippage=0.0005, impact_coef=0.0)


@pytest.fixture
def simple_bt_cfg():
    """Same-bar execution, no risk exits, no cash interest -- exact arithmetic."""
    return BacktestConfig(
        initial_cash=10_000.0,
        stop_loss=None,
        take_profit=None,
        execution="same_close",
        cash_rate_annual=0.0,
    )
