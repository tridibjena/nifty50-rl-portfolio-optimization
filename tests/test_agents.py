"""Training-protocol tests.

These do not train anything. The point is the *protocol* -- which policy gets selected
and whether selection and deployment use the same rule -- and that logic is testable with
a stub policy, without a training loop or the RL stack.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from nifty_rl.agents.train import PPO_DEFAULTS, SharpeCheckpoint, evaluate_policy_on_panel
from nifty_rl.config import BacktestConfig, CostConfig
from nifty_rl.envs.panel import build_panel_arrays

TICKERS = ["AAA.NS", "BBB.NS"]
FEATURES = ["ret", "rsi"]


def make_panel(prices_by_day):
    dates = pd.bdate_range("2024-01-01", periods=len(prices_by_day))
    rows = []
    for day, date in enumerate(dates):
        for i, ticker in enumerate(TICKERS):
            rows.append(
                {
                    "Date": date,
                    "ticker": ticker,
                    "price": float(prices_by_day[day][i]),
                    "ret": 0.0,
                    "rsi": 50.0,
                }
            )
    return build_panel_arrays(pd.DataFrame(rows), TICKERS, FEATURES)


class StubPolicy:
    """Minimal stand-in for an SB3 model: emits a fixed action, records calls."""

    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0
        self.policy = StubTorchModule()

    def predict(self, obs, deterministic=True):
        self.calls += 1
        return self.action, None


class StubTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return StubTensor(self.value)


class StubTorchModule:
    def __init__(self):
        self._state = {"w": StubTensor(0.0)}

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self._state = state


@pytest.fixture
def rising_panel():
    return make_panel([[100 + i, 100 + i] for i in range(40)])


@pytest.fixture
def falling_panel():
    return make_panel([[100 - i * 0.5, 100 - i * 0.5] for i in range(40)])


def _cfg():
    return BacktestConfig(initial_cash=100_000.0, cash_rate_annual=0.0)


# ------------------------------------------------------------------- evaluation


def test_evaluate_returns_a_standard_backtest_result(rising_panel):
    """A policy roll-out must be comparable with every other strategy in the project."""
    model = StubPolicy([5.0, 5.0, -5.0])  # concentrate in equities
    result = evaluate_policy_on_panel(model, rising_panel, _cfg(), CostConfig(), "PPO_test")

    assert result.strategy == "PPO_test"
    assert len(result.equity) == rising_panel.n_dates
    assert result.weights is not None and list(result.weights.columns) == TICKERS
    assert model.calls == rising_panel.n_dates - 1


def test_policy_that_buys_gains_in_a_rising_market(rising_panel):
    long_model = StubPolicy([5.0, 5.0, -5.0])
    cash_model = StubPolicy([-5.0, -5.0, 5.0])

    long_result = evaluate_policy_on_panel(long_model, rising_panel, _cfg(), CostConfig())
    cash_result = evaluate_policy_on_panel(cash_model, rising_panel, _cfg(), CostConfig())

    assert long_result.equity.iloc[-1] > cash_result.equity.iloc[-1]


def test_trades_are_recorded_for_a_policy_rollout(rising_panel):
    """PPO's trade log was empty in the notebook; every metric downstream needs it."""
    model = StubPolicy([5.0, 5.0, -5.0])
    result = evaluate_policy_on_panel(model, rising_panel, _cfg(), CostConfig())
    assert not result.trades.empty
    assert {"date", "ticker", "side", "shares"} <= set(result.trades.columns)


# ---------------------------------------------- bug #12: one protocol, not two


def test_checkpoint_keeps_the_best_policy_not_the_last(rising_panel):
    """The deployed model must be the best validation policy seen, not the final iterate.

    The notebook scored hyperparameter candidates on their final iterate (no callback was
    passed during tuning) but deployed the best checkpoint. Selection and deployment ran
    under different rules, so the comparison that picked the winner did not describe the
    thing that shipped.
    """
    checkpoint = SharpeCheckpoint(rising_panel, _cfg(), CostConfig(), check_freq=1)

    good = StubPolicy([5.0, 5.0, -5.0])   # invests in a rising market
    bad = StubPolicy([-5.0, -5.0, 5.0])   # sits in cash

    assert checkpoint.consider(good, step=1) is True
    best_after_good = checkpoint.best_sharpe

    assert checkpoint.consider(bad, step=2) is False
    assert checkpoint.best_sharpe == best_after_good, "a worse policy must not overwrite the best"

    assert len(checkpoint.history) == 2
    assert [h["improved"] for h in checkpoint.history] == [True, False]


def test_checkpoint_restores_the_saved_weights(rising_panel):
    checkpoint = SharpeCheckpoint(rising_panel, _cfg(), CostConfig(), check_freq=1)
    good = StubPolicy([5.0, 5.0, -5.0])
    checkpoint.consider(good, step=1)

    target = StubPolicy([0.0, 0.0, 0.0])
    target.policy._state = {"w": StubTensor(999.0)}
    checkpoint.restore(target)

    assert target.policy.state_dict()["w"].value == 0.0, "saved weights must be restored"


def test_checkpoint_restore_is_a_noop_when_nothing_improved(falling_panel):
    """With no finite improvement there is nothing to restore, and it must not crash."""
    checkpoint = SharpeCheckpoint(falling_panel, _cfg(), CostConfig(), check_freq=1)
    model = StubPolicy([0.0, 0.0, 0.0])
    restored = checkpoint.restore(model)
    assert restored is model


def test_ppo_defaults_are_internally_consistent():
    """batch_size must divide the rollout, or SB3 silently drops the remainder."""
    assert PPO_DEFAULTS["n_steps"] % PPO_DEFAULTS["batch_size"] == 0
    assert 0.0 <= PPO_DEFAULTS["ent_coef"] < 0.1
    assert 0.0 < PPO_DEFAULTS["clip_range"] <= 0.3
