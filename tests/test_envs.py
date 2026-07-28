"""Environment tests — the notebook's RL bugs, exercised without a training loop.

All of these run against :class:`PortfolioSimulator`, which has no gymnasium dependency.
That is the point of the split: the execution model is where the bugs lived, and it does
not need a reinforcement-learning framework to be checked.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from nifty_rl.backtest.costs import BUY, SELL, FlatCostModel
from nifty_rl.config import BacktestConfig, CostConfig
from nifty_rl.envs.core import PortfolioSimulator
from nifty_rl.envs.panel import build_panel_arrays
from nifty_rl.envs.rewards import (
    DifferentialSharpeReward,
    RegimeAwareReward,
    ShapedReward,
    escalating_drawdown_penalty,
)

TICKERS = ["AAA.NS", "BBB.NS"]
FEATURES = ["ret", "rsi"]


def make_panel(n_days=40, prices=None, regime=None):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for d_i, date in enumerate(dates):
        for t_i, ticker in enumerate(TICKERS):
            price = prices[d_i][t_i] if prices is not None else 100.0
            rows.append(
                {"Date": date, "ticker": ticker, "price": price, "ret": 0.001 * t_i, "rsi": 50.0}
            )
    frame = pd.DataFrame(rows)
    labels = pd.Series(regime, index=dates) if regime is not None else None
    return build_panel_arrays(frame, TICKERS, FEATURES, labels)


def make_sim(panel, **cfg_kwargs):
    cfg = BacktestConfig(initial_cash=100_000.0, cash_rate_annual=0.0, **cfg_kwargs)
    return PortfolioSimulator(panel, cfg, FlatCostModel(CostConfig()), ShapedReward())


def target_for(index: int) -> np.ndarray:
    """Softmax logits that concentrate on one slot (last slot = cash)."""
    logits = np.full(len(TICKERS) + 1, -20.0)
    logits[index] = 20.0
    return logits


# ------------------------------------------ bug #2: sells must settle before buys


def test_bug_02_a_sale_funds_a_purchase_earlier_in_the_ticker_order():
    """Rotate from the LAST ticker into the FIRST.

    The notebook ran one interleaved loop over ``self.tickers``: index 0 was processed as
    a buy while the cash to fund it was still tied up in index 1, which had not been sold
    yet. Target weights were unreachable whenever the sale sat later in the tuple.
    """
    panel = make_panel()
    sim = make_sim(panel)

    sim.step(target_for(1))  # go all-in on BBB (index 1)
    assert sim.shares[1] > 0 and sim.shares[0] == 0

    sim.step(target_for(0))  # rotate into AAA (index 0)

    assert sim.shares[0] > 0, "sale proceeds from BBB must be spendable on AAA"
    assert sim.shares[1] == 0
    assert sim.weights[0] > 0.9, f"expected near-full AAA weight, got {sim.weights[0]:.3f}"


def test_bug_02_one_pass_execution_would_fail_the_same_rotation():
    """Reference implementation of the notebook's interleaved loop, to prove the bug."""
    panel = make_panel()
    sim = make_sim(panel)
    sim.step(target_for(1))

    prices = panel.prices[sim.i].astype(float)
    value = sim.portfolio_value(prices)
    target = sim.softmax(target_for(0))
    delta = target[: sim.n] * value - sim.shares * prices

    cash, shares = sim.cash, sim.shares.copy()
    for k in range(sim.n):  # single interleaved pass, ticker order
        price = prices[k]
        if delta[k] > 0:
            quantity = int(delta[k] // price)
            cost = quantity * price * 1.0015
            if quantity > 0 and cost <= cash:
                shares[k] += quantity
                cash -= cost
        elif delta[k] < 0:
            quantity = min(int(abs(delta[k]) // price), int(shares[k]))
            shares[k] -= quantity
            cash += quantity * price * 0.9985

    assert shares[0] == 0, "one-pass execution cannot fund the buy — this is the bug"


# ------------------------------------------------ bug #3: observed weights are real


def test_bug_03_observed_weights_match_actual_holdings():
    """The agent must see what it got, not what it asked for."""
    panel = make_panel()
    sim = make_sim(panel)
    sim.step(target_for(0))

    prices = panel.prices[sim.i].astype(float)
    expected = (sim.shares * prices) / sim.net
    np.testing.assert_allclose(sim.weights[: sim.n], expected, atol=1e-6)
    assert sim.weights.sum() == pytest.approx(1.0, abs=1e-5)


def test_bug_03_weights_diverge_from_the_requested_target():
    """Integer share sizing and costs guarantee the two are not equal.

    The notebook set ``self.weights = target_w``, so this difference was invisible to the
    agent -- 11 of its 172 observation dimensions were fiction.
    """
    panel = make_panel()
    sim = make_sim(panel)
    logits = np.array([1.0, 1.0, 1.0])  # equal-ish across AAA, BBB, cash
    sim.step(logits)

    requested = sim.softmax(logits)
    assert not np.allclose(sim.weights, requested, atol=1e-3)


def test_weight_cap_is_respected():
    panel = make_panel()
    cfg = BacktestConfig(initial_cash=100_000.0, cash_rate_annual=0.0)
    sim = PortfolioSimulator(
        panel, cfg, FlatCostModel(CostConfig()), ShapedReward(), max_weight=0.25
    )
    sim.step(target_for(0))
    assert sim.weights[0] <= 0.26


# --------------------------------------------------------- bug #7: trade log exists


def test_bug_07_trades_are_logged():
    """``self.trades`` was initialised and never appended to, so PPO's win rate was NaN."""
    panel = make_panel()
    sim = make_sim(panel)
    sim.step(target_for(0))
    sim.step(target_for(1))

    assert len(sim.trades) >= 2
    frame = pd.DataFrame(sim.trades)
    assert set(frame.columns) >= {"date", "ticker", "side", "shares", "price", "value"}
    assert set(frame["side"]) <= {"buy", "sell"}
    assert (frame["shares"] > 0).all()


# ----------------------------------------------------- bug #24: observation bounds


def test_bug_24_observations_stay_inside_the_declared_box():
    """A scaled feature can exceed +-10 on a tail event; the notebook only ran nan_to_num."""
    dates = pd.bdate_range("2024-01-01", periods=20)
    rows = []
    for date in dates:
        for t_i, ticker in enumerate(TICKERS):
            rows.append(
                {"Date": date, "ticker": ticker, "price": 100.0, "ret": 500.0, "rsi": -750.0}
            )
    panel = build_panel_arrays(pd.DataFrame(rows), TICKERS, FEATURES)
    sim = make_sim(panel)

    obs = sim.observe()
    assert obs.min() >= -10.0 and obs.max() <= 10.0
    assert np.isfinite(obs).all()


def test_missing_prices_are_flagged_not_forward_filled():
    """An absent ticker gets zeroed features and availability 0, never a stale price."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    rows = [
        {"Date": d, "ticker": TICKERS[0], "price": 100.0, "ret": 0.01, "rsi": 55.0}
        for d in dates
    ]
    panel = build_panel_arrays(pd.DataFrame(rows), TICKERS, FEATURES)
    assert panel.available[:, 1].sum() == 0
    assert np.all(panel.features[:, 1, :] == 0)


# ------------------------------------------------- bug #25: regime read at decision


def test_bug_25_regime_penalty_uses_todays_regime_not_tomorrows():
    """The notebook incremented the day index *before* reading the regime.

    Today's holding was therefore penalised with tomorrow's regime label -- a one-step
    lookahead inside the reward.
    """
    n_days = 20
    switch_at = 10
    regime = [0] * switch_at + [1] * (n_days - switch_at)
    panel = make_panel(n_days=n_days, regime=regime)
    sim = make_sim(panel)

    for _ in range(switch_at - 1):
        _, _, _, _, info = sim.step(target_for(0))

    assert sim.i == switch_at - 1
    _, _, _, _, info = sim.step(target_for(0))
    assert info.regime == 0, "reward must see the regime in force when the trade was made"
    assert panel.regime[sim.i] == 1, "the next day is already the new regime"


# ------------------------------------------------------------------ reward objects


def test_shaped_reward_penalises_drawdown_and_turnover():
    reward = ShapedReward()
    kwargs = dict(net=100.0, prev_net=100.0, turnover=0.0, regime=0, holding=True)
    clean = reward(drawdown=0.0, n_trades=0, **kwargs)
    reward.reset()
    stressed = reward(drawdown=-0.20, n_trades=50, **kwargs)
    assert stressed < clean


def _accumulate_dsr(returns, **kwargs):
    """Feed a return stream through DSR; return (reward_object, final_net, total)."""
    reward = DifferentialSharpeReward(warmup_steps=20, **kwargs)
    net, total = 100.0, 0.0
    for r in returns:
        step_net = net * (1.0 + r)
        total += reward(
            net=step_net, prev_net=net, drawdown=0.0, n_trades=0,
            turnover=0.0, regime=0, holding=True,
        )
        net = step_net
    return reward, net, total


def _one_shock(reward, net, shock):
    """Apply a single return to a copy, leaving the original state untouched."""
    probe = copy.deepcopy(reward)
    return probe(
        net=net * (1.0 + shock), prev_net=net, drawdown=0.0, n_trades=0,
        turnover=0.0, regime=0, holding=True,
    )


def test_differential_sharpe_signs_track_the_running_mean():
    """DSR is a *marginal* quantity: it measures whether this step improved the Sharpe.

    Above the running mean must score positive, below it negative. Summing DSR over an
    episode does not equal the final Sharpe -- it telescopes only approximately -- so the
    defining property is tested directly rather than through a cumulative total.
    """
    rng = np.random.default_rng(0)
    reward, net, _ = _accumulate_dsr(rng.normal(0.001, 0.010, 300))

    assert _one_shock(reward, net, 0.02) > 0
    assert _one_shock(reward, net, -0.02) < 0


def test_differential_sharpe_values_a_gain_more_when_volatility_is_low():
    """The same +2% is worth more in a calm series than a turbulent one.

    This is why DSR replaces a shaped reward: risk aversion falls out of the objective
    instead of being bolted on with four hand-tuned penalty constants.
    """
    rng = np.random.default_rng(1)
    calm_reward, calm_net, _ = _accumulate_dsr(rng.normal(0.001, 0.003, 300))
    wild_reward, wild_net, _ = _accumulate_dsr(rng.normal(0.001, 0.020, 300))

    assert _one_shock(calm_reward, calm_net, 0.02) > _one_shock(wild_reward, wild_net, 0.02)


def test_differential_sharpe_is_stable_on_a_constant_return_series():
    """A constant series has zero variance, so the ratio's denominator vanishes.

    Without a warm-up and a variance floor this pinned to the clip bound on every step,
    emitting a large constant signal that had nothing to do with performance -- a
    perfectly steady gain scored -280.
    """
    _, _, total = _accumulate_dsr([0.001] * 100)
    assert abs(total) < 1e-6


def test_differential_sharpe_returns_zero_during_warmup():
    reward = DifferentialSharpeReward(warmup_steps=5)
    values = [
        reward(net=101.0, prev_net=100.0, drawdown=0.0, n_trades=0,
               turnover=0.0, regime=0, holding=True)
        for _ in range(5)
    ]
    assert all(v == 0.0 for v in values)


def test_regime_aware_reward_penalises_drawdown_harder_in_crisis():
    ladder = escalating_drawdown_penalty(4, base=0.02, top=0.10)
    reward = RegimeAwareReward(ladder)
    kwargs = dict(net=100.0, prev_net=100.0, drawdown=-0.15, n_trades=0, turnover=0.0, holding=True)

    calm = reward(regime=0, **kwargs)
    reward.reset()
    crisis = reward(regime=3, **kwargs)
    assert crisis < calm


# ------------------------------------------------------------------ panel plumbing


def test_panel_shape_and_observation_size():
    panel = make_panel(n_days=25)
    assert panel.features.shape == (25, 2, 2)
    assert panel.prices.shape == (25, 2)
    # 2 tickers x (2 features + 1 flag) + (2 + 1) weights + 1 cash ratio
    assert panel.observation_size() == 2 * 3 + 3 + 1


def test_simulator_terminates_at_the_end_of_the_panel():
    panel = make_panel(n_days=12)
    sim = make_sim(panel)
    steps, terminated = 0, False
    while not terminated and steps < 100:
        _, _, terminated, _, _ = sim.step(target_for(2))  # stay in cash
        steps += 1
    assert terminated
    assert steps == panel.n_dates - 1
