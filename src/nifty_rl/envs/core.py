"""Portfolio simulation core — pure NumPy, no gymnasium dependency.

Deliberately separated from the Gym adapter. The execution model, the weight bookkeeping
and the reward are where the notebook's environment bugs lived, and none of them needs a
reinforcement-learning framework to be exercised. Keeping them here means they are unit
tested directly rather than through a training loop.

Fixes carried over from the notebook's ``MultiStockPPOEnv``:

* **Sells settle before buys** (bug #2). The original ran one interleaved loop over
  ``self.tickers``, so a sale of the ninth name could not fund a purchase of the first,
  and whichever tickers sat early in the tuple got systematic funding priority. Target
  weights were frequently unreachable — the agent could not execute its own policy.
* **Observed weights are realised, not intended** (bug #3). The original assigned
  ``self.weights = target_w``, so 11 of 172 observation dimensions described what the
  agent asked for rather than what it got. Combined with bug #2 the two diverged
  constantly.
* **The trade log is populated** (bug #7). ``self.trades`` was initialised and never
  appended to, so PPO's win rate and payoff ratio were NaN in every results table while
  every other strategy had them.
* **Observations are clipped to the declared box** (bug #24). ``Box(-10, 10)`` was
  declared but only ``nan_to_num`` applied, so a large scaled feature silently left the
  space it advertised.
* **The regime penalty reads today, not tomorrow** (bug #25). The original incremented
  the day index before looking up ``high_vix_regime``, penalising today's holding with
  tomorrow's regime.
* **Idle cash earns the risk-free rate**, matching the backtest engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..backtest.costs import BUY, SELL, CostModel
from ..config import BacktestConfig
from .panel import PanelArrays
from .rewards import RewardFunction, ShapedReward


@dataclass
class StepInfo:
    """Diagnostics emitted alongside every step. Not part of the observation.

    The agent never sees these -- they exist so a trained policy can be *described*
    rather than only scored. ``gross_exposure`` and ``hhi`` are what revealed this
    project's central finding: PPO settled at 99% invested with a correlation of 0.993
    to buy-and-hold, meaning it had learned to be the benchmark while paying turnover.
    Without logging behaviour per step, that shows up only as a slightly worse number.
    """

    net_worth: float
    cash: float
    n_positions: int
    n_trades: int
    turnover: float
    gross_exposure: float
    hhi: float
    drawdown: float
    regime: int


class PortfolioSimulator:
    """Continuous-allocation portfolio simulation over a dense panel."""

    def __init__(
        self,
        panel: PanelArrays,
        cfg: BacktestConfig,
        cost_model: CostModel,
        reward_fn: Optional[RewardFunction] = None,
        max_weight: Optional[float] = None,
    ):
        self.panel = panel
        self.cfg = cfg
        self.cost_model = cost_model
        self.reward_fn = reward_fn or ShapedReward()
        self.max_weight = max_weight

        self.n = panel.n_tickers
        self.daily_cash_rate = (
            (1.0 + cfg.cash_rate_annual) ** (1.0 / cfg.trading_days) - 1.0
            if cfg.cash_rate_annual > 0
            else 0.0
        )
        self.observation_size = panel.observation_size()
        self.reset()

    # ------------------------------------------------------------------- lifecycle

    def reset(self) -> np.ndarray:
        """Return to day 0, fully in cash, and clear all history.

        The reward function is reset too. Rewards here are stateful -- the differential
        Sharpe reward carries running return moments -- so a stale estimate from the
        previous episode would leak across the boundary and score the first steps of a
        new episode against the last episode's volatility.
        """
        self.i = 0
        self.cash = float(self.cfg.initial_cash)
        self.shares = np.zeros(self.n, dtype=np.float64)
        self.net = self.prev_net = self.peak = float(self.cfg.initial_cash)
        self.weights = np.zeros(self.n + 1, dtype=np.float32)
        self.weights[-1] = 1.0
        self.trades: List[dict] = []
        self.equity_path: List[float] = [self.net]
        self.weight_path: List[np.ndarray] = [self.weights.copy()]
        self.reward_fn.reset()
        return self.observe()

    # ---------------------------------------------------------------- observations

    def observe(self) -> np.ndarray:
        """Feature block per ticker + availability flag, then weights and cash ratio."""
        features = self.panel.features[self.i]  # (n_tickers, n_features)
        available = self.panel.available[self.i].reshape(-1, 1)
        block = np.concatenate([features, available], axis=1).ravel()
        cash_ratio = np.float32(self.cash / max(self.cfg.initial_cash, 1e-9))
        obs = np.concatenate([block, self.weights, [cash_ratio]]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        # Declared space is Box(-10, 10); a scaled feature can exceed that on a tail
        # event, so clip rather than silently emit out-of-space observations.
        return np.clip(obs, -10.0, 10.0)

    # --------------------------------------------------------------------- helpers

    @staticmethod
    def softmax(action: np.ndarray) -> np.ndarray:
        """Turn a raw action vector into long-only weights summing to 1.

        This is how the action space stays unconstrained -- PPO emits any real numbers it
        likes and the portfolio constraints are imposed here, rather than by asking the
        optimiser to respect a simplex. The last element is the cash weight, so choosing
        to hold cash is an ordinary action rather than a special case.

        Subtracting the maximum before exponentiating changes nothing mathematically and
        prevents ``exp`` overflowing on a large action. A degenerate vector falls back to
        100% cash, which is the safe default -- never an accidental leveraged position.
        """
        a = np.asarray(action, dtype=np.float64).ravel()
        a = a - a.max()
        e = np.exp(a)
        total = e.sum()
        if not np.isfinite(total) or total <= 0:
            out = np.zeros_like(e)
            out[-1] = 1.0
            return out
        return e / total

    def portfolio_value(self, prices: np.ndarray) -> float:
        return float(max(self.cash, 0.0) + np.dot(self.shares, prices))

    def realised_weights(self, prices: np.ndarray, net: float) -> np.ndarray:
        """Weights implied by actual holdings — what the agent should observe."""
        out = np.zeros(self.n + 1, dtype=np.float32)
        if net <= 0:
            out[-1] = 1.0
            return out
        out[: self.n] = (self.shares * prices) / net
        out[-1] = max(self.cash, 0.0) / net
        return out

    def _apply_weight_cap(self, target: np.ndarray) -> np.ndarray:
        """Project onto the max-weight constraint, pushing the excess into cash."""
        if self.max_weight is None:
            return target
        capped = target.copy()
        equity_part = np.minimum(capped[: self.n], self.max_weight)
        capped[: self.n] = equity_part
        capped[-1] = max(1.0 - equity_part.sum(), 0.0)
        total = capped.sum()
        return capped / total if total > 0 else capped

    # ------------------------------------------------------------------------ step

    def step(self, action: np.ndarray):
        """Advance one trading day.

        ``action`` is a raw vector of length ``n_tickers + 1``; softmax turns it into
        target weights over the stocks plus cash, so the agent cannot request weights
        that fail to sum to 1 or go short.

        The order of operations matters and each step is here for a reason:

        1. **Convert the action to target weights** and apply the position cap.
        2. **Credit interest** on idle cash.
        3. **Work out the gap** between target and current holdings, in rupees. Tickers
           with no data today are frozen — ``delta`` is zeroed rather than traded.
        4. **Sell, then buy.** Two separate passes. Interleaving them means a sale of the
           ninth stock cannot fund a purchase of the first, so target weights become
           unreachable and whichever tickers sit early in the list get funding priority.
        5. **Buys are pro-rated** against the cash that now exists, so an over-ambitious
           target degrades proportionally instead of filling the first few names and
           starving the rest.
        6. **Advance the day, then revalue** at the next close. Trades execute at today's
           prices; the profit or loss shows up tomorrow. Revaluing at today's prices
           would let the agent bank a move it could not have known about.
        7. **Observe realised weights**, not requested ones — integer share sizing and
           partial fills mean the two differ, and the agent must see what it actually
           holds.

        Returns the gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``.
        """
        prices = self.panel.prices[self.i].astype(np.float64)
        available = self.panel.available[self.i] > 0
        regime_now = int(self.panel.regime[self.i]) if self.panel.regime is not None else 0

        target = self._apply_weight_cap(self.softmax(action))

        if self.daily_cash_rate and self.i > 0:
            self.cash *= 1.0 + self.daily_cash_rate

        value = self.portfolio_value(prices)
        target_value = target[: self.n] * value
        current_value = self.shares * prices
        delta = np.where(available, target_value - current_value, 0.0)

        n_trades = 0
        traded_value = 0.0

        # --- pass 1: sells. Must complete before any buy so proceeds are spendable.
        for k in np.flatnonzero(delta < 0):
            price = prices[k]
            if price <= 0:
                continue
            quantity = min(int(abs(delta[k]) // price), int(self.shares[k]))
            if quantity <= 0:
                continue
            fill = self.cost_model.fill_price(SELL, price, quantity)
            proceeds = quantity * fill
            self.cash += proceeds
            self.shares[k] -= quantity
            traded_value += proceeds
            n_trades += 1
            self.trades.append(
                {
                    "date": self.panel.dates[self.i],
                    "ticker": self.panel.tickers[k],
                    "side": "sell",
                    "shares": quantity,
                    "price": fill,
                    "value": proceeds,
                }
            )

        # --- pass 2: buys, pro-rated against the cash that now exists
        wanted = np.where(delta > 0, delta, 0.0)
        total_wanted = float(wanted.sum())
        budget = max(self.cash, 0.0)
        scale = min(1.0, budget / total_wanted) if total_wanted > budget > 0 else 1.0

        for k in np.flatnonzero(wanted > 0):
            price = prices[k]
            if price <= 0:
                continue
            unit = self.cost_model.fill_price(BUY, price, 1.0)
            quantity = int((wanted[k] * scale) // unit)
            if quantity <= 0:
                continue
            cost = quantity * self.cost_model.fill_price(BUY, price, quantity)
            if cost > self.cash:
                quantity = int(self.cash // unit)
                if quantity <= 0:
                    continue
                cost = quantity * self.cost_model.fill_price(BUY, price, quantity)
            self.cash = max(self.cash - cost, 0.0)
            self.shares[k] += quantity
            traded_value += cost
            n_trades += 1
            self.trades.append(
                {
                    "date": self.panel.dates[self.i],
                    "ticker": self.panel.tickers[k],
                    "side": "buy",
                    "shares": quantity,
                    "price": cost / quantity,
                    "value": cost,
                }
            )

        # --- advance one day and revalue at tomorrow's prices (no lookahead: the trade
        #     used today's prices, the P&L is realised at the next close)
        self.i += 1
        terminated = self.i >= self.panel.n_dates - 1
        next_prices = self.panel.prices[min(self.i, self.panel.n_dates - 1)].astype(np.float64)

        self.net = self.portfolio_value(next_prices)
        self.peak = max(self.peak, self.net)
        drawdown = (self.net - self.peak) / max(self.peak, 1e-9)

        # Realised weights, not the requested ones.
        self.weights = self.realised_weights(next_prices, self.net)
        holding_equity = bool(np.any(self.shares > 0))

        reward = self.reward_fn(
            net=self.net,
            prev_net=self.prev_net,
            drawdown=drawdown,
            n_trades=n_trades,
            turnover=traded_value / max(value, 1e-9),
            regime=regime_now,  # the regime in force when the decision was made
            holding=holding_equity,
        )

        self.prev_net = self.net
        self.equity_path.append(self.net)
        self.weight_path.append(self.weights.copy())

        gross = float(self.weights[: self.n].sum())
        info = StepInfo(
            net_worth=self.net,
            cash=self.cash,
            n_positions=int(np.count_nonzero(self.shares > 0)),
            n_trades=n_trades,
            turnover=traded_value / max(value, 1e-9),
            gross_exposure=gross,
            hhi=float(np.sum(self.weights[: self.n] ** 2)),
            drawdown=drawdown,
            regime=regime_now,
        )
        return self.observe(), float(reward), bool(terminated), False, info
