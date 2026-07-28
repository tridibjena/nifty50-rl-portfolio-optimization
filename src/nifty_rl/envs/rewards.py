"""Reward functions as interchangeable objects.

The notebook hard-coded one shaped reward with four hand-tuned coefficients, justified
only by "initial values caused permanent cash-hoarding; halved to allow non-trivial
policy learning". That is an anecdote, not a calibration -- and it is untestable, because
the reward could not be swapped or measured independently of the environment.

Three implementations here:

* :class:`ShapedReward` -- the notebook's formulation, reproduced exactly so old results
  stay comparable.
* :class:`DifferentialSharpeReward` -- Moody & Saffell (1998). The principled version of
  "maximise risk-adjusted return online": an exponentially-weighted estimate of the
  Sharpe ratio whose derivative gives a per-step reward, with no free penalty weights to
  tune at all.
* :class:`RegimeAwareReward` -- scales the drawdown penalty by regime instead of holding
  one constant across every market condition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Optional

import numpy as np


class RewardFunction(ABC):
    """Per-step reward. Stateful, so it must be reset between episodes."""

    name: str = "reward"

    def reset(self) -> None:
        """Clear any running state."""

    @abstractmethod
    def __call__(
        self,
        *,
        net: float,
        prev_net: float,
        drawdown: float,
        n_trades: int,
        turnover: float,
        regime: int,
        holding: bool,
    ) -> float:
        ...


def _log_return(net: float, prev_net: float) -> float:
    return float(np.log(max(net, 1e-9) / max(prev_net, 1e-9)))


class ShapedReward(RewardFunction):
    """Log return minus drawdown, downside-volatility, turnover and regime penalties.

    Reproduces the notebook's reward. Defaults are its post-calibration values.
    """

    name = "shaped"

    def __init__(
        self,
        drawdown_penalty: float = 0.04,
        downvol_penalty: float = 0.02,
        turnover_penalty: float = 0.0002,
        regime_penalty: float = 0.0002,
        downside_window: int = 20,
        penalised_regimes: Optional[List[int]] = None,
    ):
        self.drawdown_penalty = drawdown_penalty
        self.downvol_penalty = downvol_penalty
        self.turnover_penalty = turnover_penalty
        self.regime_penalty = regime_penalty
        self.downside_window = downside_window
        # Which regime indices count as "risk-off". None means the top state only,
        # resolved lazily against whatever regime labels the panel carries.
        self.penalised_regimes = penalised_regimes
        self.reset()

    def reset(self) -> None:
        self.recent: List[float] = []

    def _downside_vol(self, period_return: float) -> float:
        self.recent.append(period_return)
        window = self.recent[-self.downside_window :]
        negatives = [r for r in window if r < 0]
        return float(np.std(negatives)) if len(negatives) > 1 else 0.0

    def __call__(self, *, net, prev_net, drawdown, n_trades, turnover, regime, holding) -> float:
        period_return = (net - prev_net) / max(prev_net, 1e-9)
        down_vol = self._downside_vol(period_return)

        risk_off = regime in self.penalised_regimes if self.penalised_regimes else regime >= 2
        regime_cost = self.regime_penalty if (risk_off and holding) else 0.0

        return (
            _log_return(net, prev_net)
            - self.drawdown_penalty * abs(drawdown)
            - self.downvol_penalty * down_vol
            - self.turnover_penalty * n_trades
            - regime_cost
        )


class DifferentialSharpeReward(RewardFunction):
    """Moody & Saffell's differential Sharpe ratio.

    Maintains exponentially-weighted first and second moments of the return series and
    rewards the marginal contribution of each step to the running Sharpe estimate::

        D_t = (B_{t-1} ΔA - ½ A_{t-1} ΔB) / (B_{t-1} - A_{t-1}²)^{3/2}

    The appeal over a shaped reward is that it has no penalty weights to hand-tune: risk
    aversion falls out of the objective rather than being bolted on with four constants
    someone halved until the agent stopped hoarding cash.
    """

    name = "differential_sharpe"

    #: Below this the variance estimate is numerically indistinguishable from zero and
    #: the ratio's denominator (variance ** 1.5) explodes. A daily return standard
    #: deviation of 1e-5 is 0.001% -- no real portfolio sits under it.
    VARIANCE_FLOOR = 1e-10

    def __init__(
        self,
        adaptation_rate: float = 0.004,
        turnover_penalty: float = 0.0,
        warmup_steps: int = 20,
        clip: float = 10.0,
    ):
        self.adaptation_rate = adaptation_rate
        self.turnover_penalty = turnover_penalty
        # A Sharpe estimate from two observations is meaningless. Without a warm-up the
        # EW variance starts at exactly zero and stays near it for several steps, so the
        # first rewards are dominated by division by ~0 and pin to the clip bound --
        # which is a large *constant* signal unrelated to performance.
        self.warmup_steps = int(warmup_steps)
        self.clip = float(clip)
        self.reset()

    def reset(self) -> None:
        self.a = 0.0  # EW mean of returns
        self.b = 0.0  # EW mean of squared returns
        self.steps = 0

    def __call__(self, *, net, prev_net, drawdown, n_trades, turnover, regime, holding) -> float:
        period_return = (net - prev_net) / max(prev_net, 1e-9)
        eta = self.adaptation_rate

        # Bias correction (as in Adam). The EW accumulators start at zero, so for the
        # first ~1/eta steps they badly understate both moments -- and because the
        # variance enters as v**1.5 in the denominator, that understatement produces
        # enormous rewards. On a *constant* return series, uncorrected estimates gave a
        # cumulative reward of +103 where the correct answer is exactly 0.
        prev_bias = 1.0 - (1.0 - eta) ** self.steps if self.steps else 0.0
        if prev_bias > 0:
            a_hat = self.a / prev_bias
            b_hat = self.b / prev_bias
        else:
            a_hat = b_hat = 0.0

        delta_a = period_return - a_hat
        delta_b = period_return ** 2 - b_hat
        variance = b_hat - a_hat ** 2

        self.steps += 1
        if self.steps <= self.warmup_steps or variance <= self.VARIANCE_FLOOR:
            reward = 0.0
        else:
            reward = float((b_hat * delta_a - 0.5 * a_hat * delta_b) / (variance ** 1.5))
            reward = float(np.clip(reward, -self.clip, self.clip))

        self.a += eta * (period_return - self.a)
        self.b += eta * (period_return ** 2 - self.b)
        return reward - self.turnover_penalty * n_trades


class RegimeAwareReward(ShapedReward):
    """Shaped reward whose drawdown penalty depends on the prevailing regime.

    A single drawdown coefficient asks the agent to be equally cautious in a calm trend
    and in a crisis. Scaling it by regime is the cheapest way to express "be more
    defensive when conditions are bad" without adding another free parameter per state.
    """

    name = "regime_aware"

    def __init__(
        self,
        drawdown_penalty_by_regime: Mapping[int, float],
        default_drawdown_penalty: float = 0.04,
        **kwargs,
    ):
        super().__init__(drawdown_penalty=default_drawdown_penalty, **kwargs)
        self.drawdown_penalty_by_regime = dict(drawdown_penalty_by_regime)

    def __call__(self, *, net, prev_net, drawdown, n_trades, turnover, regime, holding) -> float:
        period_return = (net - prev_net) / max(prev_net, 1e-9)
        down_vol = self._downside_vol(period_return)

        penalty = self.drawdown_penalty_by_regime.get(int(regime), self.drawdown_penalty)
        risk_off = regime in self.penalised_regimes if self.penalised_regimes else regime >= 2
        regime_cost = self.regime_penalty if (risk_off and holding) else 0.0

        return (
            _log_return(net, prev_net)
            - penalty * abs(drawdown)
            - self.downvol_penalty * down_vol
            - self.turnover_penalty * n_trades
            - regime_cost
        )


def escalating_drawdown_penalty(n_regimes: int, base: float = 0.02, top: float = 0.10) -> Dict[int, float]:
    """Drawdown penalty rising linearly from the calmest regime to the most turbulent."""
    if n_regimes <= 1:
        return {0: base}
    return {i: float(v) for i, v in enumerate(np.linspace(base, top, n_regimes))}


REWARD_REGISTRY = {
    "shaped": ShapedReward,
    "differential_sharpe": DifferentialSharpeReward,
    "regime_aware": RegimeAwareReward,
}
