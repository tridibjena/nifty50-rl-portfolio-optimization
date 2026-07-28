"""Gymnasium adapter over :class:`~nifty_rl.envs.core.PortfolioSimulator`.

Thin by design. All execution logic, weight bookkeeping and reward computation live in
the core, which has no reinforcement-learning dependency and is unit tested directly.
This file only translates between that core and the Gym API.

``gymnasium`` is imported at module level, but the core is importable without it -- so
the portfolio mechanics remain testable in an environment with no RL stack installed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from ..backtest.costs import CostModel, build_cost_model
from ..config import BacktestConfig, CostConfig
from .core import PortfolioSimulator
from .panel import PanelArrays, build_panel_arrays
from .rewards import RewardFunction


class MultiStockPortfolioEnv(gym.Env):
    """Continuous long-only allocation across N tickers plus cash."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: PanelArrays,
        cfg: BacktestConfig,
        cost_cfg: CostConfig,
        reward_fn: Optional[RewardFunction] = None,
        cost_model: Optional[CostModel] = None,
        max_weight: Optional[float] = None,
    ):
        super().__init__()
        self.simulator = PortfolioSimulator(
            panel=panel,
            cfg=cfg,
            cost_model=cost_model or build_cost_model(cost_cfg),
            reward_fn=reward_fn,
            max_weight=max_weight,
        )
        n = panel.n_tickers
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(panel.observation_size(),), dtype=np.float32
        )
        # One logit per ticker plus cash; the simulator applies the softmax.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(n + 1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.simulator.reset(), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.simulator.step(action)
        return obs, reward, terminated, truncated, info.__dict__

    # ---------------------------------------------------------------- convenience

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        tickers: Sequence[str],
        feature_names: Sequence[str],
        cfg: BacktestConfig,
        cost_cfg: CostConfig,
        regime_labels: Optional[pd.Series] = None,
        reward_fn: Optional[RewardFunction] = None,
        max_weight: Optional[float] = None,
    ) -> "MultiStockPortfolioEnv":
        panel = build_panel_arrays(df, tickers, feature_names, regime_labels)
        return cls(panel, cfg, cost_cfg, reward_fn=reward_fn, max_weight=max_weight)

    @property
    def equity_curve(self) -> pd.Series:
        sim = self.simulator
        dates = sim.panel.dates[: len(sim.equity_path)]
        return pd.Series(sim.equity_path, index=dates, name="equity")

    @property
    def weights_frame(self) -> pd.DataFrame:
        """Realised weights per step — the allocation stack the notebook could not plot.

        Its environment never logged weights, so the published "allocation stack" chart
        was an inline momentum proxy rather than the agent's actual positions.
        """
        sim = self.simulator
        dates = sim.panel.dates[: len(sim.weight_path)]
        return pd.DataFrame(
            np.vstack(sim.weight_path),
            index=dates,
            columns=list(sim.panel.tickers) + ["CASH"],
        )

    @property
    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.simulator.trades)
