"""Reinforcement-learning environment.

``core``, ``panel`` and ``rewards`` are pure NumPy and import without gymnasium; the Gym
adapter in ``multistock`` is imported lazily so the portfolio mechanics stay testable
where no RL stack is installed.
"""

from .core import PortfolioSimulator, StepInfo
from .panel import PanelArrays, build_panel_arrays
from .rewards import (
    REWARD_REGISTRY,
    DifferentialSharpeReward,
    RegimeAwareReward,
    RewardFunction,
    ShapedReward,
    escalating_drawdown_penalty,
)

__all__ = [
    "PortfolioSimulator",
    "StepInfo",
    "PanelArrays",
    "build_panel_arrays",
    "RewardFunction",
    "ShapedReward",
    "DifferentialSharpeReward",
    "RegimeAwareReward",
    "escalating_drawdown_penalty",
    "REWARD_REGISTRY",
]


def __getattr__(name):  # pragma: no cover - import shim
    if name == "MultiStockPortfolioEnv":
        from .multistock import MultiStockPortfolioEnv

        return MultiStockPortfolioEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
