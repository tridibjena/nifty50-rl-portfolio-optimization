"""PPO training and evaluation.

Imports ``stable_baselines3`` lazily so the rest of the package works without it.
"""

from .train import (
    PPO_DEFAULTS,
    SharpeCheckpoint,
    evaluate_policy_on_panel,
    train_ppo,
    train_ppo_ensemble,
)

__all__ = [
    "PPO_DEFAULTS",
    "SharpeCheckpoint",
    "train_ppo",
    "train_ppo_ensemble",
    "evaluate_policy_on_panel",
]
