"""PPO training with a consistent selection protocol and multi-seed evaluation.

Two methodology fixes carried over from the notebook:

* **Selection and deployment use the same rule** (bug #12). The notebook's hyperparameter
  search called ``train_ppo_agent`` *without* validation data, so ``cb = None`` and each
  candidate was scored on its **final iterate**; the final model was then trained *with*
  the checkpoint callback and scored on its **best checkpoint**. Candidates were chosen
  under one protocol and deployed under another. Here the checkpoint callback is always
  active, so every number compared is the same kind of number.

* **Every result is a seed distribution, not a point** (the notebook's stated limitation).
  Policy-gradient variance on ~1,000-step episodes is large enough that a single seed says
  very little. :func:`train_ppo_ensemble` trains N seeds per window and reports the spread
  alongside the mean, plus an equal-weight action ensemble.

The environment is the vectorised :class:`~nifty_rl.envs.core.PortfolioSimulator` (≈8,000
steps/sec), which is what makes a seed sweep across every walk-forward window affordable
at all -- the notebook's pandas-per-step panel is the reason its training budget was
described as "system-constrained".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult
from ..config import BacktestConfig, CostConfig
from ..envs.panel import PanelArrays
from ..envs.rewards import RewardFunction, ShapedReward

PPO_DEFAULTS: Dict[str, object] = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def _make_env(panel, cfg, cost_cfg, reward_fn, max_weight):
    from ..envs.multistock import MultiStockPortfolioEnv

    return MultiStockPortfolioEnv(
        panel, cfg, cost_cfg, reward_fn=reward_fn, max_weight=max_weight
    )


def _sharpe(equity: pd.Series, trading_days: int = 252) -> float:
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return float("nan")
    sd = returns.std()
    if not np.isfinite(sd) or sd <= 1e-10:
        return float("nan")
    return float(returns.mean() / sd * np.sqrt(trading_days))


def evaluate_policy_on_panel(
    model,
    panel: PanelArrays,
    cfg: BacktestConfig,
    cost_cfg: CostConfig,
    strategy_name: str = "PPO",
    reward_fn: Optional[RewardFunction] = None,
    max_weight: Optional[float] = None,
    deterministic: bool = True,
) -> BacktestResult:
    """Roll a trained policy over a panel and return a standard backtest result."""
    env = _make_env(panel, cfg, cost_cfg, reward_fn or ShapedReward(), max_weight)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    weights = env.weights_frame
    equity = env.equity_curve
    positions = pd.Series(
        (weights[panel.tickers] > 1e-6).sum(axis=1), index=weights.index, name="position"
    )
    per_ticker = {t: (weights[t] > 1e-6).astype(int) for t in panel.tickers}

    return BacktestResult(
        strategy=strategy_name,
        equity=equity,
        positions=positions,
        trades=env.trades_frame,
        per_ticker_positions=per_ticker,
        weights=weights[panel.tickers],
    )


class SharpeCheckpoint:
    """Keeps the best policy by validation Sharpe and restores it after training.

    PPO can peak mid-training and regress; the deployed model should be the best policy
    seen, not the last one. Built as a plain object with a thin SB3 callback wrapper so
    the selection logic is testable without a training loop.
    """

    def __init__(self, val_panel, cfg, cost_cfg, check_freq=10_000, reward_fn=None, max_weight=None):
        self.val_panel = val_panel
        self.cfg = cfg
        self.cost_cfg = cost_cfg
        self.check_freq = int(check_freq)
        self.reward_fn = reward_fn
        self.max_weight = max_weight
        self.best_sharpe = -np.inf
        self.best_state: Optional[dict] = None
        self.history: List[Dict[str, float]] = []

    def consider(self, model, step: int) -> bool:
        result = evaluate_policy_on_panel(
            model, self.val_panel, self.cfg, self.cost_cfg,
            reward_fn=self.reward_fn, max_weight=self.max_weight,
        )
        sharpe = _sharpe(result.equity)
        # bool(), not the raw numpy scalar: np.isfinite returns np.bool_, and `and`
        # propagates it, so callers doing `is True` silently get the wrong answer.
        improved = bool(np.isfinite(sharpe) and sharpe > self.best_sharpe)
        if improved:
            self.best_sharpe = sharpe
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
        self.history.append({"step": step, "val_sharpe": sharpe, "improved": improved})
        return improved

    def restore(self, model):
        if self.best_state is not None:
            model.policy.load_state_dict(self.best_state)
        return model

    def as_sb3_callback(self):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _Callback(BaseCallback):
            def _on_step(self) -> bool:
                if self.n_calls % outer.check_freq == 0:
                    outer.consider(self.model, self.n_calls)
                return True

        return _Callback()


def train_ppo(
    train_panel: PanelArrays,
    val_panel: PanelArrays,
    cfg: BacktestConfig,
    cost_cfg: CostConfig,
    timesteps: int = 60_000,
    seed: int = 42,
    params: Optional[Dict[str, object]] = None,
    reward_fn: Optional[RewardFunction] = None,
    max_weight: Optional[float] = None,
    check_freq: int = 10_000,
    verbose: int = 0,
):
    """Train one PPO policy, checkpointing on validation Sharpe throughout."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    reward_fn = reward_fn or ShapedReward()
    settings = dict(PPO_DEFAULTS)
    settings.update(params or {})

    def _factory():
        return Monitor(_make_env(train_panel, cfg, cost_cfg, reward_fn, max_weight))

    env = DummyVecEnv([_factory])
    model = PPO(
        "MlpPolicy", env, seed=seed, device="cpu", verbose=verbose,
        policy_kwargs=dict(net_arch=[64, 64]), **settings,
    )

    checkpoint = SharpeCheckpoint(
        val_panel, cfg, cost_cfg, check_freq=check_freq,
        reward_fn=reward_fn, max_weight=max_weight,
    )
    model.learn(total_timesteps=timesteps, callback=checkpoint.as_sb3_callback(), progress_bar=False)
    # Always restore -- selection and deployment must use the same rule (bug #12).
    checkpoint.restore(model)
    return model, checkpoint


@dataclass
class EnsembleResult:
    per_seed: Dict[int, BacktestResult]
    val_sharpes: Dict[int, float]

    @property
    def seed_returns(self) -> Dict[int, pd.Series]:
        return {s: r.equity.pct_change().dropna() for s, r in self.per_seed.items()}

    def summary(self, initial_cash: float) -> Dict[str, float]:
        totals = [r.equity.iloc[-1] / initial_cash - 1.0 for r in self.per_seed.values()]
        sharpes = [_sharpe(r.equity) for r in self.per_seed.values()]
        return {
            "n_seeds": len(self.per_seed),
            "mean_return": float(np.mean(totals)),
            "std_return": float(np.std(totals)),
            "min_return": float(np.min(totals)),
            "max_return": float(np.max(totals)),
            "mean_sharpe": float(np.nanmean(sharpes)),
            "std_sharpe": float(np.nanstd(sharpes)),
        }

    def mean_equity(self, initial_cash: float) -> pd.Series:
        """Equal-weight ensemble across seeds, rebased to the starting capital.

        Averaging the *equity paths* is the portfolio interpretation: split capital
        equally across N independently trained policies. It is not the same as averaging
        their actions, and it is the version an allocator could actually run.
        """
        frame = pd.DataFrame({s: r.equity for s, r in self.per_seed.items()}).dropna()
        return frame.mean(axis=1).rename("PPO_ensemble")


def train_ppo_ensemble(
    train_panel: PanelArrays,
    val_panel: PanelArrays,
    test_panel: PanelArrays,
    cfg: BacktestConfig,
    cost_cfg: CostConfig,
    seeds: Sequence[int] = (0, 1, 2),
    timesteps: int = 60_000,
    params: Optional[Dict[str, object]] = None,
    reward_fn: Optional[RewardFunction] = None,
    max_weight: Optional[float] = None,
    progress=None,
) -> EnsembleResult:
    """Train one policy per seed and evaluate each on the held-out panel."""
    log = progress or (lambda _m: None)
    per_seed: Dict[int, BacktestResult] = {}
    val_sharpes: Dict[int, float] = {}

    for seed in seeds:
        model, checkpoint = train_ppo(
            train_panel, val_panel, cfg, cost_cfg,
            timesteps=timesteps, seed=seed, params=params,
            reward_fn=reward_fn, max_weight=max_weight,
        )
        result = evaluate_policy_on_panel(
            model, test_panel, cfg, cost_cfg,
            strategy_name=f"PPO_s{seed}", reward_fn=reward_fn, max_weight=max_weight,
        )
        per_seed[seed] = result
        val_sharpes[seed] = checkpoint.best_sharpe
        log(
            f"        seed {seed}: val_sharpe {checkpoint.best_sharpe:>6.2f}  "
            f"oos_return {result.equity.iloc[-1] / cfg.initial_cash - 1:>7.2%}"
        )

    return EnsembleResult(per_seed=per_seed, val_sharpes=val_sharpes)
