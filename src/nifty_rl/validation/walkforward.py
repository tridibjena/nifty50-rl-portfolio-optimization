"""Rolling walk-forward evaluation.

A single 70/15/15 chronological split produces exactly one out-of-sample window, and
whatever regime that window happens to land in *is* the result. In this dataset it landed
on a drawdown, so every strategy lost money and the leaderboard mostly measured which one
was least invested. Change the split ratios and the ranking changes with them. That is
not a model evaluation; it is one draw from a distribution.

This module evaluates the way a deployed model actually works:

1. Fit on everything known up to time *t* (expanding window by default -- a real desk
   does not throw away history at each refit).
2. Trade the next ``test_days`` with those parameters frozen.
3. Roll forward and refit.
4. Concatenate every out-of-sample block into **one continuous track record**.

The pooled series is the headline. Per-window numbers show consistency; the pooled series
is what a strategy would actually have returned, and it is the only thing worth putting a
confidence interval on.

Regime detectors are refit inside every window. That is the honest test of refit
stability: if "state 0" changes meaning between windows, regime-conditioned results are
not comparable across time, and the walk-forward is where that shows up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult, run_portfolio_backtest
from ..backtest.weights import run_weight_backtest
from ..config import BacktestConfig, CostConfig, MetricsConfig
from ..metrics.performance import performance_metrics
from ..metrics.stats import sharpe_from_returns
from ..strategies.meta import scale_weights_by_regime


@dataclass
class RLConfig:
    """Per-window PPO training inside the walk-forward loop.

    Feature scaling is refit on each window's training block, never on the whole sample.
    The training block is further split so the Sharpe checkpoint has a validation slice it
    has not trained on -- otherwise "best checkpoint" is chosen on the training data.
    """

    features: Sequence[str]
    seeds: Sequence[int] = (0, 1, 2)
    timesteps: int = 60_000
    val_fraction: float = 0.2
    max_weight: Optional[float] = None
    include_per_seed: bool = False
    check_freq: int = 10_000


@dataclass
class WindowSpec:
    index: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    @property
    def train_start(self) -> pd.Timestamp:
        return self.train_dates[0]

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train_dates[-1]

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test_dates[0]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test_dates[-1]


@dataclass
class WindowResult:
    spec: WindowSpec
    metrics: pd.DataFrame
    returns: Dict[str, pd.Series]
    regime_labels: pd.Series
    selected_strategy: str
    regime_names: List[str] = field(default_factory=list)


@dataclass
class WalkForwardReport:
    windows: List[WindowResult]
    per_window: pd.DataFrame
    pooled_returns: Dict[str, pd.Series]
    pooled_equity: Dict[str, pd.Series]
    pooled_regimes: pd.Series
    summary: pd.DataFrame
    regime_names: List[str]

    @property
    def n_windows(self) -> int:
        return len(self.windows)


def rolling_windows(
    dates: Sequence[pd.Timestamp],
    train_days: int = 500,
    test_days: int = 125,
    step_days: int = 125,
    expanding: bool = True,
) -> List[WindowSpec]:
    """Slice unique trading dates into (train, test) blocks.

    ``expanding=True`` anchors every train block at the start of history, which is what a
    production refit does. ``expanding=False`` gives a fixed-length rolling window, useful
    for asking whether old data still helps.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(sorted(pd.unique(dates))))
    specs: List[WindowSpec] = []
    start, index = 0, 1

    while start + train_days + test_days <= len(dates):
        train_lo = 0 if expanding else start
        train_slice = dates[train_lo : start + train_days]
        test_slice = dates[start + train_days : start + train_days + test_days]
        specs.append(WindowSpec(index=index, train_dates=train_slice, test_dates=test_slice))
        start += step_days
        index += 1

    return specs


def _select_on_train(
    train_panel: pd.DataFrame,
    strategies: Mapping[str, Callable],
    cfg: BacktestConfig,
    cost_cfg: CostConfig,
    trading_days: int,
) -> str:
    """Pick the best rule strategy on the training block by excess-return Sharpe.

    This is the decision a real system makes at each refit. Running it inside the loop is
    what makes the out-of-sample record honest: the choice never sees the block it is
    evaluated on.
    """
    best_name, best_score = None, -np.inf
    for name, fn in strategies.items():
        try:
            result = run_portfolio_backtest(train_panel, fn, name, cfg, cost_cfg)
        except Exception:
            continue
        score = sharpe_from_returns(
            result.equity.pct_change().dropna().to_numpy(), trading_days
        )
        if np.isfinite(score) and score > best_score:
            best_name, best_score = name, score
    return best_name or next(iter(strategies))


def walk_forward_evaluate(
    panel: pd.DataFrame,
    regime_features: pd.DataFrame,
    regime_feature_columns: Sequence[str],
    detector_factory: Callable[[], object],
    rule_strategies: Mapping[str, Callable],
    allocator_weights: Mapping[str, pd.DataFrame],
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    cost_cfg: CostConfig,
    metrics_cfg: MetricsConfig,
    passive_cfg: Optional[BacktestConfig] = None,
    benchmark: str = "BuyHold",
    exposure_ladder: Optional[Mapping[int, float]] = None,
    overlay_bases: Sequence[str] = ("HRP", "EqualWeight"),
    train_days: int = 500,
    test_days: int = 125,
    step_days: int = 125,
    expanding: bool = True,
    rl_config: Optional["RLConfig"] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> WalkForwardReport:
    """Run the full rolling evaluation."""
    log = progress or (lambda _msg: None)
    passive_cfg = passive_cfg or cfg

    dates = pd.DatetimeIndex(pd.to_datetime(sorted(panel["Date"].unique())))
    specs = rolling_windows(dates, train_days, test_days, step_days, expanding)
    if not specs:
        raise ValueError(
            f"No walk-forward windows: {len(dates)} trading dates cannot supply "
            f"{train_days} train + {test_days} test."
        )

    panel_dates = pd.to_datetime(panel["Date"])
    results: List[WindowResult] = []
    regime_names: List[str] = []

    for spec in specs:
        log(
            f"      window {spec.index}: train {spec.train_start.date()}→{spec.train_end.date()} "
            f"| test {spec.test_start.date()}→{spec.test_end.date()}"
        )

        train_panel = panel[panel_dates.isin(spec.train_dates)].reset_index(drop=True)
        test_panel = panel[panel_dates.isin(spec.test_dates)].reset_index(drop=True)
        if test_panel.empty or train_panel.empty:
            continue

        # --- refit the regime detector on this window's training block only
        train_features = regime_features.reindex(spec.train_dates).dropna()
        span = spec.train_dates.union(spec.test_dates)
        span_features = regime_features.reindex(span).ffill().bfill()

        detector = detector_factory()
        try:
            detector.fit(train_features)
            labels = detector.label_online(span_features)
            regime_names = list(getattr(detector, "regime_labels_", []) or regime_names)
        except Exception as exc:  # pragma: no cover - degenerate window
            log(f"        regime fit failed ({exc}); falling back to a single regime")
            labels = pd.Series(0, index=span_features.index)

        test_labels = labels.reindex(spec.test_dates).ffill().bfill()

        window_results = {}

        # --- rule-based strategies, parameters fixed
        for name, fn in rule_strategies.items():
            engine_cfg = passive_cfg if name == benchmark else cfg
            try:
                window_results[name] = run_portfolio_backtest(
                    test_panel, fn, name, engine_cfg, cost_cfg
                )
            except Exception:
                continue

        # --- the strategy this window's training block actually chose
        selected = _select_on_train(
            train_panel, rule_strategies, cfg, cost_cfg, metrics_cfg.trading_days
        )
        if selected in window_results:
            chosen = window_results[selected]
            window_results["Selected_OOS"] = type(chosen)(
                strategy="Selected_OOS",
                equity=chosen.equity.copy(),
                positions=chosen.positions.copy(),
                trades=chosen.trades.copy(),
                per_ticker_equity=chosen.per_ticker_equity,
                per_ticker_positions=chosen.per_ticker_positions,
                weights=chosen.weights,
            )

        # --- allocators (weights are already causal: trailing windows only)
        test_prices = prices.reindex(spec.test_dates).dropna(how="all")
        for name, weights in allocator_weights.items():
            block = weights.loc[weights.index.isin(spec.test_dates)]
            if block.empty:
                continue
            try:
                window_results[name] = run_weight_backtest(
                    test_prices, block, name, cfg, cost_cfg
                )
            except Exception:
                continue

        # --- regime exposure overlay
        if exposure_ladder:
            for base in overlay_bases:
                if base not in allocator_weights:
                    continue
                scaled = scale_weights_by_regime(
                    allocator_weights[base], labels, exposure_ladder
                )
                block = scaled.loc[scaled.index.isin(spec.test_dates)]
                if block.empty:
                    continue
                try:
                    window_results[f"{base}+Regime"] = run_weight_backtest(
                        test_prices, block, f"{base}+Regime", cfg, cost_cfg
                    )
                except Exception:
                    continue

        # --- PPO, retrained from scratch on this window's training block
        if rl_config is not None:
            try:
                window_results.update(
                    _train_window_ppo(
                        rl_config, train_panel, test_panel, spec, cfg, cost_cfg, log
                    )
                )
            except Exception as exc:  # pragma: no cover - RL stack optional
                log(f"        PPO skipped: {exc}")

        if benchmark not in window_results:
            continue

        bench_result = window_results[benchmark]
        rows = []
        returns = {}
        for name, result in window_results.items():
            metric = performance_metrics(
                result, bench_result, test_panel, metrics_cfg, cfg.initial_cash
            )
            record = metric.to_dict()
            record.update(
                {
                    "window": spec.index,
                    "test_start": spec.test_start,
                    "test_end": spec.test_end,
                }
            )
            rows.append(record)
            returns[name] = result.equity.pct_change().dropna()

        results.append(
            WindowResult(
                spec=spec,
                metrics=pd.DataFrame(rows),
                returns=returns,
                regime_labels=test_labels,
                selected_strategy=selected,
                regime_names=list(regime_names),
            )
        )

    if not results:
        raise RuntimeError("Walk-forward produced no usable windows.")

    per_window = pd.concat([r.metrics for r in results], ignore_index=True)
    pooled_returns = _pool_returns(results)
    pooled_equity = {
        name: cfg.initial_cash * (1.0 + series).cumprod()
        for name, series in pooled_returns.items()
    }
    pooled_regimes = pd.concat([r.regime_labels for r in results])
    pooled_regimes = pooled_regimes[~pooled_regimes.index.duplicated(keep="first")].sort_index()

    summary = aggregate_windows(per_window, pooled_returns, metrics_cfg, cfg.initial_cash)

    return WalkForwardReport(
        windows=results,
        per_window=per_window,
        pooled_returns=pooled_returns,
        pooled_equity=pooled_equity,
        pooled_regimes=pooled_regimes,
        summary=summary,
        regime_names=regime_names,
    )


def _pool_returns(results: Sequence[WindowResult]) -> Dict[str, pd.Series]:
    """Concatenate each strategy's out-of-sample blocks into one continuous series.

    Windows are consecutive and non-overlapping in test time, so chaining their daily
    returns reconstructs the track record an investor would have experienced across every
    refit -- which is the series that deserves a confidence interval.
    """
    names = sorted({name for r in results for name in r.returns})
    pooled: Dict[str, pd.Series] = {}
    for name in names:
        blocks = [r.returns[name] for r in results if name in r.returns]
        if not blocks:
            continue
        series = pd.concat(blocks).sort_index()
        pooled[name] = series[~series.index.duplicated(keep="first")]
    return pooled


def aggregate_windows(
    per_window: pd.DataFrame,
    pooled_returns: Mapping[str, pd.Series],
    metrics_cfg: MetricsConfig,
    initial_capital: float = 0.0,
) -> pd.DataFrame:
    """Per-strategy consistency across windows plus pooled out-of-sample statistics.

    ``initial_capital`` adds terminal-wealth columns. A Sharpe ratio answers "was the
    risk worth taking"; it does not answer "what would I have". Both belong in the same
    table, because a strategy can win on one and lose on the other -- and the currency
    figure is the one a non-specialist reads first.
    """
    rf_daily = (
        (1.0 + metrics_cfg.risk_free_annual) ** (1.0 / metrics_cfg.trading_days) - 1.0
        if metrics_cfg.risk_free_annual > 0
        else 0.0
    )

    rows = []
    for name, group in per_window.groupby("strategy"):
        pooled = pooled_returns.get(name)
        pooled_sharpe = (
            sharpe_from_returns(pooled.to_numpy(), metrics_cfg.trading_days, rf_daily)
            if pooled is not None
            else np.nan
        )
        pooled_total = float((1.0 + pooled).prod() - 1.0) if pooled is not None else np.nan

        years = len(pooled) / metrics_cfg.trading_days if pooled is not None else np.nan
        cagr = (
            (1.0 + pooled_total) ** (1.0 / years) - 1.0
            if np.isfinite(pooled_total) and np.isfinite(years) and years > 0
            else np.nan
        )

        rows.append(
            {
                "strategy": name,
                "n_windows": int(group["window"].nunique()),
                "pooled_total_return": pooled_total,
                "final_value": initial_capital * (1.0 + pooled_total),
                "profit": initial_capital * pooled_total,
                "cagr": cagr,
                "pooled_sharpe": pooled_sharpe,
                "mean_window_return": float(group["total_return"].mean()),
                "median_window_return": float(group["total_return"].median()),
                "worst_window_return": float(group["total_return"].min()),
                "best_window_return": float(group["total_return"].max()),
                # Consistency beats a single high number: a strategy that wins in six of
                # eight windows is a different proposition from one that wins in two.
                "windows_positive": float((group["total_return"] > 0).mean()),
                "windows_beating_benchmark": float(
                    (group["benchmark_excess_return"] > 0).mean()
                ),
                "mean_window_sharpe": float(group["Sharpe"].mean()),
                "return_dispersion": float(group["total_return"].std()),
                "mean_max_drawdown": float(group["max_drawdown"].mean()),
                "mean_exposure": float(group["exposure"].mean()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("pooled_sharpe", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def _train_window_ppo(
    rl_config: "RLConfig",
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    spec: WindowSpec,
    cfg: BacktestConfig,
    cost_cfg: CostConfig,
    log: Callable[[str], None],
) -> Dict[str, object]:
    """Scale, split, train N seeds, and return their out-of-sample results.

    The scaler is fit on the training block alone. Fitting it on the full sample would
    leak the future's mean and variance into every observation the agent ever sees -- the
    quietest possible lookahead, and invisible in any equity curve.
    """
    from sklearn.preprocessing import StandardScaler

    from ..agents.train import train_ppo_ensemble
    from ..envs.panel import build_panel_arrays

    features = list(rl_config.features)
    tickers = sorted(pd.unique(train_panel["ticker"]))

    scaler = StandardScaler().fit(train_panel[features].to_numpy())

    def _scaled(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[features] = scaler.transform(out[features].to_numpy())
        return out

    train_dates = pd.DatetimeIndex(sorted(train_panel["Date"].unique()))
    split = int(len(train_dates) * (1.0 - rl_config.val_fraction))
    fit_dates, val_dates = train_dates[:split], train_dates[split:]

    train_ts = pd.to_datetime(train_panel["Date"])
    fit_panel = _scaled(train_panel[train_ts.isin(fit_dates)])
    val_panel = _scaled(train_panel[train_ts.isin(val_dates)])
    scaled_test = _scaled(test_panel)

    fit_arrays = build_panel_arrays(fit_panel, tickers, features)
    val_arrays = build_panel_arrays(val_panel, tickers, features)
    test_arrays = build_panel_arrays(scaled_test, tickers, features)

    log(f"        PPO: {len(fit_dates)}d fit / {len(val_dates)}d val -> "
        f"{test_arrays.n_dates}d test, {len(rl_config.seeds)} seeds "
        f"x {rl_config.timesteps:,} steps")

    ensemble = train_ppo_ensemble(
        fit_arrays, val_arrays, test_arrays, cfg, cost_cfg,
        seeds=rl_config.seeds, timesteps=rl_config.timesteps,
        max_weight=rl_config.max_weight, progress=log,
    )

    results: Dict[str, object] = {}
    if rl_config.include_per_seed:
        for seed, result in ensemble.per_seed.items():
            results[f"PPO_s{seed}"] = result

    # Equal capital across the independently trained policies.
    mean_equity = ensemble.mean_equity(cfg.initial_cash)
    reference = next(iter(ensemble.per_seed.values()))
    results["PPO_ensemble"] = BacktestResult(
        strategy="PPO_ensemble",
        equity=mean_equity,
        positions=reference.positions,
        trades=pd.concat(
            [r.trades for r in ensemble.per_seed.values() if not r.trades.empty],
            ignore_index=True,
        ) if any(not r.trades.empty for r in ensemble.per_seed.values()) else pd.DataFrame(),
        per_ticker_positions=reference.per_ticker_positions,
        weights=reference.weights,
    )
    summary = ensemble.summary(cfg.initial_cash)
    log(f"        PPO seeds: return {summary['mean_return']:.2%} "
        f"+/- {summary['std_return']:.2%}  (min {summary['min_return']:.2%}, "
        f"max {summary['max_return']:.2%})")
    return results
