#!/usr/bin/env python3
"""End-to-end pipeline: data -> regimes -> rolling walk-forward -> figures -> report.

Run with::

    python3 scripts/run_pipeline.py

Evaluation is **rolling walk-forward**, not a single chronological split. A 70/15/15 cut
yields exactly one out-of-sample window, and whichever regime that window lands in *is*
the result -- change the ratios and the ranking changes with them. Here the model refits
on an expanding history, trades the next block with parameters frozen, and rolls forward;
the out-of-sample blocks are chained into one continuous track record.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:  # keep the RL stack optional
    import torch

    torch.set_num_threads(1)
except ImportError:
    pass

from nifty_rl.backtest import build_allocator_weights, price_matrix
from nifty_rl.config import RunConfig
from nifty_rl.data import build_panel
from nifty_rl.features import add_features, align_to_common_dates
from nifty_rl.metrics.stats import (
    probability_of_backtest_overfitting,
    summarise_significance,
    whites_reality_check,
)
from nifty_rl.regimes import (
    BinarySegmentation,
    GaussianHMMRegimes,
    JumpModelRegimes,
    MarkovSwitchingVariance,
    QuadrantRegimes,
    ThresholdRegimes,
    agreement_matrix,
    build_regime_features,
    detection_lag,
    lag_summary,
    persistence_summary,
    refit_stability,
    regime_conditional_stats,
    select_n_regimes,
    standardise_causally,
)
from nifty_rl.report import figures
from nifty_rl.report.build import write_results
from nifty_rl.strategies.allocators import ALLOCATORS
from nifty_rl.strategies.meta import default_exposure_ladder, performance_by_regime
from nifty_rl.strategies.signals import make_random_signal_fn, make_signal_fn
from nifty_rl.validation import RLConfig, walk_forward_evaluate

ASSETS = PROJECT_ROOT / "assets" / "v2"
RESULTS = PROJECT_ROOT / "results"

RULE_STRATEGIES = {
    "BuyHold": make_signal_fn("buy_hold"),
    "MA_20_50": make_signal_fn("ma", short=20, long=50),
    "RSI_35_60": make_signal_fn("rsi", buy_below=35, sell_above=60),
    "Breakout_20": make_signal_fn("breakout", lookback=20, exit_lookback=10),
    "MomentumPullback": make_signal_fn("momentum_pullback", rsi_max=55),
    "VIX_Regime_Mom": make_signal_fn("vix_regime_momentum"),
    "Sentiment_Momentum": make_signal_fn("sentiment_momentum"),
    "Random": make_random_signal_fn(seed=42),
}

# Observation features for the RL agent. 'technical_vix' from the original notebook --
# 15 per ticker, so a 172-dimensional observation with the availability flags, weights
# and cash ratio.
RL_FEATURES = [
    "ret", "ma_ratio", "trend_20_50", "rsi", "macd_hist", "bb_position", "bb_width",
    "atr_pct", "momentum_5", "momentum_20", "volume_change",
    "india_vix", "vix_change", "high_vix_regime", "dispersion_zscore",
]

RL_SEEDS = (0, 1, 2)
RL_TIMESTEPS = 60_000

REGIME_FEATURE_COLUMNS = [
    "realized_vol_21",
    "trend_21",
    "mean_correlation",
    "dispersion",
    "breadth",
]

TRAIN_DAYS, TEST_DAYS, STEP_DAYS = 500, 125, 125


def log(message: str) -> None:
    print(message, flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regime-aware walk-forward portfolio pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-rl", action="store_true",
        help="Skip PPO training. Runs in ~90s instead of ~20min; everything else is identical.",
    )
    parser.add_argument(
        "--seeds", type=int, default=len(RL_SEEDS),
        help="Number of PPO seeds per walk-forward window.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=RL_TIMESTEPS,
        help="PPO training steps per seed per window.",
    )
    parser.add_argument(
        "--capital", type=float, default=None,
        help="Starting capital in rupees. Changes real results, not just the scale: "
             "integer share sizing leaves proportionally less idle cash at larger sizes.",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Ignore the pinned end_date and pull data up to today. Results will "
             "no longer match the published figures.",
    )
    parser.add_argument(
        "--train-days", type=int, default=TRAIN_DAYS,
        help="Initial training block length, in trading days.",
    )
    parser.add_argument(
        "--test-days", type=int, default=TEST_DAYS,
        help="Out-of-sample block length, in trading days.",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace = None) -> None:
    args = args or parse_args([])
    cfg = RunConfig()
    if args.capital:
        cfg = cfg.with_(backtest=replace(cfg.backtest, initial_cash=args.capital))
        log(f"      capital: Rs{args.capital:,.0f}")
    if args.live:
        cfg = cfg.with_(data=replace(cfg.data, end_date=None))
        log("      NOTE: --live ignores the pinned snapshot; results will drift.")
    train_days, test_days, step_days = args.train_days, args.test_days, args.test_days
    ASSETS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1. data
    log("[1/7] Loading price panel ...")
    panel = align_to_common_dates(add_features(build_panel(cfg.data)))
    all_dates = pd.DatetimeIndex(sorted(panel["Date"].unique()))
    log(
        f"      {panel['ticker'].nunique()} tickers, {len(all_dates)} common trading dates "
        f"({all_dates[0].date()} -> {all_dates[-1].date()})"
    )

    # --------------------------------------------------------------- 2. regimes
    log("[2/7] Building regime features ...")
    regime_features_raw = build_regime_features(panel).dropna()
    first_train = regime_features_raw.index[:train_days]
    regime_features = standardise_causally(regime_features_raw, train_index=first_train)

    selection = select_n_regimes(
        regime_features.loc[first_train], candidates=(2, 3, 4),
        feature_columns=REGIME_FEATURE_COLUMNS,
    )
    selection.to_csv(RESULTS / "regime_model_selection.csv", index=False)
    best_k = int(selection.iloc[0]["n_regimes"])
    log(f"      BIC selects {best_k} regimes on the first training block")
    log(selection.to_string(index=False))

    def make_detector():
        return GaussianHMMRegimes(
            n_regimes=best_k, feature_columns=REGIME_FEATURE_COLUMNS, random_state=cfg.seed
        )

    # Detectors fitted on the FIRST training block only, then run causally over all
    # history -- the view an operator would actually have had on day one.
    display_detectors = {
        "Threshold": ThresholdRegimes(n_regimes=best_k, column="realized_vol_21"),
        "Quadrant": QuadrantRegimes(),
        "HMM": make_detector(),
        "MarkovSwitch": MarkovSwitchingVariance(n_regimes=2, return_column="trend_21"),
        "JumpModel": JumpModelRegimes(
            n_regimes=best_k, feature_columns=REGIME_FEATURE_COLUMNS, jump_penalty=2.0
        ),
    }
    labels, regime_names_map = {}, {}
    for name, detector in display_detectors.items():
        detector.fit(regime_features.loc[first_train])
        labels[name] = detector.label_online(regime_features)
        regime_names_map[name] = detector.regime_labels_

    # ------------------------------------------------- 3. validate the detectors
    log("[3/7] Validating regime models ...")
    segmenter = BinarySegmentation(min_size=40, max_breaks=8)
    breaks = segmenter.breakpoints(regime_features_raw["realized_vol_21"])
    log(f"      retrospective breaks: {[str(d.date()) for d in breaks]}")

    lag_table = {name: lag_summary(detection_lag(s, breaks)) for name, s in labels.items()}
    lag_frame = pd.DataFrame(lag_table).T
    lag_frame.to_csv(RESULTS / "regime_detection_lag.csv")
    log(f"\n{lag_frame.to_string()}\n")

    kappa = agreement_matrix(labels)
    kappa.to_csv(RESULTS / "regime_agreement.csv")

    primary_labels = labels["HMM"]
    primary_names = regime_names_map["HMM"]
    persistence = persistence_summary(primary_labels, regime_names=primary_names)
    persistence.to_csv(RESULTS / "regime_persistence.csv", index=False)
    log(persistence.to_string(index=False))
    log(f"      switch rate {persistence.attrs['switch_rate']:.3f}  "
        f"mean run {persistence.attrs['overall_mean_run']:.1f}d")

    stability = refit_stability(
        make_detector, regime_features, initial_train=train_days, step=step_days
    )
    stability.to_csv(RESULTS / "regime_refit_stability.csv", index=False)
    mean_kappa = stability["kappa_vs_previous"].mean()
    log(f"      refit stability: mean kappa vs previous fit = {mean_kappa:.3f}")

    market = price_matrix(panel).mean(axis=1)
    regime_econ = regime_conditional_stats(
        market.pct_change().dropna(), primary_labels, regime_names=primary_names
    )
    regime_econ.to_csv(RESULTS / "regime_economics.csv", index=False)
    log(f"\n{regime_econ.to_string(index=False)}\n")

    # ------------------------------------------------------- 4. allocator weights
    log("[4/7] Building allocator weight schedules ...")
    prices = price_matrix(panel)
    allocator_weights = {}
    for name, allocator in ALLOCATORS.items():
        weights = build_allocator_weights(prices, allocator, lookback=252, frequency="ME")
        if not weights.empty:
            allocator_weights[name] = weights
    log(f"      {len(allocator_weights)} allocators: {list(allocator_weights)}")

    # ------------------------------------------------------- 5. walk-forward
    log(f"[5/7] Rolling walk-forward (expanding train, {test_days}-day test blocks) "
        + ("without PPO ..." if args.no_rl
           else f"with PPO: {args.seeds} seeds x {args.timesteps:,} steps per window ..."))
    passive_cfg = replace(cfg.backtest, stop_loss=None, take_profit=None)
    ladder = default_exposure_ladder(best_k)
    log(f"      regime exposure ladder {ladder} over {primary_names}")

    report = walk_forward_evaluate(
        panel=panel,
        regime_features=regime_features,
        regime_feature_columns=REGIME_FEATURE_COLUMNS,
        detector_factory=make_detector,
        rule_strategies=RULE_STRATEGIES,
        allocator_weights=allocator_weights,
        prices=prices,
        cfg=cfg.backtest,
        cost_cfg=cfg.costs,
        metrics_cfg=cfg.metrics,
        passive_cfg=passive_cfg,
        benchmark="BuyHold",
        exposure_ladder=ladder,
        overlay_bases=("HRP", "EqualWeight", "RiskParity"),
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        expanding=True,
        rl_config=None if args.no_rl else RLConfig(
            features=RL_FEATURES,
            seeds=tuple(range(args.seeds)),
            timesteps=args.timesteps,
            val_fraction=0.2,
            include_per_seed=True,
            check_freq=10_000,
        ),
        progress=log,
    )
    report.per_window.to_csv(RESULTS / "walk_forward_windows.csv", index=False)
    report.summary.to_csv(RESULTS / "walk_forward_summary.csv", index=False)

    log(f"\n      {report.n_windows} out-of-sample windows, "
        f"{len(next(iter(report.pooled_returns.values())))} pooled OOS trading days")
    log(report.summary[
        ["strategy", "pooled_total_return", "pooled_sharpe", "windows_positive",
         "windows_beating_benchmark", "worst_window_return"]
    ].round(3).to_string(index=False))

    selections = pd.Series([w.selected_strategy for w in report.windows])
    log(f"\n      train-chosen strategy per window: {list(selections)}")

    # --------------------------------------------------------------- 6. inference
    log("\n[6/7] Pooled out-of-sample significance ...")
    exposure = dict(zip(report.summary["strategy"], report.summary["mean_exposure"]))
    n_trials = len(report.pooled_returns) + 16 + 3 + 4
    significance = summarise_significance(
        report.pooled_returns, "BuyHold", n_trials=n_trials,
        risk_free_annual=cfg.metrics.risk_free_annual,
        exposure_by_strategy=exposure,
    )
    significance.to_csv(RESULTS / "significance.csv", index=False)

    returns_matrix = pd.DataFrame(report.pooled_returns).dropna()
    pbo = probability_of_backtest_overfitting(returns_matrix, n_splits=8)
    reality = whites_reality_check(
        returns_matrix.drop(columns=["BuyHold"]), returns_matrix["BuyHold"], n_boot=500
    )
    pd.DataFrame([{**pbo, **reality}]).to_csv(RESULTS / "overfitting_tests.csv", index=False)
    log(f"      PBO = {pbo['pbo']:.2f}   White's RC p = {reality['p_value']:.3f} "
        f"(best: {reality['best_strategy']})")

    regime_performance = performance_by_regime(
        report.pooled_returns, report.pooled_regimes,
        regime_names=primary_names, risk_free_annual=cfg.metrics.risk_free_annual,
    )
    regime_performance.to_csv(RESULTS / "performance_by_regime.csv", index=False)

    # ---------------------------------------------------------------- 7. figures
    log("[7/7] Rendering figures ...")
    ranked = report.summary["strategy"].tolist()
    highlight = [s for s in ["PPO_ensemble", "MaxSharpe", "BuyHold"] if s in report.pooled_equity]
    if len(highlight) < 3:
        highlight = (ranked[:2] + ["BuyHold"])[:3]

    for mode in ("light", "dark"):
        suffix = "" if mode == "light" else "_dark"

        figures.save(
            *figures.window_returns_heatmap(
                report.per_window, mode=mode,
                subtitle=f"{report.n_windows} windows, expanding-train refit, "
                         f"{test_days}-day out-of-sample blocks.",
            ),
            ASSETS / f"walk_forward_windows{suffix}.png",
        )
        figures.save(
            *figures.consistency_scatter(report.summary, highlight, mode=mode),
            ASSETS / f"consistency{suffix}.png",
        )
        if report.per_window["strategy"].str.startswith("PPO_s").any():
            figures.save(
                *figures.ppo_seed_dispersion(report.per_window, mode=mode),
                ASSETS / f"ppo_seed_dispersion{suffix}.png",
            )
        figures.save(
            *figures.equity_curves(
                report.pooled_equity, highlight, cfg.backtest.initial_cash, mode=mode,
                title="Pooled out-of-sample equity — every walk-forward block chained",
                subtitle="Each segment was traded with parameters frozen before it began.",
            ),
            ASSETS / f"equity_curves{suffix}.png",
        )
        figures.save(
            *figures.drawdown_curves(
                report.pooled_equity, highlight, mode=mode,
                title="Pooled out-of-sample drawdown",
            ),
            ASSETS / f"drawdown{suffix}.png",
        )
        figures.save(
            *figures.sharpe_forest(significance, mode=mode,
                title="Pooled out-of-sample Sharpe with bootstrap confidence intervals"),
            ASSETS / f"sharpe_forest{suffix}.png",
        )
        figures.save(
            *figures.regime_timeline(
                market.loc[regime_features.index], labels, regime_names_map,
                breakpoints=breaks, mode=mode,
                subtitle="Fitted on the first training block only, then run causally forward.",
            ),
            ASSETS / f"regime_timeline{suffix}.png",
        )
        figures.save(
            *figures.regime_validation_panel(
                persistence, lag_table, mode=mode,
                subtitle="Both gates must pass before the regime layer is used downstream.",
            ),
            ASSETS / f"regime_validation{suffix}.png",
        )
        figures.save(
            *figures.agreement_heatmap(kappa, mode=mode,
                subtitle="Chance-corrected agreement between independent detectors."),
            ASSETS / f"regime_agreement{suffix}.png",
        )
        figures.save(
            *figures.regime_transition_heatmap(
                display_detectors["HMM"].transition_frame(), mode=mode,
                subtitle="Expected durations: " + ", ".join(
                    f"{k} {v:.0f}d"
                    for k, v in display_detectors["HMM"].expected_durations().items()
                ),
            ),
            ASSETS / f"regime_transitions{suffix}.png",
        )
        figures.save(
            *figures.regime_performance_heatmap(
                regime_performance, "sharpe", mode=mode,
                title="Pooled out-of-sample performance by regime",
                subtitle="Does a risk-penalised strategy earn its penalty when conditions are bad?",
            ),
            ASSETS / f"performance_by_regime{suffix}.png",
        )

    log(f"      figures -> {ASSETS}")

    invested = significance[~significance["cash_like"]]
    best = invested.iloc[0] if len(invested) else significance.iloc[0]
    summary = {
        "evaluation": "rolling walk-forward (expanding train)",
        "n_windows": report.n_windows,
        "train_days": train_days,
        "test_days": test_days,
        "oos_start": str(report.windows[0].spec.test_start.date()),
        "oos_end": str(report.windows[-1].spec.test_end.date()),
        "n_oos_days": int(len(next(iter(report.pooled_returns.values())))),
        "n_strategies": len(report.pooled_returns),
        "n_trials": n_trials,
        "best_strategy": best["strategy"],
        "best_pooled_sharpe": float(best["sharpe"]),
        "best_dsr": float(best["dsr"]),
        "n_ci_excludes_zero": int(significance["ci_excludes_zero"].sum()),
        "pbo": pbo["pbo"],
        "reality_check_p": reality["p_value"],
        "regime_switch_rate": persistence.attrs["switch_rate"],
        "regime_mean_run_days": persistence.attrs["overall_mean_run"],
        "regime_refit_kappa": float(mean_kappa),
    }
    pd.Series(summary).to_csv(RESULTS / "run_summary.csv")

    results_md = write_results(RESULTS, ASSETS, summary, PROJECT_ROOT / "RESULTS.md")
    log(f"      report  -> {results_md}")

    log("\n=== SUMMARY ===")
    for key, value in summary.items():
        log(f"  {key:<24s} {value}")


if __name__ == "__main__":
    main(parse_args())
