#!/usr/bin/env python3
"""Generate notebooks/01_walkthrough.ipynb from this file.

The notebook is *generated*, not hand-edited, for the same reason RESULTS.md is: a
narrative that can drift from the code it describes will drift. Editing happens here;
the .ipynb is a build artefact.

Figures are referenced as markdown links to `../assets/v2/*.png` rather than embedded as
cell outputs. That keeps the file small (a notebook with embedded PNGs is how the
original repo ended up with a 2.6 MB duplicate) and still renders on GitHub without
anyone running it.

    python3 scripts/build_notebook.py            # build
    python3 scripts/build_notebook.py --execute  # build and run it, keeping outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "notebooks" / "01_walkthrough.ipynb"

CELLS = []


def md(text: str) -> None:
    CELLS.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    CELLS.append(nbf.v4.new_code_cell(text.strip()))


# --------------------------------------------------------------------------- intro

md("""
# Regime-Aware Portfolio Research — NIFTY 50

A walkthrough of the pipeline in `src/nifty_rl/`. **This notebook contains no logic of
its own** — every function it calls is the same tested code path that produces
[`RESULTS.md`](../RESULTS.md), so the narrative cannot drift from the results.

That is deliberate. This project began as a single 44-cell notebook in which twenty
correctness bugs were found, several of them *caused* by notebook structure: globals
reassigned in place, a dispatcher that silently returned zeros, functions that could not
be exercised in isolation because they all reached for a module-level `CFG`. The logic
now lives in a package with 79 tests. What remains here is the story.

**Runtime:** about 60 seconds. The walk-forward runs live; the PPO results are loaded
from committed artefacts because training them takes ~20 minutes
(`python3 scripts/run_pipeline.py` does the whole thing).
""")

code("""
import sys, warnings
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

from nifty_rl.config import RunConfig

cfg = RunConfig()
print(f"universe   : {len(cfg.data.tickers)} NIFTY 50 names")
print(f"period     : {cfg.data.start_date} -> {cfg.data.end_date}  (pinned snapshot)")
print(f"costs      : {cfg.costs.transaction_cost:.2%} + {cfg.costs.slippage:.2%} slippage")
print(f"cash rate  : {cfg.backtest.cash_rate_annual:.1%}   risk-free: {cfg.metrics.risk_free_annual:.1%}")
""")

# ---------------------------------------------------------------------------- data

md("""
---

## 1 · Data

`end_date` is **pinned**. The original called `yf.download(start=...)` with no `end`, so
the dataset grew every day, split boundaries moved, and no published figure could ever be
reproduced. Data is cached to `data/raw/` on first fetch; every run afterwards is offline.

`auto_adjust=True` matters more than it looks: the original took `price = Adj Close` but
left `High`/`Low` raw, so true range differenced two price scales and a bonus issue put a
~50% spike into `atr_pct`.
""")

code("""
from nifty_rl.data import build_panel

panel_raw = build_panel(cfg.data)
print(f"{len(panel_raw):,} rows | {panel_raw['ticker'].nunique()} tickers | "
      f"{panel_raw['Date'].min().date()} -> {panel_raw['Date'].max().date()}")
panel_raw.head(3)
""")

# ------------------------------------------------------------------------ features

md("""
## 2 · Features

All indicators are computed **once, over full history**, inside a per-ticker groupby.

There is a rule enforced in code: **a signal function may never call `.rolling()`.** The
original recomputed moving averages on whatever slice it was handed, so the first 50 bars
of every test window collapsed to zero — on a clean uptrend it was long *11 of 60* bars
where it should have been long on all 60. Anything needing a window belongs in the
feature layer, which sees the whole series.

The cell below shows the guard firing.
""")

code("""
from nifty_rl.features import add_features, align_to_common_dates
from nifty_rl.strategies.signals import ma_crossover_signals

panel = align_to_common_dates(add_features(panel_raw))
print(f"{panel['Date'].nunique()} common trading dates x {panel['ticker'].nunique()} tickers")

# A signal handed a frame without precomputed indicators refuses to guess.
try:
    ma_crossover_signals(panel_raw[panel_raw.ticker == "INFY.NS"], short=20, long=50)
except KeyError as exc:
    print(f"\\nguard fired as intended:\\n  {str(exc)[:150]}...")
""")

# ------------------------------------------------------------------------- regimes

md("""
---

## 3 · Regime detection, and the contract that makes it believable

Five detectors run in parallel. Every one satisfies a single property:

$$P(\\text{regime}_t \\mid \\text{information up to } t)$$

— the estimate for day *t* must not change when future data arrives.

**This is where regime work usually breaks silently.** `hmmlearn.predict()` runs Viterbi
over the whole sequence. `predict_proba()` returns forward–backward *smoothed*
posteriors. Both read the future, and both are the obvious methods to reach for. So the
Gaussian HMM here implements its own forward filter: Baum-Welch for fitting, but only the
normalised forward pass at prediction time.

The contract is verified by brute force — recompute every prefix, compare against the
full-sample run.
""")

code("""
from nifty_rl.regimes import (
    GaussianHMMRegimes, assert_causal, build_regime_features, standardise_causally,
)

REGIME_FEATURES = ["realized_vol_21", "trend_21", "mean_correlation", "dispersion", "breadth"]
TRAIN_DAYS = 500

raw_features = build_regime_features(panel).dropna()
train_index = raw_features.index[:TRAIN_DAYS]
features = standardise_causally(raw_features, train_index=train_index)   # train stats only

hmm = GaussianHMMRegimes(n_regimes=4, feature_columns=REGIME_FEATURES, random_state=42)
hmm.fit(features.loc[train_index])
print(f"fitted on {len(train_index)} days, converged in {hmm.n_iter_} EM iterations")
print(f"regimes: {hmm.regime_labels_}")
""")

code("""
# The proof. Recomputes the filtered distribution on every prefix and checks the last row
# matches the full-sample run. Raises AssertionError on the first violation.
assert_causal(hmm, features, start=60, step=97)
print("CAUSAL: P(regime_t) is unchanged by the arrival of future data")
""")

code("""
# Why the forward filter, concretely: smoothed posteriors DISAGREE with filtered ones,
# because they have seen the rest of the series. They agree only at the final observation,
# where there is no future left to leak.
matrix = hmm._matrix(features)
emission, _ = hmm._scaled_emission(hmm._log_emission(matrix))
alpha, scaling = hmm._forward(emission)
beta = hmm._backward(emission, scaling)

smoothed = alpha * beta
smoothed /= smoothed.sum(axis=1, keepdims=True)

mid = len(matrix) // 2
comparison = pd.DataFrame(
    {"filtered (causal)": alpha[mid], "smoothed (uses future)": smoothed[mid]},
    index=hmm.regime_labels_,
)
print(f"day {mid} of {len(matrix)}:")
print(comparison.round(4))
print(f"\\nmax disagreement at this date : {np.abs(alpha[mid] - smoothed[mid]).max():.4f}")
print(f"max disagreement at final date: {np.abs(alpha[-1] - smoothed[-1]).max():.2e}  <- nothing left to leak")
""")

md("""
### 3b · Validating the detector before trusting it

A fitted regime path proves nothing on its own. Three gates decide whether the labels are
usable, and all three are checked before anything is conditioned on them.

**Persistence** — a model that flips every three days is untradeable after costs, however
well it fits. **Detection lag** — how many days after a structural break does the online
detector react? **Economic sanity** — does the "crisis" state actually show higher
volatility and worse drawdown, or has the model just found clusters?
""")

code("""
from nifty_rl.regimes import persistence_summary, regime_conditional_stats

labels = hmm.label_online(features)

persistence = persistence_summary(labels, regime_names=hmm.regime_labels_)
print("PERSISTENCE")
print(persistence.round(3).to_string(index=False))
print(f"\\nswitch rate {persistence.attrs['switch_rate']:.3f}  |  "
      f"mean run {persistence.attrs['overall_mean_run']:.1f} days")
print(f"\\nexpected durations from the transition matrix:")
print(hmm.expected_durations().round(1).to_string())
""")

code("""
from nifty_rl.backtest import price_matrix

market_returns = price_matrix(panel).mean(axis=1).pct_change().dropna()
econ = regime_conditional_stats(market_returns, labels, regime_names=hmm.regime_labels_)
print("ECONOMIC SANITY — volatility and drawdown must worsen from Calm to Crisis")
print(econ.round(3).to_string(index=False))
""")

md("""
Detection lag is measured against a **retrospective** change-point segmenter — a
full-sequence method that sees everything, so it can never be traded. That is precisely
what makes it a fair yardstick, and it is a separate type with no `predict_online` method
so the two can never be confused.

![Regime timeline](../assets/v2/regime_timeline.png)

![Regime validation](../assets/v2/regime_validation.png)
""")

# -------------------------------------------------------------------- walk-forward

md("""
---

## 4 · Why a single train/test split is not an evaluation

This is the most important idea in the project.

A 70/15/15 chronological split gives you exactly **one** out-of-sample window, and
whichever regime it lands in *is* your result. In this dataset it landed on a drawdown,
and the conclusion was "every strategy lost money" — true of that window, and nearly
uninformative about the strategies. Change the ratios and the ranking changes with them.

Rolling walk-forward instead: fit on everything up to *t*, trade the next 125 days with
parameters frozen, roll forward, refit. Chain every out-of-sample block into one
continuous track record.

```
│◄──────── TRAIN (expanding) ────────►│◄─ TEST 125d ─►│
                                       ▲
                              everything frozen here
```
""")

code("""
from nifty_rl.validation import rolling_windows

dates = pd.DatetimeIndex(sorted(panel["Date"].unique()))
windows = rolling_windows(dates, train_days=500, test_days=125, step_days=125)

print(f"{len(windows)} windows, {sum(len(w.test_dates) for w in windows)} out-of-sample days\\n")
for w in windows:
    print(f"  W{w.index}  train {w.train_start.date()} -> {w.train_end.date()}"
          f"   |   test {w.test_start.date()} -> {w.test_end.date()}")
""")

code("""
from dataclasses import replace

from nifty_rl.backtest import build_allocator_weights
from nifty_rl.strategies.allocators import ALLOCATORS
from nifty_rl.strategies.meta import default_exposure_ladder
from nifty_rl.strategies.signals import make_random_signal_fn, make_signal_fn
from nifty_rl.validation import walk_forward_evaluate

RULES = {
    "BuyHold": make_signal_fn("buy_hold"),
    "MA_20_50": make_signal_fn("ma", short=20, long=50),
    "RSI_35_60": make_signal_fn("rsi", buy_below=35, sell_above=60),
    "Breakout_20": make_signal_fn("breakout", lookback=20, exit_lookback=10),
    "MomentumPullback": make_signal_fn("momentum_pullback", rsi_max=55),
    "VIX_Regime_Mom": make_signal_fn("vix_regime_momentum"),
    "Sentiment_Momentum": make_signal_fn("sentiment_momentum"),
    "Random": make_random_signal_fn(seed=42),
}

prices = price_matrix(panel)
allocator_weights = {
    name: build_allocator_weights(prices, fn, lookback=252, frequency="ME")
    for name, fn in ALLOCATORS.items()
}

# PPO is excluded here (rl_config=None) purely for runtime; see section 6.
report = walk_forward_evaluate(
    panel=panel,
    regime_features=features,
    regime_feature_columns=REGIME_FEATURES,
    detector_factory=lambda: GaussianHMMRegimes(
        n_regimes=4, feature_columns=REGIME_FEATURES, random_state=42),
    rule_strategies=RULES,
    allocator_weights=allocator_weights,
    prices=prices,
    cfg=cfg.backtest,
    cost_cfg=cfg.costs,
    metrics_cfg=cfg.metrics,
    passive_cfg=replace(cfg.backtest, stop_loss=None, take_profit=None),
    exposure_ladder=default_exposure_ladder(4),
    overlay_bases=("HRP", "EqualWeight", "RiskParity"),
    train_days=500, test_days=125, step_days=125,
    rl_config=None,
)
print(f"{report.n_windows} windows complete")
""")

md("""
Note `passive_cfg` above. The original ran its buy-and-hold benchmark through a
backtester defaulting to a 6% stop-loss and 12% take-profit — so every "excess return vs
benchmark" was measured against a stop-loss strategy mislabelled as passive. Worth 11.9
percentage points.
""")

code("""
summary = report.summary[
    ["strategy", "pooled_total_return", "pooled_sharpe",
     "windows_positive", "windows_beating_benchmark", "worst_window_return"]
]
print("POOLED OUT-OF-SAMPLE  (Sharpe is excess-over-cash at 6.5%)")
summary.round(3)
""")

md("""
`windows_beating_benchmark` matters more than the pooled return. A strategy that beats
buy-and-hold in 3 of 8 windows is a strategy that had a good window.

Note also which strategies show a positive raw return and a negative Sharpe:
`MomentumPullback` returns +1.8% while cash returned ~6.5%, so it lost ~4.7 points in
real terms. The original scored everything against a risk-free rate of zero, which makes
any positive number look like success.

![Out-of-sample return by window](../assets/v2/walk_forward_windows.png)

![Pooled equity](../assets/v2/equity_curves.png)
""")

# ---------------------------------------------------------------------- statistics

md("""
---

## 5 · Does any of it survive multiple testing?

The search space here is large: SL/TP grids, PPO configurations, validation-selected
parameters, feature sets, five regime backends. The best observed Sharpe is therefore an
**order statistic**, and comparing it to zero is the wrong test.
""")

code("""
from nifty_rl.metrics.stats import (
    probability_of_backtest_overfitting, summarise_significance, whites_reality_check,
)

exposure = dict(zip(report.summary["strategy"], report.summary["mean_exposure"]))
significance = summarise_significance(
    report.pooled_returns, "BuyHold", n_trials=len(report.pooled_returns) + 23,
    risk_free_annual=cfg.metrics.risk_free_annual, exposure_by_strategy=exposure,
)

returns_matrix = pd.DataFrame(report.pooled_returns).dropna()
pbo = probability_of_backtest_overfitting(returns_matrix, n_splits=8)
rc = whites_reality_check(returns_matrix.drop(columns=["BuyHold"]),
                          returns_matrix["BuyHold"], n_boot=500)

print(f"Probability of Backtest Overfitting : {pbo['pbo']:.2f}   (0.5 = selection is noise)")
print(f"White's Reality Check p-value       : {rc['p_value']:.3f}   (best: {rc['best_strategy']})")
print(f"Confidence intervals excluding zero : {int(significance['ci_excludes_zero'].sum())}"
      f" of {len(significance)}\\n")

significance[["strategy", "sharpe", "sharpe_ci_lower", "sharpe_ci_upper",
              "ci_excludes_zero", "dsr", "cash_like"]].round(3)
""")

md("""
`cash_like` flags a strategy whose mean exposure fell below 5%. `RSI_35_60` never entered
a position: its return is entirely the cash rate, and its Sharpe is **undefined** rather
than infinite. Before that guard existed, zero excess return over zero excess volatility
produced a Sharpe of 2.1e13 and ranked a do-nothing rule first on every table.

![Sharpe with confidence intervals](../assets/v2/sharpe_forest.png)

**The honest reading:** the walk-forward record is positive and reasonably consistent, but
after correcting for how many configurations were tried, no strategy here is
statistically distinguishable from the passive benchmark.
""")

# ----------------------------------------------------------------------------- PPO

md("""
---

## 6 · The reinforcement-learning agent

PPO is retrained from scratch inside **every** walk-forward window — feature scaler refit
on that window's training block, an inner 80/20 split so the Sharpe checkpoint has
validation data it never trained on, three independent seeds. Selection and deployment use
the same rule, which the original did not do: it scored hyperparameter candidates on their
*final iterate* but deployed the *best checkpoint*.

Training all 24 runs takes ~20 minutes, so the results below are loaded from committed
artefacts. Reproduce with `python3 scripts/run_pipeline.py`.
""")

code("""
ASSETS = PROJECT_ROOT / "assets" / "v2"

dispersion = pd.read_csv(ASSETS / "ppo_seed_dispersion.csv", index_col=0)
print("PPO SEED DISPERSION — out-of-sample return per window, across 3 independent runs")
print((dispersion * 100).round(2).to_string())
print(f"\\nmean best-to-worst spread : {(dispersion['max'] - dispersion['min']).mean():.2%}")
print(f"median within-window std  : {dispersion['seed_std'].median():.2%}")
""")

code("""
equity = pd.read_csv(ASSETS / "equity_curves.csv", index_col=0, parse_dates=True)
rets = equity.pct_change().dropna()

print("WHAT THE POLICY ACTUALLY LEARNED\\n")
print(f"{'strategy':<16}{'corr w/ BuyHold':>18}")
for name in ["PPO_ensemble", "EqualWeight", "HRP", "MaxSharpe"]:
    if name in rets.columns:
        print(f"{name:<16}{rets[name].corr(rets['BuyHold']):>18.3f}")
""")

md("""
**The agent converged to buy-and-hold.** Its daily returns correlate **0.993** with the
benchmark at **99.2%** mean exposure — a tighter match than EqualWeight manages. It then
paid turnover for the privilege: behind the benchmark in 6 of 8 windows, mean −0.82pp,
best-case edge +0.15%.

**And that is not seed noise.** Three independent runs land within half a percent of each
other; in one window, within 0.06%. That tightness is the finding. It says the optimiser
found what the reward was *asking* for — so the fix is not "train longer", it is that the
reward is close to maximising log wealth with weak penalties, and full investment is very
nearly its correct answer. Changing the outcome means changing the question: the
differential Sharpe reward in `envs/rewards.py`, a turnover cost the agent can feel, or a
shorter rebalancing horizon where timing has something to decide.

The original reported a single seed and listed that as a limitation. This is what checking
it looks like.

![PPO seed dispersion](../assets/v2/ppo_seed_dispersion.png)
""")

# ------------------------------------------------------------------------- closing

md("""
---

## 7 · What this study concludes

1. **Rolling walk-forward, not a single split.** Same code and data: the single split
   said 0 of 18 strategies were profitable; walk-forward over 992 out-of-sample days says
   18 of 19. Neither is wrong — the first measured one bear market.
2. **Regimes are detectable causally, and worth detecting** — 20.6-day mean run,
   κ = 0.70 refit stability, volatility and drawdown worsening monotonically from Calm to
   Crisis. But the **exposure overlay does not pay for itself**: it cut drawdown and cost
   more return than it saved, because de-risking in high volatility means being
   underweight through the rebounds.
3. **PPO reimplemented the benchmark at a cost**, reproducibly across seeds.
4. **Nothing beats buy-and-hold with statistical significance** once the size of the
   search is accounted for.

### Limitations, stated plainly

Single seed for the regime models (the eight refits are a partial substitute). No PPO
hyperparameter search — doing it naively becomes the data-snooping problem section 5
exists to measure. Flat cost model by default; the full Indian STT/stamp/GST stack is
implemented as `CostConfig(model="india")` but not the default. Ten stocks, one market,
one six-year period.

### Reproducing

```bash
python3 -m pytest tests/ -q                    # 79 tests, ~3s, no network
python3 scripts/run_pipeline.py --no-rl        # ~37s
python3 scripts/run_pipeline.py                # ~20min, includes PPO
```

Full write-up: [`RESULTS.md`](../RESULTS.md) · plan of record: [`ROADMAP.md`](../ROADMAP.md)
""")


def build(execute: bool = False) -> Path:
    notebook = nbf.v4.new_notebook(cells=CELLS)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if execute:
        from nbconvert.preprocessors import ExecutePreprocessor

        ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(
            notebook, {"metadata": {"path": str(OUTPUT.parent)}}
        )

    nbf.write(notebook, str(OUTPUT))
    return OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="Run the notebook and keep its outputs.")
    args = parser.parse_args()

    path = build(execute=args.execute)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(PROJECT_ROOT)}  ({len(CELLS)} cells, {size_kb:.0f} KB)")
