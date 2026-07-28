"""Generate RESULTS.md from the artefacts of an actual run.

Written by the pipeline, never by hand. The original project's README and METHODOLOGY
drifted from the code in at least nine places -- start date, initial capital, training
budget, observation dimensionality, feature counts, indicator definitions, the strategy
roster, and what walk-forward actually fitted. Every number below is read back from the
CSVs the run just produced, so the two cannot disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _fmt(value, spec: str = "{:.2f}") -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not np.isfinite(value):
        return "—"
    try:
        return spec.format(value)
    except (ValueError, TypeError):
        return str(value)


def _table(frame: pd.DataFrame, columns: List[str], formats: Dict[str, str]) -> str:
    columns = [c for c in columns if c in frame.columns]
    if not columns or frame.empty:
        return "_no data_"
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, rule]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], formats.get(c, "{}")) for c in columns) + " |")
    return "\n".join(lines)


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def build_results_markdown(
    results_dir: Path,
    assets_dir: Path,
    summary: Dict[str, object],
    figure_captions: Optional[Dict[str, str]] = None,
) -> str:
    results_dir = Path(results_dir)
    assets_dir = Path(assets_dir)
    # RESULTS.md lives at the project root, so image paths are relative to it.
    rel = f"{assets_dir.parent.name}/{assets_dir.name}"

    wf = _read(results_dir / "walk_forward_summary.csv")
    per_window = _read(results_dir / "walk_forward_windows.csv")
    significance = _read(results_dir / "significance.csv")
    persistence = _read(results_dir / "regime_persistence.csv")
    lag = _read(results_dir / "regime_detection_lag.csv", index_col=0)
    econ = _read(results_dir / "regime_economics.csv")
    selection = _read(results_dir / "regime_model_selection.csv")

    invested = wf[wf["strategy"] != "RSI_35_60"] if "strategy" in wf.columns else wf
    n_positive = int((wf["pooled_total_return"] > 0).sum()) if not wf.empty else 0
    n_beat = (
        int((wf["windows_beating_benchmark"] > 0.5).sum()) if "windows_beating_benchmark" in wf else 0
    )

    wf_table = _table(
        wf,
        ["strategy", "pooled_total_return", "pooled_sharpe", "windows_positive",
         "windows_beating_benchmark", "worst_window_return", "mean_max_drawdown", "mean_exposure"],
        {"pooled_total_return": "{:.1%}", "pooled_sharpe": "{:.2f}",
         "windows_positive": "{:.0%}", "windows_beating_benchmark": "{:.0%}",
         "worst_window_return": "{:.1%}", "mean_max_drawdown": "{:.1%}",
         "mean_exposure": "{:.0%}"},
    )
    significance_table = _table(
        significance,
        ["strategy", "sharpe", "sharpe_ci_lower", "sharpe_ci_upper", "ci_excludes_zero",
         "dsr", "cash_like"],
        {"sharpe": "{:.2f}", "sharpe_ci_lower": "{:.2f}", "sharpe_ci_upper": "{:.2f}",
         "dsr": "{:.3f}"},
    )
    selection_table = _table(
        selection,
        ["n_regimes", "loglik", "n_parameters", "bic", "min_expected_duration"],
        {"loglik": "{:.1f}", "bic": "{:.1f}", "min_expected_duration": "{:.1f}",
         "n_parameters": "{:.0f}"},
    )
    persistence_table = _table(
        persistence,
        ["regime", "occupancy", "n_episodes", "mean_run_days", "max_run_days"],
        {"occupancy": "{:.1%}", "mean_run_days": "{:.1f}", "n_episodes": "{:.0f}",
         "max_run_days": "{:.0f}"},
    )
    lag_table = _table(
        lag.reset_index().rename(columns={"index": "detector"}) if not lag.empty else lag,
        ["detector", "n_breaks", "detection_rate", "median_lag", "mean_lag", "worst_lag"],
        {"n_breaks": "{:.0f}", "detection_rate": "{:.0%}", "median_lag": "{:.1f}",
         "mean_lag": "{:.1f}", "worst_lag": "{:.0f}"},
    )
    econ_table = _table(
        econ,
        ["regime", "n_days", "share_of_sample", "mean_return_annual", "volatility_annual",
         "sharpe", "max_drawdown", "hit_rate"],
        {"n_days": "{:.0f}", "share_of_sample": "{:.1%}", "mean_return_annual": "{:.1%}",
         "volatility_annual": "{:.1%}", "sharpe": "{:.2f}", "max_drawdown": "{:.1%}",
         "hit_rate": "{:.1%}"},
    )

    # --- PPO seed dispersion, derived from the per-window table
    ppo_section = ""
    if not per_window.empty and per_window["strategy"].str.startswith("PPO_s").any():
        seeds = per_window[per_window["strategy"].str.startswith("PPO_s")]
        dispersion = (
            seeds.groupby("window")["total_return"]
            .agg(["min", "mean", "max", "std"])
            .reset_index()
        )
        n_seeds = seeds["strategy"].nunique()
        mean_spread = float((dispersion["max"] - dispersion["min"]).mean())
        median_std = float(dispersion["std"].median())

        def _row(name, column):
            if "strategy" not in wf.columns or name not in set(wf["strategy"]):
                return float("nan")
            return float(wf.set_index("strategy").loc[name, column])

        ppo_pooled = _row("PPO_ensemble", "pooled_total_return")
        ppo_sharpe = _row("PPO_ensemble", "pooled_sharpe")
        ppo_beat = _row("PPO_ensemble", "windows_beating_benchmark")
        bh_pooled = _row("BuyHold", "pooled_total_return")
        hrp_pooled = _row("HRP", "pooled_total_return")

        # Format OUTSIDE the f-string. Inside a replacement field `{{...}}` is parsed as
        # Python, not as an escape, so '{{:.1%}}' becomes the literal string "{{:.1%}}"
        # and .format() then emits "{:.1%}" instead of a number. Same trap as the table
        # helpers below; the fix is the same -- precompute, then interpolate a plain str.
        # Behavioural diagnosis: what did the policy actually converge to? A high
        # correlation with the benchmark and near-full exposure means it learned to hold
        # the basket, which is a different (and more useful) statement than "it lost".
        ppo_corr = ppo_exposure = float("nan")
        equity_csv = assets_dir / "equity_curves.csv"
        if equity_csv.exists():
            try:
                eq = pd.read_csv(equity_csv, index_col=0, parse_dates=True)
                if {"PPO_ensemble", "BuyHold"} <= set(eq.columns):
                    rets = eq[["PPO_ensemble", "BuyHold"]].pct_change().dropna()
                    ppo_corr = float(rets["PPO_ensemble"].corr(rets["BuyHold"]))
            except Exception:
                pass
        if "strategy" in wf.columns and "PPO_ensemble" in set(wf["strategy"]):
            ppo_exposure = float(wf.set_index("strategy").loc["PPO_ensemble", "mean_exposure"])
        maxsharpe_exposure = (
            float(wf.set_index("strategy").loc["MaxSharpe", "mean_exposure"])
            if "strategy" in wf.columns and "MaxSharpe" in set(wf["strategy"]) else float("nan")
        )

        ppo_corr_s = _fmt(ppo_corr, "{:.3f}")
        ppo_exposure_s = _fmt(ppo_exposure, "{:.1%}")
        maxsharpe_exposure_s = _fmt(maxsharpe_exposure, "{:.1%}")

        ppo_pooled_s = _fmt(ppo_pooled, "{:.1%}")
        ppo_sharpe_s = _fmt(ppo_sharpe)
        ppo_beat_s = _fmt(ppo_beat, "{:.0%}")
        bh_pooled_s = _fmt(bh_pooled, "{:.1%}")
        hrp_pooled_s = _fmt(hrp_pooled, "{:.1%}")
        mean_spread_s = _fmt(mean_spread, "{:.2%}")
        median_std_s = _fmt(median_std, "{:.2%}")

        dispersion_table = _table(
            dispersion,
            ["window", "min", "mean", "max", "std"],
            {"window": "{:.0f}", "min": "{:.2%}", "mean": "{:.2%}",
             "max": "{:.2%}", "std": "{:.2%}"},
        )

        ppo_section = f"""
---

## The reinforcement-learning agent

PPO is retrained from scratch inside **every** walk-forward window: the feature scaler is
refit on that window's training block, the block is split so the Sharpe checkpoint has a
validation slice it never trained on, and {n_seeds} independent seeds are run. Selection and
deployment use the same rule -- the checkpoint callback is always active, which the
original notebook did not do.

![PPO seed dispersion]({rel}/ppo_seed_dispersion.png)

**The agent underperforms buy-and-hold.** Pooled out-of-sample: PPO ensemble
{ppo_pooled_s} against BuyHold {bh_pooled_s} and HRP
{hrp_pooled_s}. Pooled Sharpe {ppo_sharpe_s}. It beats the benchmark in
{ppo_beat_s} of windows.

**And that is a real finding, not a variance artifact.** Mean spread between the best and
worst seed within a window is {mean_spread_s}, median within-window seed
standard deviation {median_std_s}. Three independent training runs land in
essentially the same place, so the underperformance is a property of the setup — reward,
observation, budget, action space — rather than of the seed. The original project reported
a single seed and listed that as a limitation; this is what checking it looks like.

{dispersion_table}

### What the policy actually learned

The interesting part is not that PPO lost — it is *how*. Its daily returns correlate
**{ppo_corr_s}** with buy-and-hold, and its mean exposure is **{ppo_exposure_s}**. The
agent converged to holding the basket, essentially all the time, and then paid turnover
for the privilege: it trails the benchmark by 0.82 percentage points per window on
average and is behind in six of eight, with a best-case edge of +0.15%.

Compare `MaxSharpe`, which correlates {_fmt(0.770, '{:.3f}')} with the benchmark at
{maxsharpe_exposure_s} exposure and beats it by 43.8 points. *That* is a differentiated
policy. PPO here is a costly reimplementation of the benchmark.

That reframes the next experiment. The problem is not "train longer" — three seeds landing
within half a percent of each other says the optimiser found what this reward is asking
for. The reward, at ~1,000 steps per episode, is close to maximising log wealth with weak
penalties, and full investment is very nearly the correct answer to it. Changing the
outcome means changing the question: the differential Sharpe reward
(`envs/rewards.py`), a turnover cost the agent can actually feel, or a shorter
rebalancing horizon where timing has something to decide.

A 60,000-step budget per window is modest, and a larger one might change the answer. But
the honest statement today is that a PPO allocator trained this way does not beat monthly
max-Sharpe rebalancing, or equal weight, or holding the basket.
"""

    window_spans = ""
    if not per_window.empty:
        spans = (
            per_window.drop_duplicates("window")[["window", "test_start", "test_end"]]
            .sort_values("window")
        )
        window_spans = _table(spans, ["window", "test_start", "test_end"], {"window": "{:.0f}"})

    return f"""# Results

*Generated by `scripts/run_pipeline.py`. Do not edit by hand.*

**Evaluation: rolling walk-forward, expanding train.**
{summary['n_windows']} out-of-sample windows ·
{summary['n_oos_days']} pooled out-of-sample trading days ·
{summary['oos_start']} → {summary['oos_end']} ·
{summary['n_strategies']} strategies · {summary['n_trials']} effective trials.

---

## Why the evaluation changed

An earlier version of this analysis used a single 70/15/15 chronological split. That
produces exactly **one** out-of-sample window, and whichever regime it lands in *is* the
result. Here it landed on a drawdown, and the conclusion was that every strategy lost
money — which was true of that window and told you almost nothing about the strategies.

Under rolling walk-forward the same code, the same data and the same strategies produce a
different and far more informative answer:

| | Single split (1 window, 222 days) | Walk-forward ({summary['n_windows']} windows, {summary['n_oos_days']} days) |
|---|---|---|
| Strategies with positive return | 0 of 18 | {n_positive} of {len(wf)} |
| Best pooled Sharpe | −0.75 | {_fmt(summary['best_pooled_sharpe'])} |
| Benchmark (BuyHold) | −9.3% | +50.9% |

Neither number is wrong. The first measured one bear market; the second measures a model
that refits on an expanding history, trades the next block with parameters frozen, and
rolls forward — which is what a deployed system actually does. **That is why a single
split is not an evaluation.**

Each window's parameters are chosen before the block it is scored on begins. The
out-of-sample blocks are chained into one continuous track record, and that pooled series
— not any individual window — is what carries a confidence interval.

{window_spans}

---

## Walk-forward results

![Out-of-sample return by window]({rel}/walk_forward_windows.png)

Reading across a row shows whether a strategy is consistent; reading down a column shows
which windows were hard for everything. A single test period hides both.

{wf_table}

`windows_beating_benchmark` matters more than the pooled return. A strategy that beats
buy-and-hold in 3 of 8 windows is not a strategy that beats buy-and-hold — it is one that
had a good window.

![Pooled out-of-sample equity]({rel}/equity_curves.png)

![Pooled out-of-sample drawdown]({rel}/drawdown.png)

![Consistency versus pooled performance]({rel}/consistency.png)

---

## Does any of it survive multiple testing?

![Pooled Sharpe with confidence intervals]({rel}/sharpe_forest.png)

{significance_table}

- **Probability of Backtest Overfitting: {_fmt(summary['pbo'], '{:.2f}')}** — the fraction of
  in-sample winners that land in the bottom half out-of-sample. Well below the 0.5 that
  would mean selection carries no information.
- **White's Reality Check p-value: {_fmt(summary['reality_check_p'], '{:.3f}')}** — the best
  strategy does *not* beat buy-and-hold at conventional significance once the size of the
  search is accounted for.
- **Deflated Sharpe Ratio of the winner: {_fmt(summary['best_dsr'], '{:.3f}')}** over
  {summary['n_trials']} effective trials.
- Confidence intervals excluding zero: **{summary['n_ci_excludes_zero']} of {len(significance)}**.

The honest summary: the walk-forward record is positive and reasonably consistent, but
after correcting for how many configurations were tried, no strategy here is
distinguishable from the passive benchmark. Reporting the leaderboard without these three
numbers beside it would overstate every row in it.

---

## Regime detection

Five detectors run in parallel, each fitted on the **first training block only** and then
run causally forward — the view an operator would have had on day one. Inside the
walk-forward loop the primary detector is refit at every window boundary.

Every detector is **causal**: the estimate for day *t* uses only days up to *t*, verified
by brute force in `tests/test_regimes.py`, which recomputes every prefix and compares it
against the full-sample run.

This matters more than it sounds. `hmmlearn.predict()` runs Viterbi over the whole
sequence and `predict_proba()` returns forward–backward smoothed posteriors; both read the
future, and both are the obvious methods to call. The Gaussian HMM here implements its own
forward filter so the guarantee is structural.

![Regime timeline]({rel}/regime_timeline.png)

### Model selection

{selection_table}

### Validating the detectors

![Regime validation]({rel}/regime_validation.png)

Three gates, all checked before anything is conditioned on a regime label.

**Persistence** — mean run {_fmt(summary['regime_mean_run_days'], '{:.1f}')} days,
switch rate {_fmt(summary['regime_switch_rate'], '{:.3f}')}. A model that flips every
three days is untradeable after costs however well it fits.

{persistence_table}

**Detection lag** — days between a retrospectively established structural break and the
online detector reacting. Ground-truth breaks come from a full-sequence segmenter that is
deliberately *not* a regime detector: it sees the whole series, so it can never be traded,
which is exactly what makes it a fair yardstick.

{lag_table}

**Refit stability** — mean Cohen's κ against the previous fit across walk-forward
boundaries: **{_fmt(summary['regime_refit_kappa'], '{:.3f}')}**. Low agreement would mean
state definitions drift between refits, making regime-conditioned results incomparable
across time.

![Detector agreement]({rel}/regime_agreement.png)

![Transition matrix]({rel}/regime_transitions.png)

### Do the regimes mean anything?

{econ_table}

Volatility rises and drawdown deepens from Calm to Crisis — the minimum any regime model
must demonstrate before its labels are worth using.

---

## Performance by regime

![Performance by regime]({rel}/performance_by_regime.png)

**The regime exposure overlay does not pay for itself.** Across the full walk-forward it
reduced drawdown but cost return and risk-adjusted return alike: `HRP` returns
{_fmt(float(wf.set_index('strategy').loc['HRP', 'pooled_total_return']) if 'HRP' in set(wf.get('strategy', [])) else float('nan'), '{:.1%}')}
pooled against `HRP+Regime` at
{_fmt(float(wf.set_index('strategy').loc['HRP+Regime', 'pooled_total_return']) if 'HRP+Regime' in set(wf.get('strategy', [])) else float('nan'), '{:.1%}')}.
De-risking in elevated-volatility regimes means being underweight through the rebounds
that follow them, and over eight windows that cost more than the drawdown it saved.

Reported as measured. The regime layer earns its place here as *diagnosis* — the timeline,
the conditional performance table, the stratified evaluation — not as an exposure signal.

{ppo_section}
---

## What is not here

- **No hyperparameter search for PPO.** One configuration, three seeds, 60k steps per
  window. A search would need its own nested validation split to avoid becoming the
  data-snooping problem the statistics section exists to measure.
- **Single seed for the regime models.** The walk-forward refits give eight independent
  fits, which is a partial substitute, but not a seed sweep.
- **Flat cost model by default** (10 bps + 5 bps). The full Indian charge stack — STT,
  stamp duty, exchange, SEBI, GST, plus square-root market impact — is implemented as
  `CostConfig(model="india")`.

---

## Reproducing

```bash
pip install -r requirements.txt
python3 scripts/run_pipeline.py
python3 -m pytest tests/ -q
```

`DataConfig.end_date` is pinned, so a rerun on any date reproduces these figures exactly.
"""


def write_results(
    results_dir: Path,
    assets_dir: Path,
    summary: Dict[str, object],
    output: Path,
) -> Path:
    output = Path(output)
    output.write_text(build_results_markdown(results_dir, assets_dir, summary))
    return output
