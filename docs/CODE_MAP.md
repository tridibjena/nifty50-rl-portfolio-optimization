# Code map

How the repository fits together, module by module. Read this before reading the source.

`README.md` says what the project found. `METHODOLOGY.md` says why the methods are the
right ones. This file says **where the code is and what each piece is responsible for** —
it is the one to open when you want to change something and do not know which file to
start in.

---

## The one-paragraph version

Download daily prices for ten NIFTY 50 stocks. Turn them into features. Label each day
with a market regime using only past data. Run a set of strategies — simple rules,
classical portfolio allocators, and a PPO agent — through a realistic backtester that
charges Indian transaction costs. Do all of that inside a **rolling walk-forward loop**,
so every number reported is out-of-sample. Pool the out-of-sample blocks into one track
record, then apply statistics that account for having tried many strategies. Render
figures and a report.

The headline result is that the reinforcement-learning agent does not beat the simple
baselines, and the pipeline is built to make that conclusion trustworthy rather than to
avoid it.

---

## The flow

Each stage below corresponds to one of the `[n/7]` steps `scripts/run_pipeline.py`
prints as it runs, so the console output is itself a map of this list.

```
                      config.py            ← every knob, in frozen dataclasses
                          │
  [1]  data.py ───────────┴──────────► panel: one row per (date, ticker)
        │                                      │
        │  yfinance + on-disk cache            │
        ▼                                      ▼
  [1]  features.py                       regimes/features.py
        │  RSI, ATR, MA, VIX...           │  volatility, trend, breadth, correlation
        ▼                                      ▼
       feature panel                    [2] regimes/{hmm,threshold,jump,changepoint}.py
        │                                      │  → a regime label per day (causal)
        │                                [3] regimes/evaluate.py
        │                                      │  → is the detector stable and usable?
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
         ┌─────────────────────────────┐
         │  three families of strategy │
         ├─────────────────────────────┤
         │  strategies/signals.py      │  rules → 0/1 hold decisions
         │  strategies/allocators.py   │  [4] optimisers → portfolio weights
         │  envs/ + agents/            │  PPO → learned weights
         └─────────────┬───────────────┘
                       ▼
              backtest/engine.py        (0/1 signals, per-ticker cash)
              backtest/weights.py       (weight schedules, monthly rebalance)
              backtest/costs.py         (STT, stamp duty, GST, market impact)
                       │
                       ▼
  [5]  validation/walkforward.py  ← the spine. Everything above runs *inside* this.
                       │            expanding train → frozen test → roll → repeat
                       ▼
              pooled out-of-sample returns, one series per strategy
                       │
  [6]  metrics/performance.py     Sharpe, Sortino, drawdown, Calmar, exposure
       metrics/stats.py           deflated Sharpe, PBO, White's check, bootstrap CIs
                       │
                       ▼
  [7]  report/figures.py + report/build.py + report/narrate.py
                       │
                       ▼
              assets/v2/*.png, results/*.csv, RESULTS.md
```

**The single most important thing to understand:** `validation/walkforward.py` is not a
final evaluation step bolted on at the end. It is the outer loop. Feature scaling, regime
fitting, strategy selection and PPO training all happen *inside* each window, on that
window's training block only. That is what makes the out-of-sample record honest, and it
is the first thing an interviewer should be told.

---

## Module by module

### Foundation

| File | Responsibility | The thing to know |
|---|---|---|
| `config.py` | Every tunable parameter, in frozen dataclasses | `end_date` is **pinned**. The original notebook called `yf.download()` with no end date, so every published metric silently changed on each run. |
| `data.py` | Download, cache, assemble the panel | Caches to parquet (CSV fallback). `auto_adjust=True` keeps OHLC on one price scale — mixing adjusted closes with raw highs injected fake spikes into ATR. |
| `features.py` | Per-ticker technical features | Wilder's smoothing for RSI/ATR, not a simple moving average. Dropna is targeted at `CORE_FEATURES` so one slow-warming column does not delete a year of history. |

### Regimes

| File | Responsibility | The thing to know |
|---|---|---|
| `regimes/base.py` | `RegimeDetector` interface, `assert_causal` | `assert_causal` re-runs a detector on data prefixes and checks past labels never change. Any detector claiming to be causal must pass it. |
| `regimes/features.py` | Market-level inputs to detection | Volatility, trend, breadth, mean correlation — market-wide, not per-stock. |
| `regimes/hmm.py` | Gaussian HMM, Baum-Welch, BIC selection | Written in-package rather than using `hmmlearn` **on purpose**: that library's obvious methods (`predict`, `predict_proba`) both use future data. Here the forward-backward pass exists only inside `_fit`; prediction uses the forward filter alone. |
| `regimes/threshold.py` | Volatility-quantile rules | The transparent baseline. If the HMM cannot beat this, its complexity is not earning anything. |
| `regimes/jump.py`, `changepoint.py` | Jump model, binary segmentation | Alternative backends; agreement between them is evidence the structure is real. |
| `regimes/evaluate.py` | Detection lag, persistence, refit stability | Answers "is this detector usable?" — a regime that flips every three days is untradeable regardless of fit quality. |

### Strategies

| File | Responsibility | The thing to know |
|---|---|---|
| `strategies/signals.py` | Rule-based 0/1 signals | RSI, MA crossover, breakout, and a **random policy** — the control that shows what a coin flip earns in the same window. |
| `strategies/allocators.py` | Classical portfolio construction | Equal weight, inverse vol, min-variance, max-Sharpe, risk parity, HRP. These are the fair opponents for a weight-emitting agent; the notebook only compared it against binary timing rules, which answer a different question. |
| `strategies/meta.py` | Regime-conditioned overlays | Scale exposure by regime, or switch strategy by regime. |

### Reinforcement learning

| File | Responsibility | The thing to know |
|---|---|---|
| `envs/panel.py` | Dense NumPy arrays for the simulator | Pre-flattened so stepping does no pandas work. |
| `envs/core.py` | The portfolio simulation, pure NumPy | Deliberately free of gymnasium so it can be unit tested directly. **Sells settle before buys** — the notebook's interleaved loop meant a sale of the ninth stock could not fund a purchase of the first, so the agent frequently could not execute its own policy. |
| `envs/rewards.py` | Swappable reward functions | Includes the differential Sharpe ratio (Moody & Saffell), which has no hand-tuned penalty weights at all. |
| `envs/multistock.py` | The gymnasium adapter | Thin. All the logic is in `core.py`. |
| `agents/train.py` | PPO training, Sharpe checkpointing, seed ensembling | Checkpoints on a **validation** slice the agent has not trained on. Trains several seeds because a single-seed RL result is an anecdote. |

### Backtesting

| File | Responsibility | The thing to know |
|---|---|---|
| `backtest/costs.py` | Indian charge stack | STT, stamp duty, exchange and SEBI fees, GST, plus square-root market impact. A flat 10 bps materially understates the cost of a high-turnover strategy. |
| `backtest/engine.py` | 0/1 signal backtester | Per-ticker cash buckets. Stop-losses arm a **lockout** so an exit cannot re-enter on the next bar — the notebook's version paid round-trip costs without providing protection. |
| `backtest/weights.py` | Continuous-weight backtester | Two-pass rebalance: sells settle before buys, matching `envs/core.py`. |

### Evaluation

| File | Responsibility | The thing to know |
|---|---|---|
| `validation/walkforward.py` | **The outer loop** | Expanding train → frozen test block → roll forward → refit. Concatenates every out-of-sample block into one track record. Also wraps the NIFTY 50 index as a reference via `reference_result`. |
| `metrics/performance.py` | Descriptive metrics | Sharpe, Sortino, Calmar, Ulcer, drawdown, exposure. A no-dispersion series returns NaN, not an enormous Sharpe. |
| `metrics/stats.py` | Inferential statistics | Deflated Sharpe, PBO, White's Reality Check, stationary bootstrap. This is the file that decides how much of the leaderboard to believe. |

### Reporting

| File | Responsibility | The thing to know |
|---|---|---|
| `report/theme.py` | One matplotlib theme | Colour assigned by *role* — categorical, sequential, diverging, status — never rainbow. |
| `report/figures.py` | Every chart | Each figure writes a CSV twin so any number on a chart can be checked. |
| `report/narrate.py` | Regime episode commentary | **Presentation only.** Default path uses no model at all; `openrouter_provider` adds LLM context when `$OPENROUTER_API_KEY` is set (`--narrate-llm`). `assert_no_narration_leak` runs in CI against the real feature panel. |
| `report/build.py` | Assembles `RESULTS.md` | |

---

## Where to start reading

- **To understand the result:** `validation/walkforward.py`, then `metrics/stats.py`.
- **To understand the RL:** `envs/core.py` (the simulation), then `envs/rewards.py`,
  then `agents/train.py`. The gymnasium wrapper is not worth reading.
- **To understand the regime work:** `regimes/base.py` for the contract, then
  `regimes/hmm.py` for the substance, then `regimes/evaluate.py` for whether it works.
- **To change a parameter:** `config.py`. Nothing is hard-coded elsewhere.
- **To add a strategy:** `strategies/signals.py` if it emits hold/don't-hold,
  `strategies/allocators.py` if it emits weights. Register it in the dict at the bottom
  of the file and the pipeline picks it up.

---

## Glossary

Terms that appear in the code without definition.

**Sharpe ratio** — excess return divided by volatility, annualised. The default measure
of return per unit of risk.

**Sortino ratio** — Sharpe, but only counting downside volatility. Upside swings should
not be penalised as risk.

**Calmar ratio** — annual return divided by maximum drawdown. Return per unit of worst-case
pain.

**Ulcer index** — root-mean-square drawdown. Unlike max drawdown it accounts for how
*long* the portfolio stayed underwater, not just how deep it went.

**Maximum drawdown** — the largest peak-to-trough fall in equity.

**PSR (Probabilistic Sharpe Ratio)** — probability the true Sharpe exceeds a benchmark,
adjusted for skew and fat tails. Daily returns are not normal, and non-normality inflates
the naive Sharpe's apparent precision.

**DSR (Deflated Sharpe Ratio)** — PSR where the benchmark is the Sharpe you would expect
the *best* of N strategies to post by luck alone. Try fifty worthless strategies and the
luckiest still looks good; DSR is the correction for that.

**PBO (Probability of Backtest Overfitting)** — fraction of train/test partitions in which
the in-sample winner lands in the bottom half out-of-sample. Near 0.5 means strategy
selection is pure noise.

**White's Reality Check** — bootstrap test of whether the *best* strategy in a family beats
the benchmark, accounting for having searched the family.

**Stationary bootstrap** — resampling in random-length contiguous blocks rather than
single days, so volatility clustering and autocorrelation survive. IID resampling of daily
returns produces confidence intervals that are far too tight.

**HRP (Hierarchical Risk Parity)** — allocation by clustering assets on correlation, then
recursively splitting weight between sibling clusters by inverse variance. Needs no matrix
inversion, so it stays stable where min-variance concentrates.

**ERC / risk parity** — weights chosen so every asset contributes equal risk.

**Ledoit-Wolf shrinkage** — pulls a noisy sample covariance matrix toward a simple
structured target. With 10 assets and a 60-day window, the raw estimate is badly
conditioned.

**Walk-forward** — fit on the past, trade the next block with parameters frozen, roll
forward, repeat. The evaluation protocol a deployed model actually experiences.

**Causal / filtered vs smoothed** — a *filtered* estimate for day t uses only days up to t.
A *smoothed* estimate uses the whole series. Smoothed regime labels look far better and are
useless for trading, because they encode how the story ended.

**PPO (Proximal Policy Optimisation)** — the reinforcement-learning algorithm used here.
Constrains each policy update so training does not collapse.

**Differential Sharpe ratio** — an online reward whose per-step value is the marginal
contribution to a running Sharpe estimate. Risk aversion falls out of the objective rather
than being added as tuned penalty terms.

**STT** — Securities Transaction Tax, an Indian statutory charge on equity trades.

**ADV** — average daily traded value, used to scale market-impact cost by order size.

**HHI (Herfindahl-Hirschman Index)** — sum of squared weights, a concentration measure.
1.0 is everything in one stock; 1/N is perfectly equal weight.

**Survivorship bias** — selecting today's index members and testing them on the past,
which guarantees the universe consists of companies that did well. This project's universe
has it; see the limitations note in `README.md`.
