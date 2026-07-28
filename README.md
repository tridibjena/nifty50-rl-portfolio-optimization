# Regime-Aware Portfolio Research — NIFTY 50

Ten large-cap Indian stocks, six years of daily prices, and one question: can a
reinforcement-learning agent allocate between them better than a handful of
well-understood classical methods?

The short answer is **no** — and most of the work here goes into making that answer
trustworthy rather than into avoiding it.

![Out-of-sample return by window](assets/v2/walk_forward_windows.png)

**Everything is in one notebook: [`notebooks/00_full_pipeline.ipynb`](notebooks/00_full_pipeline.ipynb).**
Configuration, data handling, features, five regime detectors, the strategies, a realistic
Indian cost model, two backtesters, the RL environment and agent, the statistics and the
reporting. Run it top to bottom and it reproduces every figure below.

Generated numbers live in **[RESULTS.md](RESULTS.md)**.

---

## Quick start

```bash
pip install -r requirements.txt
jupyter lab notebooks/00_full_pipeline.ipynb   # then Run All
```

The last cell takes about twenty minutes, almost all of it training the agent. For a
ninety-second pass that skips only PPO, change it to:

```python
main(parse_args(["--no-rl"]))
```

Output goes to `results/` (tables), `assets/v2/` (charts, each with its numbers alongside)
and `RESULTS.md` (the write-up). The data end date is pinned, so a run next year
reproduces these figures rather than quietly evaluating a different period.

---

## Why a single train/test split is not an evaluation

This is the most important thing in the repository.

An earlier version used a 70/15/15 chronological split. That produces exactly **one**
out-of-sample window, and whichever regime it lands in *is* the result. Here it landed on
a drawdown, and the conclusion was "every strategy lost money" — true of that window, and
nearly uninformative about the strategies.

Same code, same data, same strategies, evaluated as a deployed model would be:

| | Single split (1 window, 222 days) | **Walk-forward (8 windows, 992 days)** |
|---|---|---|
| Strategies with positive return | 0 of 18 | **23 of 24** |
| Best pooled Sharpe | −0.75 | **+0.70** |
| Benchmark (BuyHold) | −9.3% | **+50.8%** |

The walk-forward refits on an expanding history, trades the next 125-day block with
parameters frozen, rolls forward, and chains every out-of-sample block into one continuous
track record. Nothing is chosen using data from the block it is scored on.

---

## Results

Starting from **₹10,00,000**, over 992 out-of-sample trading days (Mar 2022 – Apr 2026):

| Strategy | Final value | Return | Sharpe | Beat benchmark | Invested |
|---|---|---|---|---|---|
| **MaxSharpe** | **₹19,28,146** | **+92.8%** | **0.70** | **88%** | 44% |
| BuyHold *(benchmark)* | ₹15,07,648 | +50.8% | 0.39 | — | 98% |
| EqualWeight | ₹14,95,433 | +49.5% | 0.38 | 50% | 96% |
| HRP | ₹14,80,674 | +48.1% | 0.36 | 38% | 96% |
| **NIFTY 50 index** | ₹13,80,807 | +38.1% | 0.21 | 25% | 100% |
| **PPO_ensemble** | **₹13,62,429** | **+36.2%** | **0.19** | 12% | 99% |
| Random | ₹7,03,202 | −29.7% | −2.02 | 0% | 49% |

`windows_beating_benchmark` matters more than the pooled return: a strategy that beats
buy-and-hold in 3 of 8 windows had a good window, it does not beat buy-and-hold.

### Does any of it survive multiple testing?

![Pooled Sharpe with confidence intervals](assets/v2/sharpe_forest.png)

- **PBO = 0.36** — the in-sample winner lands in the bottom half out-of-sample about a
  third of the time. Under the 0.5 that would mean selection is pure noise, but not by a
  comfortable margin.
- **White's Reality Check p = 0.274** — the best strategy does **not** beat buy-and-hold
  at conventional significance once the size of the search is accounted for.
- **Deflated Sharpe Ratio of the winner = 0.108** over 47 effective trials.
- Only **2 of 24** confidence intervals exclude zero. MaxSharpe's own interval is
  **[−0.29, 1.68]**.

The honest summary: the record is positive and reasonably consistent, but no strategy is
statistically distinguishable from the passive benchmark. Publishing the leaderboard
without those numbers beside it would overstate every row.

---

## The RL agent learned to be buy-and-hold

PPO is retrained from scratch in every window — scaler refit on that window's training
block, an inner validation split for the Sharpe checkpoint, three independent seeds.

![PPO seed dispersion](assets/v2/ppo_seed_dispersion.png)

It finishes at **+36.2%** against buy-and-hold's **+50.8%**, beating the benchmark in one
window out of eight. More pointedly, it also finishes **below simply holding the NIFTY 50
index** (+38.1%) — having traded actively the entire way.

The *diagnosis* is the useful part:

| | Correlation with BuyHold | Mean exposure |
|---|---|---|
| **PPO_ensemble** | **0.993** | **99.2%** |
| EqualWeight | 0.980 | 96.1% |
| MaxSharpe | 0.770 | 44.6% |

The agent converged to holding the basket essentially all the time, then paid turnover for
it. It is a costly reimplementation of the benchmark, not a strategy.

**And that is not seed noise.** Median within-window seed standard deviation is **0.50%**,
and the mean best-to-worst spread within a window is 1.37%.
Three independent runs agreeing that closely means the optimiser found what the reward was
asking for — so the fix is not "train longer", it is to change the question. The reward is
close to maximising log wealth with weak penalties, and full investment is very nearly its
correct answer.

---

## Causal regime detection

Every online detector satisfies one contract:

```
predict_online(X[:t]).iloc[-1] == predict_online(X).iloc[t-1]
```

The label for a given day may use that day and the days before it, and nothing else.

This is where regime work usually breaks silently. `hmmlearn.predict()` runs Viterbi over
the whole sequence; `predict_proba()` returns forward–backward *smoothed* posteriors. Both
read the future, and both are the obvious methods to reach for. The Gaussian HMM here
implements its own forward filter, so the guarantee is structural rather than a matter of
remembering which function to call.

![Regime timeline](assets/v2/regime_timeline.png)

### The models are validated before anything is built on them

![Regime validation](assets/v2/regime_validation.png)

- **Persistence** — mean run 20.6 days. A model that flips every three days is untradeable
  after costs, however well it fits.
- **Detection lag** — days from a retrospectively established break to the online detector
  reacting. Threshold rules react in 2 days but flip constantly; the HMM's transition prior
  buys persistence and pays ~10 days of lag. Measured, not assumed.
- **Refit stability** — mean Cohen's κ = 0.70 against the previous fit across walk-forward
  boundaries. Low agreement would mean "state 0" changes meaning between refits, making
  regime-conditioned results incomparable across time.

Ground-truth breaks come from a full-sequence segmenter that is deliberately **not** a
regime detector — it sees the whole series, so it can never be traded, which is exactly
what makes it a fair yardstick.

### An honest negative result

**The regime exposure overlay does not pay for itself.** Across all eight windows it
reduced drawdown but cost both return and risk-adjusted return: `HRP` +48.1% pooled against
`HRP+Regime` +37.8%. De-risking in elevated-volatility regimes means being underweight
through the rebounds that follow them, and over eight windows that cost more than the
drawdown it saved.

The regime layer earns its place as **diagnosis** — the timeline, the conditional
performance table, the stratified evaluation — not as an exposure signal.

---

## What the notebook contains

Read top to bottom; each section builds on the one above it.

| Section | What it covers |
|---|---|
| Configuration | Every tunable number, in frozen dataclasses. The data end date is pinned. |
| Data and features | Download and cache, then RSI, ATR, moving averages, volatility, a VIX overlay. |
| What a trade actually costs | The full Indian charge stack — STT, stamp duty, exchange and SEBI fees, GST — plus square-root market impact, folded into the fill price. Then two backtesters that charge identically. |
| The strategies | Rules, classical allocators (min-variance, max-Sharpe, risk parity, HRP), regime overlays, and a random control. |
| Regime detection | Five detectors, all strictly causal, plus the diagnostics that decide whether they are usable. |
| The RL agent | A pure-NumPy portfolio simulation, swappable rewards including differential Sharpe, and PPO with Sharpe checkpointing across seeds. |
| Measuring the results | Descriptive metrics, then deflated Sharpe, PBO, White's Reality Check and a stationary bootstrap. |
| Walk-forward evaluation | The outer loop. Everything above runs inside it. |
| Figures and the report | Every chart saves the table behind it. |

---

## Optional: LLM regime commentary

Each detected regime episode gets a factual description computed from its own statistics —
no model, deterministic, always correct. Setting an OpenRouter key additionally attaches
what was actually happening in the market during that episode:

```bash
export OPENROUTER_API_KEY=...
```

then run the final cell with `main(parse_args(["--narrate-llm"]))`.

Strictly presentation. It is generated after the fact, written to its own artefact, and
never enters a feature, an observation or any model input — an LLM's weights encode what
happened *after* the period being described, so using it as an input would be lookahead of
a kind no prefix check can catch. Without a key the pipeline behaves identically.

---

## Layout

```
notebooks/00_full_pipeline.ipynb   the entire project
assets/v2/                         figures (light + dark), each with a CSV twin
results/                           walk-forward tables, significance, regime diagnostics
RESULTS.md                         generated by the final cell
```

A chart whose values are reachable only by reading pixels is not a result, so every figure
ships the table behind it.

The notebook draws all thirteen charts inline as it runs, and repeats them in a gallery at
the end so they are visible without running anything. Commit it with outputs cleared —
`jupyter nbconvert --clear-output --inplace notebooks/*.ipynb` — otherwise the embedded
images take it from 400 KB to about 3 MB and every rerun rewrites all of them.

---

## Limitations

- **Survivorship bias.** The universe is ten stocks that are in the NIFTY 50 *today*,
  tested on the past. That guarantees a universe of companies which did well. Every
  absolute return here is optimistic; the comparisons between strategies are not, since
  all of them trade the same universe.
- **One market, one period.** Six years of Indian large-caps, including a pandemic crash
  and the recovery after it.
- **No live trading.** Costs are modelled carefully but they are still a model. Nothing
  here has been executed against a real order book.
