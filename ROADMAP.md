# Roadmap

Plan of record for turning the current single-notebook prototype into a reproducible,
statistically defensible research artifact with regime detection as its spine.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## Why this plan exists

The committed notebook has three classes of problem:

1. **Correctness bugs** that materially change reported numbers (indicator warm-up
   recomputed on split slices, unreachable RL target weights, stop-losses that
   re-enter the next bar, phantom equity holes from `fill_value=0`).
2. **Methodology gaps** — 15k-step ablations whose conclusions are inside run-to-run
   noise yet are cited as design justification, hyperparameters selected under a
   different protocol than deployment, a single seed, and an unpinned `end_date` that
   makes every published figure unreproducible.
3. **A missing comparison** — a continuous *allocator* benchmarked only against binary
   *timing* rules, never against portfolio-construction methods (HRP, risk parity,
   min-variance) that answer the same question.

The test window (2025-06-19 → 2026-05-12) is a drawdown regime in which every strategy
lost money. That is not a failure to hide; it is the most interesting thing in the
project — but only if regimes are detected causally and results are reported per regime
with confidence intervals.

**Goal of this work is credibility, not returns.** No item below is expected to turn the
headline number green.

---

## Phase 0 — Reproducibility foundation `[x]`

Nothing downstream is verifiable until the data stops moving.

- [x] `.gitignore`
- [x] `ROADMAP.md` (this file)
- [x] Pin `Config.end_date = "2026-05-12"` to match the committed run; `--live` flag for fresh data
- [x] Parquet cache (CSV fallback when no engine) at `data/raw/{ticker}__{start}__{end}.parquet`, hash-keyed
- [x] `auto_adjust=True` so OHLC are adjusted together — fixes ATR/TR computed across
      two price scales (bug #6; RELIANCE's Oct-2024 1:1 bonus put a ~50% spike into `atr_pct`)
- [x] Drop `transformers` from requirements (dead FinBERT stub, bug #23)
- [x] Remove the 2.6 MB duplicate `nifty50opt_rl.ipynb` (source-identical to the main
      notebook, differed only in committed outputs; recoverable from git history)
- [ ] `nbstripout` pre-commit hook

**Exit criterion:** two runs on different days produce byte-identical metrics.

---

## Phase 1 — Package extraction + test scaffolding `[x]`

The 44-cell global-state notebook cannot support seed sweeps, five regime backends, or
unit tests.

```
src/nifty_rl/
  config.py          frozen dataclasses + YAML load
  data.py            download, cache, universe
  features.py        technical, macro, regime feature panel
  regimes/           base, threshold, hmm, markov_switching, jump, changepoint, evaluate
  strategies/        signals, allocators, ml, meta
  backtest/          engine, costs, constraints
  envs/              panel, multistock, rewards
  agents/            train, callbacks, evaluate
  metrics/           performance, stats
  validation/        splits, purged_cv, walkforward
  report/            figures, tables, build
conf/                YAML config
tests/
notebooks/           thin narrative, imports from src
```

**Tests written before the fixes they guard:**

```python
def test_roundtrip_pnl_exact()                  # known-answer, to the paisa
def test_stoploss_does_not_reenter_next_bar()   # bug #4
def test_ragged_dates_do_not_create_equity_holes()  # bug #5
def test_signal_uses_full_history_indicators()  # bug #1
def test_regime_is_causal()                     # Phase 5 contract
```

---

## Phase 2 — Tier 1 correctness fixes `[x]` *(non-RL bugs; #2/#3/#7/#24/#25 live in the RL env — Phase 6)*

Each ships with the test that fails before it. **Every published number will move.**

| Bug | Symptom | Fix | Module |
|---|---|---|---|
| #1 | MA/breakout recompute rolling windows on the slice → 50 of 222 test days (23%) and 50 of 120 walk-forward days (42%) forced flat | Consume precomputed `ma20`/`ma50`/`high20`; never `.rolling()` inside a signal fn | `strategies/signals` |
| #2 | Env interleaves sells and buys in one ticker-ordered loop → target weights unreachable, early tickers get cash priority | Two-pass: all sells → recompute cash → all buys, pro-rated | `envs/multistock` |
| #3 | `self.weights = target_w` records intent, not fills → 11 of 172 obs dims systematically wrong | Recompute from realized `holdings × prices / net` | `envs/multistock` |
| #4 | Risk exit sets `cur_sig=0` → `prev_sig=0` → re-buys next bar if signal persists; stop-loss provides costs without protection | Lockout until signal returns to 0 | `backtest/engine` |
| #5 | `agg_equity.add(eq, fill_value=0)` on ragged dates → phantom −10% drops | Reindex to union, ffill, then sum | `backtest/engine` |
| #7 | `env.trades` never appended → PPO win-rate/payoff NaN | Log fills, weights, turnover in `step()` | `envs/multistock` |
| #8 | Aggregate 0–10 position count compared to each ticker's return → metric measures nothing (all strategies 44–47%) | Per-ticker position series | `metrics/performance` |
| #9 | `te` already annualized, then `×√252` again → IR ~15.9× too small | Remove duplicate factor | `metrics/performance` |
| #14 | No `signal.shift(1)` anywhere despite "lookahead-free" claim | `execution: same_close \| next_open` flag, default next-bar | `backtest/engine` |
| #18 | Global `ffill()` on a ticker-sorted frame leaks across ticker boundaries | `groupby("ticker").ffill()` | `data` |
| #19 | `loss.rolling(14).mean().replace(0, nan)` → NaN RSI on 14 up-days → blanket `dropna()` deletes rows exactly during strong momentum | Wilder's smoothing; targeted `dropna(subset=...)` | `features` |
| #20 | `ma100`/`momentum_60` NaNs cost ~100 days of warm-up for columns only `full` uses | Subset-aware dropna | `features` |
| #21 | `seed=CFG.seed` for all 10 tickers → identical signals, all-in/all-out together (why it prints −23.88%) | `seed + ticker_index` | `strategies/signals` |
| #22 | `CL_F`, `USDINR_X`, `TNX`, `CNXIT_ret`, `NSEBANK_ret` downloaded, merged, never used | Drop or wire into a feature set | `data` |
| #24 | `Box(-10,10)` declared; only `nan_to_num` applied, finite outliers unclipped | `np.clip` | `envs/multistock` |
| #25 | `vix_pen` reads the panel after `self.i += 1` → tomorrow's regime | Read before increment | `envs/rewards` |
| #26 | `MA_20_50` duplicated as `Val_MA_20_50`; `select_best_on_valid(vdf, tdf)` ignores `tdf` | Dedupe leaderboard, drop dead param | `strategies/meta` |
| #27 | Cell 32 plots aggregate positions as RELIANCE arrows; cell 35's "allocation stack" is an inline momentum proxy, not PPO weights; cell 34's per-ticker sentiment heatmap is a market-wide scalar (all 10 rows +0.075) | Use real logged weights; relabel or delete | `report/figures` |

---

## Phase 3 — Financial realism `[x]`

- `backtest/costs.py`: STT (0.1% delivery, both sides), stamp duty (0.015% buy),
  exchange transaction charge, SEBI turnover fee, GST on brokerage, plus a
  square-root market-impact term scaled by ADV. Flat 10 bps becomes one selectable model.
- **Cash accrues the 91-day T-bill rate** (bug #10). At ~6.5% this is not a rounding
  error: `RSI_35_60` sits in cash nearly the whole test window and is currently scored
  as if idle capital earned zero, while `BuyHold` is fully invested. Both the 0% cash
  and the rf=0 Sharpe push the same direction — low-exposure strategies are penalized twice.
- Sharpe/Sortino/IR against that rf.
- `backtest/constraints.py`: max weight per name, turnover cap, optional sector cap —
  applied as a projection on the action so RL output is deployable.
- Calmar, Omega, Ulcer index, tail ratio, time-to-recovery.
- Benchmark against actual `^NSEI` (currently every "benchmark excess" is against
  equal-weight BuyHold of the same 10 names).

---

## Phase 4 — Portfolio-construction baselines `[x]`

The missing comparison. A weight-emitting agent must face weight-emitting methods.
All monthly-rebalanced through the same engine, estimated on trailing windows only:

- Equal weight · Inverse volatility · Minimum variance (Ledoit–Wolf shrinkage)
- Max-Sharpe mean-variance, long-only · Risk parity (full ERC)
- **Hierarchical Risk Parity** (López de Prado) — the one that matters
- Momentum-tilted equal weight

If PPO beats HRP net of realistic costs, that is a genuine result. If not, that is an
honest one.

---

## Phase 5 — Regime detection subsystem `[x]`

The centerpiece. Detect, then **validate the detector**, then apply (Phase 7).

### 5.1 Regime feature panel

| Feature | Rationale |
|---|---|
| NIFTY 21d realized vol | Primary regime axis |
| NIFTY 5d realized vol | Fast companion |
| India VIX level + 5d change | Forward-looking fear |
| NIFTY 21d return | Trend axis |
| Cross-sectional dispersion | Already computed; rotation vs coherence |
| **Mean pairwise correlation (21d)** | Correlation spikes are among the strongest crisis markers; free from data already loaded |
| Breadth (% of universe above MA50) | Participation |

### 5.2 The causal contract

```python
class RegimeDetector(ABC):
    n_regimes: int
    def fit(self, X_train) -> "RegimeDetector": ...
    def predict_online(self, X) -> pd.DataFrame:
        """Filtered P(regime_t | information up to and including t).
        Must never read rows > t."""
```

**Filtered, not smoothed, not Viterbi.** This is where regime-switching projects quietly
break: `hmmlearn.predict()` runs Viterbi over the whole sequence and `predict_proba()`
returns forward–backward smoothed posteriors. Both use the future. Only the normalized
forward pass (α_t) is admissible online.

> **Decision (from environment audit):** implement the forward filter directly rather
> than depending on `hmmlearn`. It removes a dependency, works on the installed stack,
> and — more importantly — makes the causality guarantee structural rather than a
> matter of calling the right method.

Enforced by a test that *proves* the claim instead of asserting it:

```python
def test_regime_is_causal(detector, X):
    full = detector.predict_online(X)
    for t in range(60, len(X)):
        prefix = detector.predict_online(X.iloc[:t])
        assert np.allclose(prefix.iloc[-1].values, full.iloc[t-1].values, atol=1e-8)
```

No backend ships without passing it.

### 5.3 Backends

| Backend | Method | Role |
|---|---|---|
| `threshold` | VIX tercile × trend sign → 2×2 quadrant | Transparent control, zero parameters |
| `hmm` | Gaussian HMM, diagonal covariance, 2–4 states, BIC-selected, own forward filter | Academic standard |
| `markov_switching` | Switching-variance regression on NIFTY returns | Reports transition matrix + expected durations |
| `jump` | GMM / statistical jump model with temporal penalty | No distributional assumption; penalty suppresses flapping |
| `changepoint` | PELT on realized vol | Different philosophy — breaks, not states; cross-check on HMM transitions |

All fit on train only; refit on an expanding window at each walk-forward boundary.

### 5.4 Validating the regime models

Routinely skipped; this is the differentiator.

- **Persistence / expected duration** — a model flipping every 3 days is untradeable after costs
- **Detection lag** — days from true break to flag, against hand-labeled events
  (Mar-2020 crash, 2022 chop, 2025–26 drawdown). **This is a kill criterion:** if the
  HMM flags crisis 15 days late, the Phase 7 overlay adds cost and no protection.
  Measure before building on it.
- **Refit stability** — do expanding-window refits keep labels consistent, or does
  state 0 swap meaning?
- **Cross-method agreement** — pairwise Cohen's κ across all five backends
- **Economic sanity** — regime-conditional return, vol, drawdown, correlation

**Deliverable:** regime timeline with all five backends stacked over NIFTY price, plus
an agreement matrix and a detection-lag table.

---

## Phase 6 — RL engineering `[x]`

Requires installing `gymnasium`, `stable-baselines3`, `torch` (all currently absent).

- **Vectorize the env** — replace `dict[date] → DataFrame` with a
  `(n_dates, n_tickers, n_feat)` float32 array and integer indexing. Currently ~2.5M
  pandas `.loc` calls per 250k-step run, which dominates wall clock. Expect 20–50×;
  that is what makes the 2–5M steps METHODOLOGY §8 calls for feasible on CPU.
  Ship with an equivalence test against the current implementation on a fixed action sequence.
- `SubprocVecEnv`, `n_envs=8`
- **Fix the protocol mismatch (#12):** tuning calls `train_ppo_agent` without `val_data`
  so `cb=None` and candidates are scored on the *final iterate*; the final run scores on
  the *best checkpoint*. Select and deploy under the same rule.
- **Raise budgets out of the noise floor (#11):** 15k steps at `n_steps=512` is ~29
  rollout collections. The recorded ablation spread (−0.63% to +0.86%) is inside
  single-run variance — yet METHODOLOGY §3 and the cell-18 comment cite it as the reason
  for the feature set, and they cite *three mutually contradictory* results.
  Also note: the committed run logged exactly one checkpoint improvement
  (step 25,000, val Sharpe 0.087) and never improved across the remaining 225k steps.
- Reward as a strategy object: `ShapedReward` (current), `DifferentialSharpeReward`
  (Moody & Saffell 1998), `RegimeAwareReward` (Phase 7E)
- Reward-coefficient sweep (3×3 over λ_dd × λ_to) replacing "we halved them iteratively"
- Algorithm comparison: PPO / SAC / TD3 / RecurrentPPO — converts METHODOLOGY §2's
  literature argument into evidence
- Log weights, turnover, HHI every step

> **Note:** Python 3.9.6 is the installed interpreter. Confirm torch/SB3 wheel
> availability before committing to this phase, or pin a 3.11 venv.

---

## Phase 7 — Regime × strategy integration `[x]` *(A, B, D, E, G shipped; C/F remain optional extensions)*

Ordered by risk.

- **A. Regime-conditional reporting** *(cheapest, highest value)* — slice every metric by
  regime. Directly tests the project's implicit and currently-unexamined thesis: does a
  drawdown-penalized PPO agent earn its penalty in crisis regimes?
- **B. Regime-switching strategy selection** — learn `regime → best strategy` on train,
  apply out-of-sample via the online estimate. Momentum in trending-low-vol,
  mean-reversion in chop, min-var or cash in crisis.
- **C. Regime as RL observation** — one-hot + filtered probabilities appended to obs.
  Ablate against a PPO that must infer regime from raw VIX.
- **D. Regime-conditional exposure overlay** — scale gross exposure by regime
  (100/60/25%). Applies uniformly to *every* strategy including baselines, isolating the
  regime signal's value independent of any strategy. Highest information per line of code.
- **E. Regime-aware reward** — λ_dd scaled by regime instead of one hardcoded constant.
- **F. Per-regime policies (mixture of experts)** *(highest risk)* — gated on C showing
  real signal. The crisis regime may hold only ~150 training days.
- **G. Regime-stratified purged CV** — the current chronological split trains on the
  2020–24 recovery and tests on a 2025–26 drawdown: an out-of-distribution evaluation
  presented as in-distribution. The contrast between stratified and chronological
  results is itself a finding.

---

## Phase 8 — Statistical rigor `[x]`

- **10-seed sweeps** on every RL result; mean ± std, bootstrap CI on Sharpe
- **Deflated Sharpe Ratio** and **PBO** (Bailey & López de Prado). The search space is
  16 SL/TP cells × 3 PPO configs × 10 validation candidates × 4 feature sets × 5 regime
  backends — DSR is the correct adjustment and pre-empts the first objection any quant
  reader raises.
- **Hansen SPA / White's Reality Check** for multiple-comparison-adjusted selection
- **Stationary block bootstrap** for CIs respecting autocorrelation
- **Purged K-fold with embargo** — `momentum_20` and 60-day rolling windows make
  adjacent train/test rows overlap, inflating validation
- **Factor attribution** — regress returns on NIFTY + size + value + momentum. If alpha
  survives, that's the headline; if not, saying so plainly beats another chart.

---

## Phase 9 — Reporting, docs, CI `[x]`

- `RESULTS.md` **generated from the run**, so docs cannot drift from code again
- README / METHODOLOGY rewritten against actual behavior. Current drift:

  | Doc claim | Code |
  |---|---|
  | `start_date = 2018-01-01` | `2020-01-01` |
  | `initial_cash = 1_000_000` | `100_000` |
  | tuning/ablation `50_000` steps | `30_000` / `15_000` |
  | 162-dim obs, 14 features | 172-dim, 15 features |
  | "budget raised to 500K… ~211-dim" (cell 21) | 250K, 172-dim |
  | `ma_ratio` = Price / MA20 | `ma5/ma20 − 1` |
  | `high_vix_regime` = VIX > 60th pct | 75th pct |
  | "seven rule-based strategies" | 5 rules + Random + LogReg |
  | WF "trains rule-based strategies on the train slice" (METHODOLOGY §7) | params hardcoded; only `ML_LogReg` fits anything |

- Lead with the honest framing: *the test window is a drawdown regime, every strategy
  lost money, here is which techniques limited the loss — per regime, with confidence
  intervals, and with the evidence for whether the differences are distinguishable from
  noise.* That reads as senior work. Unqualified single-seed outperformance does not.
- MLflow tracking · Dockerfile · GitHub Actions on the test suite
- Optional: Streamlit dashboard with regime timeline and strategy switcher

---

## Sequencing

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──┬──► Phase 4 ──┐
                                              │              ├──► Phase 8 ──► Phase 9
                                              ├──► Phase 5 ──┤
                                              └──► Phase 6 ──┴──► Phase 7 ──┘
```

Phases 4, 5, 6 are independent and parallelizable. **~16–22 focused days** total.

### Compressed path (~7–9 days)

Phase 0 → Phase 2 (Tier 1 fixes with tests, no full refactor) → Phase 4 → Phase 5 with
three backends (threshold, HMM, jump) → Phase 7A + 7D → Phase 8 (multi-seed + DSR +
bootstrap) → Phase 9.

Drops: package refactor, algorithm zoo, per-regime policies, factor attribution,
dashboard. Keeps everything that changes how a reviewer judges the work.

---

## Bugs found during implementation

Three defects that did not exist in the original inventory, surfaced by building and
running the pipeline. All three are pinned by tests.

| # | Bug | How it showed up |
|---|---|---|
| 28 | **"BuyHold" benchmark was not passive.** `run_portfolio_backtest` defaults to `stop_loss=0.06 / take_profit=0.12`, and the notebook ran BuyHold through it. Every published `benchmark_excess_return` was measured against a stop-loss strategy mislabelled as passive. | BuyHold returned **+2.60%** with the stops on and **−9.30%** without — an 11.9-point difference in the benchmark itself. |
| 29 | **Degenerate all-cash Sharpe.** A portfolio parked in cash earns exactly the risk-free rate, so excess return *and* excess volatility are zero. `mean/std` on float noise returned **2.1e13**, ranking a do-nothing rule first on every leaderboard and heatmap. | `RSI_35_60` never entered a position in the test window. Guard added in one shared helper; the regime and per-regime tables now import it rather than recomputing. |
| 30 | **Bootstrap CI double-subtracted the risk-free rate.** Pre-adjusted returns were handed to a statistic that also nets out rf, shifting every interval downward. | `MomentumPullback` reported a point estimate of −1.64 with a CI of [−5.84, −2.67] — outside its own interval. After the fix: [−3.02, −0.19]. Correcting it widened most intervals to include zero, which changed the headline conclusion. |

Bug 30 is the reason the reported finding is "nothing is distinguishable from noise"
rather than a ranked leaderboard. It is a good illustration of why the statistics layer
needed its own tests.

---

## Environment landmines

Things that cost real time and are not obvious from any error message.

**`stable-baselines3[extra]` aborts the interpreter on macOS.** The extras pull in
`tensorboard`, and SB3's logger imports `torch.utils.tensorboard` whenever it is present.
On macOS with torch 2.8 / Python 3.9 that import kills the process outright:

```
libc++abi: terminating due to uncaught exception of type std::__1::system_error:
mutex lock failed: Invalid argument
```

It presents as a **hang**, not an error, under any runner that buffers output — the first
three attempts here looked like "PPO is impossibly slow" rather than "the import crashed".
Diagnosis took bisecting the import chain module by module. The raw environment ran at
7,800 steps/sec throughout; nothing was ever slow.

Fix: install plain `stable-baselines3`, not `[extra]`. Neither `tensorboard` nor
`opencv-python` is needed for PPO. Recorded in `requirements.txt` so nobody rediscovers it.

**Two `libomp.dylib` copies** (torch's and scikit-learn's) coexist without incident — that
was the first hypothesis and it was wrong. `KMP_DUPLICATE_LIB_OK=TRUE` changes nothing
here.

---

## Open risks

| Risk | Mitigation |
|---|---|
| Regime overfitting — ~1,550 daily observations, a 3-state HMM on 7 features has many parameters | Start at 2 states, diagonal covariance, BIC-select, refuse any model failing the persistence test |
| Detection lag kills tradeability | Measured in 5.4 as a hard gate *before* Phase 7 is built on top |
| Per-regime policies data-starved | Gated behind 7C |
| Compute — 10 seeds × 4 algorithms × 2M steps | Phase 6 vectorization is a prerequisite for Phase 8, not a nice-to-have |
| Headline number stays negative | Expected. This plan buys credibility, not returns. |
