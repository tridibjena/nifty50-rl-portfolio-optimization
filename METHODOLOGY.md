# Methodology

## 1. Problem Formulation

Portfolio management is framed as a **Markov Decision Process (MDP)** defined by the tuple *(S, A, P, R, γ)*:

| Component | Definition |
|---|---|
| **State** *s_t* | Feature panel across all 10 tickers at time *t*, current portfolio weights, cash ratio |
| **Action** *a_t* | Continuous allocation vector over 10 stocks + cash (softmax-normalised) |
| **Transition** *P* | Deterministic given market data; stochasticity comes from price realisation |
| **Reward** *r_t* | Risk-adjusted log return (see §4) |
| **Discount** *γ* | 0.99 — long enough horizon to capture multi-day position consequences |

The action space is `Box(0, 1, shape=(11,))` — one weight per ticker plus cash. A softmax transform at each step ensures weights sum to 1 and are non-negative, avoiding the need for a constrained optimiser.

---

## 2. Why PPO

Three candidate algorithms were considered:

**DQN** — requires discretising the action space. With 10 stocks at even 5 weight levels each, the joint action space has 5¹⁰ ≈ 10M combinations. Intractable without further approximation.

**SAC (Soft Actor-Critic)** — well-suited to continuous action spaces and sample-efficient due to off-policy learning. Requires careful tuning of the entropy temperature α; in practice SAC tends to overfit to recent market regimes without a large replay buffer.

**PPO** — on-policy, stable under the clipped surrogate objective, and robust to hyperparameter choice. The clipping parameter ε prevents destructively large policy updates — important when reward variance is high (daily log returns have fat tails). PPO is the de facto standard for continuous-action finance RL in the literature (e.g. FinRL benchmark, 2021).

---

## 3. Observation Space

The observation vector at time *t* is constructed as:

```
obs = [
  for each ticker k in 1..10:
    [f_1, f_2, ..., f_14, missing_flag],   # 15 values × 10 tickers = 150
  current_weights,                          # 11 values (10 stocks + cash)
  cash_ratio,                               # 1 value
]
# Total: 162-dimensional vector
```

**Missing-price flag:** If a ticker has no data on date *t* (e.g. trading halt, index reconstitution gap), the feature vector is zeroed and `missing_flag = 1`. This allows the agent to learn to ignore stale tickers rather than acting on stale or forward-filled feature values — a common silent bug in multi-asset RL environments.

**Scaling:** All features are standardised (zero mean, unit variance) using a `StandardScaler` fitted exclusively on the training split. The same scaler is applied to validation and test splits without refitting — enforcing strict lookahead-free evaluation.

### Feature Set (`technical_vix`, 14 features)

Selected by ablation study over four candidate sets. `technical_vix` outperformed `full` (22 features) on the validation split, consistent with the curse of dimensionality at a 250K-step training budget.

| Category | Features |
|---|---|
| Returns | `ret`, `momentum_5`, `momentum_20` |
| Trend | `ma_ratio`, `trend_20_50` |
| Momentum/oscillator | `rsi`, `macd_hist` |
| Volatility | `bb_position`, `bb_width`, `atr_pct` |
| Volume | `volume_change` |
| Macro | `india_vix`, `high_vix_regime`, `dispersion_zscore` |

---

## 4. Reward Design

The per-step reward balances return maximisation against four risk penalties:

```
r_t = log(V_t / V_{t-1})
    − λ_dd  × |drawdown_t|
    − λ_dv  × σ_downside
    − λ_to  × n_trades_t
    − λ_vix × 𝟙[high_vix_regime ∧ holding]
```

| Term | Coefficient | Rationale |
|---|---|---|
| Log return | — | Log returns are additive across time; numerically stable for cumulative reward |
| Drawdown penalty | λ_dd = 0.04 | Penalises peak-to-trough loss; encourages capital preservation |
| Downside volatility | λ_dv = 0.02 | Penalises only negative return realizations (Sortino-style); upside vol is not penalised |
| Turnover | λ_to = 0.0002 | Per-trade penalty; discourages excessive rebalancing that erodes returns via TC |
| VIX regime | λ_vix = 0.0002 | Penalises holding equities when India VIX is elevated; encourages de-risking in fear regimes |

**Coefficient calibration:** Initial coefficients (v2) caused the agent to hold 100% cash throughout training — the penalty signal dominated the return signal. Coefficients were halved iteratively until the agent learned non-trivial policies while still exhibiting meaningful risk control. The downside volatility term uses a 20-step rolling window of negative returns only, consistent with the Sortino ratio definition.

---

## 5. Training Protocol

### Hyperparameter Search

Three PPO configurations are evaluated on the validation split using a composite score:

```
score = Sharpe_val + total_return_val − |max_drawdown_val|
```

This penalises drawdown explicitly so the search does not select a high-return but highly volatile policy.

| Candidate | LR | n_steps | batch | ent_coef | clip |
|---|---|---|---|---|---|
| 1 | 3e-4 | 512 | 64 | 0.00 | 0.20 |
| 2 | 1e-4 | 512 | 128 | 0.01 | 0.20 |
| 3 | 5e-5 | 512 | 128 | 0.01 | 0.15 |

### SharpeCheckpointCallback

PPO policies can peak mid-training and then regress due to policy gradient variance and non-stationarity in the reward landscape. A custom `SharpeCheckpointCallback` (Stable-Baselines3 `BaseCallback`) evaluates validation Sharpe every 25K steps, saves policy weights on any improvement, and restores the best checkpoint after `learn()` completes. The final deployed model is therefore the best policy seen during training, not the final iterate.

### Data Splits

```
Full dataset: 2018-01-01 → present

Train  │████████████████████│  ~60%  (strategy selection + PPO training)
Valid  │                    ████│  ~20%  (hyperparameter + strategy selection)
Test   │                        ████│  ~20%  (final evaluation, untouched)
```

Split is performed on **unique trading dates** (not calendar days) to ensure each split contains the same number of observations for all tickers regardless of trading halts.

---

## 6. Baseline Strategies

Seven rule-based strategies and one supervised ML baseline provide comparison points:

| Strategy | Logic |
|---|---|
| **BuyHold** | Always invested; benchmark floor |
| **MA_20_50** | Long when 20-day MA > 50-day MA |
| **RSI_35_60** | Long when RSI < 35 (oversold); exit when RSI > 60 |
| **Breakout_20** | Long on 20-day high breakout; exit on 10-day low |
| **MomentumPullback** | Long when uptrend (MA20 > MA50) + RSI pullback (35–55) |
| **Sentiment_Momentum** | Long when VIX-proxy sentiment improving + price > MA20 |
| **VIX_Regime_Momentum** | Long only when VIX < 40th percentile rolling quantile + uptrend |
| **ML_LogReg** | Logistic regression on 10 features; trained on train split per walk-forward window |

All strategies are evaluated through the same `run_portfolio_backtest` engine: equal capital per ticker, identical transaction cost (10 bps) and slippage (5 bps), stop-loss at 6%, take-profit at 12%. This makes the PPO vs. rule-based comparison fully controlled.

Rule-based hyperparameters (MA windows, RSI thresholds, breakout lookbacks) are selected on the validation split via a grid search scored by `Sharpe + total_return − |max_drawdown|`. Selected parameters are then frozen and applied to the untouched test split — a two-stage procedure that prevents in-sample optimism.

---

## 7. Evaluation Framework

### Test-split metrics

| Metric | Formula |
|---|---|
| Sharpe | `E[r] / σ(r) × √252` |
| Sortino | `E[r] / σ_downside(r) × √252` |
| CAGR | `(V_T / V_0)^(252/T) − 1` |
| Max drawdown | `min_t (V_t / max_{s≤t} V_s − 1)` |
| Information ratio | `(r_strategy − r_benchmark) / σ(r_strategy − r_benchmark) × √252` |
| VaR (95%) | 5th percentile of daily return distribution |
| CVaR (95%) | Mean of returns below VaR threshold |
| Signal accuracy | Per-ticker: fraction of invested days where price rose; equal-weighted average |

### Walk-forward validation

The full dataset is sliced into overlapping windows (500 train / 120 test / 120 step, measured in unique trading dates). Each window:

1. Trains rule-based strategies on the train slice
2. Backtests all strategies on the test slice via the shared portfolio engine
3. Computes performance metrics against a **per-window buy-hold baseline**

Using a per-window buy-hold (rather than a global one) controls for regime difficulty — a 20% PPO return in a window where buy-hold returns 22% is worse than a 12% return in a window where buy-hold returns 5%.

### Feature ablation

Four feature subsets are trained and evaluated on the validation split to quantify each feature group's marginal contribution:

| Subset | Features | Obs dim |
|---|---|---|
| `technical_only` | 11 technical indicators | ~122 |
| `technical_vix` | technical + VIX/macro (14 total) | ~162 |
| `technical_sentiment` | technical + sentiment proxies | ~162 |
| `full` | all features (19 total) | ~212 |

`technical_vix` was selected as the default after ablation confirmed that adding sentiment features over `technical_vix` provided no validation improvement, and `full` underperformed due to the curse of dimensionality at 250K steps.

---

## 8. Known Limitations

**Training budget.** 250K steps with a 162-dim obs space gives roughly 1500 policy gradient updates (at n_steps=512, batch_size=64). This is sufficient for a proof-of-concept but well below what a production RL system would require. Scaling to 2–5M steps would require GPU acceleration or a distributed training setup.

**Sentiment coverage.** VIX-proxy sentiment is a market-wide signal shared across all tickers. It captures systemic fear but not stock-specific news tone. Ticker-specific sentiment from a historical provider (GDELT, NewsAPI, RavenPack) would add genuine cross-sectional signal.

**Transaction cost model.** A fixed 10 bps + 5 bps slippage model is applied uniformly. Real Indian equity markets have SEBI turnover charges, STT, stamp duty, and bid-ask spreads that are order-size dependent. The current model understates costs for large rebalancing actions.

**Single random seed.** All results are reported for `seed=42`. A single seed means results may not be representative of the algorithm's average behaviour. A production evaluation would average over 5–10 seeds.

**Regime non-stationarity.** The PPO policy is trained on a single historical regime and evaluated on a held-out slice of the same regime. Out-of-distribution generalisation (e.g. a policy trained pre-2020 evaluated post-COVID) is not tested.
