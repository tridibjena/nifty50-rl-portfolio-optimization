"""Configuration objects.

Split into focused frozen dataclasses rather than one monolithic ``Config`` so that
sweeps can vary one concern (costs, RL hyperparameters, regime backend) without
carrying the rest along.

Reproducibility note: ``DataConfig.end_date`` is *pinned*. The original notebook called
``yf.download(start=...)`` with no ``end``, so the dataset — and therefore every split
boundary and every published metric — changed on each run. Pass ``end_date=None``
explicitly to opt into live data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Universe: 10 large-cap NIFTY 50 constituents, unchanged from the original notebook.
DEFAULT_TICKERS: Tuple[str, ...] = (
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "TCS.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "LT.NS",
    "SBIN.NS",
    "HINDUNILVR.NS",
)


@dataclass(frozen=True)
class DataConfig:
    """Data ingestion and caching."""

    tickers: Tuple[str, ...] = DEFAULT_TICKERS
    benchmark_ticker: str = "^NSEI"
    vix_ticker: str = "^INDIAVIX"

    start_date: str = "2020-01-01"
    # Pinned to the committed notebook run so published figures stay reproducible.
    # Set to None for live data (results will then drift with the calendar).
    end_date: Optional[str] = "2026-05-12"

    # auto_adjust=True keeps OHLC on a single adjusted scale. The notebook used
    # auto_adjust=False with price=Adj Close but raw High/Low, so true-range mixed two
    # price scales -- a split or bonus issue injected a spurious spike into atr_pct.
    auto_adjust: bool = True

    cache_dir: Path = field(default_factory=lambda: DATA_DIR / "raw")
    use_cache: bool = True

    # Downloaded-but-unused in the original notebook. Kept configurable but empty by
    # default; wire into a feature set before re-enabling.
    macro_assets: Tuple[str, ...] = ()
    sector_indices: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split on unique trading dates."""

    train_ratio: float = 0.70
    valid_ratio: float = 0.15

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.valid_ratio


@dataclass(frozen=True)
class CostConfig:
    """Transaction cost model.

    ``model="flat"`` reproduces the notebook (10 bps + 5 bps). ``model="india"`` applies
    the actual Indian delivery-equity charge stack; see backtest/costs.py.
    """

    model: str = "flat"  # "flat" | "india"

    # Flat model
    transaction_cost: float = 0.0010
    slippage: float = 0.0005

    # India model (delivery equity, retail)
    brokerage_rate: float = 0.0003
    brokerage_cap: float = 20.0
    stt_buy: float = 0.001
    stt_sell: float = 0.001
    stamp_duty_buy: float = 0.00015
    exchange_txn_rate: float = 0.0000297
    sebi_turnover_rate: float = 0.000001
    gst_rate: float = 0.18

    # Square-root market impact: impact_bps = coef * sqrt(order_value / ADV)
    impact_coef: float = 0.0
    impact_adv_window: int = 20


@dataclass(frozen=True)
class BacktestConfig:
    """Portfolio backtester behaviour."""

    initial_cash: float = 100_000.0
    stop_loss: Optional[float] = 0.06
    take_profit: Optional[float] = 0.12

    # The notebook re-entered on the very next bar after a stop-loss exit whenever the
    # underlying signal was still 1, so stops cost round-trips without providing
    # protection. Require the signal to go flat before re-arming.
    reenter_lockout: bool = True

    # The notebook generated a signal from bar t's close and filled at bar t's close.
    # "next_open" shifts the signal one bar, which is the defensible default for a
    # pipeline that advertises lookahead-free evaluation.
    execution: str = "next_bar"  # "next_bar" | "same_close"

    # Idle cash earned 0% in the notebook. At an Indian risk-free near 6.5% this
    # silently penalised every low-exposure strategy against fully-invested BuyHold.
    cash_rate_annual: float = 0.065
    trading_days: int = 252


@dataclass(frozen=True)
class MetricsConfig:
    risk_free_annual: float = 0.065
    trading_days: int = 252
    var_quantile: float = 0.05


@dataclass(frozen=True)
class RunConfig:
    """Top-level experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    seed: int = 42
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)

    def with_(self, **kwargs) -> "RunConfig":
        """Return a copy with top-level fields replaced (frozen-dataclass friendly)."""
        return replace(self, **kwargs)


DEFAULT_RUN = RunConfig()
