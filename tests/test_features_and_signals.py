"""Feature and signal tests covering bugs #1, #18, #19, #20, #21."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_ticker_frame

from nifty_rl.data import ffill_by_ticker
from nifty_rl.features import CORE_FEATURES, add_features, wilder_rsi
from nifty_rl.strategies.signals import (
    ma_crossover_signals,
    make_signal_fn,
    make_random_signal_fn,
)


# ------------------------------------------ bug #1: warm-up recomputed on the slice


def test_bug_01_signal_uses_full_history_indicators():
    """A split slice must not be forced flat during indicator warm-up.

    The notebook recomputed ``price.rolling(50).mean()`` inside the signal function on
    whatever slice it received, so the first 50 rows collapsed to 0 -- 23% of the test
    split and 42% of every 120-day walk-forward window.
    """
    # Steady uptrend: ma20 sits above ma50 for the whole back half of the series.
    prices = 100.0 * np.exp(np.linspace(0, 0.5, 300))
    panel = make_ticker_frame(prices)
    featured = add_features(panel)

    # Take a 60-bar slice, exactly as a test split or walk-forward window would.
    sliced = featured.tail(60).reset_index(drop=True)

    signal = ma_crossover_signals(sliced, short=20, long=50)

    assert signal.iloc[0] == 1, "first bar of a slice must not be a warm-up casualty"
    assert signal.sum() == 60, "a persistent uptrend should be long on every bar"


def test_bug_01_signal_refuses_to_recompute_windows():
    """Signals must raise rather than silently warm up on a raw frame."""
    raw = make_ticker_frame(np.linspace(100, 200, 80))
    with pytest.raises(KeyError, match="precomputed feature column"):
        ma_crossover_signals(raw, short=20, long=50)


def test_unknown_signal_kind_raises():
    """The notebook's dispatcher returned all-zeros for unregistered names.

    That is how ``VIX_Regime_Momentum`` ran as a silent no-op.
    """
    with pytest.raises(KeyError, match="Unknown signal kind"):
        make_signal_fn("does_not_exist")


# --------------------------------------------------- bug #19: RSI deleting rows


def test_bug_19_rsi_is_defined_on_a_monotonic_uptrend():
    """14 consecutive up-days must yield RSI 100, not NaN.

    The notebook's ``loss.rolling(14).mean().replace(0, np.nan)`` produced NaN, and the
    blanket ``dropna()`` then removed the row -- silently deleting observations exactly
    during the strongest momentum runs.
    """
    price = pd.Series(np.linspace(100.0, 200.0, 60))
    rsi = wilder_rsi(price, 14)

    assert rsi.iloc[20:].notna().all()
    assert rsi.iloc[-1] == pytest.approx(100.0)


def test_bug_19_rsi_is_neutral_on_a_flat_series():
    price = pd.Series([100.0] * 40)
    rsi = wilder_rsi(price, 14)
    assert rsi.iloc[-1] == pytest.approx(50.0)


def test_bug_19_no_rows_lost_to_monotonic_runs():
    """Row count must not depend on whether the series happened to trend."""
    trending = add_features(make_ticker_frame(np.linspace(100, 300, 200)))
    choppy = add_features(
        make_ticker_frame(100 + 10 * np.sin(np.linspace(0, 20, 200)))
    )
    assert len(trending) == len(choppy)


# ------------------------------------------------- bug #20: blanket dropna warm-up


def test_bug_20_warmup_is_gated_on_core_features_only():
    """``ma100``/``momentum_60`` must not cost warm-up when unused.

    The notebook's bare ``dropna()`` gated every row on all 49 columns, so a 2020-01-01
    start silently became 2020-05-29 regardless of the feature set in play.
    """
    prices = 100.0 * np.exp(np.linspace(0, 0.3, 200))
    featured = add_features(make_ticker_frame(prices))

    # Longest core window is ma50 -> ~49 rows of warm-up, not ~100.
    assert len(featured) > 200 - 60
    assert featured[list(CORE_FEATURES)].notna().all().all()
    # The long-window columns still exist; they simply do not gate row removal.
    assert "ma100" in featured.columns
    assert featured["ma100"].isna().any()


def test_bug_20_explicit_required_set_widens_the_gate():
    prices = 100.0 * np.exp(np.linspace(0, 0.3, 200))
    core = add_features(make_ticker_frame(prices))
    wide = add_features(make_ticker_frame(prices), required=list(CORE_FEATURES) + ["ma100"])
    assert len(wide) < len(core)
    assert wide["ma100"].notna().all()


# ------------------------------------------------ bug #18: ffill across tickers


def test_bug_18_ffill_does_not_leak_across_ticker_boundaries():
    """A leading NaN for one ticker must not inherit the previous ticker's value."""
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "ticker": ["A.NS", "A.NS", "B.NS", "B.NS"],
            "india_vix": [15.0, 16.0, np.nan, 22.0],
        }
    )
    filled = ffill_by_ticker(frame, ["india_vix"])

    b_first = filled.loc[filled["ticker"] == "B.NS", "india_vix"].iloc[0]
    assert pd.isna(b_first), "B's leading NaN must stay NaN, not become A's 16.0"


def test_bug_18_ffill_still_fills_within_a_ticker():
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ticker": ["A.NS"] * 3,
            "india_vix": [15.0, np.nan, 17.0],
        }
    )
    filled = ffill_by_ticker(frame, ["india_vix"])
    assert filled["india_vix"].tolist() == [15.0, 15.0, 17.0]


# ------------------------------------------------- bug #21: correlated random baseline


def test_bug_21_random_signals_are_decorrelated_across_tickers():
    """The random control must not trade all ten names in lockstep.

    The notebook seeded every ticker identically, so the "random" portfolio went all-in
    and all-out simultaneously. That concentration -- not randomness -- produced its
    -23.88% print.
    """
    signal_fn = make_random_signal_fn(seed=42)
    frame = make_ticker_frame([100.0] * 200)

    first = signal_fn(frame).to_numpy()
    second = signal_fn(frame).to_numpy()

    assert not np.array_equal(first, second)
    corr = np.corrcoef(first, second)[0, 1]
    assert abs(corr) < 0.25, f"draws should be near-independent, got corr={corr:.3f}"


def test_random_signals_are_reproducible_for_a_given_seed():
    a = make_random_signal_fn(seed=7)(make_ticker_frame([100.0] * 50)).to_numpy()
    b = make_random_signal_fn(seed=7)(make_ticker_frame([100.0] * 50)).to_numpy()
    np.testing.assert_array_equal(a, b)
