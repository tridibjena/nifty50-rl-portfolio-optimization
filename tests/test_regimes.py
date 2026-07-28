"""Regime subsystem tests.

The headline test is :func:`test_all_online_detectors_are_causal`. It does not assert
that the code *intends* to be causal -- it recomputes every prefix and checks the
filtered distribution matches, which is the only way to actually know.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_rl.regimes import (
    BinarySegmentation,
    GaussianHMMRegimes,
    JumpModelRegimes,
    MarkovSwitchingVariance,
    QuadrantRegimes,
    ThresholdRegimes,
    agreement_matrix,
    assert_causal,
    detection_lag,
    persistence_summary,
    regime_conditional_stats,
    run_lengths,
    select_n_regimes,
)


@pytest.fixture(scope="module")
def two_regime_series():
    """400 calm days then 200 turbulent days, with a known break at index 400."""
    rng = np.random.default_rng(0)
    calm = rng.normal(0.0008, 0.006, 400)
    crisis = rng.normal(-0.0015, 0.025, 200)
    returns = np.concatenate([calm, crisis])
    index = pd.bdate_range("2021-01-01", periods=len(returns))

    frame = pd.DataFrame(index=index)
    frame["ret"] = returns
    frame["realized_vol_21"] = pd.Series(returns, index=index).rolling(21).std() * np.sqrt(252)
    frame["trend_21"] = pd.Series(returns, index=index).rolling(21).mean() * 21
    frame["mean_correlation"] = np.where(np.arange(len(returns)) < 400, 0.3, 0.75)
    return frame.dropna()


DETECTOR_FACTORIES = [
    pytest.param(lambda: ThresholdRegimes(n_regimes=3, column="realized_vol_21"), id="threshold"),
    pytest.param(lambda: QuadrantRegimes(), id="quadrant"),
    pytest.param(
        lambda: GaussianHMMRegimes(
            n_regimes=2, feature_columns=["realized_vol_21", "trend_21", "mean_correlation"]
        ),
        id="hmm",
    ),
    pytest.param(lambda: MarkovSwitchingVariance(n_regimes=2, return_column="ret"), id="markov"),
    pytest.param(
        lambda: JumpModelRegimes(
            n_regimes=2, feature_columns=["realized_vol_21", "mean_correlation"]
        ),
        id="jump",
    ),
]


# ------------------------------------------------------------- the causal contract


@pytest.mark.parametrize("factory", DETECTOR_FACTORIES)
def test_all_online_detectors_are_causal(factory, two_regime_series):
    """P(regime_t) must not change when future rows are appended.

    This is the property that makes every regime-conditioned result believable. It is
    checked by brute force -- recompute each prefix, compare against the full-sample run.
    """
    detector = factory().fit(two_regime_series.iloc[:300])
    assert_causal(detector, two_regime_series, start=60, step=17)


@pytest.mark.parametrize("factory", DETECTOR_FACTORIES)
def test_probabilities_are_a_distribution(factory, two_regime_series):
    detector = factory().fit(two_regime_series.iloc[:300])
    probabilities = detector.predict_online(two_regime_series)
    assert probabilities.shape[0] == len(two_regime_series)
    np.testing.assert_allclose(probabilities.to_numpy().sum(axis=1), 1.0, atol=1e-8)
    assert (probabilities.to_numpy() >= -1e-12).all()


def test_hmm_smoothed_posterior_would_violate_causality(two_regime_series):
    """Documents *why* the forward filter is used rather than forward-backward.

    Smoothed posteriors are what ``hmmlearn.predict_proba`` returns. They read the whole
    sequence, so the estimate for day t changes when later data arrives. Asserting the
    difference keeps the distinction from being quietly optimised away.
    """
    features = ["realized_vol_21", "trend_21", "mean_correlation"]
    model = GaussianHMMRegimes(n_regimes=2, feature_columns=features).fit(
        two_regime_series.iloc[:300]
    )

    matrix = model._matrix(two_regime_series)
    emission, _ = model._scaled_emission(model._log_emission(matrix))
    alpha, scaling = model._forward(emission)
    beta = model._backward(emission, scaling)

    smoothed = alpha * beta
    smoothed /= smoothed.sum(axis=1, keepdims=True)

    # The two agree at the final observation (nothing follows it) but not before.
    np.testing.assert_allclose(alpha[-1], smoothed[-1], atol=1e-6)
    midpoint = len(matrix) // 2
    assert not np.allclose(alpha[midpoint], smoothed[midpoint], atol=1e-3)


# ----------------------------------------------------------------- economic sanity


def test_hmm_separates_calm_from_crisis(two_regime_series):
    features = ["realized_vol_21", "trend_21", "mean_correlation"]
    model = GaussianHMMRegimes(n_regimes=2, feature_columns=features).fit(two_regime_series)
    labels = model.label_online(two_regime_series)

    stats = regime_conditional_stats(
        two_regime_series["ret"], labels, regime_names=model.regime_labels_
    )
    calm = stats[stats["regime"] == "Calm"].iloc[0]
    stress = stats[stats["regime"] == "Stress"].iloc[0]

    assert stress["volatility_annual"] > calm["volatility_annual"]
    assert stress["max_drawdown"] < calm["max_drawdown"]


def test_states_are_ordered_by_the_ordering_feature(two_regime_series):
    """State 0 must always be the calmest, so a refit cannot swap label meanings."""
    features = ["realized_vol_21", "trend_21", "mean_correlation"]
    model = GaussianHMMRegimes(n_regimes=2, feature_columns=features, order_by=0).fit(
        two_regime_series
    )
    assert model.means_[0, 0] < model.means_[1, 0]


# -------------------------------------------------------------------- persistence


def test_jump_penalty_increases_persistence(two_regime_series):
    """The switching penalty must actually suppress flapping."""
    features = ["realized_vol_21", "mean_correlation"]
    flappy = JumpModelRegimes(n_regimes=2, feature_columns=features, jump_penalty=0.0).fit(
        two_regime_series
    )
    sticky = JumpModelRegimes(n_regimes=2, feature_columns=features, jump_penalty=5.0).fit(
        two_regime_series
    )

    flappy_runs = run_lengths(flappy.label_online(two_regime_series)).mean()
    sticky_runs = run_lengths(sticky.label_online(two_regime_series)).mean()
    assert sticky_runs >= flappy_runs


def test_persistence_summary_reports_switch_rate(two_regime_series):
    model = ThresholdRegimes(n_regimes=2, column="realized_vol_21").fit(two_regime_series)
    summary = persistence_summary(
        model.label_online(two_regime_series), regime_names=model.regime_labels_
    )
    assert set(summary["regime"]) == {"Calm", "Stress"}
    assert summary["occupancy"].sum() == pytest.approx(1.0)
    assert "switch_rate" in summary.attrs


# ----------------------------------------------------------------- detection lag


def test_detection_lag_is_measured_against_retrospective_breaks(two_regime_series):
    """The segmenter finds the break with hindsight; the detector is timed against it."""
    segmenter = BinarySegmentation(min_size=40, max_breaks=4)
    breaks = segmenter.breakpoints(two_regime_series["realized_vol_21"])
    assert breaks, "segmentation should find at least one structural break"

    model = ThresholdRegimes(n_regimes=2, column="realized_vol_21").fit(
        two_regime_series.iloc[:250]
    )
    lags = detection_lag(model.label_online(two_regime_series), breaks)

    assert len(lags) == len(breaks)
    assert set(lags.columns) == {"break_date", "lag_days", "detected"}


def test_segmenter_is_not_a_regime_detector():
    """The retrospective/online split must stay a type-level distinction."""
    from nifty_rl.regimes.base import RegimeDetector

    assert not isinstance(BinarySegmentation(), RegimeDetector)
    assert not hasattr(BinarySegmentation(), "predict_online")


# -------------------------------------------------------------------- agreement


def test_agreement_matrix_is_symmetric_with_unit_diagonal(two_regime_series):
    train = two_regime_series.iloc[:300]
    labels = {
        "Threshold": ThresholdRegimes(n_regimes=2, column="realized_vol_21")
        .fit(train)
        .label_online(two_regime_series),
        "Jump": JumpModelRegimes(n_regimes=2, feature_columns=["realized_vol_21", "mean_correlation"])
        .fit(train)
        .label_online(two_regime_series),
    }
    matrix = agreement_matrix(labels)
    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(np.diag(matrix.to_numpy()), 1.0)
    assert matrix.loc["Threshold", "Jump"] == pytest.approx(matrix.loc["Jump", "Threshold"])


# -------------------------------------------------------------- model selection


def test_bic_selection_returns_ranked_candidates(two_regime_series):
    table = select_n_regimes(
        two_regime_series,
        candidates=(2, 3),
        feature_columns=["realized_vol_21", "trend_21", "mean_correlation"],
    )
    assert set(table["n_regimes"]) == {2, 3}
    assert table["bic"].is_monotonic_increasing, "candidates must come back best-BIC-first"
    assert table["bic"].notna().all()
    # Persistence is reported alongside fit quality: a model can win on BIC and still be
    # untradeable, so both numbers have to be on the table when choosing.
    assert "min_expected_duration" in table.columns
    assert (table["min_expected_duration"] > 1.0).all()


def test_fit_rejects_insufficient_data():
    tiny = pd.DataFrame({"realized_vol_21": np.arange(10.0)})
    with pytest.raises(ValueError, match="too few"):
        GaussianHMMRegimes(n_regimes=3, feature_columns=["realized_vol_21"]).fit(tiny)
