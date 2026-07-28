"""Regime detection.

Four **online** (causal) detectors and one **retrospective** segmenter. The separation is
load-bearing: only the online detectors may inform a trading decision; the segmenter
supplies ground-truth breaks for measuring how late each online detector is.
"""

from .base import RegimeDetector, RetrospectiveSegmenter, assert_causal
from .changepoint import BinarySegmentation
from .evaluate import (
    agreement_matrix,
    detection_lag,
    is_tradeable,
    lag_summary,
    persistence_summary,
    refit_stability,
    regime_conditional_stats,
    run_lengths,
)
from .features import REGIME_FEATURES, build_regime_features, standardise_causally
from .hmm import GaussianHMMRegimes, MarkovSwitchingVariance, regime_names, select_n_regimes
from .jump import JumpModelRegimes
from .threshold import QuadrantRegimes, ThresholdRegimes

#: Backends used by the standard report. Keys are display names.
ONLINE_DETECTORS = {
    "Threshold": ThresholdRegimes,
    "Quadrant": QuadrantRegimes,
    "HMM": GaussianHMMRegimes,
    "MarkovSwitch": MarkovSwitchingVariance,
    "JumpModel": JumpModelRegimes,
}

__all__ = [
    "RegimeDetector",
    "RetrospectiveSegmenter",
    "assert_causal",
    "BinarySegmentation",
    "ThresholdRegimes",
    "QuadrantRegimes",
    "GaussianHMMRegimes",
    "MarkovSwitchingVariance",
    "JumpModelRegimes",
    "ONLINE_DETECTORS",
    "REGIME_FEATURES",
    "build_regime_features",
    "standardise_causally",
    "regime_names",
    "select_n_regimes",
    "agreement_matrix",
    "detection_lag",
    "lag_summary",
    "persistence_summary",
    "refit_stability",
    "regime_conditional_stats",
    "run_lengths",
    "is_tradeable",
]
