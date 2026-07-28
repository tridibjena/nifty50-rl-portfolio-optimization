"""Threshold and quadrant regime detectors.

The transparent control. No latent variables, no EM, no distributional assumption -- cut
points are quantiles of the *training* window and are then held fixed. If a probabilistic
model cannot beat this on out-of-sample economics, the extra machinery is not paying for
itself, and saying so is a result.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import RegimeDetector


class ThresholdRegimes(RegimeDetector):
    """Bucket a single feature by training-window quantiles.

    Cuts are learned once on the training slice and frozen. Using *rolling* quantiles
    instead would also be causal, but refitting the definition of "high volatility" every
    day makes regimes incomparable across time -- and makes the regime-conditional
    performance tables meaningless.
    """

    name = "threshold"

    def __init__(
        self,
        n_regimes: int = 3,
        column: str = "realized_vol_21",
        labels: Optional[Sequence[str]] = None,
    ):
        super().__init__(n_regimes=n_regimes, feature_columns=[column])
        self.column = column
        self.cut_points_: Optional[np.ndarray] = None
        self._labels = list(labels) if labels else None

    def _fit(self, X: np.ndarray) -> None:
        values = X[:, 0]
        quantiles = np.linspace(0.0, 1.0, self.n_regimes + 1)[1:-1]
        self.cut_points_ = np.quantile(values[np.isfinite(values)], quantiles)
        # Share the HMM's naming so the same regime index means the same thing in every
        # figure -- comparing detectors is impossible if one says "Crisis" and another
        # says "Q4" for the same state.
        from .hmm import regime_names

        self.regime_labels_ = list(self._labels) if self._labels else regime_names(self.n_regimes)

    def _filter(self, X: np.ndarray) -> np.ndarray:
        assignments = np.digitize(X[:, 0], self.cut_points_)
        probabilities = np.zeros((len(X), self.n_regimes))
        probabilities[np.arange(len(X)), np.clip(assignments, 0, self.n_regimes - 1)] = 1.0
        return probabilities


class QuadrantRegimes(RegimeDetector):
    """Trend sign crossed with a volatility cut -- four interpretable states.

    This is the taxonomy practitioners actually reason with, and it maps directly onto
    strategy selection: momentum wants Bull-Quiet, mean-reversion wants Bear-Quiet,
    everything wants out of Bear-Volatile.
    """

    name = "quadrant"

    LABELS: List[str] = ["Bull-Quiet", "Bull-Volatile", "Bear-Quiet", "Bear-Volatile"]

    def __init__(
        self,
        trend_column: str = "trend_21",
        vol_column: str = "realized_vol_21",
        vol_quantile: float = 0.5,
    ):
        super().__init__(n_regimes=4, feature_columns=[trend_column, vol_column])
        self.trend_column = trend_column
        self.vol_column = vol_column
        self.vol_quantile = vol_quantile
        self.vol_cut_: float = float("nan")

    def _fit(self, X: np.ndarray) -> None:
        vol = X[:, 1]
        self.vol_cut_ = float(np.quantile(vol[np.isfinite(vol)], self.vol_quantile))
        self.regime_labels_ = list(self.LABELS)

    def _filter(self, X: np.ndarray) -> np.ndarray:
        bearish = (X[:, 0] < 0).astype(int)
        volatile = (X[:, 1] >= self.vol_cut_).astype(int)
        index = bearish * 2 + volatile
        probabilities = np.zeros((len(X), 4))
        probabilities[np.arange(len(X)), index] = 1.0
        return probabilities
