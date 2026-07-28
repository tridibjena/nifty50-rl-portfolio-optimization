"""Regime detector interface and the causality contract.

Every online detector must satisfy one property::

    predict_online(X[:t]).iloc[-1] == predict_online(X).iloc[t-1]

That is, the estimate for day *t* depends only on days up to and including *t*. It is
the entire basis on which any regime-conditioned result can be believed.

This is where regime-switching projects quietly break. ``hmmlearn.predict()`` runs
Viterbi over the whole sequence; ``predict_proba()`` returns forward-backward *smoothed*
posteriors. Both read the future. Only the normalised forward pass (alpha_t) is
admissible online, which is why the Gaussian HMM here is implemented in-package rather
than delegated -- the guarantee becomes structural instead of a matter of remembering
which method to call.

Retrospective detectors (change-point segmentation) deliberately do **not** implement
``predict_online``. They are diagnostics: used to hand-label ground-truth breaks against
which the online detectors' *detection lag* is measured. Mixing the two is the mistake.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class RegimeDetector(ABC):
    """Base class for online (causal) regime detectors."""

    #: Human-readable name used in reports and figures.
    name: str = "base"

    def __init__(self, n_regimes: int = 3, feature_columns: Optional[Sequence[str]] = None):
        self.n_regimes = int(n_regimes)
        self.feature_columns = list(feature_columns) if feature_columns else None
        self._fitted = False
        self.regime_labels_: List[str] = []

    # ------------------------------------------------------------------ interface

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        """Estimate parameters from the training matrix."""

    @abstractmethod
    def _filter(self, X: np.ndarray) -> np.ndarray:
        """Return the ``(n_samples, n_regimes)`` filtered probability matrix.

        Row *t* must be computable from rows ``0..t`` alone.
        """

    # ------------------------------------------------------------------- plumbing

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_columns or list(X.columns)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise KeyError(f"{self.name}: missing regime feature column(s) {missing}")
        values = X[cols].to_numpy(dtype=float)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X: pd.DataFrame) -> "RegimeDetector":
        if self.feature_columns is None:
            self.feature_columns = list(X.columns)
        matrix = self._matrix(X)
        if len(matrix) < self.n_regimes * 10:
            raise ValueError(
                f"{self.name}: {len(matrix)} training rows is too few for "
                f"{self.n_regimes} regimes."
            )
        self._fit(matrix)
        self._fitted = True
        return self

    def predict_online(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filtered ``P(regime_t | information up to t)``, one column per regime."""
        if not self._fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict_online().")
        probabilities = self._filter(self._matrix(X))
        columns = self.regime_labels_ or [f"regime_{i}" for i in range(self.n_regimes)]
        return pd.DataFrame(probabilities, index=X.index, columns=columns)

    def label_online(self, X: pd.DataFrame) -> pd.Series:
        """Hard regime label (argmax of the filtered distribution)."""
        probabilities = self.predict_online(X)
        return pd.Series(
            np.argmax(probabilities.to_numpy(), axis=1), index=X.index, name="regime"
        )

    # ---------------------------------------------------------------- diagnostics

    def state_ordering_key(self, X: pd.DataFrame, column: str) -> List[int]:
        """Order states by their mean value of ``column`` -- used to stabilise labels."""
        labels = self.label_online(X)
        means = pd.Series(X[column].to_numpy(), index=X.index).groupby(labels).mean()
        return list(means.sort_values().index)


class RetrospectiveSegmenter(ABC):
    """Full-sequence structural-break detector.

    Explicitly *not* a :class:`RegimeDetector`: it sees the whole series and therefore
    cannot be traded. Its role is to produce ground-truth break dates for the
    detection-lag evaluation.
    """

    name: str = "segmenter"

    @abstractmethod
    def breakpoints(self, series: pd.Series) -> List[pd.Timestamp]:
        """Return the dates at which the series changes regime."""


def assert_causal(
    detector: RegimeDetector,
    X: pd.DataFrame,
    start: int = 60,
    step: int = 1,
    atol: float = 1e-8,
) -> None:
    """Verify the causality contract by brute force.

    Recomputes the filtered distribution on every prefix and checks that the last row
    matches the corresponding row of the full-sample run. Raises ``AssertionError`` on
    the first violation. Deliberately slow -- it is a proof, not a fast path.
    """
    full = detector.predict_online(X).to_numpy()
    for t in range(start, len(X) + 1, step):
        prefix = detector.predict_online(X.iloc[:t]).to_numpy()
        if not np.allclose(prefix[-1], full[t - 1], atol=atol):
            raise AssertionError(
                f"{detector.name} is not causal at t={t}: "
                f"prefix={prefix[-1]} vs full={full[t - 1]}"
            )
