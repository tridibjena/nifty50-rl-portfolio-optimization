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
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


class RegimeDetector(ABC):
    """Base class for online (causal) regime detectors.

    Subclasses implement exactly two private methods and inherit everything else:

    * ``_fit(X)`` — estimate parameters from a training matrix. May use the whole block;
      this is parameter estimation on data the model is allowed to see.
    * ``_filter(X)`` — return filtered probabilities where **row t depends only on rows
      0..t**. This is the load-bearing contract of the entire package.

    Splitting them this way is what makes causality checkable. The public methods
    (:meth:`predict_online`, :meth:`label_online`) route only through ``_filter``, so no
    subclass can accidentally expose a smoothed or Viterbi path as if it were live — and
    :func:`assert_causal` can verify any implementation by re-running it on prefixes and
    checking that past labels never change.
    """

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
        """Estimate parameters, remembering which columns were used.

        Pinning ``feature_columns`` on the first fit means a later call with differently
        ordered or extra columns raises instead of silently feeding the model a different
        feature in each slot -- a failure that produces plausible labels and invalid
        results. The row-count guard exists for the same reason: fitting three regimes on
        twenty days succeeds numerically and means nothing.
        """
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
