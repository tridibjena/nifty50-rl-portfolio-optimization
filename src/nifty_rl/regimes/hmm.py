"""Gaussian hidden Markov model with a strictly causal forward filter.

Implemented in-package rather than delegated to ``hmmlearn`` for one reason: the
causality guarantee has to be structural. ``hmmlearn.predict()`` runs Viterbi over the
whole sequence and ``predict_proba()`` returns forward-backward smoothed posteriors --
both use future observations, and both are the natural methods to reach for. A regime
series built from either silently invalidates every downstream result.

Here the smoothed quantities exist only inside :meth:`_fit` (Baum-Welch needs them), and
:meth:`_filter` -- the only path used at prediction time -- runs the forward recursion
alone.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import RegimeDetector

_VARIANCE_FLOOR = 1e-6

REGIME_NAMES = {
    2: ["Calm", "Stress"],
    3: ["Calm", "Normal", "Crisis"],
    4: ["Calm", "Normal", "Elevated", "Crisis"],
}


def regime_names(n_regimes: int) -> List[str]:
    if n_regimes in REGIME_NAMES:
        return list(REGIME_NAMES[n_regimes])
    return [f"Regime_{i}" for i in range(n_regimes)]


class GaussianHMMRegimes(RegimeDetector):
    """Diagonal-covariance Gaussian HMM fitted by Baum-Welch.

    Parameters
    ----------
    n_regimes:
        Number of hidden states. Start at 2; ``select_n_regimes`` picks by BIC.
    order_by:
        Index of the feature used to order states after fitting. States are sorted
        ascending, so with realised volatility first, state 0 is the calmest and the
        last state is the most turbulent. Without this, EM returns states in arbitrary
        order and a refit can silently swap what "state 0" means -- the refit-stability
        failure that makes regime results irreproducible.
    """

    name = "hmm"

    def __init__(
        self,
        n_regimes: int = 3,
        feature_columns: Optional[Sequence[str]] = None,
        order_by: int = 0,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        super().__init__(n_regimes=n_regimes, feature_columns=feature_columns)
        self.order_by = order_by
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.startprob_: Optional[np.ndarray] = None
        self.transmat_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.variances_: Optional[np.ndarray] = None
        self.loglikelihood_: float = float("nan")
        self.n_iter_: int = 0

    # ------------------------------------------------------------------ emissions

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """``(n_samples, n_regimes)`` log density under each state."""
        means, variances = self.means_, self.variances_
        # (T, 1, D) - (1, K, D) -> (T, K, D)
        deviation = X[:, None, :] - means[None, :, :]
        log_density = -0.5 * (
            np.log(2.0 * np.pi * variances)[None, :, :] + deviation ** 2 / variances[None, :, :]
        )
        return log_density.sum(axis=2)

    @staticmethod
    def _scaled_emission(log_emission: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Exponentiate row-wise relative to the row max.

        The per-row constant cancels out of the normalised forward/backward quantities,
        so posteriors are unaffected; it is added back to recover the log-likelihood.
        """
        row_max = log_emission.max(axis=1, keepdims=True)
        return np.exp(log_emission - row_max), row_max.ravel()

    # -------------------------------------------------------------------- forward

    def _forward(self, emission: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scaled forward recursion. Returns ``(alpha, scaling)``.

        ``alpha[t]`` is ``P(state_t | x_0..x_t)`` -- the filtered distribution. Rows
        strictly after *t* are never touched, which is the whole point.
        """
        n_obs, n_states = emission.shape
        alpha = np.zeros((n_obs, n_states))
        scaling = np.zeros(n_obs)

        current = self.startprob_ * emission[0]
        total = current.sum()
        scaling[0] = 1.0 / max(total, 1e-300)
        alpha[0] = current * scaling[0]

        for t in range(1, n_obs):
            current = (alpha[t - 1] @ self.transmat_) * emission[t]
            total = current.sum()
            scaling[t] = 1.0 / max(total, 1e-300)
            alpha[t] = current * scaling[t]

        return alpha, scaling

    def _backward(self, emission: np.ndarray, scaling: np.ndarray) -> np.ndarray:
        """Scaled backward recursion -- **fitting only**, never at prediction time."""
        n_obs, n_states = emission.shape
        beta = np.zeros((n_obs, n_states))
        beta[-1] = scaling[-1]
        for t in range(n_obs - 2, -1, -1):
            beta[t] = (self.transmat_ @ (emission[t + 1] * beta[t + 1])) * scaling[t]
        return beta

    # ------------------------------------------------------------------- fitting

    def _initialise(self, X: np.ndarray) -> None:
        from sklearn.mixture import GaussianMixture

        mixture = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="diag",
            random_state=self.random_state,
            n_init=5,
        ).fit(X)

        self.means_ = mixture.means_.copy()
        self.variances_ = np.maximum(mixture.covariances_.copy(), _VARIANCE_FLOOR)

        assignments = mixture.predict(X)
        self.startprob_ = np.full(self.n_regimes, 1.0 / self.n_regimes)

        # Empirical transition counts, Laplace-smoothed so no transition is impossible.
        counts = np.ones((self.n_regimes, self.n_regimes))
        for previous, current in zip(assignments[:-1], assignments[1:]):
            counts[previous, current] += 1.0
        self.transmat_ = counts / counts.sum(axis=1, keepdims=True)

    def _fit(self, X: np.ndarray) -> None:
        self._initialise(X)
        n_obs, n_features = X.shape
        previous_loglik = -np.inf

        for iteration in range(self.max_iter):
            log_emission = self._log_emission(X)
            emission, row_max = self._scaled_emission(log_emission)

            alpha, scaling = self._forward(emission)
            beta = self._backward(emission, scaling)

            loglik = float(-np.log(scaling).sum() + row_max.sum())

            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            xi_sum = np.zeros((self.n_regimes, self.n_regimes))
            for t in range(n_obs - 1):
                step = (
                    np.outer(alpha[t], emission[t + 1] * beta[t + 1]) * self.transmat_
                )
                xi_sum += step / max(step.sum(), 1e-300)

            self.startprob_ = np.maximum(gamma[0], 1e-12)
            self.startprob_ /= self.startprob_.sum()

            self.transmat_ = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)

            weights = np.maximum(gamma.sum(axis=0), 1e-300)
            self.means_ = (gamma.T @ X) / weights[:, None]
            deviation = X[:, None, :] - self.means_[None, :, :]
            self.variances_ = np.maximum(
                (gamma[:, :, None] * deviation ** 2).sum(axis=0) / weights[:, None],
                _VARIANCE_FLOOR,
            )

            self.n_iter_ = iteration + 1
            self.loglikelihood_ = loglik
            if abs(loglik - previous_loglik) < self.tol:
                break
            previous_loglik = loglik

        self._order_states()
        self.regime_labels_ = regime_names(self.n_regimes)

    def _order_states(self) -> None:
        """Sort states by the ordering feature so labels survive a refit."""
        order = np.argsort(self.means_[:, self.order_by])
        self.means_ = self.means_[order]
        self.variances_ = self.variances_[order]
        self.startprob_ = self.startprob_[order]
        self.transmat_ = self.transmat_[np.ix_(order, order)]

    # ----------------------------------------------------------------- prediction

    def _filter(self, X: np.ndarray) -> np.ndarray:
        """Filtered state probabilities. Forward pass only -- no backward, no Viterbi."""
        emission, _ = self._scaled_emission(self._log_emission(X))
        alpha, _ = self._forward(emission)
        return alpha

    # ---------------------------------------------------------------- diagnostics

    @property
    def n_parameters(self) -> int:
        n_states, n_features = self.means_.shape
        return (n_states - 1) + n_states * (n_states - 1) + 2 * n_states * n_features

    def bic(self, n_samples: int) -> float:
        if not np.isfinite(self.loglikelihood_):
            return float("inf")
        return float(-2.0 * self.loglikelihood_ + self.n_parameters * np.log(n_samples))

    def expected_durations(self) -> pd.Series:
        """Expected regime length in days: ``1 / (1 - a_ii)``.

        A model whose regimes last three days is untradeable after costs no matter how
        well it fits. This is the first thing to check after fitting.
        """
        diagonal = np.clip(np.diag(self.transmat_), 0.0, 1.0 - 1e-9)
        return pd.Series(
            1.0 / (1.0 - diagonal),
            index=self.regime_labels_ or [f"regime_{i}" for i in range(self.n_regimes)],
            name="expected_duration_days",
        )

    def transition_frame(self) -> pd.DataFrame:
        labels = self.regime_labels_ or [f"regime_{i}" for i in range(self.n_regimes)]
        return pd.DataFrame(self.transmat_, index=labels, columns=labels)


class MarkovSwitchingVariance(GaussianHMMRegimes):
    """Univariate switching-variance model on market returns.

    The classical Hamilton specification: one observable, states differing chiefly in
    variance. Kept as a separate backend because it is the econometric reference point,
    and because agreement between it and the multivariate HMM is evidence that the
    richer feature set is earning its keep rather than fitting noise.
    """

    name = "markov_switching"

    def __init__(
        self,
        n_regimes: int = 2,
        return_column: str = "trend_21",
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        super().__init__(
            n_regimes=n_regimes,
            feature_columns=[return_column],
            order_by=0,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

    def _order_states(self) -> None:
        """Order by variance rather than mean -- variance is what switches here."""
        order = np.argsort(self.variances_[:, 0])
        self.means_ = self.means_[order]
        self.variances_ = self.variances_[order]
        self.startprob_ = self.startprob_[order]
        self.transmat_ = self.transmat_[np.ix_(order, order)]


def select_n_regimes(
    X: pd.DataFrame,
    candidates: Sequence[int] = (2, 3, 4),
    feature_columns: Optional[Sequence[str]] = None,
    order_by: int = 0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit each candidate state count and score by BIC.

    With roughly 1,500 daily observations, a 4-state model on 8 features carries a lot
    of parameters. BIC is the guard against reading structure into noise -- but it is
    not the only one: a model that wins on BIC and still flips every three days should
    be rejected on persistence grounds.
    """
    rows = []
    for k in candidates:
        model = GaussianHMMRegimes(
            n_regimes=k,
            feature_columns=feature_columns,
            order_by=order_by,
            random_state=random_state,
        )
        try:
            model.fit(X)
        except Exception as exc:  # pragma: no cover - degenerate fits
            rows.append({"n_regimes": k, "loglik": np.nan, "bic": np.inf, "error": str(exc)})
            continue
        durations = model.expected_durations()
        rows.append(
            {
                "n_regimes": k,
                "loglik": model.loglikelihood_,
                "n_parameters": model.n_parameters,
                "bic": model.bic(len(X)),
                "min_expected_duration": float(durations.min()),
                "n_iter": model.n_iter_,
                "error": "",
            }
        )
    return pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)
