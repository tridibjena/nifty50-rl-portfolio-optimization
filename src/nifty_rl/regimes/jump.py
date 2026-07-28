"""Statistical jump model -- clustering with a switching penalty.

Nystrup et al.'s jump model adds a penalty on the *number of regime switches* to a
clustering objective, which suppresses the day-to-day flapping that makes plain
clustering untradeable.

**Deviation from the published method, stated plainly.** The original solves the
penalised objective by dynamic programming over the whole sequence, which is not causal
and therefore cannot be traded. The version here fits cluster centres on the training
window and then applies the switching penalty *online*, greedily: at each step the
incumbent regime receives a bonus, so a switch happens only when the evidence overcomes
it. That keeps the anti-flapping property while satisfying the causality contract. It is
a different estimator from the paper's and is labelled as such rather than passed off as
the original.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import RegimeDetector
from .hmm import regime_names


class JumpModelRegimes(RegimeDetector):
    """Online centroid assignment with an explicit persistence bonus.

    Parameters
    ----------
    jump_penalty:
        Bonus, in squared-distance units, awarded to the incumbent regime. Zero reduces
        this to nearest-centroid assignment, which flips constantly; larger values buy
        persistence at the cost of detection lag. That trade-off is exactly what the
        detection-lag evaluation is for.
    """

    name = "jump"

    def __init__(
        self,
        n_regimes: int = 3,
        feature_columns: Optional[Sequence[str]] = None,
        jump_penalty: float = 2.0,
        order_by: int = 0,
        random_state: int = 42,
    ):
        super().__init__(n_regimes=n_regimes, feature_columns=feature_columns)
        self.jump_penalty = float(jump_penalty)
        self.order_by = order_by
        self.random_state = random_state
        self.centres_: Optional[np.ndarray] = None
        self.scales_: Optional[np.ndarray] = None

    def _fit(self, X: np.ndarray) -> None:
        from sklearn.mixture import GaussianMixture

        mixture = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="diag",
            random_state=self.random_state,
            n_init=5,
        ).fit(X)

        centres = mixture.means_
        scales = np.sqrt(np.maximum(mixture.covariances_, 1e-8))

        order = np.argsort(centres[:, self.order_by])
        self.centres_ = centres[order]
        self.scales_ = scales[order]
        self.regime_labels_ = regime_names(self.n_regimes)

    def _filter(self, X: np.ndarray) -> np.ndarray:
        n_obs = len(X)
        probabilities = np.zeros((n_obs, self.n_regimes))

        # Mahalanobis-ish distance under the fitted diagonal scales.
        deviation = (X[:, None, :] - self.centres_[None, :, :]) / self.scales_[None, :, :]
        distances = (deviation ** 2).sum(axis=2)

        previous = int(np.argmin(distances[0]))
        probabilities[0, previous] = 1.0

        for t in range(1, n_obs):
            score = distances[t].copy()
            score[previous] -= self.jump_penalty  # incumbent advantage
            previous = int(np.argmin(score))
            probabilities[t, previous] = 1.0

        return probabilities
