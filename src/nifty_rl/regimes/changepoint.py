"""Retrospective change-point segmentation.

Deliberately **not** a :class:`~nifty_rl.regimes.base.RegimeDetector`. Binary
segmentation sees the entire series, so its breaks cannot be traded -- but that is
precisely what makes them useful as *ground truth*: the dates at which the process
demonstrably changed, established with hindsight, against which each online detector's
**detection lag** is measured.

Treating a retrospective segmentation as a tradeable signal is the single most common
way regime work becomes accidental lookahead. Keeping the two in separate classes makes
the mistake hard to make.

Implemented directly rather than via ``ruptures`` so the core evaluation has no optional
dependency; ``ruptures`` remains available for PELT if a larger search is wanted.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .base import RetrospectiveSegmenter


def _gaussian_cost(values: np.ndarray) -> float:
    """Negative log-likelihood of a segment under a Gaussian with free mean/variance."""
    n = len(values)
    if n < 2:
        return 0.0
    variance = float(np.var(values))
    if variance <= 0:
        return 0.0
    return float(n * np.log(variance))


class BinarySegmentation(RetrospectiveSegmenter):
    """Greedy binary segmentation with a BIC-style penalty per break."""

    name = "binary_segmentation"

    def __init__(self, min_size: int = 40, max_breaks: int = 12, penalty: float = 12.0):
        self.min_size = int(min_size)
        self.max_breaks = int(max_breaks)
        self.penalty = float(penalty)

    def _best_split(self, values: np.ndarray, start: int, end: int) -> Tuple[float, int]:
        baseline = _gaussian_cost(values[start:end])
        best_gain, best_index = 0.0, -1
        for split in range(start + self.min_size, end - self.min_size + 1):
            gain = baseline - (
                _gaussian_cost(values[start:split]) + _gaussian_cost(values[split:end])
            )
            if gain > best_gain:
                best_gain, best_index = gain, split
        return best_gain, best_index

    def breakpoint_indices(self, series: pd.Series) -> List[int]:
        values = np.nan_to_num(series.to_numpy(dtype=float), nan=0.0)
        n = len(values)
        if n < 2 * self.min_size:
            return []

        breaks: List[int] = []
        segments = [(0, n)]
        while len(breaks) < self.max_breaks:
            candidates = []
            for start, end in segments:
                if end - start < 2 * self.min_size:
                    continue
                gain, index = self._best_split(values, start, end)
                if index > 0:
                    candidates.append((gain, index, start, end))
            if not candidates:
                break
            gain, index, start, end = max(candidates)
            if gain < self.penalty:
                break
            breaks.append(index)
            segments.remove((start, end))
            segments.extend([(start, index), (index, end)])

        return sorted(breaks)

    def breakpoints(self, series: pd.Series) -> List[pd.Timestamp]:
        return [series.index[i] for i in self.breakpoint_indices(series)]

    def segment_labels(self, series: pd.Series) -> pd.Series:
        """Integer segment id per observation -- the retrospective 'true' regime path."""
        indices = self.breakpoint_indices(series)
        labels = np.zeros(len(series), dtype=int)
        for segment_id, start in enumerate(indices, start=1):
            labels[start:] = segment_id
        return pd.Series(labels, index=series.index, name="segment")
