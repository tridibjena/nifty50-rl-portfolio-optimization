"""Validating the regime models themselves.

Most regime work stops at "here is the fitted state path". That is the part that cannot
be checked. These diagnostics are the part that can:

* **Persistence** -- a model that flips every three days is untradeable after costs, no
  matter how well it fits.
* **Detection lag** -- days between a retrospectively established break and the online
  detector flagging it. This is a *kill criterion*: a detector that recognises a crisis
  fifteen days late adds cost and provides no protection, and no downstream overlay can
  rescue it.
* **Refit stability** -- whether an expanding-window refit keeps labels consistent, or
  whether "state 0" silently changes meaning.
* **Cross-method agreement** -- pairwise Cohen's kappa. Disagreement between backends is
  itself informative and worth reporting rather than hiding behind whichever one looked
  best.
* **Economic sanity** -- do the discovered states look like anything a portfolio manager
  would recognise?
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .base import RegimeDetector


# ------------------------------------------------------------------- persistence


def run_lengths(labels: pd.Series) -> pd.Series:
    """Length of each contiguous run of a constant label."""
    values = labels.to_numpy()
    if len(values) == 0:
        return pd.Series(dtype=float)
    change = np.flatnonzero(np.diff(values)) + 1
    boundaries = np.concatenate([[0], change, [len(values)]])
    return pd.Series(np.diff(boundaries), name="run_length")


def persistence_summary(labels: pd.Series, regime_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Occupancy and mean run length per regime, plus the overall switch rate."""
    runs = run_lengths(labels)
    values = labels.to_numpy()
    n_switches = int((np.diff(values) != 0).sum())

    rows = []
    for regime in sorted(pd.unique(values)):
        mask = values == regime
        regime_runs = []
        current = 0
        for flag in mask:
            if flag:
                current += 1
            elif current:
                regime_runs.append(current)
                current = 0
        if current:
            regime_runs.append(current)

        name = (
            regime_names[regime]
            if regime_names is not None and regime < len(regime_names)
            else f"regime_{regime}"
        )
        rows.append(
            {
                "regime": name,
                "occupancy": float(mask.mean()),
                "n_episodes": len(regime_runs),
                "mean_run_days": float(np.mean(regime_runs)) if regime_runs else 0.0,
                "median_run_days": float(np.median(regime_runs)) if regime_runs else 0.0,
                "max_run_days": int(np.max(regime_runs)) if regime_runs else 0,
            }
        )

    frame = pd.DataFrame(rows)
    frame.attrs["n_switches"] = n_switches
    frame.attrs["switch_rate"] = n_switches / max(len(values) - 1, 1)
    frame.attrs["overall_mean_run"] = float(runs.mean()) if len(runs) else 0.0
    return frame


def is_tradeable(labels: pd.Series, min_mean_run_days: float = 10.0) -> bool:
    """Crude gate: mean run length must exceed a cost-driven floor."""
    runs = run_lengths(labels)
    return bool(len(runs) and runs.mean() >= min_mean_run_days)


# ----------------------------------------------------------------- detection lag


def detection_lag(
    online_labels: pd.Series,
    break_dates: Sequence[pd.Timestamp],
    max_horizon: int = 60,
) -> pd.DataFrame:
    """Days from each ground-truth break to the next online label change.

    ``lag = NaN`` means the detector never reacted within ``max_horizon`` -- a miss, and
    strictly worse than a slow detection.
    """
    index = online_labels.index
    values = online_labels.to_numpy()
    changes = np.flatnonzero(np.diff(values)) + 1

    rows = []
    for break_date in break_dates:
        position = index.searchsorted(break_date)
        if position >= len(index):
            continue
        following = changes[changes >= position]
        following = following[following <= position + max_horizon]
        if len(following) == 0:
            rows.append({"break_date": break_date, "lag_days": np.nan, "detected": False})
        else:
            rows.append(
                {
                    "break_date": break_date,
                    "lag_days": int(following[0] - position),
                    "detected": True,
                }
            )
    return pd.DataFrame(rows)


def lag_summary(lags: pd.DataFrame) -> Dict[str, float]:
    if lags.empty:
        return {"n_breaks": 0, "detection_rate": float("nan"), "median_lag": float("nan")}
    detected = lags[lags["detected"]]
    return {
        "n_breaks": int(len(lags)),
        "detection_rate": float(lags["detected"].mean()),
        "median_lag": float(detected["lag_days"].median()) if len(detected) else float("nan"),
        "mean_lag": float(detected["lag_days"].mean()) if len(detected) else float("nan"),
        "worst_lag": float(detected["lag_days"].max()) if len(detected) else float("nan"),
    }


# ------------------------------------------------------------------- agreement


def agreement_matrix(label_map: Dict[str, pd.Series]) -> pd.DataFrame:
    """Pairwise Cohen's kappa between detectors.

    Kappa rather than raw agreement because regimes are unbalanced -- a detector sitting
    in "Normal" 70% of the time agrees with everything by chance.
    """
    from sklearn.metrics import cohen_kappa_score

    names = list(label_map)
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            aligned = pd.concat(
                [label_map[left].rename("a"), label_map[right].rename("b")],
                axis=1,
                join="inner",
            ).dropna()
            if aligned.empty:
                score = np.nan
            else:
                score = float(cohen_kappa_score(aligned["a"], aligned["b"]))
            matrix.loc[left, right] = score
            matrix.loc[right, left] = score
    return matrix


# --------------------------------------------------------------- refit stability


def refit_stability(
    make_detector,
    features: pd.DataFrame,
    initial_train: int = 500,
    step: int = 120,
) -> pd.DataFrame:
    """Refit on an expanding window and measure label agreement with the prior fit.

    Low agreement means the state definitions are moving, so regime-conditional results
    are not comparable across time -- the failure that makes regime work irreproducible.
    """
    from sklearn.metrics import cohen_kappa_score

    rows = []
    previous_labels = None
    for end in range(initial_train, len(features) + 1, step):
        window = features.iloc[:end]
        detector = make_detector()
        try:
            detector.fit(window)
        except Exception as exc:  # pragma: no cover
            rows.append({"train_end": features.index[end - 1], "kappa_vs_previous": np.nan, "error": str(exc)})
            continue
        labels = detector.label_online(features.iloc[:end])
        if previous_labels is not None:
            overlap = labels.index.intersection(previous_labels.index)
            kappa = float(
                cohen_kappa_score(labels.loc[overlap], previous_labels.loc[overlap])
            ) if len(overlap) > 1 else np.nan
        else:
            kappa = np.nan
        rows.append(
            {
                "train_end": features.index[end - 1],
                "n_train": end,
                "kappa_vs_previous": kappa,
                "error": "",
            }
        )
        previous_labels = labels
    return pd.DataFrame(rows)


# ------------------------------------------------------------- economic profile


def regime_conditional_stats(
    returns: pd.Series,
    labels: pd.Series,
    regime_names: Optional[Sequence[str]] = None,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Return, volatility, Sharpe and drawdown within each regime.

    The sanity check: if the "crisis" state does not show higher volatility and worse
    drawdown than the "calm" state, the model has not found regimes -- it has found
    clusters.
    """
    from ..metrics.stats import sharpe_from_returns

    aligned = pd.concat(
        [returns.rename("ret"), labels.rename("regime")], axis=1, join="inner"
    ).dropna()

    rows = []
    for regime, group in aligned.groupby("regime"):
        series = group["ret"]
        name = (
            regime_names[int(regime)]
            if regime_names is not None and int(regime) < len(regime_names)
            else f"regime_{int(regime)}"
        )
        equity = (1.0 + series).cumprod()
        drawdown = float((equity / equity.cummax() - 1.0).min()) if len(equity) else np.nan
        rows.append(
            {
                "regime": name,
                "regime_index": int(regime),
                "n_days": int(len(series)),
                "share_of_sample": float(len(series) / len(aligned)),
                "mean_return_annual": float(series.mean() * trading_days),
                "volatility_annual": float(series.std() * np.sqrt(trading_days)),
                "sharpe": sharpe_from_returns(series.to_numpy(), trading_days),
                "max_drawdown": drawdown,
                "hit_rate": float((series > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("regime_index").reset_index(drop=True)
