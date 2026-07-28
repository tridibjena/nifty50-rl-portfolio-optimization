"""Figures for the results report.

Every function returns ``(figure, table)`` -- the chart and the data behind it. The table
is not an afterthought: a chart whose values are reachable only by reading pixels fails
accessibility, and the CSV twin is what makes each figure checkable.

Conventions enforced here:

* **Never two y-scales on one plot.** Two measures of different scale become two panels
  or an indexed common base. The original notebook's entry/exit chart used ``twinx()``,
  which invents an alignment between price and position that is not in the data.
* **Emphasis over rainbow.** When the story is one or two series, those get hue and the
  rest recede to gray.
* **Solid hairline grids**, recessive axes, thin marks, generous padding.
* **Selective direct labels** -- endpoints and extremes only, never a number on every
  point, with a legend whenever two or more series share a panel.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .theme import (
    ALL_PAIRS_SAFE_SLOTS,
    STATUS,
    apply_theme,
    categorical,
    chrome,
    diverging_cmap,
    emphasis_colors,
    finish,
    regime_colors,
    sequential_cmap,
    series_colors,
)

FigureAndTable = Tuple[Figure, pd.DataFrame]


def _format_dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))


def _currency(ax, symbol: str = "₹") -> None:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{symbol}{v:,.0f}"))


def _percent(ax, decimals: int = 0) -> None:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.{decimals}f}%"))


# ------------------------------------------------------------------ equity curves


def equity_curves(
    equity_by_strategy: Dict[str, pd.Series],
    highlight: Sequence[str],
    initial_cash: float,
    mode: str = "light",
    title: str = "Equity curves — test period",
    subtitle: Optional[str] = None,
) -> FigureAndTable:
    """Equity paths with the strategies that matter in colour and the rest in gray."""
    apply_theme(mode)
    c = chrome(mode)
    colors = emphasis_colors(list(equity_by_strategy), highlight, mode)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Context first so the highlighted paths sit on top.
    for name, equity in equity_by_strategy.items():
        if name in highlight:
            continue
        ax.plot(equity.index, equity.to_numpy(), color=colors[name], linewidth=1.0, alpha=0.55, zorder=2)

    for name in highlight:
        if name not in equity_by_strategy:
            continue
        equity = equity_by_strategy[name]
        ax.plot(equity.index, equity.to_numpy(), color=colors[name], linewidth=2.2, zorder=4, label=name)
        # Direct-label the endpoint only.
        ax.annotate(
            f" {name}",
            xy=(equity.index[-1], equity.iloc[-1]),
            color=colors[name],
            fontsize=8.5,
            va="center",
            fontweight="bold",
        )

    ax.axhline(initial_cash, color=c["axis"], linewidth=1.0, zorder=1)
    ax.annotate(
        "  initial capital",
        xy=(list(equity_by_strategy.values())[0].index[0], initial_cash),
        color=c["muted"], fontsize=8, va="bottom",
    )

    _format_dates(ax)
    _currency(ax)
    ax.set_ylabel("Portfolio value")
    handles = [plt.Line2D([], [], color=colors[n], linewidth=2.2, label=n) for n in highlight if n in equity_by_strategy]
    handles.append(plt.Line2D([], [], color=c["muted"], linewidth=1.0, label="other strategies"))
    ax.legend(handles=handles, loc="best", ncol=1)
    finish(ax, title, subtitle, mode)
    fig.tight_layout()

    table = pd.DataFrame(equity_by_strategy)
    return fig, table


def drawdown_curves(
    equity_by_strategy: Dict[str, pd.Series],
    highlight: Sequence[str],
    mode: str = "light",
    title: str = "Drawdown — test period",
    subtitle: Optional[str] = None,
) -> FigureAndTable:
    apply_theme(mode)
    c = chrome(mode)
    colors = emphasis_colors(list(equity_by_strategy), highlight, mode)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    table = {}
    for name, equity in equity_by_strategy.items():
        dd = (equity / equity.cummax() - 1.0) * 100.0
        table[name] = dd
        is_highlight = name in highlight
        ax.plot(
            dd.index, dd.to_numpy(),
            color=colors[name],
            linewidth=2.0 if is_highlight else 0.9,
            alpha=1.0 if is_highlight else 0.5,
            zorder=4 if is_highlight else 2,
            label=name if is_highlight else None,
        )

    ax.axhline(0, color=c["axis"], linewidth=1.0)
    _format_dates(ax)
    _percent(ax)
    ax.set_ylabel("Drawdown")
    handles = [plt.Line2D([], [], color=colors[n], linewidth=2.0, label=n) for n in highlight if n in equity_by_strategy]
    handles.append(plt.Line2D([], [], color=c["muted"], linewidth=0.9, label="other strategies"))
    ax.legend(handles=handles, loc="lower left")
    finish(ax, title, subtitle, mode)
    fig.tight_layout()
    return fig, pd.DataFrame(table)


# ---------------------------------------------------------------- regime timeline


def regime_timeline(
    market_equity: pd.Series,
    labels_by_detector: Dict[str, pd.Series],
    regime_names_by_detector: Dict[str, Sequence[str]],
    breakpoints: Optional[Sequence[pd.Timestamp]] = None,
    mode: str = "light",
    title: str = "Regime timeline — five detectors over the market path",
    subtitle: Optional[str] = None,
) -> FigureAndTable:
    """The centrepiece figure: market path on top, one causal detector strip per row.

    Small multiples rather than overlays -- five regime paths on one axis would be
    unreadable, and putting them on a second y-scale against price would invent a
    relationship. Retrospective break dates are drawn as vertical rules across every
    panel so each detector's lag is visible directly.
    """
    apply_theme(mode)
    c = chrome(mode)
    n_detectors = len(labels_by_detector)

    fig, axes = plt.subplots(
        n_detectors + 1, 1,
        figsize=(11, 2.8 + 0.72 * n_detectors),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2] + [0.75] * n_detectors, "hspace": 0.32},
    )
    price_ax = axes[0]

    normalised = market_equity / market_equity.iloc[0] * 100.0
    price_ax.plot(normalised.index, normalised.to_numpy(), color=c["primary"], linewidth=1.6)
    price_ax.set_ylabel("Market (=100)")
    price_ax.grid(True, axis="y")

    if breakpoints:
        for date in breakpoints:
            for ax in axes:
                ax.axvline(date, color=STATUS["critical"], linewidth=1.0, alpha=0.55, zorder=1)
        price_ax.annotate(
            "vertical rules = retrospective structural breaks (ground truth for detection lag)",
            xy=(0.0, -0.16), xycoords="axes fraction",
            fontsize=7.5, color=c["muted"],
        )

    rows = []
    for ax, (detector_name, labels) in zip(axes[1:], labels_by_detector.items()):
        names = list(regime_names_by_detector.get(detector_name, []))
        if not names:
            names = [f"R{i}" for i in range(int(labels.max()) + 1)]
        palette = regime_colors(names, mode)

        values = labels.to_numpy()
        dates = labels.index
        start = 0
        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[start]:
                regime_index = int(values[start])
                label_name = names[regime_index] if regime_index < len(names) else f"R{regime_index}"
                ax.axvspan(
                    dates[start],
                    dates[min(i, len(dates) - 1)],
                    color=palette[label_name],
                    linewidth=0,
                )
                rows.append(
                    {
                        "detector": detector_name,
                        "start": dates[start],
                        "end": dates[min(i, len(dates) - 1)],
                        "regime": label_name,
                        "days": i - start,
                    }
                )
                start = i

        ax.set_yticks([])
        ax.set_ylabel(detector_name, rotation=0, ha="right", va="center", fontsize=8.5, color=c["secondary"])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        handles = [Patch(facecolor=palette[n], label=n) for n in names]
        ax.legend(
            handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5),
            fontsize=7, handlelength=1.0, handleheight=0.9, borderpad=0.2, labelspacing=0.25,
        )

    _format_dates(axes[-1])
    finish(price_ax, title, subtitle, mode)
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


def regime_transition_heatmap(
    transitions: pd.DataFrame,
    mode: str = "light",
    title: str = "Regime transition matrix",
    subtitle: Optional[str] = None,
) -> FigureAndTable:
    """Transition probabilities. Magnitude, so a single-hue sequential ramp."""
    apply_theme(mode)
    c = chrome(mode)

    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    data = transitions.to_numpy()
    image = ax.imshow(data, cmap=sequential_cmap(mode), vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(transitions.columns)), transitions.columns, fontsize=8)
    ax.set_yticks(range(len(transitions.index)), transitions.index, fontsize=8)
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    ax.grid(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=8,
                color="#ffffff" if value > 0.55 else c["primary"],
            )

    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.outline.set_visible(False)
    bar.set_label("probability", fontsize=8)
    finish(ax, title, subtitle, mode)
    fig.tight_layout()
    return fig, transitions


def agreement_heatmap(
    matrix: pd.DataFrame,
    mode: str = "light",
    title: str = "Cross-detector agreement (Cohen's κ)",
    subtitle: Optional[str] = None,
) -> FigureAndTable:
    apply_theme(mode)
    c = chrome(mode)

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    data = matrix.to_numpy(dtype=float)
    image = ax.imshow(data, cmap=sequential_cmap(mode), vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(matrix.columns)), matrix.columns, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    ax.grid(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if not np.isfinite(value):
                continue
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=8,
                color="#ffffff" if value > 0.55 else c["primary"],
            )

    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.outline.set_visible(False)
    bar.set_label("κ", fontsize=8)
    finish(ax, title, subtitle, mode)
    fig.tight_layout()
    return fig, matrix


def regime_validation_panel(
    persistence: pd.DataFrame,
    lag_by_detector: Dict[str, dict],
    min_run_days: float = 10.0,
    mode: str = "light",
    title: str = "Regime model validation",
    subtitle: Optional[str] = None,
) -> FigureAndTable:
    """Persistence and detection lag side by side, each against its gate.

    These two numbers decide whether the regime layer is usable at all. A model that
    flips every few days is untradeable after costs; a model that recognises a crisis
    weeks late adds cost and no protection. Both gates are drawn on the chart, and the
    status colours carry a label so meaning never rests on hue.
    """
    apply_theme(mode)
    c = chrome(mode)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))

    # --- mean run length vs the tradeability floor
    names = persistence["regime"].tolist()
    runs = persistence["mean_run_days"].to_numpy()
    colors = [STATUS["good"] if v >= min_run_days else STATUS["critical"] for v in runs]
    bars = left.barh(names, runs, color=colors, height=0.6)
    left.axvline(min_run_days, color=c["primary"], linewidth=1.2)
    left.annotate(
        f" tradeability floor ({min_run_days:.0f}d)",
        xy=(min_run_days, len(names) - 0.4), color=c["secondary"], fontsize=8, va="top",
    )
    for bar, value in zip(bars, runs):
        left.annotate(
            f" {value:.0f}d", xy=(value, bar.get_y() + bar.get_height() / 2),
            va="center", fontsize=8, color=c["secondary"],
        )
    left.set_xlabel("Mean run length (days)")
    left.grid(True, axis="x")
    left.set_axisbelow(True)
    finish(left, "Persistence", "pass = regime lasts long enough to trade", mode)

    # --- median detection lag
    detectors = list(lag_by_detector)
    medians = [lag_by_detector[d].get("median_lag", np.nan) for d in detectors]
    rates = [lag_by_detector[d].get("detection_rate", np.nan) for d in detectors]
    lag_colors = [
        STATUS["good"] if (np.isfinite(m) and m <= 10) else
        STATUS["warning"] if (np.isfinite(m) and m <= 20) else STATUS["critical"]
        for m in medians
    ]
    bars = right.barh(detectors, [0 if not np.isfinite(m) else m for m in medians], color=lag_colors, height=0.6)
    for bar, median, rate in zip(bars, medians, rates):
        text = "no detection" if not np.isfinite(median) else f" {median:.0f}d  ({rate:.0%} found)"
        right.annotate(
            text, xy=(0 if not np.isfinite(median) else median, bar.get_y() + bar.get_height() / 2),
            va="center", fontsize=8, color=c["secondary"],
        )
    right.set_xlabel("Median detection lag (days)")
    right.grid(True, axis="x")
    right.set_axisbelow(True)
    finish(right, "Detection lag", "green ≤ 10d  ·  amber ≤ 20d  ·  red slower", mode)

    fig.tight_layout()
    table = persistence.copy()
    return fig, table


# ------------------------------------------------------------- honest leaderboard


def sharpe_forest(
    significance: pd.DataFrame,
    mode: str = "light",
    title: str = "Sharpe with bootstrap confidence intervals",
    subtitle: Optional[str] = "Interval = stationary-block bootstrap 95%. Dashed rule = deflated-Sharpe threshold for this many trials.",
) -> FigureAndTable:
    """Dot-and-interval leaderboard.

    A bare ranked table of point estimates invites a conclusion ~220 trading days cannot
    support. Showing the interval makes the overlap obvious, and the deflation threshold
    marks the bar the *best* of many trials actually has to clear.
    """
    apply_theme(mode)
    c = chrome(mode)
    frame = significance.sort_values("sharpe", na_position="first").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(frame) + 2.4))
    y = np.arange(len(frame))

    for i, row in frame.iterrows():
        lower, upper = row["sharpe_ci_lower"], row["sharpe_ci_upper"]
        if np.isfinite(lower) and np.isfinite(upper):
            ax.plot([lower, upper], [i, i], color=c["axis"], linewidth=2.0, solid_capstyle="round", zorder=2)
        elif not np.isfinite(row["sharpe"]):
            # A portfolio that never left cash has an undefined Sharpe. Say so on the
            # chart rather than leaving a blank row the reader has to interpret.
            ax.annotate(
                "never invested — Sharpe undefined",
                xy=(0, i), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=c["muted"], style="italic",
            )

    significant = frame["ci_excludes_zero"].to_numpy()
    benchmark = frame["is_benchmark"].to_numpy()
    point_colors = [
        categorical(mode)[1] if b else (categorical(mode)[0] if s else c["muted"])
        for s, b in zip(significant, benchmark)
    ]
    ax.scatter(frame["sharpe"], y, s=64, color=point_colors, zorder=4, edgecolor=chrome(mode)["surface"], linewidth=1.5)

    ax.axvline(0, color=c["primary"], linewidth=1.1, zorder=1)
    threshold = frame["deflation_threshold"].dropna()
    if len(threshold):
        ax.axvline(float(threshold.iloc[0]), color=STATUS["critical"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)

    ax.set_yticks(y, frame["strategy"], fontsize=8.5)
    ax.set_xlabel("Annualised Sharpe")
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=categorical(mode)[1], label="benchmark", markersize=8),
        plt.Line2D([], [], marker="o", linestyle="", color=categorical(mode)[0], label="CI excludes zero", markersize=8),
        plt.Line2D([], [], marker="o", linestyle="", color=c["muted"], label="indistinguishable from zero", markersize=8),
    ]
    ax.legend(handles=handles, loc="lower right")
    finish(ax, title, subtitle, mode)
    fig.tight_layout()
    return fig, significance


def regime_performance_heatmap(
    performance: pd.DataFrame,
    value_column: str = "sharpe",
    mode: str = "light",
    title: str = "Strategy performance by regime",
    subtitle: Optional[str] = None,
) -> FigureAndTable:
    """Strategy x regime grid. Polarity matters, so a diverging ramp around zero.

    This is the table the whole project was missing: it answers whether a
    drawdown-penalised agent actually earns its penalty when conditions are bad, rather
    than reporting one blended number over a window that happened to be a drawdown.
    """
    apply_theme(mode)
    c = chrome(mode)
    grid = performance.pivot(index="strategy", columns="regime", values=value_column)

    # Regimes are ordinal (calm -> crisis). Pivot returns them alphabetically, which puts
    # "Crisis" second and makes the gradient across the row meaningless.
    if "regime_index" in performance.columns:
        order = (
            performance.drop_duplicates("regime")
            .sort_values("regime_index")["regime"]
            .tolist()
        )
        grid = grid[[r for r in order if r in grid.columns]]

    magnitude = float(np.nanmax(np.abs(grid.to_numpy()))) if grid.size else 1.0
    magnitude = magnitude if np.isfinite(magnitude) and magnitude > 0 else 1.0

    fig, ax = plt.subplots(figsize=(1.5 * len(grid.columns) + 4.0, 0.42 * len(grid.index) + 2.2))
    image = ax.imshow(grid.to_numpy(), cmap=diverging_cmap(mode), vmin=-magnitude, vmax=magnitude, aspect="auto")

    ax.set_xticks(range(len(grid.columns)), grid.columns, fontsize=8.5)
    ax.set_yticks(range(len(grid.index)), grid.index, fontsize=8.5)
    ax.grid(False)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.to_numpy()[i, j]
            if not np.isfinite(value):
                continue
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=8,
                color="#ffffff" if abs(value) > 0.62 * magnitude else c["primary"],
            )

    bar = fig.colorbar(image, ax=ax, shrink=0.85)
    bar.outline.set_visible(False)
    bar.set_label(value_column, fontsize=8)
    finish(ax, title, subtitle, mode)
    fig.tight_layout()
    return fig, grid


def risk_return_scatter(
    metrics: pd.DataFrame,
    highlight: Sequence[str],
    mode: str = "light",
    title: str = "Risk and return — test period",
    subtitle: Optional[str] = None,
) -> FigureAndTable:
    """Scatter compares every pair of colours at once, so highlights are capped at three."""
    apply_theme(mode)
    c = chrome(mode)
    capped = list(highlight)[:ALL_PAIRS_SAFE_SLOTS]
    colors = emphasis_colors(metrics["strategy"].tolist(), capped, mode)

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for _, row in metrics.iterrows():
        name = row["strategy"]
        is_highlight = name in capped
        ax.scatter(
            row["annual_volatility"] * 100.0,
            row["total_return"] * 100.0,
            s=150 if is_highlight else 46,
            color=colors[name],
            zorder=4 if is_highlight else 2,
            edgecolor=chrome(mode)["surface"],
            linewidth=1.6,
            label=name if is_highlight else None,
        )
        if is_highlight:
            ax.annotate(
                f"  {name}",
                xy=(row["annual_volatility"] * 100.0, row["total_return"] * 100.0),
                fontsize=8.5, color=colors[name], va="center", fontweight="bold",
            )

    ax.axhline(0, color=c["primary"], linewidth=1.1)
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("Total return (%)")
    ax.legend(loc="best")
    ax.set_axisbelow(True)
    finish(ax, title, subtitle, mode)
    fig.tight_layout()
    return fig, metrics[["strategy", "annual_volatility", "total_return"]]


def allocation_area(
    weights: pd.DataFrame,
    mode: str = "light",
    title: str = "Portfolio allocation",
    subtitle: Optional[str] = None,
) -> FigureAndTable:
    """Realised weights through time.

    The notebook's 'allocation stack' plotted an inline momentum proxy rather than the
    agent's weights, because the environment never logged them. This one takes the
    logged weights, so it shows what the policy actually did.
    """
    apply_theme(mode)
    c = chrome(mode)

    columns = list(weights.columns)
    if len(columns) > 8:
        ranked = weights.mean().sort_values(ascending=False)
        keep = list(ranked.index[:7])
        folded = weights[keep].copy()
        folded["Other"] = weights.drop(columns=keep).sum(axis=1)
        weights = folded
        columns = list(weights.columns)

    colors = series_colors(columns, mode)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.stackplot(
        weights.index,
        [weights[col].to_numpy() for col in columns],
        colors=[colors[col] for col in columns],
        labels=columns,
        linewidth=0.8,
        edgecolor=chrome(mode)["surface"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Weight")
    _format_dates(ax)
    ax.grid(True, axis="y")
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8)
    finish(ax, title, subtitle, mode)
    fig.tight_layout()
    return fig, weights


def save(fig: Figure, table: pd.DataFrame, path, also_csv: bool = True) -> None:
    """Write the figure and, beside it, the table that makes its values readable."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if also_csv and table is not None and not table.empty:
        table.to_csv(path.with_suffix(".csv"))
    plt.close(fig)


# ------------------------------------------------------------------- walk-forward


def window_returns_heatmap(
    per_window: pd.DataFrame,
    mode: str = "light",
    title: str = "Out-of-sample return by walk-forward window",
    subtitle: Optional[str] = None,
) -> FigureAndTable:
    """Strategy x window returns — the figure a single split cannot produce.

    Reading across a row shows whether a strategy is consistent or whether its headline
    number came from one lucky block. Reading down a column shows which windows were hard
    for everything, which is the context a single test period silently hides.
    """
    apply_theme(mode)
    c = chrome(mode)
    grid = per_window.pivot(index="strategy", columns="window", values="total_return") * 100.0
    grid = grid.loc[grid.mean(axis=1).sort_values(ascending=False).index]

    magnitude = float(np.nanmax(np.abs(grid.to_numpy()))) if grid.size else 1.0
    magnitude = magnitude if np.isfinite(magnitude) and magnitude > 0 else 1.0

    fig, ax = plt.subplots(
        figsize=(1.05 * len(grid.columns) + 5.0, 0.40 * len(grid.index) + 2.4)
    )
    image = ax.imshow(
        grid.to_numpy(), cmap=diverging_cmap(mode), vmin=-magnitude, vmax=magnitude, aspect="auto"
    )

    ax.set_xticks(range(len(grid.columns)), [f"W{c_}" for c_ in grid.columns], fontsize=8.5)
    ax.set_yticks(range(len(grid.index)), grid.index, fontsize=8.5)
    ax.grid(False)

    values = grid.to_numpy()
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            ax.text(
                j, i, f"{value:.1f}",
                ha="center", va="center", fontsize=7.5,
                color="#ffffff" if abs(value) > 0.62 * magnitude else c["primary"],
            )

    bar = fig.colorbar(image, ax=ax, shrink=0.85)
    bar.outline.set_visible(False)
    bar.set_label("return (%)", fontsize=8)
    finish(ax, title, subtitle, mode)
    fig.tight_layout()
    return fig, grid


def consistency_scatter(
    summary: pd.DataFrame,
    highlight: Sequence[str],
    mode: str = "light",
    title: str = "Consistency versus pooled performance",
    subtitle: Optional[str] = "A high pooled Sharpe from one good window is not a strategy.",
) -> FigureAndTable:
    """Share of windows profitable against pooled out-of-sample Sharpe."""
    apply_theme(mode)
    c = chrome(mode)
    capped = list(highlight)[:ALL_PAIRS_SAFE_SLOTS]
    colors = emphasis_colors(summary["strategy"].tolist(), capped, mode)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for _, row in summary.iterrows():
        name = row["strategy"]
        is_highlight = name in capped
        ax.scatter(
            row["windows_positive"] * 100.0,
            row["pooled_sharpe"],
            s=150 if is_highlight else 44,
            color=colors[name],
            zorder=4 if is_highlight else 2,
            edgecolor=chrome(mode)["surface"],
            linewidth=1.6,
        )
        if is_highlight:
            ax.annotate(
                f"  {name}",
                xy=(row["windows_positive"] * 100.0, row["pooled_sharpe"]),
                fontsize=8.5, color=colors[name], va="center", fontweight="bold",
            )

    ax.axhline(0, color=c["primary"], linewidth=1.1)
    ax.axvline(50, color=c["axis"], linewidth=1.0)
    ax.set_xlabel("Windows with a positive return (%)")
    ax.set_ylabel("Pooled out-of-sample Sharpe")
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[n], label=n, markersize=8)
        for n in capped
    ]
    handles.append(
        plt.Line2D([], [], marker="o", linestyle="", color=c["muted"], label="other strategies", markersize=7)
    )
    ax.legend(handles=handles, loc="best")
    finish(ax, title, subtitle, mode)
    fig.tight_layout()
    return fig, summary[["strategy", "windows_positive", "pooled_sharpe"]]


def ppo_seed_dispersion(
    per_window: pd.DataFrame,
    benchmark: str = "BuyHold",
    mode: str = "light",
    title: str = "PPO seed dispersion by window",
    subtitle: Optional[str] = "Every seed is an independent training run on the same data.",
) -> FigureAndTable:
    """Per-window PPO seed spread against the benchmark.

    The original project reported a single seed and listed that as a limitation. Showing
    the spread answers the question that limitation raises: is the agent's result a draw
    from a wide distribution, or is the policy genuinely reproducible? Tight dispersion
    means an underperforming agent is underperforming for real reasons, not variance.
    """
    apply_theme(mode)
    c = chrome(mode)

    seed_rows = per_window[per_window["strategy"].str.startswith("PPO_s")]
    if seed_rows.empty:
        raise ValueError("no PPO seed rows in per_window")

    windows = sorted(seed_rows["window"].unique())
    fig, ax = plt.subplots(figsize=(1.05 * len(windows) + 3.6, 4.6))
    palette = categorical(mode)

    for window in windows:
        block = seed_rows[seed_rows["window"] == window]["total_return"] * 100.0
        lo, hi = block.min(), block.max()
        ax.plot([window, window], [lo, hi], color=c["axis"], linewidth=2.0,
                solid_capstyle="round", zorder=2)
        ax.scatter([window] * len(block), block, s=42, color=palette[0],
                   zorder=4, edgecolor=chrome(mode)["surface"], linewidth=1.2)

    ensemble = per_window[per_window["strategy"] == "PPO_ensemble"]
    if not ensemble.empty:
        ax.plot(ensemble["window"], ensemble["total_return"] * 100.0,
                color=palette[0], linewidth=1.4, alpha=0.5, zorder=3, label="PPO ensemble")

    bench = per_window[per_window["strategy"] == benchmark]
    if not bench.empty:
        ax.plot(bench["window"], bench["total_return"] * 100.0, color=palette[1],
                linewidth=2.2, marker="o", markersize=6, zorder=5, label=benchmark)

    ax.axhline(0, color=c["primary"], linewidth=1.1)
    ax.set_xticks(windows, [f"W{w}" for w in windows])
    ax.set_xlabel("Walk-forward window")
    ax.set_ylabel("Out-of-sample return (%)")
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=palette[0],
                   label="individual seeds", markersize=7),
        plt.Line2D([], [], color=palette[1], marker="o", linewidth=2.2,
                   label=benchmark, markersize=6),
    ]
    ax.legend(handles=handles, loc="best")
    finish(ax, title, subtitle, mode)
    fig.tight_layout()

    table = (
        seed_rows.groupby("window")["total_return"]
        .agg(["min", "mean", "max", "std"])
        .rename(columns={"std": "seed_std"})
    )
    return fig, table
