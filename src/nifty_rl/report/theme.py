"""Chart theme: palette slots, matplotlib styling, and the emphasis helper.

Colour is assigned **by the job it does**, not by row order:

* *categorical* -- identity (which strategy, which regime). Fixed slot order, never
  cycled, never reassigned when a filter changes the series count.
* *sequential* -- magnitude (agreement, transition probability). One hue, light to dark.
* *diverging* -- polarity (excess return, alpha). Two opposite hues around a neutral
  gray midpoint, never a hue at the midpoint.
* *status* -- state (pass/fail on a validation gate). Reserved; never reused as a series
  colour, and always paired with a label so meaning never rests on hue alone.

The slot ordering is the colour-vision-deficiency safety mechanism, not decoration.
Adjacent pairs clear a CVD separation gate in both light and dark; that is why series are
assigned in order rather than by whichever colour looks nice for a given chart.

The notebook it replaces used ``sns.set_palette("tab10")`` and let matplotlib cycle,
which meant a strategy's colour changed whenever the strategy list changed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------- categorical

CATEGORICAL_LIGHT: List[str] = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

CATEGORICAL_DARK: List[str] = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

# Scatter, bubble and small-multiple forms compare *all* pairs rather than adjacent
# ones; only the first three slots clear the floors under that stricter test.
ALL_PAIRS_SAFE_SLOTS = 3

# --------------------------------------------------------------------- sequential

SEQUENTIAL_BLUE: List[str] = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

# Diverging: warm/cool poles that read as opposite, neutral gray between them.
DIVERGING_LOW = "#2a78d6"
DIVERGING_HIGH = "#e34948"
DIVERGING_MID_LIGHT = "#f0efec"
DIVERGING_MID_DARK = "#383835"

STATUS: Dict[str, str] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

CHROME = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}


def categorical(mode: str = "light") -> List[str]:
    return CATEGORICAL_DARK if mode == "dark" else CATEGORICAL_LIGHT


def chrome(mode: str = "light") -> Dict[str, str]:
    return CHROME["dark" if mode == "dark" else "light"]


def sequential_cmap(mode: str = "light"):
    """One-hue light-to-dark ramp for magnitude."""
    from matplotlib.colors import LinearSegmentedColormap

    steps = SEQUENTIAL_BLUE if mode == "light" else list(reversed(SEQUENTIAL_BLUE))
    return LinearSegmentedColormap.from_list("seq_blue", steps)


def diverging_cmap(mode: str = "light", negative_is_red: bool = True):
    """Two opposite hues around a neutral gray midpoint.

    The midpoint is gray, never a hue, so "no change" reads as nothing.

    ``negative_is_red`` puts red at the low end and blue at the high end, which is the
    orientation every financial reader expects for returns and Sharpe ratios. The
    palette's nominal pole order is the reverse; using it unflipped for a returns
    heatmap paints losses blue and gains red, and the chart reads backwards at a glance.
    """
    from matplotlib.colors import LinearSegmentedColormap

    midpoint = DIVERGING_MID_LIGHT if mode == "light" else DIVERGING_MID_DARK
    low, high = (
        (DIVERGING_HIGH, DIVERGING_LOW) if negative_is_red else (DIVERGING_LOW, DIVERGING_HIGH)
    )
    return LinearSegmentedColormap.from_list("div_returns", [low, midpoint, high])


def apply_theme(mode: str = "light") -> None:
    """Set recessive chrome: hairline solid grid, no top/right spines, generous space.

    Gridlines are **solid** hairlines. Dashing them adds noise and reads as
    "projection" or "threshold" when it is only a grid -- a habit the original notebook
    had on every axhline.
    """
    c = chrome(mode)
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "figure.facecolor": c["surface"],
            "axes.facecolor": c["surface"],
            "savefig.facecolor": c["surface"],
            "axes.edgecolor": c["axis"],
            "axes.labelcolor": c["secondary"],
            "axes.titlecolor": c["primary"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": c["grid"],
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": c["muted"],
            "ytick.color": c["muted"],
            "xtick.labelcolor": c["secondary"],
            "ytick.labelcolor": c["secondary"],
            "text.color": c["primary"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "figure.autolayout": False,
        }
    )


def emphasis_colors(
    names: Sequence[str],
    highlight: Sequence[str],
    mode: str = "light",
) -> Dict[str, str]:
    """Give highlighted series categorical hues; everything else recedes to gray.

    Eight saturated hues when the story is one or two series is the most common way a
    chart misses its own point. Emphasis keeps the context visible without competing
    with it.
    """
    palette = categorical(mode)
    muted = chrome(mode)["muted"]
    mapping: Dict[str, str] = {}
    slot = 0
    for name in names:
        if name in highlight:
            mapping[name] = palette[slot % len(palette)]
            slot += 1
        else:
            mapping[name] = muted
    return mapping


def series_colors(names: Sequence[str], mode: str = "light") -> Dict[str, str]:
    """Stable name-to-hue map.

    Colour follows the entity, not its rank, so a chart that drops a strategy does not
    repaint the survivors. Past eight names the tail folds to gray rather than cycling --
    a generated ninth hue is indistinguishable from an existing slot under CVD.
    """
    palette = categorical(mode)
    muted = chrome(mode)["muted"]
    return {
        name: (palette[i] if i < len(palette) else muted) for i, name in enumerate(names)
    }


def regime_colors(labels: Sequence[str], mode: str = "light") -> Dict[str, str]:
    """Regimes are ordered (calm -> crisis), so they take an ordinal ramp, not identity hues.

    On the light surface the ramp starts at step 250 rather than the lightest step: an
    ordinal ramp's nearest-to-surface step still has to clear a contrast floor, unlike a
    sequential ramp where "near zero" is allowed to recede.
    """
    n = max(len(labels), 1)
    if mode == "light":
        usable = SEQUENTIAL_BLUE[3:]
    else:
        usable = list(reversed(SEQUENTIAL_BLUE[:-2]))
    if n == 1:
        picks = [usable[len(usable) // 2]]
    else:
        step = (len(usable) - 1) / (n - 1)
        picks = [usable[int(round(i * step))] for i in range(n)]
    return dict(zip(labels, picks))


def finish(ax, title: Optional[str] = None, subtitle: Optional[str] = None, mode: str = "light"):
    """Apply the shared title treatment and trim chart junk.

    Title and subtitle are offset in **points**, not axes fractions. An axes-fraction
    offset is a fraction of the plot's height, so on a tall figure the gap collapses and
    the two strings overlap -- which is exactly what happened on the 19-row forest plot.
    """
    c = chrome(mode)
    if title:
        ax.set_title(title, loc="left", pad=26 if subtitle else 12)
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(0, 8), textcoords="offset points",
            fontsize=8.5, color=c["secondary"], va="bottom", ha="left",
        )
    ax.tick_params(length=0)
    return ax
