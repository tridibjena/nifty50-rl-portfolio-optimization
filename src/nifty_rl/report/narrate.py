"""Regime narration — commentary attached to detected episodes, for readers.

The detector says `Crisis, 2022-06-13 to 2022-07-08`. True, causal, and opaque. This
module attaches *what was happening* so a reader can judge whether the label is
plausible, without having to remember four years of market history.

**This is presentation, and the boundary is enforced rather than intended.**

* Narration is generated **after** the fact, from the completed episode table.
* It is written to its own artefact (`results/regime_narration.csv`) and never joined
  into the feature panel, the observation vector, or any model input.
* :func:`assert_no_narration_leak` is called by the test suite against the real feature
  frame, so a future edit that pipes commentary into a feature fails CI.

The reason for that severity: an LLM's weights encode what happened *after* the period
being narrated. Feeding its output into a model is lookahead of a kind no prefix test can
catch, because the contamination lives inside a model you did not train. As commentary it
is harmless and useful. As a feature it would silently invalidate every result in the
project.

Without a provider the module still works: :func:`describe_from_data` produces a factual
description computed from the episode's own statistics, with no model involved at all.
That is the default, and it is deterministic.

To enable model commentary, pass a provider -- :func:`openrouter_provider` is included.
It is opt-in, and the pipeline runs identically without a key.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

#: A provider takes a prompt and returns prose. Any LLM client can be adapted to this.
NarrationProvider = Callable[[str], str]

FORBIDDEN_COLUMNS = ("narration", "regime_narration", "commentary", "llm_note")

#: Environment variable holding the OpenRouter API key. Never passed on the command line,
#: so it cannot end up in shell history or a process listing.
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_MODEL_ENV = "OPENROUTER_MODEL"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

#: OpenRouter's catalogue changes over time; override with ``OPENROUTER_MODEL`` or the
#: ``model=`` argument if this identifier is retired.
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"


@dataclass
class RegimeEpisode:
    """One contiguous run of a single regime, with the market context that defines it."""

    regime: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_days: int
    market_return: float
    volatility: float
    max_drawdown: float
    best_day: float
    worst_day: float

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "start": self.start,
            "end": self.end,
            "n_days": self.n_days,
            "market_return": self.market_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "best_day": self.best_day,
            "worst_day": self.worst_day,
        }


def extract_episodes(
    labels: pd.Series,
    market_returns: pd.Series,
    regime_names: Optional[Sequence[str]] = None,
    min_days: int = 5,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Collapse a regime label path into episodes with their realised market statistics.

    Useful on its own, before any narration: a table of "here are the twelve crisis
    episodes and what the market did in each" is a far more checkable object than a
    coloured strip.
    """
    aligned = pd.concat(
        [labels.rename("regime"), market_returns.rename("ret")], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        return pd.DataFrame()

    values = aligned["regime"].to_numpy()
    boundaries = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(values)]])

    episodes: List[RegimeEpisode] = []
    for lo, hi in zip(starts, ends):
        if hi - lo < min_days:
            continue
        block = aligned.iloc[lo:hi]
        returns = block["ret"]
        equity = (1.0 + returns).cumprod()
        index = int(block["regime"].iloc[0])
        name = (
            regime_names[index]
            if regime_names is not None and index < len(regime_names)
            else f"regime_{index}"
        )
        episodes.append(
            RegimeEpisode(
                regime=name,
                start=block.index[0],
                end=block.index[-1],
                n_days=len(block),
                market_return=float(equity.iloc[-1] - 1.0),
                volatility=float(returns.std() * np.sqrt(trading_days)),
                max_drawdown=float((equity / equity.cummax() - 1.0).min()),
                best_day=float(returns.max()),
                worst_day=float(returns.min()),
            )
        )

    return pd.DataFrame([e.to_dict() for e in episodes])


def describe_from_data(episode: pd.Series) -> str:
    """Factual description computed from the episode's own numbers. No model involved.

    This is the default narration. It cannot be wrong about the market because it only
    restates what the data says, and it makes the LLM path optional rather than load-bearing.
    """
    direction = "rose" if episode["market_return"] >= 0 else "fell"
    return (
        f"{episode['n_days']} trading days. The market {direction} "
        f"{abs(episode['market_return']):.1%} at {episode['volatility']:.0%} annualised "
        f"volatility, with a peak-to-trough drawdown of {abs(episode['max_drawdown']):.1%} "
        f"(worst day {episode['worst_day']:.1%}, best {episode['best_day']:+.1%})."
    )


def build_prompt(episode: pd.Series, market: str = "Indian equities (NIFTY 50)") -> str:
    """Prompt asking for factual context, not interpretation.

    Deliberately narrow. It supplies the dates and the realised statistics and asks only
    what was happening in the market — a recall task. It does not ask whether the regime
    label is correct, what to do about it, or what happens next; the moment commentary
    starts making claims about the future it stops being presentation.
    """
    return (
        f"In two sentences, state the main market and macroeconomic events affecting "
        f"{market} between {episode['start']:%d %B %Y} and {episode['end']:%d %B %Y}.\n\n"
        f"Realised over that window: return {episode['market_return']:+.1%}, "
        f"annualised volatility {episode['volatility']:.0%}, "
        f"maximum drawdown {abs(episode['max_drawdown']):.1%}.\n\n"
        f"Report only what occurred. Do not evaluate any trading strategy, do not "
        f"forecast, and do not comment on whether the period was a good time to invest. "
        f"If you are not confident about the events, say so instead of guessing."
    )


# ------------------------------------------------------------------ LLM provider


def openrouter_provider(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    max_tokens: int = 220,
    temperature: float = 0.0,
    max_retries: int = 2,
    backoff: float = 2.0,
) -> NarrationProvider:
    """Build a :data:`NarrationProvider` backed by OpenRouter.

    Uses :mod:`urllib` from the standard library rather than an SDK. Commentary is a
    presentation nicety, and it should not add a dependency that everyone installing this
    project has to carry -- particularly since CI runs with no network at all.

    The key is read from ``$OPENROUTER_API_KEY`` and never accepted as a CLI argument, so
    it stays out of shell history and ``ps`` output. A missing key raises **here**, at
    construction, rather than during the run: 45 episodes silently falling back to the
    data description would look like success.

    ``temperature=0`` because this is a research artefact. Two runs over the same
    episodes should produce the same report, and sampling would make the commentary
    churn on every rebuild for no benefit.

    Transient failures (429, 5xx, connection drops) are retried with exponential backoff.
    Anything still failing after that propagates to :func:`narrate`, which falls back to
    the data description for that episode and marks its ``source`` as ``"data"`` -- so a
    partial outage degrades one row rather than the report.
    """
    key = api_key or os.environ.get(OPENROUTER_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{OPENROUTER_KEY_ENV} is not set. Export it to enable LLM narration, or "
            "omit the flag to use the deterministic data-only descriptions."
        )
    chosen = model or os.environ.get(OPENROUTER_MODEL_ENV) or DEFAULT_OPENROUTER_MODEL

    def provider(prompt: str) -> str:
        payload = json.dumps(
            {
                "model": chosen,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ).encode("utf-8")

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            request = urllib.request.Request(
                OPENROUTER_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    # OpenRouter uses these for attribution on its dashboard; harmless
                    # and it keeps the traffic identifiable as coming from this project.
                    "HTTP-Referer": "https://github.com/tridibjena/nifty50-rl-portfolio-optimization",
                    "X-Title": "nifty50-regime-narration",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return str(body["choices"][0]["message"]["content"]).strip()
            except urllib.error.HTTPError as exc:
                last_error = exc
                # 4xx other than rate-limiting means the request itself is wrong --
                # a bad model id or a revoked key. Retrying cannot fix it.
                if exc.code != 429 and exc.code < 500:
                    raise
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                last_error = exc

            if attempt < max_retries:
                time.sleep(backoff ** attempt)

        raise RuntimeError(f"OpenRouter request failed after {max_retries + 1} attempts") from last_error

    return provider


def narrate(
    episodes: pd.DataFrame,
    provider: Optional[NarrationProvider] = None,
    market: str = "Indian equities (NIFTY 50)",
) -> pd.DataFrame:
    """Attach commentary to each episode.

    With no provider, every row is described from its own statistics and ``source`` is
    ``"data"``. With a provider, ``source`` is ``"llm"`` and a failed call falls back to
    the data description rather than dropping the row -- the report should never depend
    on a network call succeeding.
    """
    if episodes.empty:
        return episodes.assign(narration=pd.Series(dtype=str), source=pd.Series(dtype=str))

    narrations, sources = [], []
    for _, episode in episodes.iterrows():
        fallback = describe_from_data(episode)
        if provider is None:
            narrations.append(fallback)
            sources.append("data")
            continue
        try:
            text = provider(build_prompt(episode, market)).strip()
            narrations.append(f"{fallback} {text}" if text else fallback)
            sources.append("llm" if text else "data")
        except Exception:
            narrations.append(fallback)
            sources.append("data")

    return episodes.assign(narration=narrations, source=sources)


def to_markdown(narrated: pd.DataFrame, regime_order: Optional[Sequence[str]] = None) -> str:
    """Render narrated episodes as a readable timeline for the report."""
    if narrated.empty:
        return "_no episodes met the minimum length_"

    frame = narrated.sort_values("start")
    lines: List[str] = []
    for _, row in frame.iterrows():
        lines.append(
            f"**{row['regime']}** · {row['start']:%d %b %Y} → {row['end']:%d %b %Y}  \n"
            f"{row['narration']}\n"
        )
    if "source" in frame.columns and (frame["source"] == "llm").any():
        lines.append(
            "\n*Commentary is generated after the fact for readability. It is written to "
            "its own artefact and never enters any feature, observation or model input.*"
        )
    return "\n".join(lines)


def assert_no_narration_leak(frame: pd.DataFrame, context: str = "feature panel") -> None:
    """Fail loudly if commentary has found its way into modelling data.

    Called by the test suite against the real feature frame. The point is that a future
    edit which merges narration into features breaks the build rather than quietly
    producing better-looking and completely invalid results.
    """
    present = [c for c in frame.columns if c.lower() in FORBIDDEN_COLUMNS]
    if present:
        raise AssertionError(
            f"Narration column(s) {present} found in the {context}. Commentary is "
            "presentation only -- an LLM's weights encode what happened after the period "
            "it describes, so using it as a feature is lookahead that no prefix test can "
            "detect."
        )
