"""Narration is presentation. These tests keep it that way.

The risk is not that commentary is wrong -- it is that someone later finds it useful and
joins it into the feature panel. An LLM's weights encode what happened *after* the period
it describes, so a narration-derived feature is lookahead that `assert_causal` cannot
detect: the contamination lives inside a model nobody in this repo trained.
"""

from __future__ import annotations

import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

from nifty_rl.report import narrate as narrate_module
from nifty_rl.report.narrate import (
    OPENROUTER_KEY_ENV,
    assert_no_narration_leak,
    build_prompt,
    extract_episodes,
    narrate,
    openrouter_provider,
    to_markdown,
)


@pytest.fixture
def labelled():
    dates = pd.bdate_range("2022-01-03", periods=60)
    rng = np.random.default_rng(0)
    returns = pd.Series(
        np.concatenate([rng.normal(0.002, 0.004, 30), rng.normal(-0.004, 0.020, 30)]),
        index=dates,
    )
    labels = pd.Series([0] * 30 + [2] * 30, index=dates)
    return labels, returns


def test_episodes_carry_their_market_context(labelled):
    labels, returns = labelled
    episodes = extract_episodes(labels, returns, regime_names=["Calm", "Normal", "Crisis"])

    assert list(episodes["regime"]) == ["Calm", "Crisis"]
    calm, crisis = episodes.iloc[0], episodes.iloc[1]
    assert crisis["volatility"] > calm["volatility"]
    assert crisis["max_drawdown"] < calm["max_drawdown"]


def test_short_episodes_are_dropped(labelled):
    """A one-day blip is dropped, but it still splits the run it interrupts.

    Episodes are contiguous by definition, so the 30-day crisis becomes a 15-day and a
    14-day episode either side of the blip -- three surviving episodes, not two. Dropping
    a short episode is not the same as smoothing it away, and conflating the two would
    misreport how often the detector actually changed its mind.
    """
    labels, returns = labelled
    labels.iloc[45] = 1  # one-day blip inside the crisis run

    episodes = extract_episodes(labels, returns, min_days=10)

    assert len(episodes) == 3
    assert (episodes["n_days"] >= 10).all(), "the one-day blip must not survive"
    assert list(episodes["n_days"]) == [30, 15, 14]


def test_default_narration_uses_no_model(labelled):
    """With no provider the description is computed from the episode's own numbers."""
    labels, returns = labelled
    narrated = narrate(extract_episodes(labels, returns, regime_names=["Calm", "N", "Crisis"]))

    assert set(narrated["source"]) == {"data"}
    assert all("trading days" in n for n in narrated["narration"])


def test_provider_failure_falls_back_rather_than_breaking_the_report(labelled):
    """A report must not depend on a network call succeeding."""
    labels, returns = labelled

    def broken(_prompt):
        raise RuntimeError("api down")

    narrated = narrate(extract_episodes(labels, returns), provider=broken)
    assert set(narrated["source"]) == {"data"}
    assert narrated["narration"].notna().all()


def test_provider_output_is_appended_to_the_factual_description(labelled):
    labels, returns = labelled
    narrated = narrate(extract_episodes(labels, returns), provider=lambda p: "Rates rose.")

    assert set(narrated["source"]) == {"llm"}
    for text in narrated["narration"]:
        assert "trading days" in text and "Rates rose." in text


def test_prompt_asks_for_recall_not_forecasts_or_advice(labelled):
    labels, returns = labelled
    prompt = build_prompt(extract_episodes(labels, returns).iloc[0])

    assert "Do not evaluate any trading strategy" in prompt
    assert "do not forecast" in prompt.lower()
    # The prompt supplies realised statistics and dates only.
    assert "Report only what occurred" in prompt


# --------------------------------------------------------------- the actual guard


def test_narration_columns_are_rejected_in_modelling_data():
    frame = pd.DataFrame({"ret": [0.01], "rsi": [50.0], "narration": ["Crisis, rates rose"]})
    with pytest.raises(AssertionError, match="presentation only"):
        assert_no_narration_leak(frame)


def test_clean_feature_frame_passes_the_guard():
    assert_no_narration_leak(pd.DataFrame({"ret": [0.01], "rsi": [50.0]}))


def test_real_feature_panel_contains_no_commentary():
    """Run the guard against the actual feature builder, not a hand-made frame."""
    from conftest import make_panel
    from nifty_rl.features import add_features

    featured = add_features(make_panel({f"T{i}.NS": list(np.linspace(100, 130, 200)) for i in range(2)}))
    assert_no_narration_leak(featured, context="add_features output")


# ------------------------------------------------------- the OpenRouter provider
#
# No test here touches the network. They exercise request construction, failure
# handling and secret hygiene against a stubbed urlopen.


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_urlopen(monkeypatch, handler):
    """Replace urlopen and record every request the provider makes."""
    captured = []

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        return handler(request, len(captured) - 1)

    monkeypatch.setattr(narrate_module.urllib.request, "urlopen", fake_urlopen)
    return captured


def _ok(_request, _attempt):
    return _FakeResponse({"choices": [{"message": {"content": "Rates rose sharply."}}]})


def test_missing_key_fails_at_construction_not_mid_run(monkeypatch):
    """45 episodes silently falling back to data would look like a successful run."""
    monkeypatch.delenv(OPENROUTER_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=OPENROUTER_KEY_ENV):
        openrouter_provider()


def test_key_is_read_from_the_environment_and_sent_as_a_bearer_token(monkeypatch):
    monkeypatch.setenv(OPENROUTER_KEY_ENV, "sk-test-secret")
    captured = _stub_urlopen(monkeypatch, _ok)

    assert openrouter_provider()("prompt") == "Rates rose sharply."

    assert captured[0].headers["Authorization"] == "Bearer sk-test-secret"


def test_request_is_deterministic_and_uses_the_configured_model(monkeypatch):
    """Temperature 0 so rebuilding the report does not churn the commentary."""
    monkeypatch.setenv(OPENROUTER_KEY_ENV, "k")
    captured = _stub_urlopen(monkeypatch, _ok)

    openrouter_provider(model="vendor/some-model")("the prompt")

    body = json.loads(captured[0].data)
    assert body["model"] == "vendor/some-model"
    assert body["temperature"] == 0.0
    assert body["messages"] == [{"role": "user", "content": "the prompt"}]


def test_explicit_model_beats_env_which_beats_the_default(monkeypatch):
    monkeypatch.setenv(OPENROUTER_KEY_ENV, "k")
    monkeypatch.setenv(narrate_module.OPENROUTER_MODEL_ENV, "vendor/from-env")
    captured = _stub_urlopen(monkeypatch, _ok)

    openrouter_provider()("p")
    openrouter_provider(model="vendor/explicit")("p")

    assert json.loads(captured[0].data)["model"] == "vendor/from-env"
    assert json.loads(captured[1].data)["model"] == "vendor/explicit"


def test_rate_limits_are_retried_but_bad_requests_are_not(monkeypatch):
    monkeypatch.setenv(OPENROUTER_KEY_ENV, "k")
    monkeypatch.setattr(narrate_module.time, "sleep", lambda _s: None)

    def rate_limited_once(_request, attempt):
        if attempt == 0:
            raise urllib.error.HTTPError(narrate_module.OPENROUTER_URL, 429, "slow down", {}, None)
        return _ok(_request, attempt)

    calls = _stub_urlopen(monkeypatch, rate_limited_once)
    assert openrouter_provider()("p") == "Rates rose sharply."
    assert len(calls) == 2, "a 429 should be retried"

    # A revoked key or a retired model id is not transient; retrying cannot fix it.
    def unauthorised(_request, _attempt):
        raise urllib.error.HTTPError(narrate_module.OPENROUTER_URL, 401, "nope", {}, None)

    calls = _stub_urlopen(monkeypatch, unauthorised)
    with pytest.raises(urllib.error.HTTPError):
        openrouter_provider()("p")
    assert len(calls) == 1, "a 401 must not be retried"


def test_a_dead_endpoint_degrades_the_row_not_the_report(monkeypatch, labelled):
    """The whole point of the fallback: an outage costs prose, never numbers."""
    monkeypatch.setenv(OPENROUTER_KEY_ENV, "k")
    monkeypatch.setattr(narrate_module.time, "sleep", lambda _s: None)

    def always_down(_request, _attempt):
        raise urllib.error.URLError("connection refused")

    _stub_urlopen(monkeypatch, always_down)
    labels, returns = labelled

    narrated = narrate(extract_episodes(labels, returns), provider=openrouter_provider())

    assert set(narrated["source"]) == {"data"}
    assert narrated["narration"].notna().all()
    assert all("trading days" in text for text in narrated["narration"])


def test_the_api_key_never_reaches_a_published_artefact(monkeypatch, labelled):
    monkeypatch.setenv(OPENROUTER_KEY_ENV, "sk-secret-value")
    _stub_urlopen(monkeypatch, _ok)
    labels, returns = labelled

    narrated = narrate(extract_episodes(labels, returns), provider=openrouter_provider())

    assert set(narrated["source"]) == {"llm"}
    rendered = to_markdown(narrated) + narrated.to_csv()
    assert "sk-secret-value" not in rendered


def test_markdown_flags_generated_commentary(labelled):
    labels, returns = labelled
    narrated = narrate(extract_episodes(labels, returns), provider=lambda p: "Context.")
    rendered = to_markdown(narrated)
    assert "generated after the fact" in rendered
    assert "never enters any feature" in rendered
