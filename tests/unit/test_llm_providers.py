"""Provider-layer contract tests.

Both providers must produce identical validated structures and identical
error types. Nothing here makes a network call or spawns a real subprocess.
"""

import subprocess

import pytest

from jobsearch.config.schemas import LlmSettings
from jobsearch.llm.provider import (
    LLMInvocationError, LLMParseError, LLMTimeoutError, LLMUnavailableError,
    extract_json_object, get_provider, validate_into,
)
from jobsearch.llm.schemas import TriageResult

VALID = {
    "fit_score": 72, "verdict": "worth_applying",
    "matched_requirements": ["python", "fastapi"],
    "missing_requirements": ["kubernetes"],
    "seniority_assessment": "early_career", "red_flags": [],
    "one_line_rationale": "Good stack overlap.",
}


# --- factory ---------------------------------------------------------------

def test_provider_selection_is_config_driven():
    assert type(get_provider(LlmSettings(provider="claude_cli"))).__name__ == "ClaudeCLIProvider"
    assert type(get_provider(LlmSettings(provider="anthropic"))).__name__ == "AnthropicProvider"


def test_unknown_provider_raises_unavailable():
    class Bad:
        provider = "openai"
    with pytest.raises(LLMUnavailableError):
        get_provider(Bad())


def test_config_rejects_unknown_provider():
    with pytest.raises(Exception):
        LlmSettings(provider="gpt")


# --- shared output handling ------------------------------------------------

@pytest.mark.parametrize("raw", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '```\n{"a": 1}\n```',
    'Here is the result:\n{"a": 1}\nHope that helps.',
    '  \n {"a": 1}  ',
])
def test_extract_json_handles_provider_wrapping(raw):
    assert extract_json_object(raw).strip() == '{"a": 1}'


def test_extract_json_handles_nested_and_stringed_braces():
    raw = 'text {"a": {"b": 2}, "c": "} not the end"} tail'
    assert '"c"' in extract_json_object(raw)


@pytest.mark.parametrize("raw", ["", "   ", "no json here", "{unbalanced"])
def test_extract_json_rejects_garbage(raw):
    with pytest.raises(LLMParseError):
        extract_json_object(raw)


def test_validate_into_accepts_valid_payload():
    import json
    r = validate_into(TriageResult, json.dumps(VALID))
    assert isinstance(r, TriageResult) and r.fit_score == 72


def test_validate_into_rejects_bad_enum_and_range():
    import json
    with pytest.raises(LLMParseError):
        validate_into(TriageResult, json.dumps({**VALID, "verdict": "amazing"}))
    with pytest.raises(LLMParseError):
        validate_into(TriageResult, json.dumps({**VALID, "fit_score": 500}))


def test_validate_into_rejects_non_object():
    with pytest.raises(LLMParseError):
        validate_into(TriageResult, "[1, 2, 3]")


def test_both_providers_share_one_validation_path():
    """Guards against the two providers drifting apart."""
    from jobsearch.llm import anthropic_provider, claude_cli
    for module in (claude_cli, anthropic_provider):
        src = __import__("pathlib").Path(module.__file__).read_text()
        assert "validate_into(schema," in src, f"{module.__name__} must use validate_into"


# --- claude CLI provider error handling ------------------------------------

def _cli(**kw):
    from jobsearch.llm.claude_cli import ClaudeCLIProvider
    return ClaudeCLIProvider(LlmSettings(provider="claude_cli", **kw))


def test_cli_unavailable_when_binary_missing():
    p = _cli(cli_binary="definitely-not-a-real-binary-xyz")
    ok, reason = p.available()
    assert not ok and "not found" in reason
    with pytest.raises(LLMUnavailableError):
        p.complete(system="s", user="u", schema=TriageResult)


def test_cli_timeout_is_translated(monkeypatch):
    p = _cli()
    monkeypatch.setattr(p, "available", lambda: (True, "stub"))
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)
    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(LLMTimeoutError):
        p.complete(system="s", user="u", schema=TriageResult)


def test_cli_nonzero_exit_is_translated(monkeypatch):
    p = _cli()
    monkeypatch.setattr(p, "available", lambda: (True, "stub"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        args=a, returncode=2, stdout="", stderr="boom"))
    with pytest.raises(LLMInvocationError):
        p.complete(system="s", user="u", schema=TriageResult)


def test_cli_malformed_envelope_is_translated(monkeypatch):
    p = _cli()
    monkeypatch.setattr(p, "available", lambda: (True, "stub"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        args=a, returncode=0, stdout="not json at all", stderr=""))
    with pytest.raises(LLMParseError):
        p.complete(system="s", user="u", schema=TriageResult)


def test_cli_error_envelope_is_translated(monkeypatch):
    import json
    p = _cli()
    monkeypatch.setattr(p, "available", lambda: (True, "stub"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        args=a, returncode=0, stderr="",
        stdout=json.dumps({"is_error": True, "result": "rate limited"})))
    with pytest.raises(LLMInvocationError):
        p.complete(system="s", user="u", schema=TriageResult)


def test_cli_returns_validated_schema_from_fenced_output(monkeypatch):
    """The CLI wraps JSON in markdown fences; that must still validate."""
    import json
    p = _cli()
    monkeypatch.setattr(p, "available", lambda: (True, "stub"))
    envelope = {"is_error": False, "subtype": "success",
                "result": "```json\n" + json.dumps(VALID) + "\n```",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "total_cost_usd": 0.001}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        args=a, returncode=0, stdout=json.dumps(envelope), stderr=""))
    res = p.complete(system="s", user="u", schema=TriageResult)
    assert isinstance(res.data, TriageResult)
    assert res.data.verdict == "worth_applying"
    assert res.provider == "claude_cli"
    assert res.usage.cost_is_estimated is True


def test_cli_child_env_never_carries_secrets(monkeypatch):
    from jobsearch.llm.claude_cli import _safe_env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-propagate")
    monkeypatch.setenv("ADZUNA_APP_KEY", "secret")
    env = _safe_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "ADZUNA_APP_KEY" not in env


def test_error_text_is_scrubbed_of_key_shapes():
    from jobsearch.llm.claude_cli import _scrub
    out = _scrub("failed with sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIII rest")
    assert "sk-ant-" not in out and "[redacted]" in out


# --- anthropic provider ----------------------------------------------------

def test_anthropic_unavailable_without_env_key(monkeypatch):
    from jobsearch.llm.anthropic_provider import AnthropicProvider
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = AnthropicProvider(LlmSettings(provider="anthropic"))
    ok, reason = p.available()
    assert not ok and "ANTHROPIC_API_KEY" in reason
    with pytest.raises(LLMUnavailableError):
        p.complete(system="s", user="u", schema=TriageResult)


def test_anthropic_key_is_never_a_config_field():
    fields = set(LlmSettings.model_fields)
    for banned in ("api_key", "anthropic_api_key", "key", "token", "secret"):
        assert banned not in fields, f"{banned} must not be configurable"


# --- cost ------------------------------------------------------------------

def test_cost_uses_published_rates_and_cache_multipliers():
    from jobsearch.llm.cost import estimate_cost_usd
    plain = estimate_cost_usd("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=0)
    assert plain == pytest.approx(1.00, rel=1e-3)
    out = estimate_cost_usd("claude-haiku-4-5", input_tokens=0, output_tokens=1_000_000)
    assert out == pytest.approx(5.00, rel=1e-3)
    cached = estimate_cost_usd("claude-haiku-4-5", input_tokens=1_000_000,
                               output_tokens=0, cached_input_tokens=1_000_000)
    assert cached < plain
