"""Provider-layer contract tests.

Both providers must produce identical validated structures and identical
error types. Nothing here makes a network call or spawns a real subprocess.
"""

import json
import subprocess
import sys
import types

import pytest

from jobsearch.config.schemas import LlmSettings
from jobsearch.llm.provider import (
    LLMInvocationError, LLMParseError, LLMTimeoutError, LLMUnavailableError,
    extract_json_object, get_provider, validate_into,
)
from jobsearch.llm.schemas import LlmJobInsightExtraction, TriageResult

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
    assert type(get_provider(LlmSettings(provider="openai"))).__name__ == "OpenAIProvider"


def test_unknown_provider_raises_unavailable():
    class Bad:
        provider = "bogus"
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
    r = validate_into(TriageResult, json.dumps(VALID))
    assert isinstance(r, TriageResult) and r.fit_score == 72


def test_validate_into_rejects_bad_enum_and_range():
    with pytest.raises(LLMParseError):
        validate_into(TriageResult, json.dumps({**VALID, "verdict": "amazing"}))
    with pytest.raises(LLMParseError):
        validate_into(TriageResult, json.dumps({**VALID, "fit_score": 500}))


def test_validate_into_rejects_non_object():
    with pytest.raises(LLMParseError):
        validate_into(TriageResult, "[1, 2, 3]")


def test_both_providers_share_one_validation_path():
    """Guards against the two providers drifting apart."""
    from jobsearch.llm import anthropic_provider, claude_cli, openai_provider
    for module in (claude_cli, anthropic_provider, openai_provider):
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
    for banned in ("api_key", "anthropic_api_key", "openai_api_key", "key", "token", "secret"):
        assert banned not in fields, f"{banned} must not be configurable"


# --- openai provider -------------------------------------------------------

def _openai(**kw):
    from jobsearch.llm.openai_provider import OpenAIProvider
    return OpenAIProvider(LlmSettings(provider="openai", model="gpt-4.1-mini", **kw))


def _openai_response(text=None, *, usage=None, status=None, output=None):
    return types.SimpleNamespace(
        output_text=text,
        output=output or [],
        usage=usage,
        status=status,
    )


def _install_fake_openai(monkeypatch, *, response=None):
    calls = []
    state = {"response": response, "error": None}

    class APIError(Exception):
        pass

    class APITimeoutError(APIError):
        pass

    class APIConnectionError(APIError):
        pass

    class APIStatusError(APIError):
        def __init__(self, message="status", status_code=500, body=None):
            super().__init__(message)
            self.status_code = status_code
            self.body = body

    class AuthenticationError(APIStatusError):
        pass

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            if state["error"] is not None:
                raise state["error"]
            return state["response"]

    class OpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.responses = Responses()

    module = types.SimpleNamespace(
        OpenAI=OpenAI,
        APIError=APIError,
        APITimeoutError=APITimeoutError,
        APIConnectionError=APIConnectionError,
        APIStatusError=APIStatusError,
        AuthenticationError=AuthenticationError,
    )
    monkeypatch.setitem(sys.modules, "openai", module)
    return module, calls, state


def test_openai_unavailable_without_env_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = _openai()
    ok, reason = p.available()
    assert not ok and "OPENAI_API_KEY" in reason
    with pytest.raises(LLMUnavailableError):
        p.complete(system="s", user="u", schema=TriageResult)


def test_openai_available_with_env_key_and_sdk(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _install_fake_openai(monkeypatch)
    p = _openai()
    ok, reason = p.available()
    assert ok and "gpt-4.1-mini" in reason


def test_openai_returns_validated_schema_and_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    usage = types.SimpleNamespace(
        input_tokens=1000,
        output_tokens=250,
        input_tokens_details=types.SimpleNamespace(cached_tokens=100),
    )
    _module, calls, state = _install_fake_openai(monkeypatch)
    state["response"] = _openai_response(json.dumps(VALID), usage=usage)

    res = _openai(max_output_tokens=321).complete(
        system="s", user="u", schema=TriageResult, purpose="triage"
    )

    assert isinstance(res.data, TriageResult)
    assert res.provider == "openai"
    assert res.model == "gpt-4.1-mini"
    assert res.usage.input_tokens == 1000
    assert res.usage.cached_input_tokens == 100
    assert res.usage.output_tokens == 250
    assert res.usage.cost_usd > 0
    assert calls[0]["max_output_tokens"] == 321
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert calls[0]["text"]["format"]["name"] == "TriageResult"
    assert [m["role"] for m in calls[0]["input"]] == ["system", "user"]


def test_openai_structured_output_request_uses_strict_responses_schema(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    valid = {
        "role_summary": "Backend SDE role building data products.",
        "must_have_requirements": [
            {
                "text": "Backend software engineering",
                "category": "skill",
                "candidate_match": "matched",
                "evidence_refs": ["fact:1"],
                "rationale": "Verified backend evidence is present.",
            }
        ],
        "preferred_requirements": [],
        "role_priorities": ["backend engineering"],
        "grounded_fit_assessment": "The supplied evidence supports backend work.",
        "fit_verdict": "worth_applying",
        "candidate_match_evidence_refs": ["fact:1"],
        "uncertainties": [],
        "gaps": [],
    }
    _module, calls, state = _install_fake_openai(monkeypatch)
    state["response"] = _openai_response(json.dumps(valid))

    _openai().complete(
        system="system prompt", user="user prompt",
        schema=LlmJobInsightExtraction, purpose="job_intelligence"
    )

    fmt = calls[0]["text"]["format"]
    schema = fmt["schema"]
    assert fmt == {
        "type": "json_schema",
        "name": "LlmJobInsightExtraction",
        "schema": schema,
        "strict": True,
    }
    assert '"default"' not in json.dumps(schema)
    assert '"title"' not in json.dumps(schema)
    object_schemas = _object_schemas(schema)
    assert object_schemas
    for item in object_schemas:
        assert item["additionalProperties"] is False
        assert set(item["required"]) == set(item["properties"])


def test_openai_reads_nested_output_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    output = [
        {
            "content": [
                {"type": "output_text", "text": json.dumps(VALID)},
            ]
        }
    ]
    _module, _calls, state = _install_fake_openai(monkeypatch)
    state["response"] = _openai_response(output=output)

    res = _openai().complete(system="s", user="u", schema=TriageResult)

    assert res.data.fit_score == 72


def test_openai_malformed_response_is_translated(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _module, _calls, state = _install_fake_openai(monkeypatch)
    state["response"] = _openai_response("not json")
    with pytest.raises(LLMParseError):
        _openai().complete(system="s", user="u", schema=TriageResult)


def test_openai_authentication_failure_is_unavailable(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    module, _calls, state = _install_fake_openai(monkeypatch)
    state["error"] = module.AuthenticationError("bad auth", status_code=401)
    with pytest.raises(LLMUnavailableError):
        _openai().complete(system="s", user="u", schema=TriageResult)


def test_openai_provider_failure_is_invocation_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    module, _calls, state = _install_fake_openai(monkeypatch)
    state["error"] = module.APIStatusError("server error", status_code=500)
    with pytest.raises(LLMInvocationError):
        _openai().complete(system="s", user="u", schema=TriageResult, purpose="triage")


def test_openai_status_error_surfaces_safe_debug_fields(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    module, _calls, state = _install_fake_openai(monkeypatch)
    state["error"] = module.APIStatusError(
        "bad request includes sk-should-not-leak-1234567890",
        status_code=400,
        body={
            "error": {
                "message": (
                    "Invalid schema for response_format "
                    "sk-should-not-leak-1234567890"
                ),
                "type": "invalid_request_error",
                "param": "text.format.schema",
                "code": "invalid_json_schema",
            }
        },
    )

    with pytest.raises(LLMInvocationError) as exc:
        _openai().complete(
            system="candidate prompt must not appear",
            user="job prompt must not appear",
            schema=TriageResult,
            purpose="job_intelligence",
        )

    message = str(exc.value)
    assert "openai API returned 400" in message
    assert "purpose='job_intelligence'" in message
    assert "type=invalid_request_error" in message
    assert "code=invalid_json_schema" in message
    assert "param=text.format.schema" in message
    assert "Invalid schema for response_format" in message
    assert "sk-" not in message
    assert "candidate prompt" not in message
    assert "job prompt" not in message


def test_openai_timeout_is_translated(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    module, _calls, state = _install_fake_openai(monkeypatch)
    state["error"] = module.APITimeoutError("timeout")
    with pytest.raises(LLMTimeoutError):
        _openai().complete(system="s", user="u", schema=TriageResult)


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


def _object_schemas(schema):
    found = []
    if isinstance(schema, dict):
        if isinstance(schema.get("properties"), dict):
            found.append(schema)
        for value in schema.values():
            found.extend(_object_schemas(value))
    elif isinstance(schema, list):
        for value in schema:
            found.extend(_object_schemas(value))
    return found
