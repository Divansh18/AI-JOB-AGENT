"""LLM provider abstraction.

Contract for every provider:
  * same input  - (system, user, schema, purpose)
  * same output - a ProviderResult carrying an instance of the requested
    pydantic schema, validated through validate_into() below
  * same errors - the LLMError hierarchy, never a provider-specific exception

Switching providers is a config change. No caller outside this package may
import a concrete provider or shell out to a model runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


# --- errors ----------------------------------------------------------------


class LLMError(Exception):
    """Base for every provider failure. Callers only ever catch this."""


class LLMUnavailableError(LLMError):
    """Provider cannot run: binary missing, key absent, package not installed."""


class LLMTimeoutError(LLMError):
    """Provider exceeded its time budget."""


class LLMInvocationError(LLMError):
    """Provider ran but failed: non-zero exit, API error, empty response."""


class LLMParseError(LLMError):
    """Provider returned output that does not satisfy the requested schema."""


class LLMBudgetError(LLMError):
    """The configured spend cap would be exceeded."""


# --- results ---------------------------------------------------------------


@dataclass
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_is_estimated: bool = True


@dataclass
class ProviderResult:
    """Uniform return value across providers."""

    data: BaseModel
    usage: Usage = field(default_factory=Usage)
    provider: str = ""
    model: str = ""
    prompt_hash: str = ""
    from_cache: bool = False
    raw_text: str = ""


# --- shared output handling ------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json_object(text: str) -> str:
    """Pull a JSON object out of model output.

    Handles: bare JSON, markdown-fenced JSON, and JSON preceded or followed by
    prose. Providers differ in how much wrapping they add, so this normalises
    them before validation.
    """
    if not text or not text.strip():
        raise LLMParseError("empty response")
    candidate = text.strip()

    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate

    # Fall back to the first balanced {...} span.
    start = candidate.find("{")
    if start == -1:
        raise LLMParseError(f"no JSON object in response: {candidate[:160]!r}")
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : i + 1]
    raise LLMParseError(f"unbalanced JSON in response: {candidate[:160]!r}")


def validate_into(schema: type[T], raw_text: str) -> T:
    """Parse and validate provider output into the requested schema.

    Single choke point: every provider calls this, so both return byte-identical
    validated structures or fail identically.
    """
    payload = extract_json_object(raw_text)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMParseError(f"expected a JSON object, got {type(parsed).__name__}")
    try:
        return schema(**parsed)
    except ValidationError as exc:
        raise LLMParseError(f"schema validation failed: {exc.errors()[:3]}") from exc


def prompt_hash(system: str, user: str, model: str) -> str:
    return hashlib.sha256(f"{model}\x00{system}\x00{user}".encode("utf-8")).hexdigest()


# --- protocol --------------------------------------------------------------


class LLMProvider(Protocol):
    name: str
    model: str

    def available(self) -> tuple[bool, str]:
        """Whether this provider can run right now, plus a human reason."""
        ...

    def complete(self, *, system: str, user: str, schema: type[T],
                 purpose: str = "") -> ProviderResult:
        ...


# --- factory ---------------------------------------------------------------

PROVIDERS = ("claude_cli", "anthropic", "openai")


def get_provider(llm_settings) -> LLMProvider:
    """Build the configured provider.

    Concrete providers are imported lazily so that a missing optional
    dependency (an API SDK, or the claude binary) can never break
    import of the deterministic pipeline.
    """
    name = getattr(llm_settings, "provider", "claude_cli")
    if name == "claude_cli":
        from .claude_cli import ClaudeCLIProvider

        return ClaudeCLIProvider(llm_settings)
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(llm_settings)
    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(llm_settings)
    raise LLMUnavailableError(
        f"unknown llm.provider {name!r}; expected one of {list(PROVIDERS)}"
    )
