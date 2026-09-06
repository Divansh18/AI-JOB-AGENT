"""OpenAI API provider via the official SDK.

The API key is read from OPENAI_API_KEY by the SDK/environment only. It is
never accepted as a config value, written to disk, or included in errors.
"""

from __future__ import annotations

import os
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from .cost import estimate_cost_usd
from .provider import (
    LLMInvocationError,
    LLMParseError,
    LLMTimeoutError,
    LLMUnavailableError,
    ProviderResult,
    Usage,
    prompt_hash,
    validate_into,
)

T = TypeVar("T", bound=BaseModel)

API_KEY_ENV = "OPENAI_API_KEY"
_SECRET_SHAPE = re.compile(r"sk-[A-Za-z0-9_-]{12,}")


class OpenAIProvider:
    """Calls the Responses API and validates structured JSON output."""

    name = "openai"

    def __init__(self, settings):
        self.settings = settings
        self.model = getattr(settings, "model", "gpt-4.1-mini")
        self.max_tokens = int(getattr(settings, "max_output_tokens", 1024))
        self.timeout = float(getattr(settings, "api_timeout_seconds", 60))
        self._client = None

    # -- availability -------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        if not os.environ.get(API_KEY_ENV, "").strip():
            return False, f"{API_KEY_ENV} is not set in the environment"
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "the 'openai' package is not installed"
        return True, f"openai SDK ready (model={self.model})"

    def _get_client(self):
        if self._client is None:
            import openai

            # Key comes from OPENAI_API_KEY via the SDK's own resolution.
            self._client = openai.OpenAI(timeout=self.timeout)
        return self._client

    # -- completion ---------------------------------------------------------

    def complete(self, *, system: str, user: str, schema: type[T],
                 purpose: str = "") -> ProviderResult:
        ok, reason = self.available()
        if not ok:
            raise LLMUnavailableError(reason)

        import openai

        client = self._get_client()
        try:
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_output_tokens=self.max_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": _schema_name(schema),
                        "schema": _openai_strict_json_schema(schema),
                        "strict": True,
                    }
                },
            )
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError(f"openai API timed out after {self.timeout:.0f}s") from exc
        except openai.AuthenticationError as exc:
            raise LLMUnavailableError("openai API authentication failed") from exc
        except openai.APIStatusError as exc:
            raise LLMInvocationError(_status_error_message(exc, purpose=purpose)) from exc
        except openai.APIConnectionError as exc:
            raise LLMInvocationError(
                f"could not reach the openai API: {_scrub(str(exc))}"
            ) from exc
        except openai.APIError as exc:
            raise LLMInvocationError(f"openai API error for purpose={purpose!r}") from exc

        if getattr(response, "status", None) == "failed":
            raise LLMInvocationError("openai API failed the request")

        text = _response_text(response)
        if not text.strip():
            raise LLMParseError("openai API returned no text content")

        data = validate_into(schema, text)  # shared validation path

        usage = _usage(response, self.model)
        return ProviderResult(
            data=data, usage=usage, provider=self.name, model=self.model,
            prompt_hash=prompt_hash(system, user, self.model),
            raw_text=text[:2000],
        )


def _schema_name(schema: type[BaseModel]) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", schema.__name__).strip("_")
    return (name or "structured_output")[:64]


def _openai_strict_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Return the Responses API strict JSON-schema subset for a Pydantic model."""
    try:
        from openai.lib._pydantic import to_strict_json_schema

        raw = to_strict_json_schema(schema)
    except Exception:
        raw = schema.model_json_schema()

    return _strip_schema_unsupported_keywords(_ensure_openai_strict_schema(raw))


def _ensure_openai_strict_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [_ensure_openai_strict_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    properties = node.get("properties")
    if isinstance(properties, dict):
        node["required"] = list(properties.keys())
        node["additionalProperties"] = False
        node["properties"] = {
            key: _ensure_openai_strict_schema(value)
            for key, value in properties.items()
        }

    for key in ("$defs", "definitions"):
        definitions = node.get(key)
        if isinstance(definitions, dict):
            node[key] = {
                name: _ensure_openai_strict_schema(value)
                for name, value in definitions.items()
            }

    items = node.get("items")
    if isinstance(items, dict):
        node["items"] = _ensure_openai_strict_schema(items)

    for key in ("anyOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            node[key] = [_ensure_openai_strict_schema(value) for value in variants]

    if node.get("type") == "object":
        node["additionalProperties"] = False

    return node


def _strip_schema_unsupported_keywords(node: Any) -> Any:
    if isinstance(node, list):
        return [_strip_schema_unsupported_keywords(item) for item in node]
    if not isinstance(node, dict):
        return node

    # The Responses structured-output subset rejects Pydantic defaults. Titles
    # are also not useful to generation and only bloat the request.
    node.pop("default", None)
    node.pop("title", None)
    for value in node.values():
        _strip_schema_unsupported_keywords(value)
    return node


def _response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str):
        return direct

    chunks: list[str] = []
    for item in _get(response, "output", []) or []:
        for content in _get(item, "content", []) or []:
            text = _get(content, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


def _usage(response: Any, model: str) -> Usage:
    u = _get(response, "usage", None)
    input_tokens = _int(_get(u, "input_tokens", _get(u, "prompt_tokens", 0)))
    output_tokens = _int(_get(u, "output_tokens", _get(u, "completion_tokens", 0)))
    details = _get(u, "input_tokens_details", _get(u, "prompt_tokens_details", None))
    cached = _int(_get(details, "cached_tokens", 0))
    return Usage(
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        cost_usd=estimate_cost_usd(
            model, input_tokens=input_tokens, output_tokens=output_tokens,
            cached_input_tokens=cached,
        ),
        cost_is_estimated=False,
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _status_error_message(exc: Exception, *, purpose: str) -> str:
    details = _status_error_details(exc)
    parts = [
        f"openai API returned {details['status']} for purpose={purpose!r}",
    ]
    for key in ("type", "code", "param", "message"):
        value = details.get(key)
        if value:
            parts.append(f"{key}={value}")
    return "; ".join(parts)


def _status_error_details(exc: Exception) -> dict[str, str]:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    body = getattr(exc, "body", None)
    if body is None and response is not None:
        json_method = getattr(response, "json", None)
        if callable(json_method):
            try:
                body = json_method()
            except Exception:
                body = None

    error = body.get("error", body) if isinstance(body, dict) else {}
    message = _safe_text(
        _get(error, "message", "") or getattr(exc, "message", "") or str(exc)
    )
    return {
        "status": _safe_text(str(status or "unknown"), limit=40),
        "type": _safe_text(str(_get(error, "type", "") or ""), limit=120),
        "code": _safe_text(str(_get(error, "code", "") or ""), limit=120),
        "param": _safe_text(str(_get(error, "param", "") or ""), limit=160),
        "message": message,
    }


def _safe_text(text: str, *, limit: int = 700) -> str:
    compact = " ".join(_scrub(str(text or "")).split())
    return compact[:limit]


def _scrub(text: str) -> str:
    return _SECRET_SHAPE.sub("[redacted]", text)
