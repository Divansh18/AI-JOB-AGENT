"""Long-term provider: the Anthropic API via the official SDK.

The API key is read from the environment only. It is never accepted as a
config value, never written to disk, and never included in an error message.
"""

from __future__ import annotations

import os
from typing import TypeVar

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

API_KEY_ENV = "ANTHROPIC_API_KEY"


class AnthropicProvider:
    """Calls the Messages API and validates the response into the schema."""

    name = "anthropic"

    def __init__(self, settings):
        self.settings = settings
        self.model = getattr(settings, "model", "claude-haiku-4-5")
        self.max_tokens = int(getattr(settings, "max_output_tokens", 1024))
        self.timeout = float(getattr(settings, "api_timeout_seconds", 60))
        self._client = None

    # -- availability -------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        if not os.environ.get(API_KEY_ENV, "").strip():
            return False, f"{API_KEY_ENV} is not set in the environment"
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "the 'anthropic' package is not installed"
        return True, f"anthropic SDK ready (model={self.model})"

    def _get_client(self):
        if self._client is None:
            import anthropic

            # Key comes from the environment via the SDK's own resolution.
            self._client = anthropic.Anthropic(timeout=self.timeout)
        return self._client

    # -- completion ---------------------------------------------------------

    def complete(self, *, system: str, user: str, schema: type[T],
                 purpose: str = "") -> ProviderResult:
        ok, reason = self.available()
        if not ok:
            raise LLMUnavailableError(reason)

        import anthropic

        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        # Stable prefix: the profile block is identical across
                        # every job, so caching it cuts input cost materially.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(f"anthropic API timed out after {self.timeout:.0f}s") from exc
        except anthropic.APIStatusError as exc:
            raise LLMInvocationError(
                f"anthropic API returned {exc.status_code} for purpose={purpose!r}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMInvocationError(f"could not reach the anthropic API: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMInvocationError("anthropic API declined the request")

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        if not text.strip():
            raise LLMParseError("anthropic API returned no text content")

        data = validate_into(schema, text)  # shared validation path

        u = response.usage
        input_tokens = int(getattr(u, "input_tokens", 0) or 0)
        cached = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        output_tokens = int(getattr(u, "output_tokens", 0) or 0)

        usage = Usage(
            input_tokens=input_tokens,
            cached_input_tokens=cached,
            output_tokens=output_tokens,
            cost_usd=estimate_cost_usd(
                self.model, input_tokens=input_tokens, output_tokens=output_tokens,
                cached_input_tokens=cached, cache_write_tokens=cache_write,
            ),
            cost_is_estimated=False,
        )
        return ProviderResult(
            data=data, usage=usage, provider=self.name, model=self.model,
            prompt_hash=prompt_hash(system, user, self.model),
            raw_text=text[:2000],
        )
