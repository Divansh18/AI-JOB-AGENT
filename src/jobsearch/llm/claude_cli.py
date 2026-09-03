"""Temporary provider: the local Claude Code CLI.

This is the ONLY module in the codebase permitted to invoke the `claude`
executable. It exists so development can proceed against a local runtime
before the Anthropic API key is in place, and is expected to be retired -
nothing here is a permanent dependency, and no other module imports it
directly (see llm.provider.get_provider).

Cost note: the CLI reports session cost that includes the harness's own
cached context, so its USD figures are NOT representative of API pricing.
They are recorded as estimates and flagged as such.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import TypeVar

from pydantic import BaseModel

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

# Environment variables that must never be forwarded or echoed.
_SENSITIVE_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ADZUNA_APP_KEY")


def _safe_env() -> dict:
    """Child environment with secrets stripped.

    The CLI authenticates through its own stored credentials; it does not need
    our API key, and not forwarding it removes any chance of leaking it into a
    subprocess argument list or crash dump.
    """
    env = {k: v for k, v in os.environ.items() if k not in _SENSITIVE_ENV}
    env["CLAUDE_CODE_DISABLE_TELEMETRY"] = env.get("CLAUDE_CODE_DISABLE_TELEMETRY", "1")
    return env


def _scrub(text: str, limit: int = 400) -> str:
    """Truncate provider output for logging and remove anything key-shaped."""
    if not text:
        return ""
    cleaned = text
    for marker in ("sk-ant-", "sk-"):
        idx = cleaned.find(marker)
        while idx != -1:
            cleaned = cleaned[:idx] + "[redacted]" + cleaned[idx + 40 :]
            idx = cleaned.find(marker)
    return cleaned.strip()[:limit]


class ClaudeCLIProvider:
    """Drives `claude -p` and validates its output into the requested schema."""

    name = "claude_cli"

    def __init__(self, settings):
        self.settings = settings
        self.model = getattr(settings, "model", "claude-haiku-4-5")
        self.binary = getattr(settings, "cli_binary", "claude")
        self.timeout = float(getattr(settings, "cli_timeout_seconds", 120))

    # -- availability -------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        path = shutil.which(self.binary)
        if not path:
            return False, f"{self.binary!r} not found on PATH"
        return True, f"{self.binary} at {path}"

    def _cli_model(self) -> str:
        """Map a full model id onto the CLI's short alias."""
        m = (self.model or "").lower()
        for alias in ("haiku", "sonnet", "opus"):
            if alias in m:
                return alias
        return self.model

    # -- completion ---------------------------------------------------------

    def complete(self, *, system: str, user: str, schema: type[T],
                 purpose: str = "") -> ProviderResult:
        ok, reason = self.available()
        if not ok:
            raise LLMUnavailableError(reason)

        prompt = (
            f"{system}\n\n{user}\n\n"
            "Respond with a single JSON object and nothing else. "
            "No preamble, no explanation, no markdown fences."
        )
        cmd = [
            self.binary, "-p", prompt,
            "--output-format", "json",
            "--model", self._cli_model(),
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout, env=_safe_env(), check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMTimeoutError(
                f"claude CLI exceeded {self.timeout:.0f}s for purpose={purpose!r}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise LLMInvocationError(f"could not run claude CLI: {exc}") from exc

        if proc.returncode != 0:
            raise LLMInvocationError(
                f"claude CLI exited {proc.returncode}: {_scrub(proc.stderr or proc.stdout)}"
            )

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise LLMParseError(
                f"claude CLI did not return a JSON envelope: {_scrub(proc.stdout)}"
            ) from exc

        if envelope.get("is_error") or envelope.get("subtype") not in (None, "success"):
            raise LLMInvocationError(
                f"claude CLI reported an error: {_scrub(str(envelope.get('result', '')))}"
            )

        text = envelope.get("result") or ""
        if not isinstance(text, str) or not text.strip():
            raise LLMParseError("claude CLI returned an empty result field")

        data = validate_into(schema, text)  # shared validation path

        raw_usage = envelope.get("usage") or {}
        usage = Usage(
            input_tokens=int(raw_usage.get("input_tokens") or 0),
            cached_input_tokens=int(raw_usage.get("cache_read_input_tokens") or 0),
            output_tokens=int(raw_usage.get("output_tokens") or 0),
            cost_usd=float(envelope.get("total_cost_usd") or 0.0),
            cost_is_estimated=True,  # includes harness context; not API-equivalent
        )
        return ProviderResult(
            data=data, usage=usage, provider=self.name, model=self.model,
            prompt_hash=prompt_hash(system, user, self.model),
            raw_text=text[:2000],
        )
