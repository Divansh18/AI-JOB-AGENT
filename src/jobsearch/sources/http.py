"""Shared HTTP client: polite rate limiting, retries, identifying UA."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx


@dataclass
class RateLimiter:
    requests_per_second: float = 1.0
    _last: dict[str, float] = field(default_factory=dict)

    def wait(self, host: str) -> None:
        if self.requests_per_second <= 0:
            return
        interval = 1.0 / self.requests_per_second
        now = time.monotonic()
        last = self._last.get(host, 0.0)
        delta = now - last
        if delta < interval:
            time.sleep(interval - delta)
        self._last[host] = time.monotonic()


class HttpClient:
    """Thin wrapper over httpx with retry/backoff and per-host pacing."""

    def __init__(self, *, user_agent: str, timeout: float = 20.0, max_retries: int = 3,
                 requests_per_second: float = 1.0):
        self.limiter = RateLimiter(requests_per_second)
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": user_agent, "Accept": "application/json, text/html;q=0.8"},
        )

    def get(self, url: str, **kwargs) -> httpx.Response | None:
        """GET with backoff. Returns None if all attempts fail."""
        host = httpx.URL(url).host or "unknown"
        delay = 1.0
        for attempt in range(self.max_retries):
            self.limiter.wait(host)
            try:
                resp = self._client.get(url, **kwargs)
            except httpx.HTTPError:
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(delay)
                delay *= 2
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt == self.max_retries - 1:
                    return resp
                retry_after = resp.headers.get("Retry-After")
                time.sleep(float(retry_after) if (retry_after or "").isdigit() else delay)
                delay *= 2
                continue
            return resp
        return None

    def get_json(self, url: str, **kwargs) -> tuple[dict | list | None, int | None]:
        resp = self.get(url, **kwargs)
        if resp is None:
            return None, None
        if resp.status_code != 200:
            return None, resp.status_code
        try:
            return resp.json(), 200
        except Exception:
            return None, resp.status_code

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
