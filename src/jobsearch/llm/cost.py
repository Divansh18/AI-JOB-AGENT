"""Token cost accounting.

Prices are USD per million tokens. Cache reads bill at 10% of base input and
cache writes at 125% where the provider exposes those token classes.
"""

from __future__ import annotations

PRICING_USD_PER_MTOK = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


def estimate_cost_usd(model: str, *, input_tokens: int, output_tokens: int,
                      cached_input_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    rates = PRICING_USD_PER_MTOK.get(model)
    if rates is None:
        base = model.rsplit("-", 1)[0]
        rates = PRICING_USD_PER_MTOK.get(base, PRICING_USD_PER_MTOK["claude-haiku-4-5"])
    per = 1_000_000
    uncached = max(0, input_tokens - cached_input_tokens - cache_write_tokens)
    cost = (
        uncached * rates["input"]
        + cached_input_tokens * rates["input"] * CACHE_READ_MULTIPLIER
        + cache_write_tokens * rates["input"] * CACHE_WRITE_MULTIPLIER
        + output_tokens * rates["output"]
    ) / per
    return round(cost, 6)


def to_inr(usd: float, rate: float = 88.0) -> float:
    return round(usd * rate, 4)


class BudgetGuard:
    """Hard stop on daily spend. Checked before every paid call."""

    def __init__(self, repo, *, daily_cap_inr: float, usd_to_inr: float):
        self.repo = repo
        self.daily_cap_inr = daily_cap_inr
        self.usd_to_inr = usd_to_inr

    def spent_today_inr(self) -> float:
        from datetime import datetime, timezone

        start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0).isoformat()
        return self.repo.spend_inr(since=start)

    def remaining_inr(self) -> float:
        return max(0.0, self.daily_cap_inr - self.spent_today_inr())

    def check(self) -> tuple[bool, str]:
        spent = self.spent_today_inr()
        if spent >= self.daily_cap_inr:
            return False, f"daily cap reached: INR {spent:.2f} / {self.daily_cap_inr:.2f}"
        return True, f"INR {spent:.2f} of {self.daily_cap_inr:.2f} used today"
