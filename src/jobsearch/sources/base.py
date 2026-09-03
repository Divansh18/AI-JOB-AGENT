"""Source adapter contract.

Adding a job source means implementing this protocol and registering it.
Nothing else in the system changes - that is the whole point of the seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

from ..domain.models import RawPosting


@dataclass
class FetchContext:
    """Everything an adapter needs, passed in rather than imported."""

    companies: list[dict] = field(default_factory=list)
    settings: object | None = None
    limit_per_company: int | None = None


@dataclass
class FetchReport:
    source: str
    fetched: int = 0
    errors: int = 0
    error_detail: list[str] = field(default_factory=list)


class SourceAdapter(Protocol):
    name: str
    kind: str

    def available(self) -> tuple[bool, str]:
        """Whether this adapter can run (e.g. API key present)."""
        ...

    def fetch(self, ctx: FetchContext, report: FetchReport) -> Iterator[RawPosting]:
        ...
