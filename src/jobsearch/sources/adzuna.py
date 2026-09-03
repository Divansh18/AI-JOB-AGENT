"""Adzuna aggregator API. Optional: disabled unless credentials are present.

Key-gated by design - Phase 0 must run fully without it.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterator

from ..domain.models import RawPosting
from .base import FetchContext, FetchReport

SEARCH_URL = (
    "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
    "?app_id={app_id}&app_key={app_key}&results_per_page=50&what={what}"
    "&max_days_old={max_days}&content-type=application/json"
)

DEFAULT_QUERIES = [
    "software engineer", "software developer", "full stack developer",
    "backend developer", "frontend developer", "ai engineer",
]


class AdzunaAdapter:
    name = "adzuna"
    kind = "aggregator"

    def __init__(self, client, *, country: str = "in", queries: list[str] | None = None,
                 pages: int = 2, max_days_old: int = 14):
        self.client = client
        self.country = country
        self.queries = queries or DEFAULT_QUERIES
        self.pages = pages
        self.max_days_old = max_days_old
        self.app_id = os.environ.get("ADZUNA_APP_ID", "").strip()
        self.app_key = os.environ.get("ADZUNA_APP_KEY", "").strip()

    def available(self) -> tuple[bool, str]:
        if not self.app_id or not self.app_key:
            return False, "ADZUNA_APP_ID / ADZUNA_APP_KEY not set - source skipped"
        return True, "credentials present"

    def fetch(self, ctx: FetchContext, report: FetchReport) -> Iterator[RawPosting]:
        ok, _ = self.available()
        if not ok:
            return
        for query in self.queries:
            for page in range(1, self.pages + 1):
                url = SEARCH_URL.format(
                    country=self.country, page=page, app_id=self.app_id,
                    app_key=self.app_key, what=query.replace(" ", "%20"),
                    max_days=self.max_days_old,
                )
                payload, status = self.client.get_json(url)
                if not isinstance(payload, dict):
                    report.errors += 1
                    report.error_detail.append(f"{query} p{page}: HTTP {status}")
                    continue
                results = payload.get("results", []) or []
                if not results:
                    break
                for job in results:
                    posted = None
                    if job.get("created"):
                        try:
                            posted = datetime.fromisoformat(job["created"].replace("Z", "+00:00"))
                        except ValueError:
                            posted = None
                    report.fetched += 1
                    yield RawPosting(
                        source="adzuna",
                        external_id=f"az:{job.get('id')}",
                        title=job.get("title", "") or "",
                        company_name=(job.get("company") or {}).get("display_name", "") or "Unknown",
                        location_raw=(job.get("location") or {}).get("display_name", "") or "",
                        description_html=job.get("description") or "",
                        apply_url=job.get("redirect_url", "") or "",
                        canonical_url=job.get("redirect_url"),
                        posted_at=posted,
                        employment_hint=job.get("contract_time"),
                        raw={"id": job.get("id"), "category": (job.get("category") or {}).get("label")},
                    )
