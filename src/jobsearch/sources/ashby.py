"""Ashby posting API (public, unauthenticated)."""

from __future__ import annotations

from datetime import datetime
from typing import Iterator

from ..domain.models import RawPosting
from .base import FetchContext, FetchReport

BOARD_URL = "https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true"
PROBE_URL = "https://api.ashbyhq.com/posting-api/job-board/{token}"


def parse_board(payload: dict, company_name: str, company_id: int | None) -> Iterator[RawPosting]:
    for job in payload.get("jobs", []) or []:
        posted = None
        if job.get("publishedAt"):
            try:
                posted = datetime.fromisoformat(job["publishedAt"].replace("Z", "+00:00"))
            except ValueError:
                posted = None
        yield RawPosting(
            source="ashby",
            external_id=f"ab:{job.get('id')}",
            title=job.get("title", "") or "",
            company_name=company_name,
            location_raw=job.get("location", "") or "",
            description_html=job.get("descriptionHtml") or job.get("descriptionPlain") or "",
            apply_url=job.get("applyUrl") or job.get("jobUrl") or "",
            canonical_url=job.get("jobUrl"),
            posted_at=posted,
            employment_hint=job.get("employmentType"),
            company_id=company_id,
            raw={"id": job.get("id"), "team": job.get("team"),
                 "isRemote": job.get("isRemote")},
        )


class AshbyAdapter:
    name = "ashby"
    kind = "ats"

    def __init__(self, client):
        self.client = client

    def available(self) -> tuple[bool, str]:
        return True, "public API"

    def fetch(self, ctx: FetchContext, report: FetchReport) -> Iterator[RawPosting]:
        for company in ctx.companies:
            if company["ats_type"] != "ashby":
                continue
            payload, status = self.client.get_json(BOARD_URL.format(token=company["ats_token"]))
            if not isinstance(payload, dict):
                report.errors += 1
                report.error_detail.append(f"{company['name']}: HTTP {status}")
                company["_ok"] = False
                continue
            company["_ok"] = True
            count = 0
            for posting in parse_board(payload, company["name"], company["id"]):
                count += 1
                report.fetched += 1
                yield posting
                if ctx.limit_per_company and count >= ctx.limit_per_company:
                    break
