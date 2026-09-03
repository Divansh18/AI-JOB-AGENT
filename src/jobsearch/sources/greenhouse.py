"""Greenhouse job boards API (public, unauthenticated)."""

from __future__ import annotations

from datetime import datetime
from typing import Iterator

from ..domain.models import RawPosting
from .base import FetchContext, FetchReport

BOARD_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
PROBE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"


def parse_board(payload: dict, company_name: str, company_id: int | None) -> Iterator[RawPosting]:
    for job in payload.get("jobs", []) or []:
        loc = (job.get("location") or {}).get("name", "") or ""
        updated = job.get("updated_at") or job.get("first_published")
        posted = None
        if updated:
            try:
                posted = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            except ValueError:
                posted = None
        offices = ", ".join(
            o.get("name", "") for o in (job.get("offices") or []) if o.get("name")
        )
        yield RawPosting(
            source="greenhouse",
            external_id=f"gh:{job.get('id')}",
            title=job.get("title", "") or "",
            company_name=company_name,
            location_raw=loc or offices,
            description_html=job.get("content") or "",
            apply_url=job.get("absolute_url", "") or "",
            canonical_url=job.get("absolute_url"),
            posted_at=posted,
            employment_hint=None,
            company_id=company_id,
            raw={"id": job.get("id"), "departments": [
                d.get("name") for d in (job.get("departments") or [])]},
        )


class GreenhouseAdapter:
    name = "greenhouse"
    kind = "ats"

    def __init__(self, client):
        self.client = client

    def available(self) -> tuple[bool, str]:
        return True, "public API"

    def fetch(self, ctx: FetchContext, report: FetchReport) -> Iterator[RawPosting]:
        for company in ctx.companies:
            if company["ats_type"] != "greenhouse":
                continue
            payload, status = self.client.get_json(BOARD_URL.format(token=company["ats_token"]))
            if not isinstance(payload, dict):
                report.errors += 1
                report.error_detail.append(f"{company['name']}: HTTP {status}")
                yield from ()
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
