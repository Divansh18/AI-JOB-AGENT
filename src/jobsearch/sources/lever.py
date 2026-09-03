"""Lever postings API (public, unauthenticated)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from ..domain.models import RawPosting
from .base import FetchContext, FetchReport

BOARD_URL = "https://api.lever.co/v0/postings/{token}?mode=json"
PROBE_URL = BOARD_URL


def parse_board(payload: list, company_name: str, company_id: int | None) -> Iterator[RawPosting]:
    for job in payload or []:
        cats = job.get("categories") or {}
        posted = None
        if job.get("createdAt"):
            try:
                posted = datetime.fromtimestamp(job["createdAt"] / 1000, tz=timezone.utc)
            except (ValueError, OSError, TypeError):
                posted = None
        description = job.get("descriptionPlain") or job.get("description") or ""
        for lst in job.get("lists") or []:
            description += " " + (lst.get("text") or "") + " " + (lst.get("content") or "")
        description += " " + (job.get("additionalPlain") or job.get("additional") or "")
        yield RawPosting(
            source="lever",
            external_id=f"lv:{job.get('id')}",
            title=job.get("text", "") or "",
            company_name=company_name,
            location_raw=cats.get("location", "") or "",
            description_html=description,
            apply_url=job.get("hostedUrl", "") or job.get("applyUrl", "") or "",
            canonical_url=job.get("hostedUrl"),
            posted_at=posted,
            employment_hint=cats.get("commitment"),
            company_id=company_id,
            raw={"id": job.get("id"), "team": cats.get("team"),
                 "commitment": cats.get("commitment")},
        )


class LeverAdapter:
    name = "lever"
    kind = "ats"

    def __init__(self, client):
        self.client = client

    def available(self) -> tuple[bool, str]:
        return True, "public API"

    def fetch(self, ctx: FetchContext, report: FetchReport) -> Iterator[RawPosting]:
        for company in ctx.companies:
            if company["ats_type"] != "lever":
                continue
            payload, status = self.client.get_json(BOARD_URL.format(token=company["ats_token"]))
            if not isinstance(payload, list):
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
