"""Adapter registry: the single place a new source gets wired in."""

from __future__ import annotations

from .adzuna import AdzunaAdapter
from .ashby import AshbyAdapter
from .greenhouse import GreenhouseAdapter
from .lever import LeverAdapter

ATS_ADAPTERS = {
    "greenhouse": GreenhouseAdapter,
    "lever": LeverAdapter,
    "ashby": AshbyAdapter,
}


def build_adapters(client, *, include_aggregators: bool = True) -> list:
    adapters = [cls(client) for cls in ATS_ADAPTERS.values()]
    if include_aggregators:
        adapters.append(AdzunaAdapter(client))
    return adapters


# Probe endpoints used by company discovery. Kept next to the adapters so a
# new ATS only ever needs touching in one module.
PROBE_ENDPOINTS = {
    "greenhouse": "https://boards-api.greenhouse.io/v1/boards/{token}/jobs",
    "lever": "https://api.lever.co/v0/postings/{token}?mode=json",
    "ashby": "https://api.ashbyhq.com/posting-api/job-board/{token}",
}


def extract_probe_result(ats: str, payload) -> tuple[int, list[str]]:
    """Return (job_count, sample_titles) from a probe response."""
    if ats == "lever":
        if not isinstance(payload, list):
            return 0, []
        return len(payload), [j.get("text", "") for j in payload[:5]]
    if not isinstance(payload, dict):
        return 0, []
    jobs = payload.get("jobs", []) or []
    return len(jobs), [j.get("title", "") for j in jobs[:5]]
