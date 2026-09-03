"""Duplicate detection across sources.

Two passes:
  1. exact  - identical identity fingerprint or identical canonical URL
  2. fuzzy  - same normalized company + high title similarity + compatible
              location, which is what catches the same role arriving from an
              ATS board and an aggregator with slightly different wording.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from .models import Job
from .normalize import normalize_location

TITLE_SIMILARITY_THRESHOLD = 88
COMPANY_SIMILARITY_THRESHOLD = 92


@dataclass(frozen=True)
class DuplicateLink:
    canonical_fingerprint: str
    duplicate_fingerprint: str
    method: str
    similarity: float


def _locations_compatible(a: Job, b: Job) -> bool:
    """Whether two same-title postings are plausibly the same requisition.

    Deliberately conservative. A large employer posts the identical title in
    many cities; treating those as one record would hide a Bengaluru opening
    behind, say, a Mexico City canonical - the exact false negative this
    system is built to avoid. Merging therefore requires positive evidence
    that the locations agree, not merely an absence of evidence that they
    differ.
    """
    # Different known countries are never the same requisition.
    if a.country and b.country and a.country != b.country:
        return False

    # Both sides resolved to canonical cities: they must overlap.
    if a.locations and b.locations:
        return bool(set(a.locations) & set(b.locations))

    # No canonical city on at least one side. Compare the raw strings when we
    # have both; substring matching absorbs "Remote" vs "Remote - India".
    ra, rb = normalize_location(a.location_raw), normalize_location(b.location_raw)
    if ra and rb:
        return ra == rb or ra in rb or rb in ra

    # Genuinely unknown on one side - allow the merge, since the identity
    # fingerprint and canonical URL checks have already run.
    return True


def find_duplicates(jobs: list[Job]) -> list[DuplicateLink]:
    """Return links pointing duplicates at a canonical job.

    Canonical choice is the earliest ``first_seen_at`` - we want to preserve
    the moment we first had the chance to apply, not the moment a slower
    source echoed it.
    """
    ordered = sorted(jobs, key=lambda j: (j.first_seen_at, j.fingerprint))
    links: list[DuplicateLink] = []

    by_identity: dict[str, Job] = {}
    by_url: dict[str, Job] = {}
    by_company: dict[str, list[Job]] = {}
    claimed: set[str] = set()

    for job in ordered:
        if job.fingerprint in by_identity:
            links.append(DuplicateLink(by_identity[job.fingerprint].fingerprint,
                                       job.fingerprint, "identity", 100.0))
            claimed.add(job.fingerprint)
            continue
        by_identity[job.fingerprint] = job

        if job.canonical_url:
            prior = by_url.get(job.canonical_url)
            if prior is not None and prior.fingerprint != job.fingerprint:
                links.append(DuplicateLink(prior.fingerprint, job.fingerprint,
                                           "canonical_url", 100.0))
                claimed.add(job.fingerprint)
                continue
            by_url[job.canonical_url] = job

        matched = False
        for candidate in by_company.get(job.company_normalized, []):
            if candidate.fingerprint in claimed:
                continue
            sim = fuzz.token_set_ratio(candidate.title_normalized, job.title_normalized)
            if sim >= TITLE_SIMILARITY_THRESHOLD and _locations_compatible(candidate, job):
                links.append(DuplicateLink(candidate.fingerprint, job.fingerprint,
                                           "fuzzy_title", float(sim)))
                claimed.add(job.fingerprint)
                matched = True
                break
        if not matched:
            by_company.setdefault(job.company_normalized, []).append(job)

    return links


def company_name_matches(a: str, b: str) -> float:
    """Similarity used when verifying a discovered ATS board against a name."""
    return float(fuzz.token_set_ratio(a.lower(), b.lower()))
