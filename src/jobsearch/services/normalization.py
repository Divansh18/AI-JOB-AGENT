"""Turn RawPosting objects into normalized Job records."""

from __future__ import annotations

from datetime import datetime, timezone

from ..domain.extract import (
    extract_location_facts,
    extract_employment_type,
)
from ..domain.fingerprint import canonicalize_url, content_hash, identity_fingerprint
from ..domain.models import Job, RawPosting
from ..domain.normalize import (
    html_to_text,
    normalize_company,
    normalize_location,
    normalize_title,
)

MAX_DESCRIPTION_CHARS = 40000


def normalize_posting(posting: RawPosting, *, now: datetime | None = None) -> Job:
    now = now or datetime.now(timezone.utc)

    description = posting.description_text or html_to_text(posting.description_html or "")
    description = description[:MAX_DESCRIPTION_CHARS]

    title_norm = normalize_title(posting.title)
    company_norm = normalize_company(posting.company_name)
    loc_norm = normalize_location(posting.location_raw)

    location = extract_location_facts(posting.title, posting.location_raw, description)
    employment = extract_employment_type(posting.title, description, posting.employment_hint)

    fingerprint = identity_fingerprint(
        source=posting.source,
        external_id=posting.external_id,
        company_normalized=company_norm,
        title_normalized=title_norm,
        location_normalized=loc_norm,
    )

    return Job(
        fingerprint=fingerprint,
        content_hash=content_hash(posting.title, description, posting.location_raw),
        source=posting.source,
        title=posting.title.strip(),
        title_normalized=title_norm,
        company_name_raw=posting.company_name.strip(),
        company_normalized=company_norm,
        location_raw=posting.location_raw.strip(),
        locations=location.cities,
        country=location.country,
        remote_type=location.remote_type.value,
        remote_scope=location.remote_scope.value,
        employment_type=employment.value,
        description_text=description,
        apply_url=posting.apply_url,
        canonical_url=canonicalize_url(posting.canonical_url or posting.apply_url),
        posted_at=posting.posted_at,
        first_seen_at=now,
        external_id=posting.external_id,
        company_id=posting.company_id,
        raw=posting.raw,
    )
