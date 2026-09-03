"""Manual single-URL ingest for jobs found outside the automated sources."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from ..domain.models import RawPosting
from ..domain.normalize import html_to_text


def _meta(html: str, prop: str) -> str | None:
    m = re.search(
        rf'<meta[^>]+(?:property|name)=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']+)',
        html, re.I,
    )
    return m.group(1).strip() if m else None


def _title(html: str) -> str:
    for prop in ("og:title", "twitter:title"):
        v = _meta(html, prop)
        if v:
            return v
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    return m.group(1).strip() if m else ""


def build_posting(url: str, html: str, *, company: str | None = None,
                  title: str | None = None, location: str | None = None) -> RawPosting:
    """Build a posting from fetched HTML, with optional manual overrides.

    Overrides always win: when extraction is uncertain the human's value is
    authoritative, and nothing here is ever invented.
    """
    page_title = title or _title(html)
    site = _meta(html, "og:site_name")
    inferred_company = company or site or ""
    if not company and " at " in page_title:
        parts = page_title.split(" at ")
        page_title, inferred_company = parts[0].strip(), parts[-1].strip()
    text = html_to_text(html)
    return RawPosting(
        source="manual",
        external_id=None,
        title=page_title.strip(),
        company_name=(inferred_company or "Unknown").strip(),
        location_raw=(location or _meta(html, "og:locality") or "").strip(),
        description_html=None,
        description_text=text,
        apply_url=url,
        canonical_url=url,
        posted_at=datetime.now(timezone.utc),
        raw={"ingested": "manual"},
    )
