"""Manual single-URL ingest for jobs found outside the automated sources."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from ..domain.extract import INDIA_CITIES
from ..domain.models import RawPosting
from ..domain.normalize import html_to_text


_GENERIC_SITE_NAMES = {"lever", "greenhouse", "greenhouse.io", "ashby", "ashbyhq"}
_TITLE_COMPANY_SEPARATORS = (" - ", " – ", " — ")
_HEADER_CHARS = 700


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


def _is_lever_url(url: str) -> bool:
    host = urlparse(url or "").netloc.lower()
    return host == "jobs.lever.co" or host.endswith(".lever.co")


def _lever_title_company(url: str, page_title: str, text: str) -> tuple[str | None, str | None]:
    if not _is_lever_url(url):
        return None, None
    for separator in _TITLE_COMPANY_SEPARATORS:
        if separator not in page_title:
            continue
        company, job_title = [part.strip() for part in page_title.split(separator, 1)]
        if company and job_title and _text_starts_with_title(text, job_title):
            return job_title, company
    return None, None


def _text_starts_with_title(text: str, title: str) -> bool:
    header = re.sub(r"\s+", " ", (text or "")[:_HEADER_CHARS]).strip().lower()
    normalized_title = re.sub(r"\s+", " ", title or "").strip().lower()
    return bool(normalized_title and header.startswith(normalized_title))


def _header_location(text: str) -> str | None:
    header = (text or "")[:_HEADER_CHARS]
    if not header:
        return None
    cities = "|".join(re.escape(city) for city in sorted(INDIA_CITIES, key=len, reverse=True))
    patterns = (
        rf"\b({cities})\s*,?\s*India\b",
        r"\bRemote\s*[-–,/()]?\s*India\b",
        r"\bIndia\s*[-–,/()]?\s*Remote\b",
        r"\bAnywhere in India\b",
    )
    for pattern in patterns:
        match = re.search(pattern, header, re.I)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip(" -–—/,")
    return None


def build_posting(url: str, html: str, *, company: str | None = None,
                  title: str | None = None, location: str | None = None) -> RawPosting:
    """Build a posting from fetched HTML, with optional manual overrides.

    Overrides always win: when extraction is uncertain the human's value is
    authoritative, and nothing here is ever invented.
    """
    extracted_title = _title(html)
    text = html_to_text(html)
    lever_title, lever_company = _lever_title_company(url, extracted_title, text)
    page_title = title or lever_title or extracted_title
    site = _meta(html, "og:site_name")
    if site and site.strip().lower() in _GENERIC_SITE_NAMES:
        site = None
    inferred_company = company or site or lever_company or ""
    if not company and " at " in page_title:
        parts = page_title.split(" at ")
        page_title, inferred_company = parts[0].strip(), parts[-1].strip()
    return RawPosting(
        source="manual",
        external_id=None,
        title=page_title.strip(),
        company_name=(inferred_company or "Unknown").strip(),
        location_raw=(location or _meta(html, "og:locality") or _header_location(text) or "").strip(),
        description_html=None,
        description_text=text,
        apply_url=url,
        canonical_url=url,
        posted_at=datetime.now(timezone.utc),
        raw={"ingested": "manual"},
    )
