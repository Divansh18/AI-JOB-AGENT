"""Stable identity and content hashing for postings."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse, urlunparse

TRACKING_PARAMS_PREFIXES = ("utm_", "gh_", "ref", "source", "src", "lever-", "trk")


def canonicalize_url(url: str | None) -> str | None:
    """Strip query strings, fragments and trailing slashes for identity use."""
    if not url:
        return None
    try:
        p = urlparse(url.strip())
    except ValueError:
        return url
    if not p.scheme:
        return url
    path = p.path.rstrip("/") or "/"
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return urlunparse((p.scheme.lower(), netloc, path, "", "", ""))


def _h(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def identity_fingerprint(
    *,
    source: str,
    external_id: str | None,
    company_normalized: str,
    title_normalized: str,
    location_normalized: str,
) -> str:
    """Primary identity for a posting.

    Prefers a source-stable external id. Falls back to a company/title/location
    triple, which is also what lets the same job from two different sources
    collapse onto one record.
    """
    if external_id:
        return _h(f"src:{source}:{external_id}")
    loc = re.sub(r"\s+", " ", location_normalized or "").strip()
    return _h(f"cmp:{company_normalized}|{title_normalized}|{loc}")


def cross_source_key(company_normalized: str, title_normalized: str) -> str:
    """Looser key used to detect the same role arriving from another source."""
    return _h(f"x:{company_normalized}|{title_normalized}")


def content_hash(title: str, description_text: str, location_raw: str) -> str:
    """Detect meaningful edits to a posting we already track."""
    body = re.sub(r"\s+", " ", (description_text or "")).strip().lower()
    return _h(f"{title.strip().lower()}|{location_raw.strip().lower()}|{body}")
