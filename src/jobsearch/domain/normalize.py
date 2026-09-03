"""Text normalization for titles, company names and locations."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s\.\+#/&-]")
_HTML_TAG = re.compile(r"<[^>]+>")

COMPANY_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt limited", "p ltd",
    "limited", "ltd", "llc", "inc", "incorporated", "corp", "corporation",
    "gmbh", "b.v.", "bv", "plc", "co", "company", "technologies",
    "technology", "tech", "labs", "lab", "software", "solutions",
    "systems", "services", "india", "global",
]


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def normalize_title(title: str) -> str:
    """Lowercase, de-accent and collapse a job title for pattern matching.

    Keeps '.', '+', '#', '/' and '-' because they carry meaning in titles
    (node.js, c++, c#, ai/ml, front-end).
    """
    if not title:
        return ""
    t = strip_accents(title).lower()
    t = t.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def normalize_company(name: str) -> str:
    """Normalize a company name for identity matching across sources."""
    if not name:
        return ""
    n = strip_accents(name).lower()
    n = re.sub(r"[^\w\s&-]", " ", n)
    n = _WS.sub(" ", n).strip()
    # Strip trailing legal/generic suffixes, longest first, repeatedly.
    changed = True
    while changed:
        changed = False
        for suffix in sorted(COMPANY_SUFFIXES, key=len, reverse=True):
            if n.endswith(" " + suffix):
                n = n[: -(len(suffix) + 1)].strip()
                changed = True
                break
    return n


def normalize_location(loc: str) -> str:
    if not loc:
        return ""
    l = strip_accents(loc).lower()
    l = re.sub(r"[^\w\s,/-]", " ", l)
    return _WS.sub(" ", l).strip()


def html_to_text(html: str) -> str:
    """Best-effort HTML -> plain text.

    Handles entity-escaped markup as well as literal markup. Greenhouse, for
    example, returns descriptions where the tags themselves are escaped
    (``&lt;p&gt;``), so unescaping has to happen *before* tag stripping or the
    markup survives into the text used for embedding and skill matching.

    Uses selectolax when available and falls back to a regex strip, so the
    domain layer never hard-depends on an optional parser.
    """
    if not html:
        return ""

    text = html
    # Two passes covers escaped and double-escaped markup; more would risk
    # mangling legitimate text that merely mentions an entity.
    for _ in range(2):
        if "&lt;" in text or "&gt;" in text or "&amp;" in text:
            text = _unescape(text)
        if "<" in text:
            text = _strip_tags(text)
    text = _unescape(text)
    return _WS.sub(" ", text).strip()


def _unescape(text: str) -> str:
    import html as _html

    return _html.unescape(text)


def _strip_tags(markup: str) -> str:
    try:
        from selectolax.parser import HTMLParser

        return HTMLParser(markup).text(separator=" ")
    except Exception:  # pragma: no cover - fallback path
        return _HTML_TAG.sub(" ", markup)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\.\+#]+", text.lower())
