"""Company list management and ATS board discovery.

Central safety property: an ATS token is never assumed or invented. Every
token that reaches companies.yaml has been probed against the live public
board endpoint AND passed a name-similarity check against the input name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import yaml

from ..config.schemas import CompaniesFile, CompanyEntry
from ..domain.dedupe import company_name_matches
from ..domain.normalize import normalize_company
from ..sources.registry import PROBE_ENDPOINTS, extract_probe_result

NAME_MATCH_THRESHOLD = 60.0
MIN_JOBS_FOR_VERIFICATION = 1

DROP_WORDS = {
    "inc", "ltd", "limited", "pvt", "private", "llc", "corp", "corporation",
    "technologies", "technology", "labs", "software", "solutions", "systems",
    "services", "the", "co", "company", "gmbh", "plc", "india", "global",
}


def slug_candidates(name: str) -> list[str]:
    """Generate plausible board slugs for a company name.

    These are hypotheses to be tested against the live API - never written to
    config without verification.
    """
    base = re.sub(r"[^\w\s-]", " ", name.lower()).strip()
    words = [w for w in re.split(r"[\s-]+", base) if w]
    core = [w for w in words if w not in DROP_WORDS] or words
    if not core:
        return []
    candidates = [
        "".join(core),
        "-".join(core),
        core[0],
    ]
    if len(core) > 1:
        candidates.append("".join(core[:2]))
        candidates.append("-".join(core[:2]))
    seen, out = set(), []
    for c in candidates:
        c = c.strip("-")
        if c and c not in seen and len(c) >= 2:
            seen.add(c)
            out.append(c)
    return out


@dataclass
class ProbeHit:
    name: str
    ats: str
    token: str
    job_count: int
    sample_titles: list[str] = field(default_factory=list)
    name_similarity: float = 0.0
    board_url: str = ""

    def to_entry(self, priority: str = "normal", tags: list[str] | None = None,
                 hq_country: str | None = None) -> CompanyEntry:
        return CompanyEntry(
            name=self.name, ats=self.ats, token=self.token, priority=priority,
            tags=tags or [], hq_country=hq_country, active=True,
            verified_at=date.today().isoformat(),
            notes=f"verified: {self.job_count} postings; e.g. {self.sample_titles[0][:60]}"
            if self.sample_titles else f"verified: {self.job_count} postings",
        )


def probe_company(client, name: str, ats_types: list[str] | None = None) -> ProbeHit | None:
    """Probe a company name against public ATS boards.

    Returns the best verified hit, or None. A hit requires: HTTP 200, parsable
    payload, at least one live posting, and a name-similarity check so that a
    generic slug cannot silently bind to an unrelated company.
    """
    ats_types = ats_types or list(PROBE_ENDPOINTS.keys())
    best: ProbeHit | None = None

    for token in slug_candidates(name):
        for ats in ats_types:
            url = PROBE_ENDPOINTS[ats].format(token=token)
            payload, status = client.get_json(url)
            if payload is None or status != 200:
                continue
            job_count, titles = extract_probe_result(ats, payload)
            if job_count < MIN_JOBS_FOR_VERIFICATION:
                continue
            similarity = company_name_matches(normalize_company(name), token)
            if similarity < NAME_MATCH_THRESHOLD:
                continue
            hit = ProbeHit(
                name=name, ats=ats, token=token, job_count=job_count,
                sample_titles=[t for t in titles if t],
                name_similarity=similarity,
                board_url=url,
            )
            if best is None or hit.name_similarity > best.name_similarity:
                best = hit
        if best is not None:
            break  # first slug variant that verifies wins
    return best


def load_companies_file(path) -> CompaniesFile:
    if not path.exists():
        return CompaniesFile()
    with open(path, "r", encoding="utf-8") as fh:
        return CompaniesFile(**(yaml.safe_load(fh) or {}))


def write_companies_file(path, file: CompaniesFile) -> None:
    payload = {
        "version": file.version,
        "companies": [
            {k: v for k, v in c.model_dump().items() if v not in (None, [], "")}
            for c in sorted(file.companies, key=lambda c: c.name.lower())
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True, width=100)


def merge_entries(file: CompaniesFile, new: list[CompanyEntry]) -> tuple[CompaniesFile, int]:
    existing = {(c.ats, c.token) for c in file.companies}
    added = 0
    for entry in new:
        if (entry.ats, entry.token) in existing:
            continue
        file.companies.append(entry)
        existing.add((entry.ats, entry.token))
        added += 1
    return file, added


def sync_to_db(conn, file: CompaniesFile) -> dict:
    """Reconcile companies.yaml into the database.

    Removing a company from YAML deactivates it but never deletes history.
    """
    from ..persistence.repositories import CompanyRepo

    repo = CompanyRepo(conn)
    keep: set[tuple[str, str]] = set()
    for entry in file.companies:
        if entry.active:
            keep.add((entry.ats, entry.token))
        repo.upsert(
            name=entry.name,
            name_normalized=normalize_company(entry.name),
            ats_type=entry.ats,
            ats_token=entry.token,
            board_url=PROBE_ENDPOINTS[entry.ats].format(token=entry.token),
            priority=entry.priority,
            tags=entry.tags,
            hq_country=entry.hq_country,
            active=entry.active,
            verified_at=entry.verified_at,
            added_via="yaml",
            notes=entry.notes,
        )
    deactivated = repo.deactivate_missing(keep)
    return {"synced": len(file.companies), "deactivated": deactivated}
