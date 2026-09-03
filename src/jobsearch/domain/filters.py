"""Hard-filter rule engine.

Design rule: every exclusion carries a stable rule id, and every "signal is
missing" case defaults to KEEP. Missing a good job costs far more than
reading a bad card, so ambiguity always resolves in favour of showing it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from ..config.schemas import Filters
from .extract import (
    EmploymentType,
    RemoteScope,
    RemoteType,
    YoeAnalysis,
    YoeVerdict,
    detect_work_auth_blocker,
    extract_yoe,
)
from .models import FilterResult, Job
from .taxonomy import TitleAnalysis, TitleTier, classify_title

RULE_DESCRIPTIONS = {
    "title_role": "Title is a role type we never target (QA, DevOps, sales, ...)",
    "title_seniority": "Title indicates senior/staff/principal/lead/manager level",
    "title_tier": "Title tier is not in the configured include list",
    "yoe_hard_high": "Description explicitly requires 4+ years of experience",
    "employment_type": "Employment type is excluded (contract/freelance/part-time)",
    "remote_scope": "Remote role is restricted to a region I cannot apply from",
    "work_auth": "Description states a work-authorization requirement I do not meet",
    "country": "Location country is outside the allowed list and not remote-eligible",
    "company_blocked": "Company is on the blocklist",
    "content_block": "Description contains an excluded phrase",
    "too_short": "Description is too short to evaluate",
    "stale": "Posting is older than the freshness window",
}


def _company_blocked(job: Job, f: Filters) -> bool:
    name = job.company_normalized
    if any(name == b.strip().lower() for b in f.companies.blocklist):
        return True
    return any(re.search(p, name) for p in f.companies.blocklist_patterns)


def evaluate(
    job: Job,
    filters: Filters,
    *,
    now: datetime | None = None,
    title_analysis: TitleAnalysis | None = None,
    yoe_analysis: YoeAnalysis | None = None,
) -> FilterResult:
    """Apply all hard filters, returning pass/fail plus full explanation."""
    now = now or datetime.now(timezone.utc)
    failed: list[str] = []
    notes: list[str] = []

    ta = title_analysis or classify_title(job.title_normalized)
    ya = yoe_analysis or extract_yoe(job.description_text)

    # --- title ---
    if ta.tier == TitleTier.EXCLUDED:
        failed.append("title_role")
    else:
        allowed = set(filters.titles.include_tiers)
        if filters.titles.keep_ambiguous:
            allowed.add(TitleTier.AMBIGUOUS.value)
        if filters.titles.keep_adjacent:
            allowed.add(TitleTier.ADJACENT.value)
        if ta.tier.value not in allowed:
            failed.append("title_tier")

    if filters.titles.exclude_senior_titles and ta.is_senior:
        failed.append("title_seniority")

    # --- experience (graded) ---
    if ya.verdict.value in filters.experience.exclude_verdicts:
        failed.append("yoe_hard_high")
    if ya.verdict == YoeVerdict.UNKNOWN and filters.experience.unknown_policy == "pass":
        notes.append("no experience requirement stated - kept")

    # --- employment type ---
    if job.employment_type in filters.employment.exclude:
        failed.append("employment_type")
    elif job.employment_type in filters.employment.deprioritize:
        notes.append(f"{job.employment_type} - kept but deprioritized in ranking")

    # --- geography ---
    scope = job.remote_scope
    is_remote = job.remote_type in (RemoteType.REMOTE.value, RemoteType.HYBRID.value)
    if scope in filters.locations.remote.exclude_scopes:
        failed.append("remote_scope")
    elif job.country and job.country not in filters.locations.allow_countries:
        # Not in India. Keep only if it is genuinely remote and open to a
        # region I can apply from.
        eligible = is_remote and scope in filters.locations.remote.allowed_scopes
        if not eligible:
            failed.append("country")
    elif not job.country and filters.locations.unknown_policy != "pass":
        failed.append("country")

    # --- work authorization ---
    blocker = detect_work_auth_blocker(job.description_text)
    if blocker and filters.work_authorization.exclude_blocked:
        failed.append("work_auth")

    # --- company ---
    if _company_blocked(job, filters):
        failed.append("company_blocked")

    # --- content ---
    d = (job.description_text or "").lower()
    for phrase in filters.content.exclude_phrases:
        if phrase.lower() in d:
            failed.append("content_block")
            notes.append(f"excluded phrase: {phrase}")
            break
    if job.description_chars < filters.content.min_description_chars:
        failed.append("too_short")

    # --- freshness ---
    reference = job.posted_at or job.first_seen_at
    if reference:
        ref = reference if reference.tzinfo else reference.replace(tzinfo=timezone.utc)
        age_days = (now - ref).total_seconds() / 86400
        if age_days > filters.freshness.max_age_days:
            failed.append("stale")

    signals = {
        "title_tier": ta.tier.value,
        "seniority_token": ta.seniority_token,
        "level_token": ta.level_token,
        "early_career_signal": ta.early_career_signal,
        "yoe": ya.as_signal(),
        "employment_type": job.employment_type,
        "remote_type": job.remote_type,
        "remote_scope": scope,
        "country": job.country,
        "work_auth_blocker": blocker,
    }
    return FilterResult(passed=not failed, rules_failed=failed, signals=signals, notes=notes)


def explain(result: FilterResult) -> list[str]:
    return [f"{rid}: {RULE_DESCRIPTIONS.get(rid, 'unknown rule')}" for rid in result.rules_failed]
