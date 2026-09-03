"""Hard-filter rule engine.

Design rule: every exclusion carries a stable rule id, and every "signal is
missing" case defaults to KEEP. Missing a good job costs far more than
reading a bad card, so ambiguity always resolves in favour of showing it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from ..config.schemas import Filters, LocationFilters
from .extract import (
    COUNTRY_NAMES,
    EmploymentType,
    LocationFacts,
    REGION_NAMES,
    RemoteScope,
    RemoteType,
    YoeAnalysis,
    YoeVerdict,
    detect_work_auth_blocker,
    extract_location_facts,
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


def _country_names(codes: list[str]) -> str:
    return ", ".join(COUNTRY_NAMES.get(code, code) for code in codes)


def _scope_reason(scope: str) -> str:
    return {
        RemoteScope.US_ONLY.value: "remote restricted to the United States",
        RemoteScope.UK_ONLY.value: "remote restricted to the United Kingdom",
        RemoteScope.EU_ONLY.value: "remote restricted to the EU/Europe",
        RemoteScope.EMEA_ONLY.value: "remote restricted to EMEA",
        RemoteScope.CANADA_ONLY.value: "remote restricted to Canada",
        RemoteScope.OTHER_RESTRICTED.value: "remote restricted to a foreign region",
    }.get(scope, f"remote restricted to {scope}")


def _region_names(codes: list[str]) -> str:
    return ", ".join(REGION_NAMES.get(code, code) for code in codes)


def _location_result(
    facts: LocationFacts,
    *,
    status: str,
    bucket: str,
    reason: str,
    rule_id: str | None,
) -> dict:
    return {
        "status": status,
        "bucket": bucket,
        "reason": reason,
        "rule_id": rule_id,
        "cities": facts.cities,
        "countries": facts.countries,
        "regions": facts.regions,
        "country": facts.country,
        "remote_type": facts.remote_type.value,
        "remote_scope": facts.remote_scope.value,
        "multi_location": facts.multi_location,
        "used_description_fallback": facts.used_description_fallback,
        "evidence": facts.evidence,
    }


def _assess_location(job: Job, location_filters: LocationFilters) -> dict:
    facts = extract_location_facts(job.title, job.location_raw, job.description_text)
    allowed = {code.upper() for code in location_filters.allow_countries}
    allowed_countries = [code for code in facts.countries if code in allowed]
    foreign_countries = [code for code in facts.countries if code not in allowed]
    foreign_regions = [code for code in facts.regions if code not in {"APAC"}]

    if allowed_countries:
        if facts.remote_scope == RemoteScope.INDIA and facts.remote_type in (
            RemoteType.REMOTE, RemoteType.HYBRID
        ):
            return _location_result(
                facts,
                status="eligible",
                bucket="india_remote",
                reason="remote within India",
                rule_id=None,
            )
        if foreign_countries or facts.multi_location:
            return _location_result(
                facts,
                status="eligible",
                bucket="india_multi_location",
                reason="India included among listed locations",
                rule_id=None,
            )
        if facts.cities:
            return _location_result(
                facts,
                status="eligible",
                bucket="india_city",
                reason="India location stated",
                rule_id=None,
            )
        return _location_result(
            facts,
            status="eligible",
            bucket="india_country",
            reason="India location stated",
            rule_id=None,
        )

    if facts.remote_type == RemoteType.REMOTE:
        if not location_filters.remote.allow:
            return _location_result(
                facts,
                status="ineligible",
                bucket="remote_blocked",
                reason="remote roles disabled by config",
                rule_id="remote_scope",
            )
        if facts.remote_scope.value in location_filters.remote.exclude_scopes:
            return _location_result(
                facts,
                status="ineligible",
                bucket="remote_restricted",
                reason=_scope_reason(facts.remote_scope.value),
                rule_id="remote_scope",
            )
        if foreign_countries:
            return _location_result(
                facts,
                status="ineligible",
                bucket="remote_restricted",
                reason=f"remote restricted to {_country_names(foreign_countries)}",
                rule_id="remote_scope",
            )
        if foreign_regions:
            return _location_result(
                facts,
                status="ineligible",
                bucket="remote_restricted",
                reason=f"remote restricted to {_region_names(foreign_regions)}",
                rule_id="remote_scope",
            )
        if facts.remote_scope.value in location_filters.remote.allowed_scopes:
            if facts.remote_scope == RemoteScope.GLOBAL:
                return _location_result(
                    facts,
                    status="eligible",
                    bucket="remote_global",
                    reason="explicitly global remote",
                    rule_id=None,
                )
            if facts.remote_scope == RemoteScope.APAC:
                return _location_result(
                    facts,
                    status="eligible",
                    bucket="remote_apac",
                    reason="explicitly APAC remote",
                    rule_id=None,
                )
            if facts.remote_scope == RemoteScope.INDIA:
                return _location_result(
                    facts,
                    status="eligible",
                    bucket="india_remote",
                    reason="remote within India",
                    rule_id=None,
                )
        return _location_result(
            facts,
            status="unknown",
            bucket="unknown_remote",
            reason="remote eligibility unclear",
            rule_id=None,
        )

    if facts.remote_type == RemoteType.HYBRID:
        if facts.remote_scope.value in location_filters.remote.exclude_scopes:
            return _location_result(
                facts,
                status="ineligible",
                bucket="hybrid_restricted",
                reason=_scope_reason(facts.remote_scope.value),
                rule_id="remote_scope",
            )
        if foreign_countries:
            return _location_result(
                facts,
                status="ineligible",
                bucket="foreign_office",
                reason=f"listed location outside allowed countries: {_country_names(foreign_countries)}",
                rule_id="country",
            )
        if foreign_regions:
            return _location_result(
                facts,
                status="ineligible",
                bucket="foreign_office",
                reason=f"listed location outside allowed countries: {_region_names(foreign_regions)}",
                rule_id="country",
            )
        if facts.remote_scope.value in location_filters.remote.allowed_scopes:
            return _location_result(
                facts,
                status="eligible",
                bucket="remote_apac" if facts.remote_scope == RemoteScope.APAC else "remote_global",
                reason="explicitly APAC remote" if facts.remote_scope == RemoteScope.APAC else "explicitly global remote",
                rule_id=None,
            )
        return _location_result(
            facts,
            status="unknown",
            bucket="unknown_hybrid",
            reason="hybrid eligibility unclear",
            rule_id=None,
        )

    if foreign_countries:
        return _location_result(
            facts,
            status="ineligible",
            bucket="foreign_office",
            reason=f"listed location outside allowed countries: {_country_names(foreign_countries)}",
            rule_id="country",
        )
    if foreign_regions:
        return _location_result(
            facts,
            status="ineligible",
            bucket="foreign_office",
            reason=f"listed location outside allowed countries: {_region_names(foreign_regions)}",
            rule_id="country",
        )

    if location_filters.unknown_policy != "pass":
        return _location_result(
            facts,
            status="ineligible",
            bucket="unknown_location",
            reason="location eligibility unknown",
            rule_id="country",
        )

    return _location_result(
        facts,
        status="unknown",
        bucket="unknown_location",
        reason="location eligibility unknown",
        rule_id=None,
    )


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
    location = _assess_location(job, filters.locations)
    if location["status"] == "ineligible" and location["rule_id"]:
        failed.append(location["rule_id"])
    elif location["status"] == "unknown":
        notes.append(location["reason"])

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
        "remote_type": location["remote_type"],
        "remote_scope": location["remote_scope"],
        "country": location["country"],
        "location": location,
        "work_auth_blocker": blocker,
    }
    return FilterResult(passed=not failed, rules_failed=failed, signals=signals, notes=notes)


def explain(result: FilterResult) -> list[str]:
    return [f"{rid}: {RULE_DESCRIPTIONS.get(rid, 'unknown rule')}" for rid in result.rules_failed]
