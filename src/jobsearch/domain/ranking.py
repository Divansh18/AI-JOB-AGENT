"""Deterministic ranking.

No LLM. Every component is bounded, explainable and reproducible, and the
breakdown is persisted so the digest can show *why* a job ranked where it did.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from ..config.schemas import Profile, RankingConfig
from .extract import RemoteScope, RemoteType, YoeVerdict
from .models import Job, RankedJob, ScoreComponents
from .normalize import tokenize
from .taxonomy import TitleTier, canonical_skill, classify_title

TIER_SCORES = {
    TitleTier.CORE_SWE: 1.00,
    TitleTier.AI: 0.90,
    TitleTier.STACK: 0.90,
    TitleTier.ADJACENT: 0.50,
    TitleTier.AMBIGUOUS: 0.25,
    TitleTier.EXCLUDED: 0.0,
}


def _skill_surface_forms(skill: str) -> list[str]:
    """Surface forms to search for in a JD, given a canonical skill name."""
    forms = {skill}
    if "." in skill:
        forms.add(skill.replace(".", ""))
        forms.add(skill.replace(".", " "))
    if skill.endswith(".js"):
        forms.add(skill[:-3])
    forms.add(skill.replace("-", " "))
    return [f for f in forms if len(f) >= 2]


def match_skills(description: str, profile: Profile) -> tuple[list[str], list[str]]:
    """Find which of the candidate's declared skills appear in the JD.

    Strong and familiar skills are returned separately and never merged: a
    developing familiarity must not be presented as production experience.
    """
    text = " " + re.sub(r"[^a-z0-9\.\+#\s]", " ", (description or "").lower()) + " "

    def hits(skills: list[str]) -> list[str]:
        found = []
        for raw in skills:
            skill = canonical_skill(raw)
            for form in _skill_surface_forms(skill):
                pattern = r"(?<![a-z0-9])" + re.escape(form) + r"(?![a-z0-9])"
                if re.search(pattern, text):
                    found.append(raw)
                    break
        return found

    return hits(profile.skills.strong), hits(profile.skills.familiar)


def location_score(job: Job, profile: Profile, signals: dict | None = None) -> tuple[float, str]:
    """0..1 location fit plus a human-readable reason."""
    location = (signals or {}).get("location") or {}
    preferred = {p.strip().lower() for p in profile.preferred_locations}
    job_cities = {c.lower() for c in (location.get("cities") or job.locations)}
    if job_cities & preferred:
        return 1.0, f"preferred city: {', '.join(sorted(job_cities & preferred))}"
    bucket = location.get("bucket")
    reason = location.get("reason")
    status = location.get("status")
    if bucket == "india_remote":
        return 0.95, reason or "remote within India"
    if bucket == "india_multi_location":
        return 0.92, reason or "India included among listed locations"
    if bucket in {"india_city", "india_country"}:
        return 0.88, reason or "India location stated"
    if bucket == "remote_global":
        return 0.68, reason or "explicitly global remote"
    if bucket == "remote_apac":
        return 0.6, reason or "explicitly APAC remote"
    if status == "unknown":
        return 0.3, reason or "location eligibility unknown"

    # Fallback for direct unit tests that do not pass filter signals.
    if job.country == "IN" and job.remote_type == RemoteType.REMOTE.value:
        return 0.95, "remote within India"
    if job.country == "IN":
        return 0.88, "India location stated"
    if job.remote_scope == RemoteScope.INDIA.value:
        return 0.95, "remote within India"
    if job.remote_scope == RemoteScope.GLOBAL.value:
        return 0.68, "explicitly global remote"
    if job.remote_scope == RemoteScope.APAC.value:
        return 0.6, "explicitly APAC remote"
    if job.remote_type == RemoteType.REMOTE.value:
        return 0.3, "remote eligibility unclear"
    return 0.25, "location eligibility unknown"


def freshness_score(job: Job, now: datetime, halflife_hours: float) -> tuple[float, float]:
    """Exponential decay on age. Returns (0..1 score, age_hours)."""
    reference = job.posted_at or job.first_seen_at
    ref = reference if reference.tzinfo else reference.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (now - ref).total_seconds() / 3600)
    return math.exp(-age_hours / halflife_hours), age_hours


def title_score(job: Job) -> tuple[float, str]:
    ta = classify_title(job.title_normalized)
    return TIER_SCORES.get(ta.tier, 0.0), ta.tier.value


def percentile_normalize(values: list[float]) -> list[float]:
    """Map raw similarities onto 0..1 by rank.

    bge-small cosine similarities for profile-vs-JD text sit in a narrow band
    (~0.60-0.85), so raw values carry almost no discriminating power. Ranking
    against the current pool restores it.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    for rank, idx in enumerate(order):
        out[idx] = rank / (n - 1)
    return out


def score_job(
    job: Job,
    *,
    profile: Profile,
    config: RankingConfig,
    semantic_norm: float,
    signals: dict,
    company_priority: str = "normal",
    now: datetime | None = None,
) -> RankedJob:
    """Compute the final 0-100 score with a full component breakdown."""
    now = now or datetime.now(timezone.utc)
    w = config.weights
    comp = ScoreComponents()
    explanation: list[str] = []
    flags: list[str] = []

    comp.semantic = w.semantic * max(0.0, min(1.0, semantic_norm))
    explanation.append(f"semantic fit {semantic_norm:.2f} of pool")

    t_raw, tier = title_score(job)
    comp.title = w.title * t_raw
    explanation.append(f"title tier {tier}")

    f_raw, age_hours = freshness_score(job, now, config.freshness_halflife_hours)
    comp.freshness = w.freshness * f_raw
    explanation.append(f"{age_hours:.0f}h old")

    l_raw, l_reason = location_score(job, profile, signals)
    comp.location = w.location * l_raw
    explanation.append(l_reason)

    strong, familiar = match_skills(job.description_text, profile)
    denom = max(1, len(profile.skills.strong))
    raw_skill = (len(strong) + config.familiar_skill_weight * len(familiar)) / denom
    comp.skills = w.skills * min(1.0, raw_skill)
    if strong:
        explanation.append(f"{len(strong)} core skills matched")

    comp.priority = {"high": w.priority, "normal": w.priority * 0.4, "low": 0.0}.get(
        company_priority, w.priority * 0.4
    )

    # --- adjustments ---
    adj = 0.0
    yoe_verdict = (signals.get("yoe") or {}).get("verdict", YoeVerdict.UNKNOWN.value)
    adj += getattr(config.yoe_adjustments, yoe_verdict, 0.0)
    if yoe_verdict == YoeVerdict.PENALTY.value:
        flags.append("3 yrs required")
    elif yoe_verdict == YoeVerdict.SOFT_HIGH.value:
        flags.append("4+ yrs preferred")
    elif yoe_verdict == YoeVerdict.STRONG.value:
        flags.append("early-career friendly")

    emp = job.employment_type
    adj += getattr(config.employment_adjustments, emp, 0.0)
    if emp == "internship":
        flags.append("internship")
    elif emp == "contract":
        flags.append("contract")

    location = signals.get("location") or {}
    if signals.get("early_career_signal"):
        adj += config.early_career_title_bonus
    if location.get("status") == "unknown":
        flags.append("location eligibility unknown")
    elif location.get("bucket") == "remote_global":
        flags.append("global remote")
    elif location.get("bucket") == "remote_apac":
        flags.append("APAC remote")
    if (location.get("remote_type") or job.remote_type) == RemoteType.HYBRID.value:
        flags.append("hybrid")
    if signals.get("work_auth_blocker"):
        flags.append("work-auth language present")

    comp.adjustments = adj
    total = max(0.0, min(100.0, comp.total()))

    return RankedJob(
        job=job,
        components=comp,
        score=round(total, 2),
        matched_skills_strong=strong,
        matched_skills_familiar=familiar,
        flags=flags,
        explanation=explanation,
    )
