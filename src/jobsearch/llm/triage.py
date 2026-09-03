"""Optional LLM triage of top-ranked jobs.

Disabled by default. This is a *second opinion* layer: it never replaces the
deterministic rank, and the pipeline is fully functional without it.

This module talks only to llm.provider.get_provider(). It has no knowledge of
which provider is configured and never touches a binary or an SDK directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..config.loader import AppConfig
from ..persistence.repositories import JobRepo, LlmCallRepo, ScoreRepo
from ..services.audit import Audit
from .cache import ResponseCache
from .cost import BudgetGuard, to_inr
from .provider import (
    LLMBudgetError,
    LLMError,
    LLMUnavailableError,
    get_provider,
    prompt_hash,
)
from .schemas import TRIAGE_JSON_SPEC, TriageResult

SYSTEM_TEMPLATE = """You assess whether a job posting is a good match for one specific candidate.

CANDIDATE PROFILE (the only facts you may rely on):
{profile}

RULES:
- Judge fit only from the profile above. Never assume experience that is not listed.
- Skills listed as "developing familiarity" are NOT production experience; weigh them lower.
- The candidate is early-career. A role demanding substantial senior experience is a stretch or a poor fit, however appealing.
- Be accurate rather than encouraging. A wrong "strong" costs the candidate time.

Return exactly this JSON shape:
{spec}"""

USER_TEMPLATE = """JOB POSTING

Title: {title}
Company: {company}
Location: {location} ({remote_type})
Employment type: {employment_type}

Description:
{description}"""


@dataclass
class TriageReport:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    from_cache: int = 0
    cost_inr: float = 0.0
    provider: str = ""
    errors: list[str] = field(default_factory=list)
    skipped_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "attempted": self.attempted, "succeeded": self.succeeded,
            "failed": self.failed, "from_cache": self.from_cache,
            "cost_inr": round(self.cost_inr, 4), "provider": self.provider,
            "skipped_reason": self.skipped_reason,
        }


def run_triage(conn, config: AppConfig, *, top_n: int = 25,
               dry_run: bool = False) -> TriageReport:
    """Triage the top N ranked jobs. Never raises on provider failure."""
    report = TriageReport()
    llm = config.settings.llm

    if not llm.enabled:
        report.skipped_reason = "llm.enabled is false in config/settings.yaml"
        return report

    try:
        provider = get_provider(llm)
    except LLMUnavailableError as exc:
        report.skipped_reason = str(exc)
        return report

    report.provider = provider.name
    ok, reason = provider.available()
    if not ok:
        report.skipped_reason = f"provider {provider.name} unavailable: {reason}"
        return report

    call_repo = LlmCallRepo(conn)
    guard = BudgetGuard(call_repo, daily_cap_inr=llm.daily_cap_inr,
                        usd_to_inr=llm.usd_to_inr)
    within, budget_msg = guard.check()
    if not within:
        report.skipped_reason = budget_msg
        return report

    cache = ResponseCache(config.root / "data" / "cache" / "llm")
    score_repo, job_repo = ScoreRepo(conn), JobRepo(conn)
    audit = Audit(conn)

    rows = score_repo.top(top_n, exclude_applied=True)
    system = SYSTEM_TEMPLATE.format(
        profile=config.profile.matching_text(), spec=TRIAGE_JSON_SPEC)

    for row in rows:
        if score_repo.get(row["id"], stage="triage") is not None:
            continue
        report.attempted += 1

        user = USER_TEMPLATE.format(
            title=row["title"], company=row["company_name_raw"],
            location=row["location_raw"] or "unspecified",
            remote_type=row["remote_type"], employment_type=row["employment_type"],
            description=(row["description_text"] or "")[: llm.max_jd_chars],
        )

        if dry_run:
            continue

        phash = prompt_hash(system, user, provider.model)
        cached = cache.get(phash, provider.model)
        if cached is not None:
            try:
                result = TriageResult(**cached["data"])
                _save(score_repo, row["id"], result, provider.model)
                report.succeeded += 1
                report.from_cache += 1
                continue
            except Exception:
                pass  # stale cache entry; fall through to a live call

        within, budget_msg = guard.check()
        if not within:
            report.errors.append(budget_msg)
            break

        try:
            res = provider.complete(system=system, user=user,
                                    schema=TriageResult, purpose="triage")
        except LLMError as exc:
            # A provider failure degrades triage only. The deterministic
            # rank for this job is already saved and remains authoritative.
            report.failed += 1
            report.errors.append(f"job {row['id']}: {type(exc).__name__}: {exc}")
            audit.system("triage_failed", job_id=row["id"], error=type(exc).__name__)
            continue

        cost_inr = to_inr(res.usage.cost_usd, llm.usd_to_inr)
        report.cost_inr += cost_inr
        call_repo.log(
            purpose="triage", model=res.model, prompt_hash=res.prompt_hash,
            job_id=row["id"], input_tokens=res.usage.input_tokens,
            cached_input_tokens=res.usage.cached_input_tokens,
            output_tokens=res.usage.output_tokens, cost_usd=res.usage.cost_usd,
            cost_inr=cost_inr, from_cache=False,
        )
        cache.put(phash, provider.model,
                  {"data": res.data.model_dump(), "provider": res.provider})
        _save(score_repo, row["id"], res.data, res.model)
        report.succeeded += 1

    audit.system("triage_complete", **report.as_dict())
    return report


def _save(score_repo: ScoreRepo, job_id: int, result: TriageResult, model: str) -> None:
    score_repo.conn.execute(
        """INSERT OR REPLACE INTO job_scores
           (job_id, stage, score, components, rank_version, model, rationale,
            matched, missing, flags, explanation, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        (job_id, "triage", float(result.fit_score), "{}", 1, model,
         result.one_line_rationale, json.dumps(result.matched_requirements),
         json.dumps(result.missing_requirements),
         json.dumps(result.red_flags + [result.verdict]),
         json.dumps([f"seniority: {result.seniority_assessment}"])),
    )
