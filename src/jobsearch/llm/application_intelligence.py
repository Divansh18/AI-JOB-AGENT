"""Optional provider-backed JD intelligence for one already-promising job."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from ..domain.application_intelligence import (
    ApplicationIntelligenceIssue,
    ApplicationIntelligenceReport,
    LlmJobInsight,
    LlmRequirement,
    make_requirement,
    validate_llm_job_insight,
)
from ..persistence.repositories import LlmArtifactRepo, LlmCallRepo
from ..services.application_intelligence_context import (
    DEFAULT_PROMPT_VERSION,
    PURPOSE_JOB_INTELLIGENCE,
    ApplicationIntelligenceContext,
    ApplicationIntelligenceContextError,
    build_application_intelligence_context,
    context_fingerprint,
)
from ..services.audit import Audit
from .cache import ResponseCache
from .cost import BudgetGuard, estimate_cost_usd, to_inr
from .provider import LLMError, LLMUnavailableError, get_provider, prompt_hash
from .schemas import APPLICATION_INTELLIGENCE_JSON_SPEC, LlmJobInsightExtraction

SYSTEM_TEMPLATE = """You provide bounded JD intelligence for one already-screened job.

Hard rules:
- Deterministic filtering, fit scoring, and resume safety gates remain authoritative.
- Do not override deterministic blockers or poor_fit decisions.
- Use only candidate evidence refs present in allowed_evidence_refs.
- Evidence refs are opaque canonical IDs such as fact:123; never cite category:key selectors or refs that are absent from allowed_evidence_refs.
- Every matched or partial candidate-match claim must cite evidence_refs.
- Never invent experience, skills, metrics, dates, YOE, education, visa/work authorization, salary, notice period, relocation, legal answers, or unsupported facts.
- For salary, work authorization, sponsorship, relocation, notice period, demographic, or legal/attestation topics: identify the requirement if the JD states it, but do not infer the candidate answer.
- Treat job title, company, location, department, employment type, and ATS/navigation text such as "Apply for this job" as metadata, not as skill/experience requirements.
- If the JD has an explicit location constraint, include it only as a concise category="location" requirement, not as a copied page header.
- Return one JSON object only, with this exact structure:
{spec}"""

USER_TEMPLATE = """DETERMINISTIC_CONTEXT_JSON
{context_json}"""


class ApplicationIntelligenceError(Exception):
    pass


def analyze_job_intelligence(conn, config, job_id: int) -> ApplicationIntelligenceReport:
    """Run Phase 2C.1 JD intelligence for one explicit job id.

    This function never runs when llm.enabled is false, never batches jobs, and
    returns a deterministic status instead of raising on provider/cost failures.
    """
    settings = config.settings.llm
    purpose = PURPOSE_JOB_INTELLIGENCE
    prompt_version = str(getattr(settings, "prompt_version", DEFAULT_PROMPT_VERSION) or DEFAULT_PROMPT_VERSION)
    model = str(getattr(settings, "model", "") or "unknown")
    provider_name = str(getattr(settings, "provider", "") or "unknown")
    max_input_chars = int(getattr(settings, "max_input_chars", 16000))

    if not getattr(settings, "enabled", False):
        return ApplicationIntelligenceReport(
            job_id=job_id,
            status="skipped",
            purpose=purpose,
            prompt_version=prompt_version,
            context_fingerprint="",
            source_truth_hash="",
            provider=provider_name,
            model=model,
            errors=["llm_disabled"],
        )

    try:
        context = build_application_intelligence_context(
            conn,
            config,
            job_id,
            purpose=purpose,
            prompt_version=prompt_version,
            max_jd_chars=int(getattr(settings, "max_jd_chars", 6000)),
            max_context_chars=_context_payload_budget(max_input_chars),
        )
    except ApplicationIntelligenceContextError as exc:
        return ApplicationIntelligenceReport(
            job_id=job_id,
            status="failed",
            purpose=purpose,
            prompt_version=prompt_version,
            context_fingerprint="",
            source_truth_hash="",
            provider=provider_name,
            model=model,
            errors=[str(exc)],
        )

    fingerprint = context_fingerprint(context, model=model)
    system, user = _prompts(context)
    phash = prompt_hash(system, user, model)
    gate_errors = _gate_errors(context)
    if gate_errors:
        return _save_report(
            conn,
            _base_report(context, provider_name=provider_name, model=model, fingerprint=fingerprint, prompt_hash_value=phash),
            context=context,
            status="blocked",
            errors=gate_errors,
        )

    if len(system) + len(user) > max_input_chars:
        return _save_report(
            conn,
            _base_report(context, provider_name=provider_name, model=model, fingerprint=fingerprint, prompt_hash_value=phash),
            context=context,
            status="blocked",
            errors=[f"input_context_too_large: {len(system) + len(user)} chars > {max_input_chars}"],
        )

    artifact_repo = LlmArtifactRepo(conn)
    existing = artifact_repo.find_valid(
        job_id=job_id,
        purpose=purpose,
        prompt_version=prompt_version,
        provider=provider_name,
        model=model,
        context_fingerprint=fingerprint,
    )
    if existing is not None:
        return replace(_report_from_row(existing), from_cache=True)

    cache = ResponseCache(config.root / "data" / "cache" / "llm")
    cached = cache.get(phash, model)
    if cached is not None:
        cached_report = _report_from_cached(context, cached, provider_name, model, fingerprint, phash)
        if cached_report is not None:
            return _save_report(conn, cached_report, context=context, from_cache=True)

    budget_error = _budget_error(conn, settings, model, system, user)
    if budget_error:
        return _save_report(
            conn,
            _base_report(context, provider_name=provider_name, model=model, fingerprint=fingerprint, prompt_hash_value=phash),
            context=context,
            status="blocked",
            errors=[budget_error],
        )

    try:
        provider = get_provider(settings)
    except LLMUnavailableError as exc:
        return _save_report(
            conn,
            _base_report(context, provider_name=provider_name, model=model, fingerprint=fingerprint, prompt_hash_value=phash),
            context=context,
            status="failed",
            errors=[f"provider_unavailable: {exc}"],
        )

    ok, reason = provider.available()
    if not ok:
        return _save_report(
            conn,
            _base_report(context, provider_name=provider.name, model=provider.model, fingerprint=fingerprint, prompt_hash_value=phash),
            context=context,
            status="failed",
            errors=[f"provider_unavailable: {reason}"],
        )

    try:
        result = provider.complete(
            system=system,
            user=user,
            schema=LlmJobInsightExtraction,
            purpose=purpose,
        )
    except LLMError as exc:
        return _save_report(
            conn,
            _base_report(context, provider_name=provider.name, model=provider.model, fingerprint=fingerprint, prompt_hash_value=phash),
            context=context,
            status="failed",
            errors=[f"{type(exc).__name__}: {exc}"],
        )

    cost_inr = to_inr(result.usage.cost_usd, getattr(settings, "usd_to_inr", 88.0))
    LlmCallRepo(conn).log(
        purpose=purpose,
        model=result.model or provider.model,
        prompt_hash=result.prompt_hash or phash,
        job_id=job_id,
        input_tokens=result.usage.input_tokens,
        cached_input_tokens=result.usage.cached_input_tokens,
        output_tokens=result.usage.output_tokens,
        cost_usd=result.usage.cost_usd,
        cost_inr=cost_inr,
        from_cache=False,
    )

    report = _report_from_schema(
        context,
        result.data,
        provider_name=result.provider or provider.name,
        model=result.model or provider.model,
        fingerprint=fingerprint,
        prompt_hash_value=result.prompt_hash or phash,
        from_cache=False,
        cost_inr=cost_inr,
    )
    if report.ok:
        cache.put(phash, model, {"data": result.data.model_dump(), "provider": report.provider})
    return _save_report(conn, report, context=context)


def get_artifact(conn, artifact_id: int) -> dict[str, Any]:
    row = LlmArtifactRepo(conn).get(artifact_id)
    if row is None:
        raise ApplicationIntelligenceError(f"LLM artifact {artifact_id} not found")
    return _artifact_payload(row)


def list_artifacts(conn, *, job_id: int | None = None, limit: int = 20) -> list[dict[str, Any]]:
    rows = LlmArtifactRepo(conn).list(job_id=job_id, purpose=PURPOSE_JOB_INTELLIGENCE, limit=limit)
    return [_artifact_payload(row) for row in rows]


def _gate_errors(context: ApplicationIntelligenceContext) -> list[str]:
    errors: list[str] = []
    filter_payload = context.deterministic.get("filter") or {}
    if not filter_payload.get("present"):
        errors.append("job_has_no_deterministic_filter_result")
    elif not filter_payload.get("passed"):
        errors.append(
            "job_did_not_pass_deterministic_filters: "
            + ", ".join(filter_payload.get("rules_failed") or ["unknown"])
        )
    location = ((filter_payload.get("signals") or {}).get("location") or {})
    if location.get("status") == "ineligible":
        errors.append(f"location_ineligible: {location.get('reason') or 'deterministic location gate'}")
    resume = context.deterministic.get("resume") or {}
    if resume.get("decision") == "poor_fit":
        errors.append("poor_fit_resume_decision: " + "; ".join(resume.get("reasons") or []))
    hard_blockers = (context.deterministic.get("fit") or {}).get("hard_blockers") or []
    if hard_blockers:
        errors.append("deterministic_fit_blockers: " + "; ".join(str(item) for item in hard_blockers))
    return _dedupe_strings(errors)


def _budget_error(conn, settings, model: str, system: str, user: str) -> str | None:
    daily_guard = BudgetGuard(
        LlmCallRepo(conn),
        daily_cap_inr=float(getattr(settings, "daily_cap_inr", 15.0)),
        usd_to_inr=float(getattr(settings, "usd_to_inr", 88.0)),
    )
    ok, msg = daily_guard.check()
    if not ok:
        return msg

    repo = LlmCallRepo(conn)
    month_spend = repo.spend_inr(since=_month_start_iso())
    monthly_cap = float(getattr(settings, "monthly_cap_inr", 500.0))
    if month_spend >= monthly_cap:
        return f"monthly cap reached: INR {month_spend:.2f} / {monthly_cap:.2f}"

    estimated = _estimated_call_cost_inr(settings, model, system, user)
    per_call_cap = float(getattr(settings, "per_call_cap_inr", 25.0))
    if estimated > per_call_cap:
        return f"per-call cap would be exceeded: estimated INR {estimated:.2f} / {per_call_cap:.2f}"
    if month_spend + estimated > monthly_cap:
        return f"monthly cap would be exceeded: estimated INR {month_spend + estimated:.2f} / {monthly_cap:.2f}"
    return None


def _estimated_call_cost_inr(settings, model: str, system: str, user: str) -> float:
    input_tokens = _approx_tokens(system) + _approx_tokens(user)
    output_tokens = int(getattr(settings, "max_output_tokens", 1024))
    cost_usd = estimate_cost_usd(model, input_tokens=input_tokens, output_tokens=output_tokens)
    return to_inr(cost_usd, getattr(settings, "usd_to_inr", 88.0))


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text or "") / 4))


def _month_start_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()


def _prompts(context: ApplicationIntelligenceContext) -> tuple[str, str]:
    system = SYSTEM_TEMPLATE.format(spec=APPLICATION_INTELLIGENCE_JSON_SPEC)
    context_json = json.dumps(context.as_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    user = USER_TEMPLATE.format(context_json=context_json)
    return system, user


def _context_payload_budget(max_input_chars: int) -> int:
    system = SYSTEM_TEMPLATE.format(spec=APPLICATION_INTELLIGENCE_JSON_SPEC)
    empty_user = USER_TEMPLATE.format(context_json="")
    return max(1000, max_input_chars - len(system) - len(empty_user) - 64)


def _base_report(
    context: ApplicationIntelligenceContext,
    *,
    provider_name: str,
    model: str,
    fingerprint: str,
    prompt_hash_value: str,
) -> ApplicationIntelligenceReport:
    return ApplicationIntelligenceReport(
        job_id=context.job_id,
        status="failed",
        purpose=context.purpose,
        prompt_version=context.prompt_version,
        context_fingerprint=fingerprint,
        source_truth_hash=context.source_truth_hash,
        provider=provider_name,
        model=model,
        prompt_hash=prompt_hash_value,
        allowed_evidence_refs=context.allowed_evidence_refs,
        deterministic_fit_score=context.deterministic_fit_score,
        deterministic_resume_decision=context.deterministic_resume_decision,
    )


def _report_from_cached(
    context: ApplicationIntelligenceContext,
    cached: dict[str, Any],
    provider_name: str,
    model: str,
    fingerprint: str,
    phash: str,
) -> ApplicationIntelligenceReport | None:
    try:
        output = LlmJobInsightExtraction(**(cached.get("data") or {}))
    except Exception:
        return None
    return _report_from_schema(
        context,
        output,
        provider_name=str(cached.get("provider") or provider_name),
        model=model,
        fingerprint=fingerprint,
        prompt_hash_value=phash,
        from_cache=True,
        cost_inr=0.0,
    )


def _report_from_schema(
    context: ApplicationIntelligenceContext,
    output: LlmJobInsightExtraction,
    *,
    provider_name: str,
    model: str,
    fingerprint: str,
    prompt_hash_value: str,
    from_cache: bool,
    cost_inr: float,
) -> ApplicationIntelligenceReport:
    insight = _insight_from_schema(context.job_id, output)
    issues = validate_llm_job_insight(
        insight,
        allowed_evidence_refs=context.allowed_evidence_refs,
        deterministic_blockers=context.deterministic_blockers,
    )
    return ApplicationIntelligenceReport(
        job_id=context.job_id,
        status="invalid" if issues else "valid",
        purpose=context.purpose,
        prompt_version=context.prompt_version,
        context_fingerprint=fingerprint,
        source_truth_hash=context.source_truth_hash,
        provider=provider_name,
        model=model,
        prompt_hash=prompt_hash_value,
        insight=insight,
        validation_errors=issues,
        allowed_evidence_refs=context.allowed_evidence_refs,
        deterministic_fit_score=context.deterministic_fit_score,
        deterministic_resume_decision=context.deterministic_resume_decision,
        from_cache=from_cache,
        cost_inr=cost_inr,
    )


def _insight_from_schema(job_id: int, output: LlmJobInsightExtraction) -> LlmJobInsight:
    must = [
        make_requirement(
            text=item.text,
            category=item.category,
            required=True,
            candidate_match=item.candidate_match,
            evidence_refs=item.evidence_refs,
            rationale=item.rationale,
            prefix="must",
        )
        for item in output.must_have_requirements
    ]
    preferred = [
        make_requirement(
            text=item.text,
            category=item.category,
            required=False,
            candidate_match=item.candidate_match,
            evidence_refs=item.evidence_refs,
            rationale=item.rationale,
            prefix="preferred",
        )
        for item in output.preferred_requirements
    ]
    return LlmJobInsight(
        job_id=job_id,
        role_summary=output.role_summary,
        must_have_requirements=must,
        preferred_requirements=preferred,
        role_priorities=output.role_priorities,
        grounded_fit_assessment=output.grounded_fit_assessment,
        fit_verdict=output.fit_verdict,
        candidate_match_evidence_refs=_dedupe_strings(output.candidate_match_evidence_refs),
        uncertainties=output.uncertainties,
        gaps=output.gaps,
    )


def _save_report(
    conn,
    report: ApplicationIntelligenceReport,
    *,
    context: ApplicationIntelligenceContext | None = None,
    status: str | None = None,
    errors: list[str] | None = None,
    from_cache: bool | None = None,
) -> ApplicationIntelligenceReport:
    stored = replace(
        report,
        status=status or report.status,
        errors=errors if errors is not None else report.errors,
        from_cache=report.from_cache if from_cache is None else from_cache,
    )
    context_input = context.as_dict() if context is not None else _input_json_for_report(stored)
    artifact_id = LlmArtifactRepo(conn).save(
        job_id=stored.job_id,
        purpose=stored.purpose,
        prompt_version=stored.prompt_version,
        provider=stored.provider or "unknown",
        model=stored.model or "unknown",
        context_fingerprint=stored.context_fingerprint,
        prompt_hash=stored.prompt_hash or "",
        source_truth_hash=stored.source_truth_hash,
        status=stored.status,
        input_json=context_input,
        output_json=stored.insight.as_dict() if stored.insight else {},
        validation_errors=[issue.as_dict() for issue in stored.validation_errors],
        evidence_refs=_report_evidence_refs(stored),
        from_cache=stored.from_cache,
        cost_inr=stored.cost_inr,
    )
    Audit(conn).human(
        "llm_application_intelligence",
        "llm_artifact",
        artifact_id,
        job_id=stored.job_id,
        purpose=stored.purpose,
        status=stored.status,
        provider=stored.provider,
        model=stored.model,
        from_cache=stored.from_cache,
        cost_inr=stored.cost_inr,
    )
    return replace(stored, artifact_id=artifact_id)


def _input_json_for_report(report: ApplicationIntelligenceReport) -> dict[str, Any]:
    return {
        "job_id": report.job_id,
        "purpose": report.purpose,
        "prompt_version": report.prompt_version,
        "context_fingerprint": report.context_fingerprint,
        "source_truth_hash": report.source_truth_hash,
        "allowed_evidence_refs": report.allowed_evidence_refs,
        "deterministic_fit_score": report.deterministic_fit_score,
        "deterministic_resume_decision": report.deterministic_resume_decision,
    }


def _report_evidence_refs(report: ApplicationIntelligenceReport) -> list[str]:
    if report.insight is None:
        return []
    refs: list[str] = list(report.insight.candidate_match_evidence_refs)
    for requirement in report.insight.must_have_requirements + report.insight.preferred_requirements:
        refs.extend(requirement.evidence_refs)
    return _dedupe_strings(refs)


def _report_from_row(row) -> ApplicationIntelligenceReport:
    output = _loads(row["output_json"], {})
    issues = [
        ApplicationIntelligenceIssue(
            code=item.get("code", ""),
            message=item.get("message", ""),
            evidence_refs=item.get("evidence_refs") or [],
        )
        for item in _loads(row["validation_errors"], [])
    ]
    insight = _insight_from_dict(int(row["job_id"]), output) if output else None
    input_json = _loads(row["input_json"], {})
    return ApplicationIntelligenceReport(
        artifact_id=int(row["id"]),
        job_id=int(row["job_id"]),
        status=row["status"],
        purpose=row["purpose"],
        prompt_version=row["prompt_version"],
        context_fingerprint=row["context_fingerprint"],
        source_truth_hash=row["source_truth_hash"],
        provider=row["provider"],
        model=row["model"],
        prompt_hash=row["prompt_hash"],
        insight=insight,
        validation_errors=issues,
        errors=[],
        allowed_evidence_refs=input_json.get("allowed_evidence_refs") or [],
        deterministic_fit_score=_stored_fit_score(input_json),
        deterministic_resume_decision=_stored_resume_decision(input_json),
        from_cache=bool(row["from_cache"]),
        cost_inr=float(row["cost_inr"] or 0.0),
        created_at=_parse_dt(row["created_at"]),
    )


def _insight_from_dict(job_id: int, payload: dict[str, Any]) -> LlmJobInsight:
    must = [
        LlmRequirement(
            id=item.get("id", ""),
            text=item.get("text", ""),
            category=item.get("category", "other"),
            required=bool(item.get("required", True)),
            candidate_match=item.get("candidate_match", "unknown"),
            evidence_refs=item.get("evidence_refs") or [],
            rationale=item.get("rationale", ""),
        )
        for item in payload.get("must_have_requirements", [])
    ]
    preferred = [
        LlmRequirement(
            id=item.get("id", ""),
            text=item.get("text", ""),
            category=item.get("category", "other"),
            required=bool(item.get("required", False)),
            candidate_match=item.get("candidate_match", "unknown"),
            evidence_refs=item.get("evidence_refs") or [],
            rationale=item.get("rationale", ""),
        )
        for item in payload.get("preferred_requirements", [])
    ]
    return LlmJobInsight(
        job_id=job_id,
        role_summary=payload.get("role_summary", ""),
        must_have_requirements=must,
        preferred_requirements=preferred,
        role_priorities=payload.get("role_priorities") or [],
        grounded_fit_assessment=payload.get("grounded_fit_assessment", ""),
        fit_verdict=payload.get("fit_verdict", "unknown"),
        candidate_match_evidence_refs=payload.get("candidate_match_evidence_refs") or [],
        uncertainties=payload.get("uncertainties") or [],
        gaps=payload.get("gaps") or [],
    )


def _artifact_payload(row) -> dict[str, Any]:
    base = dict(row)
    base["input_json"] = _loads(row["input_json"], {})
    base["output_json"] = _loads(row["output_json"], {})
    base["validation_errors"] = _loads(row["validation_errors"], [])
    base["evidence_refs"] = _loads(row["evidence_refs"], [])
    base["from_cache"] = bool(row["from_cache"])
    return base


def _stored_fit_score(input_json: dict[str, Any]) -> int | None:
    direct = input_json.get("deterministic_fit_score")
    if direct is not None:
        return int(direct)
    nested = ((input_json.get("deterministic") or {}).get("fit") or {}).get("overall_fit_score")
    return int(nested) if nested is not None else None


def _stored_resume_decision(input_json: dict[str, Any]) -> str | None:
    direct = input_json.get("deterministic_resume_decision")
    if direct:
        return str(direct)
    nested = ((input_json.get("deterministic") or {}).get("resume") or {}).get("decision")
    return str(nested) if nested else None


def _loads(raw: str | None, fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out
