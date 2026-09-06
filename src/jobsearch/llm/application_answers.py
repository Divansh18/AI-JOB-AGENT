"""Optional provider-backed grounded application-answer drafting."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from ..domain.application_intelligence import (
    AnswerDraftSet,
    ApplicationIntelligenceIssue,
    ApplicationQuestion,
    GroundedAnswerDraft,
    validate_grounded_answer_draft,
)
from ..persistence.repositories import ApplicationAnswerDraftRepo, LlmCallRepo
from ..services.application_intelligence_context import (
    DEFAULT_ANSWER_PROMPT_VERSION,
    PURPOSE_ANSWER_DRAFTING,
    ApplicationAnswerContext,
    ApplicationIntelligenceContextError,
    answer_context_fingerprint,
    build_application_answer_context,
)
from ..services.audit import Audit
from .application_intelligence import _budget_error
from .cache import ResponseCache
from .cost import to_inr
from .provider import LLMError, LLMUnavailableError, get_provider, prompt_hash
from .schemas import APPLICATION_ANSWER_DRAFT_JSON_SPEC, GroundedAnswerDraftSetExtraction


SYSTEM_TEMPLATE = """You draft grounded free-text job application answers.

Hard rules:
- Draft only for the safe_free_text questions supplied in safe_questions.
- Use only candidate evidence refs present in allowed_evidence_refs.
- Every candidate-specific claim must be supported by evidence_refs.
- Use only the supplied job/JD context for company-specific reasoning.
- Do not use public web knowledge, assumptions, or external company research.
- Never answer salary, work authorization, sponsorship, relocation, notice period, demographic/self-identification, criminal history, legal, or attestation questions.
- Never invent experience, skills, metrics, dates, years of experience, education, visa/work authorization, compensation, notice period, or unsupported facts.
- Keep answers concise, specific, and human-review ready.
- Return one JSON object only, with this exact structure:
{spec}"""

USER_TEMPLATE = """ANSWER_DRAFT_CONTEXT_JSON
{context_json}"""


class ApplicationAnswerDraftingError(Exception):
    pass


def draft_application_answers(conn, config, application_id: int) -> AnswerDraftSet:
    settings = config.settings.llm
    purpose = PURPOSE_ANSWER_DRAFTING
    prompt_version = str(
        getattr(settings, "answer_prompt_version", DEFAULT_ANSWER_PROMPT_VERSION)
        or DEFAULT_ANSWER_PROMPT_VERSION
    )
    model = str(getattr(settings, "model", "") or "unknown")
    provider_name = str(getattr(settings, "provider", "") or "unknown")

    if not getattr(settings, "enabled", False):
        return AnswerDraftSet(
            application_id=application_id,
            job_id=0,
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
        context = build_application_answer_context(
            conn,
            config,
            application_id,
            purpose=purpose,
            prompt_version=prompt_version,
            max_jd_chars=int(getattr(settings, "max_jd_chars", 6000)),
        )
    except ApplicationIntelligenceContextError as exc:
        return AnswerDraftSet(
            application_id=application_id,
            job_id=0,
            status="failed",
            purpose=purpose,
            prompt_version=prompt_version,
            context_fingerprint="",
            source_truth_hash="",
            provider=provider_name,
            model=model,
            errors=[str(exc)],
        )

    fingerprint = answer_context_fingerprint(context, model=model)
    system, user = _prompts(context)
    phash = prompt_hash(system, user, model)
    blocked_drafts = _save_blocked_question_drafts(
        conn,
        context,
        provider=provider_name,
        model=model,
        prompt_version=prompt_version,
        context_fingerprint=fingerprint,
        prompt_hash_value=phash,
    )
    base = _base_set(context, provider_name, model, fingerprint, phash)
    plan_blockers = context.application_plan.get("blockers") or []
    if plan_blockers:
        return replace(
            base,
            status="blocked",
            blocked_questions=context.blocked_questions,
            drafts=blocked_drafts,
            errors=[f"plan_blocked:{blocker}" for blocker in plan_blockers],
        )

    if not context.questions:
        return replace(base, status="skipped", errors=["no_application_questions"])
    if not context.safe_questions:
        return replace(
            base,
            status="blocked",
            blocked_questions=context.blocked_questions,
            drafts=blocked_drafts,
            errors=["no_llm_draftable_questions"],
        )

    max_input_chars = int(getattr(settings, "max_input_chars", 16000))
    if len(system) + len(user) > max_input_chars:
        failed = _save_failed_safe_questions(
            conn,
            context,
            provider=provider_name,
            model=model,
            prompt_version=prompt_version,
            context_fingerprint=fingerprint,
            prompt_hash_value=phash,
            status="blocked",
            message=f"input_context_too_large: {len(system) + len(user)} chars > {max_input_chars}",
        )
        return replace(
            base,
            status="blocked",
            drafts=blocked_drafts + failed,
            blocked_questions=context.blocked_questions,
            errors=[f"input_context_too_large: {len(system) + len(user)} chars > {max_input_chars}"],
        )

    repo = ApplicationAnswerDraftRepo(conn)
    existing = repo.find_by_fingerprint(
        application_id=application_id,
        prompt_version=prompt_version,
        provider=provider_name,
        model=model,
        context_fingerprint=fingerprint,
    )
    if _covers_safe_questions(existing, context.safe_questions):
        return replace(
            base,
            status=_draft_set_status(existing),
            drafts=[replace(draft, from_cache=True) for draft in existing],
            blocked_questions=context.blocked_questions,
            from_cache=True,
        )

    cache = ResponseCache(config.root / "data" / "cache" / "llm")
    cached = cache.get(phash, model)
    if cached is not None:
        cached_set = _draft_set_from_cached(
            conn,
            context,
            cached,
            provider_name=provider_name,
            model=model,
            fingerprint=fingerprint,
            prompt_hash_value=phash,
            blocked_drafts=blocked_drafts,
        )
        if cached_set is not None:
            return cached_set

    budget_error = _budget_error(conn, settings, model, system, user)
    if budget_error:
        failed = _save_failed_safe_questions(
            conn,
            context,
            provider=provider_name,
            model=model,
            prompt_version=prompt_version,
            context_fingerprint=fingerprint,
            prompt_hash_value=phash,
            status="blocked",
            message=budget_error,
        )
        return replace(
            base,
            status="blocked",
            drafts=blocked_drafts + failed,
            blocked_questions=context.blocked_questions,
            errors=[budget_error],
        )

    try:
        provider = get_provider(settings)
    except LLMUnavailableError as exc:
        failed = _save_failed_safe_questions(
            conn,
            context,
            provider=provider_name,
            model=model,
            prompt_version=prompt_version,
            context_fingerprint=fingerprint,
            prompt_hash_value=phash,
            status="failed",
            message=f"provider_unavailable: {exc}",
        )
        return replace(
            base,
            status="failed",
            drafts=blocked_drafts + failed,
            blocked_questions=context.blocked_questions,
            errors=[f"provider_unavailable: {exc}"],
        )

    ok, reason = provider.available()
    if not ok:
        failed = _save_failed_safe_questions(
            conn,
            context,
            provider=provider.name,
            model=provider.model,
            prompt_version=prompt_version,
            context_fingerprint=fingerprint,
            prompt_hash_value=phash,
            status="failed",
            message=f"provider_unavailable: {reason}",
        )
        return replace(
            base,
            status="failed",
            drafts=blocked_drafts + failed,
            blocked_questions=context.blocked_questions,
            provider=provider.name,
            model=provider.model,
            errors=[f"provider_unavailable: {reason}"],
        )

    try:
        result = provider.complete(
            system=system,
            user=user,
            schema=GroundedAnswerDraftSetExtraction,
            purpose=purpose,
        )
    except LLMError as exc:
        failed = _save_failed_safe_questions(
            conn,
            context,
            provider=provider.name,
            model=provider.model,
            prompt_version=prompt_version,
            context_fingerprint=fingerprint,
            prompt_hash_value=phash,
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
        )
        return replace(
            base,
            status="failed",
            drafts=blocked_drafts + failed,
            blocked_questions=context.blocked_questions,
            provider=provider.name,
            model=provider.model,
            errors=[f"{type(exc).__name__}: {exc}"],
        )

    cost_inr = to_inr(result.usage.cost_usd, getattr(settings, "usd_to_inr", 88.0))
    LlmCallRepo(conn).log(
        purpose=purpose,
        model=result.model or provider.model,
        prompt_hash=result.prompt_hash or phash,
        job_id=context.job_id,
        input_tokens=result.usage.input_tokens,
        cached_input_tokens=result.usage.cached_input_tokens,
        output_tokens=result.usage.output_tokens,
        cost_usd=result.usage.cost_usd,
        cost_inr=cost_inr,
        from_cache=False,
    )

    draft_set = _draft_set_from_schema(
        conn,
        context,
        result.data,
        provider_name=result.provider or provider.name,
        model=result.model or provider.model,
        fingerprint=fingerprint,
        prompt_hash_value=result.prompt_hash or phash,
        from_cache=False,
        cost_inr=cost_inr,
        blocked_drafts=blocked_drafts,
    )
    if draft_set.status in {"valid", "invalid"}:
        cache.put(phash, model, {"data": result.data.model_dump(), "provider": draft_set.provider})
    return draft_set


def list_answer_drafts(conn, application_id: int) -> list[dict[str, Any]]:
    return [draft.as_dict() for draft in ApplicationAnswerDraftRepo(conn).list_by_application(application_id)]


def get_answer_draft(conn, draft_id: int) -> dict[str, Any]:
    draft = ApplicationAnswerDraftRepo(conn).get(draft_id)
    if draft is None:
        raise ApplicationAnswerDraftingError(f"application answer draft {draft_id} not found")
    return draft.as_dict()


def _prompts(context: ApplicationAnswerContext) -> tuple[str, str]:
    system = SYSTEM_TEMPLATE.format(spec=APPLICATION_ANSWER_DRAFT_JSON_SPEC)
    context_json = json.dumps(_provider_context(context), ensure_ascii=True, sort_keys=True, indent=2)
    return system, USER_TEMPLATE.format(context_json=context_json)


def _provider_context(context: ApplicationAnswerContext) -> dict[str, Any]:
    payload = context.as_dict()
    payload["questions"] = [question.as_dict() for question in context.safe_questions]
    payload["safe_questions"] = [question.as_dict() for question in context.safe_questions]
    payload.pop("blocked_questions", None)
    return payload


def _base_set(
    context: ApplicationAnswerContext,
    provider_name: str,
    model: str,
    fingerprint: str,
    prompt_hash_value: str,
) -> AnswerDraftSet:
    return AnswerDraftSet(
        application_id=context.application_id,
        job_id=context.job_id,
        status="failed",
        purpose=context.purpose,
        prompt_version=context.prompt_version,
        context_fingerprint=fingerprint,
        source_truth_hash=context.source_truth_hash,
        provider=provider_name,
        model=model,
        prompt_hash=prompt_hash_value,
        questions=context.questions,
        blocked_questions=context.blocked_questions,
        allowed_evidence_refs=context.allowed_evidence_refs,
    )


def _draft_set_from_cached(
    conn,
    context: ApplicationAnswerContext,
    cached: dict[str, Any],
    *,
    provider_name: str,
    model: str,
    fingerprint: str,
    prompt_hash_value: str,
    blocked_drafts: list[GroundedAnswerDraft],
) -> AnswerDraftSet | None:
    try:
        output = GroundedAnswerDraftSetExtraction(**(cached.get("data") or {}))
    except Exception:
        return None
    return _draft_set_from_schema(
        conn,
        context,
        output,
        provider_name=str(cached.get("provider") or provider_name),
        model=model,
        fingerprint=fingerprint,
        prompt_hash_value=prompt_hash_value,
        from_cache=True,
        cost_inr=0.0,
        blocked_drafts=blocked_drafts,
    )


def _draft_set_from_schema(
    conn,
    context: ApplicationAnswerContext,
    output: GroundedAnswerDraftSetExtraction,
    *,
    provider_name: str,
    model: str,
    fingerprint: str,
    prompt_hash_value: str,
    from_cache: bool,
    cost_inr: float,
    blocked_drafts: list[GroundedAnswerDraft],
) -> AnswerDraftSet:
    by_key = {item.question_key: item for item in output.drafts}
    drafts: list[GroundedAnswerDraft] = []
    repo = ApplicationAnswerDraftRepo(conn)
    for question in context.safe_questions:
        item = by_key.get(question.key)
        if item is None:
            draft = _failed_draft(
                context,
                question,
                provider=provider_name,
                model=model,
                prompt_version=context.prompt_version,
                context_fingerprint=fingerprint,
                prompt_hash_value=prompt_hash_value,
                status="invalid",
                message="provider did not return a draft for this safe question",
                from_cache=from_cache,
                cost_inr=0.0,
            )
        else:
            draft = GroundedAnswerDraft(
                application_id=context.application_id,
                job_id=context.job_id,
                question_key=question.key,
                question_text=question.text,
                question_classification=question.classification,
                answer_text=item.answer_text,
                evidence_refs=list(item.evidence_refs),
                confidence=float(item.confidence),
                review_required=bool(item.review_required) or item.confidence < 0.75,
                validation_status="valid",
                provider=provider_name,
                model=model,
                prompt_version=context.prompt_version,
                context_fingerprint=fingerprint,
                prompt_hash=prompt_hash_value,
                source_truth_hash=context.source_truth_hash,
                from_cache=from_cache,
                cost_inr=cost_inr,
            )
            issues = validate_grounded_answer_draft(
                draft,
                allowed_evidence_refs=context.allowed_evidence_refs,
                evidence_text_by_ref=context.evidence_text_by_ref,
                job_context_text=context.job_context_text,
            )
            draft = replace(
                draft,
                validation_status="invalid" if issues else "valid",
                validation_errors=issues,
                review_required=draft.review_required or bool(issues),
            )
        drafts.append(repo.save(draft))

    status = _draft_set_status(blocked_drafts + drafts)
    Audit(conn).human(
        "llm_application_answer_drafts",
        "application",
        context.application_id,
        job_id=context.job_id,
        status=status,
        provider=provider_name,
        model=model,
        drafts=len(drafts),
        blocked_questions=len(context.blocked_questions),
        from_cache=from_cache,
        cost_inr=cost_inr,
    )
    return AnswerDraftSet(
        application_id=context.application_id,
        job_id=context.job_id,
        status=status,
        purpose=context.purpose,
        prompt_version=context.prompt_version,
        context_fingerprint=fingerprint,
        source_truth_hash=context.source_truth_hash,
        provider=provider_name,
        model=model,
        prompt_hash=prompt_hash_value,
        questions=context.questions,
        drafts=blocked_drafts + drafts,
        blocked_questions=context.blocked_questions,
        allowed_evidence_refs=context.allowed_evidence_refs,
        from_cache=from_cache,
        cost_inr=cost_inr,
    )


def _save_blocked_question_drafts(
    conn,
    context: ApplicationAnswerContext,
    *,
    provider: str,
    model: str,
    prompt_version: str,
    context_fingerprint: str,
    prompt_hash_value: str,
) -> list[GroundedAnswerDraft]:
    repo = ApplicationAnswerDraftRepo(conn)
    drafts: list[GroundedAnswerDraft] = []
    for question in context.blocked_questions:
        draft = GroundedAnswerDraft(
            application_id=context.application_id,
            job_id=context.job_id,
            question_key=question.key,
            question_text=question.text,
            question_classification=question.classification,
            answer_text="",
            evidence_refs=[],
            confidence=0.0,
            review_required=True,
            validation_status="blocked",
            validation_errors=[
                ApplicationIntelligenceIssue(
                    code="llm_question_not_safe",
                    message=f"LLM drafting is not allowed for {question.classification} questions",
                )
            ],
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            context_fingerprint=context_fingerprint,
            prompt_hash=prompt_hash_value,
            source_truth_hash=context.source_truth_hash,
            from_cache=False,
            cost_inr=0.0,
        )
        drafts.append(repo.save(draft))
    return drafts


def _save_failed_safe_questions(
    conn,
    context: ApplicationAnswerContext,
    *,
    provider: str,
    model: str,
    prompt_version: str,
    context_fingerprint: str,
    prompt_hash_value: str,
    status: str,
    message: str,
) -> list[GroundedAnswerDraft]:
    return [
        ApplicationAnswerDraftRepo(conn).save(
            _failed_draft(
                context,
                question,
                provider=provider,
                model=model,
                prompt_version=prompt_version,
                context_fingerprint=context_fingerprint,
                prompt_hash_value=prompt_hash_value,
                status=status,
                message=message,
                from_cache=False,
                cost_inr=0.0,
            )
        )
        for question in context.safe_questions
    ]


def _failed_draft(
    context: ApplicationAnswerContext,
    question: ApplicationQuestion,
    *,
    provider: str,
    model: str,
    prompt_version: str,
    context_fingerprint: str,
    prompt_hash_value: str,
    status: str,
    message: str,
    from_cache: bool,
    cost_inr: float,
) -> GroundedAnswerDraft:
    return GroundedAnswerDraft(
        application_id=context.application_id,
        job_id=context.job_id,
        question_key=question.key,
        question_text=question.text,
        question_classification=question.classification,
        answer_text="",
        evidence_refs=[],
        confidence=0.0,
        review_required=True,
        validation_status=status,
        validation_errors=[ApplicationIntelligenceIssue(code=status, message=message)],
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        context_fingerprint=context_fingerprint,
        prompt_hash=prompt_hash_value,
        source_truth_hash=context.source_truth_hash,
        from_cache=from_cache,
        cost_inr=cost_inr,
    )


def _covers_safe_questions(drafts: list[GroundedAnswerDraft], questions: list[ApplicationQuestion]) -> bool:
    if not drafts or not questions:
        return False
    valid_keys = {draft.question_key for draft in drafts if draft.validation_status in {"valid", "invalid"}}
    return {question.key for question in questions}.issubset(valid_keys)


def _draft_set_status(drafts: list[GroundedAnswerDraft]) -> str:
    active = [draft for draft in drafts if draft.validation_status != "blocked"]
    if not active:
        return "blocked" if drafts else "skipped"
    if any(draft.validation_status == "failed" for draft in active):
        return "failed"
    if any(draft.validation_status == "invalid" for draft in active):
        return "invalid"
    if all(draft.validation_status == "valid" for draft in active):
        return "valid"
    return "blocked"
