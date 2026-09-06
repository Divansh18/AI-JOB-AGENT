"""Optional provider-backed resume wording suggestions.

The deterministic resume engine chooses the evidence and validates final PDF
output. This module only asks an enabled provider for wording/reordering drafts
for that already-selected evidence.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from ..domain.application_intelligence import (
    ApplicationIntelligenceIssue,
    ResumeWordingArtifact,
    ResumeWordingSuggestion,
    make_resume_wording_suggestion,
    validate_resume_wording_suggestion,
)
from ..domain.resume_intelligence import (
    PAGE_STATUS_VALID,
    ResumeChange,
    ResumeContentItem,
    ResumeExperienceEntry,
    ResumeProjectEntry,
    ResumeSourceModel,
    validate_tailored_resume,
)
from ..persistence.repositories import LlmCallRepo, ResumeWordingArtifactRepo
from ..services import resume_output as resume_output_service
from ..services.application_intelligence_context import (
    DEFAULT_RESUME_WORDING_PROMPT_VERSION,
    PURPOSE_RESUME_WORDING,
    ApplicationIntelligenceContextError,
    ResumeWordingContext,
    build_resume_wording_context,
    resume_wording_provider_context,
    resume_wording_context_fingerprint,
)
from ..services.audit import Audit
from .application_intelligence import _budget_error
from .cache import ResponseCache
from .cost import to_inr
from .provider import LLMError, LLMUnavailableError, get_provider, prompt_hash
from .schemas import RESUME_WORDING_JSON_SPEC, ResumeWordingArtifactExtraction


SYSTEM_TEMPLATE = """You suggest bounded resume wording and ordering improvements.

Hard rules:
- Deterministic resume selection is authoritative. Do not add new evidence.
- Use only selected_resume_items and allowed_evidence_refs.
- Treat item_key as the immutable source identifier. Do not reproduce original text.
- Preserve facts, dates, titles, company names, project names, metrics, skills, education, and scope.
- You may rewrite wording for clarity, shorten wording, and reorder selected bullets/projects.
- Do not introduce new skills, numbers, metrics, scale, users, revenue, performance, YOE, dates, companies, titles, leadership, education, certifications, or responsibilities.
- Every suggestion must cite evidence_refs from allowed_evidence_refs.
- Prefer meaningful bullet rewrites relevant to the JD. Omit items that are already strong.
- Do not rewrite project/section summaries merely to append names or make cosmetic wording changes.
- Return one JSON object only, with this exact structure:
{spec}"""

USER_TEMPLATE = """RESUME_WORDING_CONTEXT_JSON
{context_json}"""

VALID_ACTIONS = {"rewrite", "shorten", "reorder"}
IGNORABLE_SUGGESTION_ISSUES = {
    "cosmetic_resume_wording_suggestion",
    "no_op_resume_wording_suggestion",
}


class ResumeWordingError(Exception):
    pass


def improve_resume_wording(conn, config, job_id: int) -> ResumeWordingArtifact:
    """Run Phase 2C.3 for one explicit job.

    If LLMs are disabled or unavailable, the existing deterministic resume flow
    remains the fallback and this function returns a non-fabricated status.
    """
    settings = config.settings.llm
    purpose = PURPOSE_RESUME_WORDING
    prompt_version = str(
        getattr(settings, "resume_wording_prompt_version", DEFAULT_RESUME_WORDING_PROMPT_VERSION)
        or DEFAULT_RESUME_WORDING_PROMPT_VERSION
    )
    model = str(getattr(settings, "model", "") or "unknown")
    provider_name = str(getattr(settings, "provider", "") or "unknown")
    max_input_chars = int(getattr(settings, "max_input_chars", 16000))

    if not getattr(settings, "enabled", False):
        return ResumeWordingArtifact(
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
        context, prepared = build_resume_wording_context(
            conn,
            config,
            job_id,
            purpose=purpose,
            prompt_version=prompt_version,
            max_jd_chars=int(getattr(settings, "max_jd_chars", 6000)),
            max_context_chars=_context_payload_budget(max_input_chars),
        )
    except ApplicationIntelligenceContextError as exc:
        return ResumeWordingArtifact(
            job_id=job_id,
            status="blocked",
            purpose=purpose,
            prompt_version=prompt_version,
            context_fingerprint="",
            source_truth_hash="",
            provider=provider_name,
            model=model,
            errors=[str(exc)],
        )

    fingerprint = resume_wording_context_fingerprint(context, model=model)
    system, user = _prompts(context)
    phash = prompt_hash(system, user, model)
    base = _base_artifact(context, provider_name, model, fingerprint, phash)

    deterministic_validation = context.deterministic.get("validation") or {}
    if not deterministic_validation.get("ok", False):
        return _save_artifact(
            conn,
            replace(base, status="blocked", errors=["deterministic_resume_validation_failed"]),
        )

    if len(system) + len(user) > max_input_chars:
        return _save_artifact(
            conn,
            replace(
                base,
                status="blocked",
                errors=[f"input_context_too_large: {len(system) + len(user)} chars > {max_input_chars}"],
            ),
        )

    repo = ResumeWordingArtifactRepo(conn)
    existing = repo.find_by_fingerprint(
        job_id=job_id,
        purpose=purpose,
        prompt_version=prompt_version,
        provider=provider_name,
        model=model,
        context_fingerprint=fingerprint,
    )
    if existing is not None and existing.status in {"valid", "invalid"}:
        return replace(existing, from_cache=True)

    cache = ResponseCache(config.root / "data" / "cache" / "llm")
    cached = cache.get(phash, model)
    if cached is not None:
        cached_artifact = _artifact_from_cached(
            conn,
            config,
            context,
            prepared,
            cached,
            provider_name=provider_name,
            model=model,
            fingerprint=fingerprint,
            prompt_hash_value=phash,
        )
        if cached_artifact is not None:
            return cached_artifact

    budget_error = _budget_error(conn, settings, model, system, user)
    if budget_error:
        return _save_artifact(conn, replace(base, status="blocked", errors=[budget_error]))

    try:
        provider = get_provider(settings)
    except LLMUnavailableError as exc:
        return _save_artifact(conn, replace(base, status="failed", errors=[f"provider_unavailable: {exc}"]))

    ok, reason = provider.available()
    if not ok:
        return _save_artifact(
            conn,
            replace(base, status="failed", provider=provider.name, model=provider.model, errors=[f"provider_unavailable: {reason}"]),
        )

    try:
        result = provider.complete(
            system=system,
            user=user,
            schema=ResumeWordingArtifactExtraction,
            purpose=purpose,
        )
    except LLMError as exc:
        return _save_artifact(
            conn,
            replace(base, status="failed", provider=provider.name, model=provider.model, errors=[f"{type(exc).__name__}: {exc}"]),
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

    artifact = _artifact_from_schema(
        conn,
        config,
        context,
        prepared,
        result.data,
        provider_name=result.provider or provider.name,
        model=result.model or provider.model,
        fingerprint=fingerprint,
        prompt_hash_value=result.prompt_hash or phash,
        from_cache=False,
        cost_inr=cost_inr,
    )
    if artifact.status in {"valid", "invalid"}:
        cache.put(phash, model, {"data": result.data.model_dump(), "provider": artifact.provider})
    return artifact


def get_resume_wording_artifact(conn, artifact_id: int) -> dict[str, Any]:
    artifact = ResumeWordingArtifactRepo(conn).get(artifact_id)
    if artifact is None:
        raise ResumeWordingError(f"resume wording artifact {artifact_id} not found")
    return artifact.as_dict()


def list_resume_wording_artifacts(conn, job_id: int, *, limit: int = 20) -> list[dict[str, Any]]:
    return [artifact.as_dict() for artifact in ResumeWordingArtifactRepo(conn).list_by_job(job_id, limit=limit)]


def _prompts(context: ResumeWordingContext) -> tuple[str, str]:
    system = SYSTEM_TEMPLATE.format(spec=RESUME_WORDING_JSON_SPEC)
    context_json = json.dumps(
        _provider_context(context),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return system, USER_TEMPLATE.format(context_json=context_json)


def _provider_context(context: ResumeWordingContext) -> dict[str, Any]:
    return resume_wording_provider_context(context)


def _context_payload_budget(max_input_chars: int) -> int:
    system = SYSTEM_TEMPLATE.format(spec=RESUME_WORDING_JSON_SPEC)
    empty_user = USER_TEMPLATE.format(context_json="")
    return max(1000, max_input_chars - len(system) - len(empty_user) - 64)


def _base_artifact(
    context: ResumeWordingContext,
    provider_name: str,
    model: str,
    fingerprint: str,
    prompt_hash_value: str,
) -> ResumeWordingArtifact:
    return ResumeWordingArtifact(
        job_id=context.job_id,
        status="failed",
        purpose=context.purpose,
        prompt_version=context.prompt_version,
        context_fingerprint=fingerprint,
        source_truth_hash=context.source_truth_hash,
        provider=provider_name,
        model=model,
        prompt_hash=prompt_hash_value,
        request=context.request,
        allowed_evidence_refs=context.allowed_evidence_refs,
    )


def _artifact_from_cached(
    conn,
    config,
    context: ResumeWordingContext,
    prepared,
    cached: dict[str, Any],
    *,
    provider_name: str,
    model: str,
    fingerprint: str,
    prompt_hash_value: str,
) -> ResumeWordingArtifact | None:
    try:
        output = ResumeWordingArtifactExtraction(**(cached.get("data") or {}))
    except Exception:
        return None
    return _artifact_from_schema(
        conn,
        config,
        context,
        prepared,
        output,
        provider_name=str(cached.get("provider") or provider_name),
        model=model,
        fingerprint=fingerprint,
        prompt_hash_value=prompt_hash_value,
        from_cache=True,
        cost_inr=0.0,
    )


def _artifact_from_schema(
    conn,
    config,
    context: ResumeWordingContext,
    prepared,
    output: ResumeWordingArtifactExtraction,
    *,
    provider_name: str,
    model: str,
    fingerprint: str,
    prompt_hash_value: str,
    from_cache: bool,
    cost_inr: float,
) -> ResumeWordingArtifact:
    suggestions = [_validated_suggestion(context, item) for item in output.suggestions]
    artifact_issues = _artifact_validation_errors(suggestions)
    valid_suggestions = [suggestion for suggestion in suggestions if suggestion.ok]

    resume_variant_id: int | None = None
    if valid_suggestions:
        try:
            rendered_payload = _render_with_suggestions(
                conn,
                config,
                context,
                prepared,
                valid_suggestions,
                fingerprint,
            )
        except Exception as exc:
            return _save_artifact(
                conn,
                ResumeWordingArtifact(
                    job_id=context.job_id,
                    status="failed",
                    purpose=context.purpose,
                    prompt_version=context.prompt_version,
                    context_fingerprint=fingerprint,
                    source_truth_hash=context.source_truth_hash,
                    provider=provider_name,
                    model=model,
                    prompt_hash=prompt_hash_value,
                    request=context.request,
                    suggestions=suggestions,
                    validation_errors=artifact_issues,
                    errors=[f"resume_render_failed: {exc}"],
                    allowed_evidence_refs=context.allowed_evidence_refs,
                    from_cache=from_cache,
                    cost_inr=cost_inr,
                ),
            )
        resume_variant_id = int(rendered_payload["resume_id"])
        render_validation = rendered_payload.get("validation") or {}
        if not render_validation.get("ok", False):
            artifact_issues.extend(_resume_validation_issues(render_validation.get("issues") or []))
        tailored = rendered_payload.get("tailored_resume") or {}
        if tailored.get("page_validation_status") != PAGE_STATUS_VALID or tailored.get("page_count") != 1:
            artifact_issues.append(
                ApplicationIntelligenceIssue(
                    code="page_validation_failed",
                    message=(
                        "LLM-worded resume variant must render to exactly one page; "
                        f"status={tailored.get('page_validation_status')}, page_count={tailored.get('page_count')}"
                    ),
                )
            )
    elif not suggestions:
        artifact_issues.append(
            ApplicationIntelligenceIssue(
                code="no_resume_wording_suggestions",
                message="provider returned no resume wording suggestions",
            )
        )
    elif not artifact_issues:
        artifact_issues.append(
            ApplicationIntelligenceIssue(
                code="no_actionable_resume_wording_suggestions",
                message="provider returned no valid actionable resume wording suggestions",
            )
        )

    status = "invalid" if artifact_issues else "valid"
    artifact = ResumeWordingArtifact(
        job_id=context.job_id,
        resume_variant_id=resume_variant_id,
        status=status,
        purpose=context.purpose,
        prompt_version=context.prompt_version,
        context_fingerprint=fingerprint,
        source_truth_hash=context.source_truth_hash,
        provider=provider_name,
        model=model,
        prompt_hash=prompt_hash_value,
        request=context.request,
        suggestions=suggestions,
        validation_errors=artifact_issues,
        allowed_evidence_refs=context.allowed_evidence_refs,
        from_cache=from_cache,
        cost_inr=cost_inr,
    )
    return _save_artifact(conn, artifact)


def _validated_suggestion(context: ResumeWordingContext, item) -> ResumeWordingSuggestion:
    selected_item = context.item_by_key.get(item.item_key)
    suggestion = make_resume_wording_suggestion(
        section=selected_item.section if selected_item is not None else "",
        item_key=item.item_key,
        action=item.action,
        original_text=selected_item.text if selected_item is not None else "",
        suggested_text=item.rewritten_text,
        evidence_refs=item.evidence_refs,
    )
    issues: list[ApplicationIntelligenceIssue] = []
    if suggestion.action not in VALID_ACTIONS:
        issues.append(
            ApplicationIntelligenceIssue(
                code="invalid_resume_wording_action",
                message=f"resume wording action must be one of {sorted(VALID_ACTIONS)}",
            )
        )
    if selected_item is None:
        issues.append(
            ApplicationIntelligenceIssue(
                code="invalid_resume_item_key",
                message=f"item_key is not in selected resume context: {suggestion.item_key}",
            )
        )
    else:
        item_refs = set(selected_item.evidence_refs)
        for ref in suggestion.evidence_refs:
            if ref not in item_refs:
                issues.append(
                    ApplicationIntelligenceIssue(
                        code="item_evidence_ref_mismatch",
                        message=f"evidence ref {ref} is not attached to selected item {suggestion.item_key}",
                        evidence_refs=[ref],
                    )
                )
    issues.extend(
        validate_resume_wording_suggestion(
            suggestion,
            allowed_evidence_refs=context.allowed_evidence_refs,
            evidence_text_by_ref=context.evidence_text_by_ref,
            protected_terms=context.protected_terms,
        )
    )
    if selected_item is not None:
        issues.extend(_cosmetic_suggestion_issues(suggestion, selected_item.text))
    return replace(
        suggestion,
        validation_status="invalid" if issues else "valid",
        validation_errors=issues,
        review_required=True,
    )


def _render_with_suggestions(
    conn,
    config,
    context: ResumeWordingContext,
    prepared,
    suggestions: list[ResumeWordingSuggestion],
    fingerprint: str,
) -> dict[str, Any]:
    if prepared.tailored_resume.preview is None:
        raise ResumeWordingError("deterministic tailored resume preview is missing")
    updated_preview = _apply_suggestions_to_preview(prepared.tailored_resume.preview, suggestions)
    change_items = [
        ResumeChange(
            section=suggestion.section,
            action=f"llm_{suggestion.action}",
            text=suggestion.suggested_text,
            evidence_refs=suggestion.evidence_refs,
            rationale="Validated LLM wording suggestion using deterministic selected evidence.",
        )
        for suggestion in suggestions
    ]
    tailored = replace(
        prepared.tailored_resume,
        preview=updated_preview,
        changes=[*prepared.tailored_resume.changes, *change_items],
        recommendation=replace(
            prepared.tailored_resume.recommendation,
            reasons=[
                *prepared.tailored_resume.recommendation.reasons,
                "LLM wording suggestions applied after deterministic evidence validation.",
            ],
        ),
    )
    validation = validate_tailored_resume(tailored, prepared.source_model)
    updated_prepared = replace(prepared, tailored_resume=tailored, validation=validation)
    return resume_output_service.build_resume_output_from_prepared(
        conn,
        config,
        updated_prepared,
        render_fingerprint_extra={
            "resume_wording_context_fingerprint": fingerprint,
            "suggestions": [suggestion.as_dict() for suggestion in suggestions],
        },
    )


def _apply_suggestions_to_preview(
    preview: ResumeSourceModel,
    suggestions: list[ResumeWordingSuggestion],
) -> ResumeSourceModel:
    replacements = {
        suggestion.item_key: suggestion.suggested_text
        for suggestion in suggestions
        if suggestion.suggested_text and suggestion.action in {"rewrite", "shorten", "reorder"}
    }
    order_rank = {
        suggestion.item_key: index
        for index, suggestion in enumerate(suggestions)
        if suggestion.action == "reorder"
    }
    summary = _replace_item(preview.summary, "summary", replacements) if preview.summary else None
    experience = [
        _apply_experience_entry(entry, index, replacements, order_rank)
        for index, entry in enumerate(preview.experience)
    ]
    projects = [
        _apply_project_entry(entry, index, replacements, order_rank)
        for index, entry in enumerate(preview.projects)
    ]
    projects = _reorder_projects(projects, order_rank)
    return replace(preview, summary=summary, experience=experience, projects=projects)


def _apply_experience_entry(
    entry: ResumeExperienceEntry,
    entry_index: int,
    replacements: dict[str, str],
    order_rank: dict[str, int],
) -> ResumeExperienceEntry:
    summary = _replace_item(entry.summary, f"experience:{entry_index}:summary", replacements) if entry.summary else None
    bullets = [
        _replace_item(bullet, f"experience:{entry_index}:bullet:{bullet_index}", replacements) or bullet
        for bullet_index, bullet in enumerate(entry.bullets)
    ]
    bullets = _reorder_items(bullets, f"experience:{entry_index}:bullet:", order_rank)
    return replace(entry, summary=summary, bullets=bullets)


def _apply_project_entry(
    entry: ResumeProjectEntry,
    entry_index: int,
    replacements: dict[str, str],
    order_rank: dict[str, int],
) -> ResumeProjectEntry:
    summary = _replace_item(entry.summary, f"projects:{entry_index}:summary", replacements) if entry.summary else None
    bullets = [
        _replace_item(bullet, f"projects:{entry_index}:bullet:{bullet_index}", replacements) or bullet
        for bullet_index, bullet in enumerate(entry.bullets)
    ]
    bullets = _reorder_items(bullets, f"projects:{entry_index}:bullet:", order_rank)
    return replace(entry, summary=summary, bullets=bullets)


def _replace_item(
    item: ResumeContentItem | None,
    item_key: str,
    replacements: dict[str, str],
) -> ResumeContentItem | None:
    if item is None:
        return None
    replacement = replacements.get(item_key)
    return replace(item, text=replacement) if replacement else item


def _reorder_items(
    items: list[ResumeContentItem],
    prefix: str,
    order_rank: dict[str, int],
) -> list[ResumeContentItem]:
    if not any(f"{prefix}{index}" in order_rank for index in range(len(items))):
        return items
    return [
        item
        for _, item in sorted(
            enumerate(items),
            key=lambda pair: (order_rank.get(f"{prefix}{pair[0]}", 10_000 + pair[0]), pair[0]),
        )
    ]


def _reorder_projects(
    projects: list[ResumeProjectEntry],
    order_rank: dict[str, int],
) -> list[ResumeProjectEntry]:
    project_order: dict[int, int] = {}
    for key, rank in order_rank.items():
        if not key.startswith("projects:"):
            continue
        parts = key.split(":")
        if len(parts) < 2:
            continue
        try:
            index = int(parts[1])
        except ValueError:
            continue
        project_order[index] = min(project_order.get(index, rank), rank)
    if not project_order:
        return projects
    return [
        project
        for _, project in sorted(
            enumerate(projects),
            key=lambda pair: (project_order.get(pair[0], 10_000 + pair[0]), pair[0]),
        )
    ]


def _artifact_validation_errors(suggestions: list[ResumeWordingSuggestion]) -> list[ApplicationIntelligenceIssue]:
    issues: list[ApplicationIntelligenceIssue] = []
    for suggestion in suggestions:
        if suggestion.validation_errors and all(
            issue.code in IGNORABLE_SUGGESTION_ISSUES
            for issue in suggestion.validation_errors
        ):
            continue
        issues.extend(suggestion.validation_errors)
    return issues


def _resume_validation_issues(items: list[dict[str, Any]]) -> list[ApplicationIntelligenceIssue]:
    return [
        ApplicationIntelligenceIssue(
            code=f"resume_{item.get('code') or 'validation_error'}",
            message=str(item.get("message") or "resume validation failed"),
            evidence_refs=list(item.get("evidence_refs") or []),
        )
        for item in items
    ]


def _save_artifact(conn, artifact: ResumeWordingArtifact) -> ResumeWordingArtifact:
    stored = ResumeWordingArtifactRepo(conn).save(artifact)
    Audit(conn).human(
        "llm_resume_wording",
        "resume_wording_artifact",
        stored.artifact_id,
        job_id=stored.job_id,
        resume_variant_id=stored.resume_variant_id,
        status=stored.status,
        provider=stored.provider,
        model=stored.model,
        from_cache=stored.from_cache,
        cost_inr=stored.cost_inr,
    )
    return stored


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _cosmetic_suggestion_issues(
    suggestion: ResumeWordingSuggestion,
    original_text: str,
) -> list[ApplicationIntelligenceIssue]:
    original = _normalize_text(original_text)
    rewritten = _normalize_text(suggestion.suggested_text)
    if not rewritten:
        return []
    if suggestion.action == "reorder":
        return []
    if _normalize_for_cosmetic(original) == _normalize_for_cosmetic(rewritten):
        return [
            ApplicationIntelligenceIssue(
                code="no_op_resume_wording_suggestion",
                message="rewritten_text does not materially change the selected resume item",
            )
        ]
    if suggestion.item_key.endswith(":summary") and _summary_name_append_only(original, rewritten):
        return [
            ApplicationIntelligenceIssue(
                code="cosmetic_resume_wording_suggestion",
                message="summary rewrite only appends/prepends naming text without improving content",
            )
        ]
    return []


def _normalize_for_cosmetic(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _summary_name_append_only(original: str, rewritten: str) -> bool:
    original_tokens = _token_set(original)
    rewritten_tokens = _token_set(rewritten)
    if not original_tokens or not rewritten_tokens:
        return False
    if not original_tokens.issubset(rewritten_tokens):
        return False
    added = rewritten_tokens - original_tokens
    return 0 < len(added) <= 4


def _token_set(value: str) -> set[str]:
    return {
        token
        for token in _normalize_text(value).lower().replace("/", " ").split()
        if token.strip(".,:;()[]{}-")
        for token in [token.strip(".,:;()[]{}-")]
    }
