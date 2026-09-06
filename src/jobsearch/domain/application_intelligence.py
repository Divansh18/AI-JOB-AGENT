"""Grounded application intelligence domain models and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any


REQUIREMENT_CATEGORIES = {
    "skill",
    "experience",
    "responsibility",
    "domain",
    "education",
    "location",
    "work_authorization",
    "sponsorship",
    "salary",
    "relocation",
    "notice_period",
    "legal",
    "demographic",
    "other",
}
MATCH_STATUSES = {"matched", "partial", "unsupported", "unknown"}
FIT_VERDICTS = {"strong_match", "worth_applying", "stretch", "poor_fit", "unknown"}
REPORT_STATUSES = {"valid", "invalid", "blocked", "failed", "skipped"}
QUESTION_CLASSIFICATIONS = {
    "safe_free_text",
    "factual_candidate_field",
    "sensitive",
    "legal_attestation",
    "self_identification",
    "unknown",
}
DRAFT_VALIDATION_STATUSES = {"valid", "invalid", "blocked", "skipped", "failed"}
RESUME_WORDING_STATUSES = {"valid", "invalid", "blocked", "skipped", "failed"}
REVIEW_STATUSES = {"pending_review", "approved", "rejected"}
SENSITIVE_REQUIREMENT_CATEGORIES = {
    "work_authorization",
    "sponsorship",
    "salary",
    "relocation",
    "notice_period",
    "legal",
    "demographic",
}

_SAFE_FREE_TEXT_PATTERNS = (
    "why do you want to work",
    "why are you interested",
    "why this company",
    "why this role",
    "why do you want this role",
    "describe relevant experience",
    "relevant experience",
    "tell us about a project",
    "project relevant",
    "what makes you a good fit",
    "why are you a good fit",
    "cover letter",
    "anything else",
)
_FACTUAL_FIELD_TERMS = {
    "first_name",
    "last_name",
    "full_name",
    "name",
    "email",
    "phone",
    "linkedin",
    "github",
    "portfolio",
    "website",
    "current_location",
    "location",
    "earliest_start_date",
    "start_date",
}
_SENSITIVE_TERMS = {
    "salary",
    "compensation",
    "pay",
    "ctc",
    "notice_period",
    "notice",
    "work_authorization",
    "right_to_work",
    "authorized",
    "visa",
    "sponsorship",
    "sponsor",
    "relocation",
    "relocate",
}
_LEGAL_TERMS = {
    "attest",
    "attestation",
    "certify",
    "certification",
    "declare",
    "declaration",
    "legal",
    "accurate",
    "truthful",
    "terms",
    "agreement",
    "background_check",
    "criminal",
    "conviction",
}
_SELF_ID_TERMS = {
    "disability",
    "veteran",
    "gender",
    "race",
    "ethnicity",
    "demographic",
    "self_identification",
    "self_id",
    "pronouns",
    "sexual_orientation",
}
_COMPANY_RESEARCH_RISK_PHRASES = (
    "recent funding",
    "series a",
    "series b",
    "series c",
    "ipo",
    "founding story",
    "your mission to",
    "your blog",
    "your customers include",
    "industry-leading culture",
)
_KNOWN_TECH_TERMS = {
    "python",
    "fastapi",
    "django",
    "flask",
    "java",
    "javascript",
    "typescript",
    "react",
    "node",
    "go",
    "golang",
    "rust",
    "kubernetes",
    "docker",
    "aws",
    "gcp",
    "azure",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "sqlite",
    "spark",
    "kafka",
    "machine learning",
    "ml",
    "ai",
    "llm",
}


@dataclass(frozen=True)
class LlmRequirement:
    id: str
    text: str
    category: str
    required: bool
    candidate_match: str = "unknown"
    evidence_refs: list[str] = field(default_factory=list)
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category,
            "required": self.required,
            "candidate_match": self.candidate_match,
            "evidence_refs": self.evidence_refs,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class LlmJobInsight:
    job_id: int
    role_summary: str
    must_have_requirements: list[LlmRequirement]
    preferred_requirements: list[LlmRequirement]
    role_priorities: list[str]
    grounded_fit_assessment: str
    fit_verdict: str
    candidate_match_evidence_refs: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "role_summary": self.role_summary,
            "must_have_requirements": [item.as_dict() for item in self.must_have_requirements],
            "preferred_requirements": [item.as_dict() for item in self.preferred_requirements],
            "role_priorities": self.role_priorities,
            "grounded_fit_assessment": self.grounded_fit_assessment,
            "fit_verdict": self.fit_verdict,
            "candidate_match_evidence_refs": self.candidate_match_evidence_refs,
            "uncertainties": self.uncertainties,
            "gaps": self.gaps,
        }


@dataclass(frozen=True)
class ApplicationIntelligenceIssue:
    code: str
    message: str
    evidence_refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "evidence_refs": self.evidence_refs,
        }


@dataclass(frozen=True)
class ApplicationIntelligenceReport:
    job_id: int
    status: str
    purpose: str
    prompt_version: str
    context_fingerprint: str
    source_truth_hash: str
    provider: str | None = None
    model: str | None = None
    prompt_hash: str | None = None
    artifact_id: int | None = None
    insight: LlmJobInsight | None = None
    validation_errors: list[ApplicationIntelligenceIssue] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    allowed_evidence_refs: list[str] = field(default_factory=list)
    deterministic_fit_score: int | None = None
    deterministic_resume_decision: str | None = None
    from_cache: bool = False
    cost_inr: float = 0.0
    created_at: datetime | None = None

    @property
    def ok(self) -> bool:
        return self.status == "valid" and not self.validation_errors and self.insight is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "job_id": self.job_id,
            "status": self.status,
            "purpose": self.purpose,
            "prompt_version": self.prompt_version,
            "context_fingerprint": self.context_fingerprint,
            "source_truth_hash": self.source_truth_hash,
            "provider": self.provider,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "insight": self.insight.as_dict() if self.insight else None,
            "validation_errors": [issue.as_dict() for issue in self.validation_errors],
            "errors": self.errors,
            "allowed_evidence_refs": self.allowed_evidence_refs,
            "deterministic_fit_score": self.deterministic_fit_score,
            "deterministic_resume_decision": self.deterministic_resume_decision,
            "from_cache": self.from_cache,
            "cost_inr": round(self.cost_inr, 4),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class ApplicationQuestion:
    key: str
    text: str
    classification: str
    source: str = "application_plan"
    required: bool = False

    @property
    def llm_draft_allowed(self) -> bool:
        return self.classification == "safe_free_text"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "text": self.text,
            "classification": self.classification,
            "source": self.source,
            "required": self.required,
            "llm_draft_allowed": self.llm_draft_allowed,
        }


@dataclass(frozen=True)
class GroundedAnswerDraft:
    application_id: int
    job_id: int
    question_key: str
    question_text: str
    answer_text: str
    evidence_refs: list[str]
    confidence: float
    review_required: bool
    validation_status: str
    validation_errors: list[ApplicationIntelligenceIssue] = field(default_factory=list)
    question_classification: str = "unknown"
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    context_fingerprint: str | None = None
    prompt_hash: str | None = None
    source_truth_hash: str = ""
    from_cache: bool = False
    cost_inr: float = 0.0
    review_status: str = "pending_review"
    reviewer_source: str | None = None
    reviewed_at: datetime | None = None
    review_note: str | None = None
    draft_id: int | None = None
    created_at: datetime | None = None

    @property
    def ok(self) -> bool:
        return self.validation_status == "valid" and not self.validation_errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "application_id": self.application_id,
            "job_id": self.job_id,
            "question_key": self.question_key,
            "question_text": self.question_text,
            "question_classification": self.question_classification,
            "answer_text": self.answer_text,
            "evidence_refs": self.evidence_refs,
            "confidence": round(self.confidence, 3),
            "review_required": self.review_required,
            "validation_status": self.validation_status,
            "validation_errors": [issue.as_dict() for issue in self.validation_errors],
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "context_fingerprint": self.context_fingerprint,
            "prompt_hash": self.prompt_hash,
            "source_truth_hash": self.source_truth_hash,
            "from_cache": self.from_cache,
            "cost_inr": round(self.cost_inr, 4),
            "review_status": self.review_status,
            "reviewer_source": self.reviewer_source,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "review_note": self.review_note,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class AnswerDraftSet:
    application_id: int
    job_id: int
    status: str
    purpose: str
    prompt_version: str
    context_fingerprint: str
    source_truth_hash: str
    provider: str | None = None
    model: str | None = None
    prompt_hash: str | None = None
    questions: list[ApplicationQuestion] = field(default_factory=list)
    drafts: list[GroundedAnswerDraft] = field(default_factory=list)
    blocked_questions: list[ApplicationQuestion] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    allowed_evidence_refs: list[str] = field(default_factory=list)
    from_cache: bool = False
    cost_inr: float = 0.0

    @property
    def ok(self) -> bool:
        active = [draft for draft in self.drafts if draft.validation_status != "blocked"]
        return self.status == "valid" and bool(active) and all(draft.ok for draft in active)

    def as_dict(self) -> dict[str, Any]:
        return {
            "application_id": self.application_id,
            "job_id": self.job_id,
            "status": self.status,
            "purpose": self.purpose,
            "prompt_version": self.prompt_version,
            "context_fingerprint": self.context_fingerprint,
            "source_truth_hash": self.source_truth_hash,
            "provider": self.provider,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "questions": [question.as_dict() for question in self.questions],
            "drafts": [draft.as_dict() for draft in self.drafts],
            "blocked_questions": [question.as_dict() for question in self.blocked_questions],
            "errors": self.errors,
            "allowed_evidence_refs": self.allowed_evidence_refs,
            "from_cache": self.from_cache,
            "cost_inr": round(self.cost_inr, 4),
            "ok": self.ok,
        }


@dataclass(frozen=True)
class ResumeWordingRequest:
    job_id: int
    target_role: str
    sections: list[str]
    allowed_evidence_refs: list[str]
    prompt_version: str
    source_truth_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "target_role": self.target_role,
            "sections": self.sections,
            "allowed_evidence_refs": self.allowed_evidence_refs,
            "prompt_version": self.prompt_version,
            "source_truth_hash": self.source_truth_hash,
        }


@dataclass(frozen=True)
class ResumeWordingSuggestion:
    section: str
    original_text: str
    suggested_text: str
    evidence_refs: list[str]
    validation_status: str = "valid"
    validation_errors: list[ApplicationIntelligenceIssue] = field(default_factory=list)
    review_required: bool = True
    action: str = "rewrite"
    item_key: str = ""

    @property
    def ok(self) -> bool:
        return self.validation_status == "valid" and not self.validation_errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "item_key": self.item_key,
            "action": self.action,
            "original_text": self.original_text,
            "suggested_text": self.suggested_text,
            "evidence_refs": self.evidence_refs,
            "validation_status": self.validation_status,
            "validation_errors": [issue.as_dict() for issue in self.validation_errors],
            "review_required": self.review_required,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class ResumeWordingArtifact:
    job_id: int
    status: str
    purpose: str
    prompt_version: str
    context_fingerprint: str
    source_truth_hash: str
    provider: str | None = None
    model: str | None = None
    prompt_hash: str | None = None
    artifact_id: int | None = None
    resume_variant_id: int | None = None
    request: ResumeWordingRequest | None = None
    suggestions: list[ResumeWordingSuggestion] = field(default_factory=list)
    validation_errors: list[ApplicationIntelligenceIssue] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    allowed_evidence_refs: list[str] = field(default_factory=list)
    from_cache: bool = False
    cost_inr: float = 0.0
    review_status: str = "pending_review"
    reviewer_source: str | None = None
    reviewed_at: datetime | None = None
    review_note: str | None = None
    promoted_application_id: int | None = None
    created_at: datetime | None = None

    @property
    def ok(self) -> bool:
        active = [suggestion for suggestion in self.suggestions if suggestion.validation_status == "valid"]
        return self.status == "valid" and bool(active) and not self.validation_errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "job_id": self.job_id,
            "resume_variant_id": self.resume_variant_id,
            "status": self.status,
            "purpose": self.purpose,
            "prompt_version": self.prompt_version,
            "context_fingerprint": self.context_fingerprint,
            "source_truth_hash": self.source_truth_hash,
            "provider": self.provider,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "request": self.request.as_dict() if self.request else None,
            "suggestions": [suggestion.as_dict() for suggestion in self.suggestions],
            "validation_errors": [issue.as_dict() for issue in self.validation_errors],
            "errors": self.errors,
            "allowed_evidence_refs": self.allowed_evidence_refs,
            "from_cache": self.from_cache,
            "cost_inr": round(self.cost_inr, 4),
            "review_status": self.review_status,
            "reviewer_source": self.reviewer_source,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "review_note": self.review_note,
            "promoted_application_id": self.promoted_application_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ok": self.ok,
        }


def make_requirement(
    *,
    text: str,
    category: str,
    required: bool,
    candidate_match: str = "unknown",
    evidence_refs: list[str] | None = None,
    rationale: str = "",
    prefix: str | None = None,
) -> LlmRequirement:
    normalized_category = normalize_category(category)
    return LlmRequirement(
        id=f"{prefix or ('must' if required else 'preferred')}:{_slug(text)[:48] or 'requirement'}",
        text=_clean_text(text)[:260],
        category=normalized_category,
        required=required,
        candidate_match=normalize_match_status(candidate_match),
        evidence_refs=_dedupe_refs(evidence_refs or []),
        rationale=_clean_text(rationale)[:300],
    )


def validate_llm_job_insight(
    insight: LlmJobInsight,
    *,
    allowed_evidence_refs: set[str] | list[str],
    deterministic_blockers: list[str] | None = None,
) -> list[ApplicationIntelligenceIssue]:
    allowed = set(allowed_evidence_refs)
    blockers = [item for item in (deterministic_blockers or []) if item]
    issues: list[ApplicationIntelligenceIssue] = []

    if not _clean_text(insight.role_summary):
        issues.append(_issue("missing_role_summary", "role_summary is required"))
    if insight.fit_verdict not in FIT_VERDICTS:
        issues.append(_issue("invalid_fit_verdict", f"fit_verdict must be one of {sorted(FIT_VERDICTS)}"))
    if blockers and insight.fit_verdict not in {"poor_fit", "stretch", "unknown"}:
        issues.append(
            _issue(
                "deterministic_blocker_override",
                "LLM fit verdict cannot override deterministic blockers",
            )
        )

    seen_requirement_ids: set[str] = set()
    for collection_name, required_value, requirements in (
        ("must_have_requirements", True, insight.must_have_requirements),
        ("preferred_requirements", False, insight.preferred_requirements),
    ):
        for requirement in requirements:
            if requirement.id in seen_requirement_ids:
                issues.append(_issue("duplicate_requirement_id", f"duplicate requirement id {requirement.id!r}"))
            seen_requirement_ids.add(requirement.id)
            issues.extend(
                validate_llm_requirement(
                    requirement,
                    allowed_evidence_refs=allowed,
                    expected_required=required_value,
                    collection_name=collection_name,
                )
            )

    for ref in insight.candidate_match_evidence_refs:
        if ref not in allowed:
            issues.append(
                _issue("invalid_candidate_match_evidence_ref", f"candidate evidence ref is not in context: {ref}", [ref])
            )

    return issues


def validate_llm_requirement(
    requirement: LlmRequirement,
    *,
    allowed_evidence_refs: set[str],
    expected_required: bool,
    collection_name: str,
) -> list[ApplicationIntelligenceIssue]:
    issues: list[ApplicationIntelligenceIssue] = []
    if not _clean_text(requirement.text):
        issues.append(_issue("missing_requirement_text", f"{collection_name} entry has empty text"))
    if requirement.category not in REQUIREMENT_CATEGORIES:
        issues.append(_issue("invalid_requirement_category", f"unknown requirement category {requirement.category!r}"))
    if requirement.required != expected_required:
        issues.append(
            _issue(
                "requirement_required_mismatch",
                f"{collection_name} entry has required={requirement.required}; expected {expected_required}",
            )
        )
    if requirement.candidate_match not in MATCH_STATUSES:
        issues.append(_issue("invalid_candidate_match", f"unknown candidate_match {requirement.candidate_match!r}"))

    refs = _dedupe_refs(requirement.evidence_refs)
    for ref in refs:
        if ref not in allowed_evidence_refs:
            issues.append(_issue("invalid_evidence_ref", f"evidence ref is not in context: {ref}", [ref]))
    if requirement.candidate_match in {"matched", "partial"} and not refs:
        issues.append(
            _issue(
                "candidate_claim_missing_evidence_refs",
                "matched or partial candidate claims require evidence_refs",
            )
        )
    if requirement.category in SENSITIVE_REQUIREMENT_CATEGORIES:
        if refs:
            issues.append(
                _issue(
                    "sensitive_evidence_ref_forbidden",
                    f"{requirement.category} must stay deterministic/human-review only",
                    refs,
                )
            )
        if requirement.candidate_match in {"matched", "partial"}:
            issues.append(
                _issue(
                    "sensitive_match_claim_forbidden",
                    f"{requirement.category} candidate match cannot be inferred by the LLM",
                )
            )
    return issues


def normalize_category(value: str) -> str:
    slug = _slug(value)
    aliases = {
        "must_have": "other",
        "preferred": "other",
        "authorization": "work_authorization",
        "visa": "sponsorship",
        "compensation": "salary",
        "legal_attestation": "legal",
        "self_identification": "demographic",
    }
    return aliases.get(slug, slug if slug in REQUIREMENT_CATEGORIES else "other")


def normalize_match_status(value: str) -> str:
    slug = _slug(value)
    aliases = {
        "match": "matched",
        "supported": "matched",
        "support": "matched",
        "partially_matched": "partial",
        "partial_match": "partial",
        "missing": "unsupported",
        "not_supported": "unsupported",
        "no_evidence": "unsupported",
        "unclear": "unknown",
    }
    return aliases.get(slug, slug if slug in MATCH_STATUSES else "unknown")


def make_application_question(
    *,
    key: str,
    text: str,
    source: str = "application_plan",
    required: bool = False,
) -> ApplicationQuestion:
    question_key = _slug(key or text) or "question"
    question_text = _clean_text(text or key)
    return ApplicationQuestion(
        key=question_key[:120],
        text=question_text[:500],
        classification=classify_application_question(question_key, question_text),
        source=_clean_text(source)[:80] or "application_plan",
        required=required,
    )


def classify_application_question(question_key: str, question_text: str = "") -> str:
    """Classify a question before any LLM call.

    The allowlist is intentionally narrow: uncertain questions are not drafted.
    """
    haystack = f"{question_key} {question_text}".lower().replace("-", "_")
    slug = _slug(haystack)
    if _contains_any(slug, _SELF_ID_TERMS):
        return "self_identification"
    if _contains_any(slug, _LEGAL_TERMS):
        return "legal_attestation"
    if _contains_any(slug, _SENSITIVE_TERMS):
        return "sensitive"
    if _contains_any(slug, _FACTUAL_FIELD_TERMS):
        return "factual_candidate_field"
    plain = " ".join(haystack.replace("_", " ").split())
    if any(pattern in plain for pattern in _SAFE_FREE_TEXT_PATTERNS):
        return "safe_free_text"
    return "unknown"


def validate_grounded_answer_draft(
    draft: GroundedAnswerDraft,
    *,
    allowed_evidence_refs: set[str] | list[str],
    evidence_text_by_ref: dict[str, str],
    job_context_text: str = "",
) -> list[ApplicationIntelligenceIssue]:
    allowed = set(allowed_evidence_refs)
    issues: list[ApplicationIntelligenceIssue] = []
    classification = draft.question_classification
    if classification not in QUESTION_CLASSIFICATIONS:
        issues.append(_issue("invalid_question_classification", f"unknown classification {classification!r}"))
    if classification != "safe_free_text":
        if draft.answer_text.strip():
            issues.append(
                _issue(
                    "blocked_question_answered",
                    f"LLM drafts are not allowed for {classification} questions",
                )
            )
        return issues
    if not _clean_text(draft.question_text):
        issues.append(_issue("missing_question_text", "question_text is required"))
    if not _clean_text(draft.answer_text):
        issues.append(_issue("missing_answer_text", "answer_text is required"))
    refs = _dedupe_refs(draft.evidence_refs)
    for ref in refs:
        if ref not in allowed:
            issues.append(_issue("invalid_evidence_ref", f"evidence ref is not in context: {ref}", [ref]))
    if draft.answer_text.strip() and not refs:
        issues.append(
            _issue(
                "answer_missing_evidence_refs",
                "grounded application answers require evidence_refs",
            )
        )
    if refs:
        supported_text = _support_text(refs, evidence_text_by_ref)
        issues.extend(_unsupported_number_issues(draft.answer_text, supported_text))
        issues.extend(_unsupported_tech_issues(draft.answer_text, supported_text + " " + job_context_text))
    issues.extend(_company_research_issues(draft.answer_text, job_context_text))
    if not 0 <= draft.confidence <= 1:
        issues.append(_issue("invalid_confidence", "confidence must be between 0 and 1"))
    return issues


def make_resume_wording_suggestion(
    *,
    section: str,
    original_text: str,
    suggested_text: str,
    evidence_refs: list[str] | None = None,
    action: str = "rewrite",
    item_key: str = "",
) -> ResumeWordingSuggestion:
    return ResumeWordingSuggestion(
        section=_slug(section)[:80] or "unknown",
        item_key=_clean_text(item_key)[:160],
        action=_slug(action)[:40] or "rewrite",
        original_text=_clean_text(original_text)[:1200],
        suggested_text=_clean_text(suggested_text)[:1200],
        evidence_refs=_dedupe_refs(evidence_refs or []),
        review_required=True,
    )


def validate_resume_wording_suggestion(
    suggestion: ResumeWordingSuggestion,
    *,
    allowed_evidence_refs: set[str] | list[str],
    evidence_text_by_ref: dict[str, str],
    protected_terms: list[str] | None = None,
) -> list[ApplicationIntelligenceIssue]:
    allowed = set(allowed_evidence_refs)
    issues: list[ApplicationIntelligenceIssue] = []
    if not _clean_text(suggestion.original_text):
        issues.append(_issue("missing_original_text", "original_text is required"))
    if not _clean_text(suggestion.suggested_text):
        issues.append(_issue("missing_suggested_text", "suggested_text is required"))
    refs = _dedupe_refs(suggestion.evidence_refs)
    for ref in refs:
        if ref not in allowed:
            issues.append(_issue("invalid_evidence_ref", f"evidence ref is not selected for this resume: {ref}", [ref]))
    if suggestion.suggested_text.strip() and not refs:
        issues.append(_issue("suggestion_missing_evidence_refs", "resume wording suggestions require evidence_refs"))
    supported = " ".join([suggestion.original_text, _support_text(refs, evidence_text_by_ref)])
    issues.extend(_unsupported_number_issues(suggestion.suggested_text, supported))
    issues.extend(_unsupported_tech_issues(suggestion.suggested_text, supported))
    issues.extend(_unsupported_identity_issues(suggestion.suggested_text, supported, protected_terms or []))
    issues.extend(_unsupported_role_company_phrase_issues(suggestion.suggested_text, supported))
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", suggestion.suggested_text.lower()):
        claim = match.group(0)
        if claim not in supported.lower():
            issues.append(_issue("unsupported_years_claim", f"YOE claim is not supported by selected evidence: {claim}"))
    return issues


def _issue(code: str, message: str, refs: list[str] | None = None) -> ApplicationIntelligenceIssue:
    return ApplicationIntelligenceIssue(code=code, message=message, evidence_refs=refs or [])


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")


def _contains_any(haystack: str, terms: set[str]) -> bool:
    wrapped = f"_{haystack}_"
    return any(f"_{_slug(term)}_" in wrapped for term in terms)


def _support_text(refs: list[str], evidence_text_by_ref: dict[str, str]) -> str:
    return " ".join(evidence_text_by_ref.get(ref, "") for ref in refs)


def _unsupported_number_issues(answer_text: str, supported_text: str) -> list[ApplicationIntelligenceIssue]:
    issues: list[ApplicationIntelligenceIssue] = []
    supported_claims = {claim.key for claim in _numeric_claims(supported_text)}
    seen: set[tuple[str, str, str, str]] = set()
    for claim in _numeric_claims(answer_text):
        if claim.key in seen:
            continue
        seen.add(claim.key)
        if claim.key not in supported_claims:
            issues.append(
                _issue(
                    "unsupported_numeric_claim",
                    f"numeric/date claim is not supported by evidence: {claim.display}",
                )
            )
    return issues


def _unsupported_tech_issues(answer_text: str, supported_text: str) -> list[ApplicationIntelligenceIssue]:
    issues: list[ApplicationIntelligenceIssue] = []
    supported = supported_text.lower()
    answer = answer_text.lower()
    for term in sorted(_KNOWN_TECH_TERMS, key=len, reverse=True):
        if not _term_in_text(term, answer):
            continue
        if _term_in_text(term, supported):
            continue
        issues.append(_issue("unsupported_skill_claim", f"skill claim is not supported by evidence/JD: {term}"))
    return issues


def _company_research_issues(answer_text: str, job_context_text: str) -> list[ApplicationIntelligenceIssue]:
    issues: list[ApplicationIntelligenceIssue] = []
    answer = answer_text.lower()
    job = job_context_text.lower()
    for phrase in _COMPANY_RESEARCH_RISK_PHRASES:
        if phrase in answer and phrase not in job:
            issues.append(
                _issue(
                    "unsupported_company_claim",
                    f"company-specific claim is not present in the job context: {phrase}",
                )
            )
    return issues


def _unsupported_identity_issues(
    text: str,
    supported_text: str,
    protected_terms: list[str],
) -> list[ApplicationIntelligenceIssue]:
    issues: list[ApplicationIntelligenceIssue] = []
    normalized_supported = supported_text.lower()
    for term in _dedupe_refs([_clean_text(value) for value in protected_terms]):
        if len(term) < 3:
            continue
        if _term_in_text(term.lower(), text.lower()) and term.lower() not in normalized_supported:
            issues.append(
                _issue(
                    "unsupported_identity_claim",
                    f"title/company/project claim is not supported by selected evidence: {term}",
                )
            )
    return issues


def _unsupported_role_company_phrase_issues(text: str, supported_text: str) -> list[ApplicationIntelligenceIssue]:
    issues: list[ApplicationIntelligenceIssue] = []
    supported = supported_text.lower()
    for pattern, label in (
        (r"\b(?:worked|interned|engineered|built|shipped|developed)\s+at\s+([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,3})", "company"),
        (r"\bas\s+(?:an?\s+)?([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4})", "title"),
    ):
        for match in re.finditer(pattern, text or ""):
            claim = match.group(1).strip()
            if claim and claim.lower() not in supported:
                issues.append(_issue(f"unsupported_{label}_claim", f"{label} claim is not supported by selected evidence: {claim}"))
    return issues


@dataclass(frozen=True)
class _NumericClaim:
    display: str
    key: tuple[str, str, str, str]


_NUMERIC_CLAIM_RE = re.compile(
    r"(?<![A-Za-z0-9,])"
    r"(?P<currency>[$₹€£])?\s*"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?P<suffix>[kKmMbB])?"
    r"(?P<plus>\+)?"
    r"(?P<percent>%)?"
    r"(?![A-Za-z0-9,])"
)
_DURATION_UNITS = {
    "second": "seconds",
    "seconds": "seconds",
    "sec": "seconds",
    "secs": "seconds",
    "minute": "minutes",
    "minutes": "minutes",
    "min": "minutes",
    "mins": "minutes",
    "hour": "hours",
    "hours": "hours",
    "day": "days",
    "days": "days",
    "week": "weeks",
    "weeks": "weeks",
    "month": "months",
    "months": "months",
    "year": "years",
    "years": "years",
    "yr": "years",
    "yrs": "years",
}


def _numeric_claims(text: str) -> list[_NumericClaim]:
    claims: list[_NumericClaim] = []
    seen: set[tuple[str, str, str, str]] = set()
    for match in _NUMERIC_CLAIM_RE.finditer(text or ""):
        claim = _numeric_claim_from_match(text or "", match)
        if claim is None or claim.key in seen:
            continue
        claims.append(claim)
        seen.add(claim.key)
    return claims


def _numeric_claim_from_match(text: str, match: re.Match[str]) -> _NumericClaim | None:
    raw_number = match.group("number") or ""
    suffix = (match.group("suffix") or "").lower()
    currency = match.group("currency") or ""
    is_percent = bool(match.group("percent"))
    display = match.group(0).strip()
    unit = _duration_unit_after(text, match.end())

    if unit:
        value = _canonical_numeric_value(raw_number, suffix)
        return _NumericClaim(
            display=f"{display} {unit}",
            key=("duration", value, unit, ""),
        )

    if is_percent:
        value = _canonical_numeric_value(raw_number, suffix)
        return _NumericClaim(display=display, key=("percent", value, "", ""))

    if currency:
        value = _canonical_numeric_value(raw_number, suffix)
        return _NumericClaim(display=display, key=("money", value, "", currency))

    if _is_year_like(raw_number, suffix):
        return _NumericClaim(display=display, key=("date_year", raw_number, "", ""))

    if "." in raw_number and not suffix and "," not in raw_number:
        # Decimals are often versions; keep exact to avoid equating version-like claims.
        return _NumericClaim(display=display, key=("decimal", raw_number, "", ""))

    value = _canonical_numeric_value(raw_number, suffix)
    return _NumericClaim(display=display, key=("number", value, "", ""))


def _canonical_numeric_value(raw_number: str, suffix: str = "") -> str:
    try:
        value = Decimal(raw_number.replace(",", ""))
    except InvalidOperation:
        return raw_number.lower()
    multiplier = {
        "k": Decimal(1000),
        "m": Decimal(1_000_000),
        "b": Decimal(1_000_000_000),
    }.get((suffix or "").lower(), Decimal(1))
    value *= multiplier
    out = format(value.normalize(), "f")
    return out.rstrip("0").rstrip(".") if "." in out else out


def _duration_unit_after(text: str, end: int) -> str:
    match = re.match(
        r"\s*(seconds?|secs?|minutes?|mins?|hours?|days?|weeks?|months?|years?|yrs?)\b",
        text[end:] or "",
        flags=re.I,
    )
    if not match:
        return ""
    return _DURATION_UNITS.get(match.group(1).lower(), "")


def _is_year_like(raw_number: str, suffix: str) -> bool:
    if suffix or "," in raw_number or "." in raw_number or len(raw_number) != 4:
        return False
    try:
        value = int(raw_number)
    except ValueError:
        return False
    return 1900 <= value <= 2099


def _term_in_text(term: str, text: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def _dedupe_refs(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        ref = str(raw or "").strip()
        if not ref or ref in seen:
            continue
        out.append(ref)
        seen.add(ref)
    return out
