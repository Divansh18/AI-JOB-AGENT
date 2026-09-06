"""Deterministic context assembly for optional LLM application intelligence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..config.loader import AppConfig
from ..domain.application_intelligence import (
    ApplicationQuestion,
    ResumeWordingRequest,
    make_application_question,
)
from ..domain.application_planner import is_sensitive_application_question
from ..persistence.repositories import (
    ApplicationAutofillRunRepo,
    ApplicationPlanRepo,
    CandidateAnswerRepo,
    CandidateFactRepo,
    FilterRepo,
    JobRepo,
    LlmArtifactRepo,
    MasterResumeRepo,
    ScoreRepo,
)
from . import resume_intelligence as resume_service


PURPOSE_JOB_INTELLIGENCE = "job_intelligence"
PURPOSE_ANSWER_DRAFTING = "application_answer_drafting"
PURPOSE_RESUME_WORDING = "resume_wording"
DEFAULT_PROMPT_VERSION = "application_intelligence_v1"
DEFAULT_ANSWER_PROMPT_VERSION = "application_answer_drafting_v1"
DEFAULT_RESUME_WORDING_PROMPT_VERSION = "resume_wording_v2"
RESUME_WORDING_PROTOCOL_VERSION = "resume_wording_protocol_v3"
DEFAULT_MAX_EVIDENCE_CHARS = 900
DEFAULT_MAX_JD_INTELLIGENCE_EVIDENCE_CHARS = 500
MAX_JD_INTELLIGENCE_EVIDENCE_ITEMS = 18

SAFE_PERSONAL_PROFILE_KEYS = {
    "summary",
    "professional_summary",
    "headline",
    "current_location",
    "years_experience",
    "experience_stage",
}
SAFE_CONTEXT_CATEGORIES = {
    "work_experience",
    "projects",
    "education",
    "skills",
    "evidence",
}
EXCLUDED_CONTEXT_CATEGORIES = {
    "availability",
    "work_authorization",
    "location_preferences",
    "links",
}


class ApplicationIntelligenceContextError(Exception):
    pass


@dataclass(frozen=True)
class CandidateEvidenceContext:
    ref: str
    aliases: list[str]
    category: str
    key: str
    source: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "aliases": self.aliases,
            "category": self.category,
            "key": self.key,
            "source": self.source,
            "text": self.text,
        }


@dataclass(frozen=True)
class ApplicationIntelligenceContext:
    job_id: int
    purpose: str
    prompt_version: str
    source_truth_hash: str
    job: dict[str, Any]
    deterministic: dict[str, Any]
    candidate_evidence: list[CandidateEvidenceContext] = field(default_factory=list)

    @property
    def allowed_evidence_refs(self) -> list[str]:
        return _dedupe_strings([item.ref for item in self.candidate_evidence])

    @property
    def deterministic_resume_decision(self) -> str | None:
        return self.deterministic.get("resume", {}).get("decision")

    @property
    def deterministic_fit_score(self) -> int | None:
        value = self.deterministic.get("fit", {}).get("overall_fit_score")
        return int(value) if value is not None else None

    @property
    def deterministic_blockers(self) -> list[str]:
        blockers: list[str] = []
        blockers.extend(self.deterministic.get("filter", {}).get("rules_failed") or [])
        blockers.extend(self.deterministic.get("fit", {}).get("hard_blockers") or [])
        return _dedupe_strings(blockers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "purpose": self.purpose,
            "prompt_version": self.prompt_version,
            "source_truth_hash": self.source_truth_hash,
            "job": self.job,
            "deterministic": _canonicalize_payload_evidence_refs(
                self.deterministic,
                self.candidate_evidence,
            ),
            "candidate_evidence": [
                _candidate_evidence_provider_payload(item)
                for item in self.candidate_evidence
            ],
            "allowed_evidence_refs": self.allowed_evidence_refs,
        }


@dataclass(frozen=True)
class VerifiedAnswerContext:
    question_key: str
    category: str
    answer_text: str
    evidence_refs: list[str]
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_key": self.question_key,
            "category": self.category,
            "answer_text": self.answer_text,
            "evidence_refs": self.evidence_refs,
            "source": self.source,
        }


@dataclass(frozen=True)
class ApplicationAnswerContext:
    application_id: int
    job_id: int
    purpose: str
    prompt_version: str
    source_truth_hash: str
    job: dict[str, Any]
    application_plan: dict[str, Any]
    job_intelligence: dict[str, Any] | None
    questions: list[ApplicationQuestion]
    candidate_evidence: list[CandidateEvidenceContext] = field(default_factory=list)
    verified_answers: list[VerifiedAnswerContext] = field(default_factory=list)

    @property
    def allowed_evidence_refs(self) -> list[str]:
        refs: list[str] = []
        for item in self.candidate_evidence:
            refs.append(item.ref)
            refs.extend(item.aliases)
        return _dedupe_strings(refs)

    @property
    def safe_questions(self) -> list[ApplicationQuestion]:
        return [question for question in self.questions if question.llm_draft_allowed]

    @property
    def blocked_questions(self) -> list[ApplicationQuestion]:
        return [question for question in self.questions if not question.llm_draft_allowed]

    @property
    def evidence_text_by_ref(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in self.candidate_evidence:
            out[item.ref] = item.text
            for alias in item.aliases:
                out[alias] = item.text
        return out

    @property
    def job_context_text(self) -> str:
        parts = [
            self.job.get("title") or "",
            self.job.get("company") or "",
            self.job.get("location_raw") or "",
            self.job.get("description_text") or "",
        ]
        if self.job_intelligence:
            parts.append(json.dumps(self.job_intelligence, ensure_ascii=True, sort_keys=True))
        return " ".join(str(part) for part in parts if part)

    def as_dict(self) -> dict[str, Any]:
        return {
            "application_id": self.application_id,
            "job_id": self.job_id,
            "purpose": self.purpose,
            "prompt_version": self.prompt_version,
            "source_truth_hash": self.source_truth_hash,
            "job": self.job,
            "application_plan": self.application_plan,
            "job_intelligence": self.job_intelligence,
            "questions": [question.as_dict() for question in self.questions],
            "candidate_evidence": [item.as_dict() for item in self.candidate_evidence],
            "verified_answers": [item.as_dict() for item in self.verified_answers],
            "allowed_evidence_refs": self.allowed_evidence_refs,
        }


@dataclass(frozen=True)
class SelectedResumeItemContext:
    item_key: str
    section: str
    text: str
    evidence_refs: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_key": self.item_key,
            "section": self.section,
            "text": self.text,
            "evidence_refs": self.evidence_refs,
        }


@dataclass(frozen=True)
class ResumeWordingContext:
    job_id: int
    purpose: str
    prompt_version: str
    source_truth_hash: str
    request: ResumeWordingRequest
    job: dict[str, Any]
    master_resume: dict[str, Any]
    deterministic: dict[str, Any]
    job_intelligence: dict[str, Any] | None
    selected_resume_items: list[SelectedResumeItemContext]
    selected_evidence: list[CandidateEvidenceContext]
    protected_terms: list[str] = field(default_factory=list)

    @property
    def allowed_evidence_refs(self) -> list[str]:
        return _dedupe_strings([item.ref for item in self.selected_evidence])

    @property
    def allowed_item_keys(self) -> list[str]:
        allowed = set(self.allowed_evidence_refs)
        return _dedupe_strings(
            [
                item.item_key
                for item in self.selected_resume_items
                if any(ref in allowed for ref in item.evidence_refs)
            ]
        )

    @property
    def evidence_text_by_ref(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in self.selected_evidence:
            out[item.ref] = item.text
        for item in self.selected_resume_items:
            for ref in item.evidence_refs:
                out.setdefault(ref, item.text)
        return out

    @property
    def item_by_key(self) -> dict[str, SelectedResumeItemContext]:
        return {item.item_key: item for item in _selected_items_for_evidence(self.selected_resume_items, self.selected_evidence)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "purpose": self.purpose,
            "prompt_version": self.prompt_version,
            "source_truth_hash": self.source_truth_hash,
            **resume_wording_provider_context(self),
            "master_resume": _master_resume_metadata_only(self.master_resume),
            "allowed_evidence_refs": self.allowed_evidence_refs,
            "allowed_item_keys": self.allowed_item_keys,
            "protected_terms": self.protected_terms,
        }


def build_application_intelligence_context(
    conn,
    config: AppConfig,
    job_id: int,
    *,
    purpose: str = PURPOSE_JOB_INTELLIGENCE,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    max_jd_chars: int = 6000,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    max_context_chars: int | None = None,
) -> ApplicationIntelligenceContext:
    job = JobRepo(conn).get(job_id)
    if job is None:
        raise ApplicationIntelligenceContextError(f"job {job_id} not found")

    filter_row = FilterRepo(conn).get(job_id)
    score_row = ScoreRepo(conn).get(job_id)
    prepared = resume_service.prepare_resume_variant(conn, config, job_id)
    verified_facts = CandidateFactRepo(conn).list(verified=True)
    jd_evidence_chars = min(max_evidence_chars, DEFAULT_MAX_JD_INTELLIGENCE_EVIDENCE_CHARS)
    all_evidence = [
        item
        for item in (
            _candidate_evidence(fact, max_chars=jd_evidence_chars)
            for fact in verified_facts
        )
        if item is not None
    ]

    filter_payload = _filter_payload(filter_row)
    fit_payload = prepared.fit_report.as_dict()
    relevant_refs = _relevant_evidence_refs(fit_payload, prepared.tailored_resume.recommendation.evidence_refs)
    evidence = _select_job_relevant_evidence(
        all_evidence,
        relevant_refs,
        _job_relevance_text(job, prepared.analysis, prepared.fit_report),
    )
    context = ApplicationIntelligenceContext(
        job_id=job_id,
        purpose=purpose,
        prompt_version=prompt_version,
        source_truth_hash=prepared.source_truth_hash,
        job=_compact_job_payload(job, job_id=job_id, max_jd_chars=max_jd_chars),
        deterministic=_compact_deterministic_payload(prepared, filter_payload, score_row),
        candidate_evidence=evidence,
    )
    if max_context_chars:
        return pack_application_intelligence_context(context, max_context_chars=max_context_chars)
    return context


def pack_application_intelligence_context(
    context: ApplicationIntelligenceContext,
    *,
    max_context_chars: int,
) -> ApplicationIntelligenceContext:
    """Shrink JD-intelligence context with deterministic priority rules."""
    if max_context_chars <= 0 or application_intelligence_context_size(context) <= max_context_chars:
        return context

    evidence_count = len(context.candidate_evidence)
    jd_len = len(str(context.job.get("description_text") or ""))
    steps = [
        (min(evidence_count, 14), 420, min(jd_len, 5200), 6, 220, True),
        (min(evidence_count, 10), 320, min(jd_len, 4200), 5, 180, True),
        (min(evidence_count, 8), 240, min(jd_len, 3200), 4, 150, False),
        (min(evidence_count, 6), 180, min(jd_len, 2400), 3, 120, False),
        (min(evidence_count, 4), 140, min(jd_len, 1800), 2, 100, False),
        (min(evidence_count, 3), 110, min(jd_len, 1200), 2, 80, False),
        (min(evidence_count, 1), 90, min(jd_len, 800), 1, 70, False),
        (0, 0, min(jd_len, 500), 1, 60, False),
    ]
    packed = context
    for evidence_limit, evidence_chars, jd_chars, match_limit, match_chars, keep_rank_text in steps:
        packed = _repack_context(
            context,
            evidence_limit=evidence_limit,
            evidence_chars=evidence_chars,
            jd_chars=jd_chars,
            match_limit=match_limit,
            match_chars=match_chars,
            keep_rank_text=keep_rank_text,
        )
        if application_intelligence_context_size(packed) <= max_context_chars:
            return packed
    return packed


def application_intelligence_context_size(context: ApplicationIntelligenceContext) -> int:
    return len(_compact_json(context.as_dict()))


def build_resume_wording_context(
    conn,
    config: AppConfig,
    job_id: int,
    *,
    purpose: str = PURPOSE_RESUME_WORDING,
    prompt_version: str = DEFAULT_RESUME_WORDING_PROMPT_VERSION,
    max_jd_chars: int = 6000,
    max_master_chars: int = 8000,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    max_context_chars: int | None = None,
) -> tuple[ResumeWordingContext, Any]:
    job = JobRepo(conn).get(job_id)
    if job is None:
        raise ApplicationIntelligenceContextError(f"job {job_id} not found")
    master = MasterResumeRepo(conn).get_active()
    if master is None:
        raise ApplicationIntelligenceContextError("no active master resume is registered")
    prepared = resume_service.prepare_resume_variant(conn, config, job_id)
    tailored = prepared.tailored_resume
    if tailored.recommendation.decision != "tailor":
        raise ApplicationIntelligenceContextError(
            f"resume wording requires deterministic tailor decision, got {tailored.recommendation.decision}"
        )
    if tailored.preview is None:
        raise ApplicationIntelligenceContextError("deterministic tailored resume preview is missing")

    allowed_refs = _dedupe_strings(tailored.recommendation.evidence_refs)
    if not allowed_refs:
        raise ApplicationIntelligenceContextError("deterministic resume selection has no evidence refs")
    selected_items = _selected_resume_items(tailored.preview, set(allowed_refs))
    if not selected_items:
        raise ApplicationIntelligenceContextError("no selected resume text maps to deterministic evidence refs")

    verified_facts = CandidateFactRepo(conn).list(verified=True)
    selected_evidence = [
        item
        for item in (_candidate_evidence(fact, max_chars=max_evidence_chars) for fact in verified_facts)
        if item is not None and (item.ref in allowed_refs or any(alias in allowed_refs for alias in item.aliases))
    ]
    selected_items, selected_evidence = _canonical_resume_wording_boundary(
        selected_items,
        selected_evidence,
        evidence_chars=max_evidence_chars,
    )
    if not selected_items:
        raise ApplicationIntelligenceContextError("no selected resume items have canonical verified evidence refs")
    if not selected_evidence:
        raise ApplicationIntelligenceContextError("no canonical verified evidence is available for selected resume items")

    final_allowed_refs = _dedupe_strings([item.ref for item in selected_evidence])
    sections = _dedupe_strings([item.section for item in selected_items])
    request = ResumeWordingRequest(
        job_id=job_id,
        target_role=prepared.analysis.role_title,
        sections=sections,
        allowed_evidence_refs=final_allowed_refs,
        prompt_version=prompt_version,
        source_truth_hash=prepared.source_truth_hash,
    )
    context = ResumeWordingContext(
        job_id=job_id,
        purpose=purpose,
        prompt_version=prompt_version,
        source_truth_hash=prepared.source_truth_hash,
        request=request,
        job={
            "id": job_id,
            "title": job.title,
            "company": job.company_name_raw,
            "location_raw": job.location_raw,
            "remote_type": job.remote_type,
            "remote_scope": job.remote_scope,
            "employment_type": job.employment_type,
            "description_text": _compact_jd_text_for_provider(
                job.description_text or "",
                title=job.title,
                location=job.location_raw,
                max_chars=max_jd_chars,
            ),
            "content_hash": job.content_hash,
        },
        master_resume={
            "id": master.id,
            "identity": master.identity,
            "version": master.version,
            "page_limit": master.page_limit,
            "file_hash": master.file_hash,
            "text": _truncate(master.extracted_text or "", max_master_chars),
        },
        deterministic={
            "analysis": prepared.analysis.as_dict(),
            "fit_report": prepared.fit_report.as_dict(),
            "resume_recommendation": tailored.recommendation.as_dict(),
            "selected_resume_item_keys": [item.item_key for item in selected_items],
            "validation": prepared.validation.as_dict(),
        },
        job_intelligence=None,
        selected_resume_items=selected_items,
        selected_evidence=selected_evidence,
        protected_terms=_preview_protected_terms(tailored.preview),
    )
    if max_context_chars:
        context = pack_resume_wording_context(context, max_context_chars=max_context_chars)
    return context, prepared


def build_application_answer_context(
    conn,
    config: AppConfig,
    application_id: int,
    *,
    purpose: str = PURPOSE_ANSWER_DRAFTING,
    prompt_version: str = DEFAULT_ANSWER_PROMPT_VERSION,
    max_jd_chars: int = 6000,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
) -> ApplicationAnswerContext:
    plan_row = ApplicationPlanRepo(conn).get_by_application(application_id)
    if plan_row is None:
        raise ApplicationIntelligenceContextError(f"application plan {application_id} not found")
    plan_json = _loads(plan_row["plan_json"], {})
    job_id = int(plan_row["job_id"])
    job = JobRepo(conn).get(job_id)
    if job is None:
        raise ApplicationIntelligenceContextError(f"job {job_id} not found")

    verified_facts = CandidateFactRepo(conn).list(verified=True)
    evidence = [
        item
        for item in (_candidate_evidence(fact, max_chars=max_evidence_chars) for fact in verified_facts)
        if item is not None
    ]
    verified_answers = [
        item
        for item in (_verified_answer_context(answer) for answer in CandidateAnswerRepo(conn).list(verified=True))
        if item is not None
    ]
    questions = _application_questions(conn, application_id, plan_json)
    intelligence = _latest_valid_job_intelligence(conn, job_id)
    source_truth_hash = _answer_source_truth_hash(
        evidence=evidence,
        verified_answers=verified_answers,
        questions=questions,
        job_content_hash=job.content_hash,
        plan_updated_at=plan_row["updated_at"],
        intelligence=intelligence,
    )
    return ApplicationAnswerContext(
        application_id=application_id,
        job_id=job_id,
        purpose=purpose,
        prompt_version=prompt_version,
        source_truth_hash=source_truth_hash,
        job={
            "id": job_id,
            "title": job.title,
            "company": job.company_name_raw,
            "location_raw": job.location_raw,
            "remote_type": job.remote_type,
            "remote_scope": job.remote_scope,
            "employment_type": job.employment_type,
            "apply_url": job.apply_url,
            "description_text": (job.description_text or "")[:max(0, max_jd_chars)],
            "content_hash": job.content_hash,
        },
        application_plan={
            "application_id": application_id,
            "job_id": job_id,
            "state": plan_row["state"],
            "blockers": plan_json.get("blockers") or [],
            "review_required": bool(plan_row["review_required"]),
            "resume_id": plan_row["resume_variant_id"],
            "resume_path": plan_row["resume_path"],
            "resume_page_validation_status": plan_json.get("resume_page_validation_status"),
            "resume_page_count": plan_json.get("resume_page_count"),
            "known_answer_keys": [answer.get("question_key") for answer in plan_json.get("known_answers") or []],
        },
        job_intelligence=intelligence,
        questions=questions,
        candidate_evidence=evidence,
        verified_answers=verified_answers,
    )


def resume_wording_context_fingerprint(context: ResumeWordingContext, *, model: str) -> str:
    payload = {
        "purpose": context.purpose,
        "prompt_version": context.prompt_version,
        "protocol_version": RESUME_WORDING_PROTOCOL_VERSION,
        "model": model,
        "job_content_hash": context.job.get("content_hash"),
        "source_truth_hash": context.source_truth_hash,
        "selected_resume_evidence": [item.as_dict() for item in context.selected_resume_items],
        "master_file_hash": context.master_resume.get("file_hash"),
        "provider_context": resume_wording_provider_context(context),
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resume_wording_provider_context(context: ResumeWordingContext) -> dict[str, Any]:
    deterministic = _resume_wording_deterministic_payload(context)
    selected_items = _selected_items_for_evidence(context.selected_resume_items, context.selected_evidence)
    allowed_item_keys = _dedupe_strings([item.item_key for item in selected_items])
    return {
        "job_id": context.job_id,
        "purpose": context.purpose,
        "prompt_version": context.prompt_version,
        "protocol_version": RESUME_WORDING_PROTOCOL_VERSION,
        "job": {
            "id": context.job.get("id"),
            "title": context.job.get("title"),
            "company": context.job.get("company"),
            "location_raw": context.job.get("location_raw"),
            "remote_type": context.job.get("remote_type"),
            "remote_scope": context.job.get("remote_scope"),
            "employment_type": context.job.get("employment_type"),
            "description_text": context.job.get("description_text"),
            "content_hash": context.job.get("content_hash"),
        },
        "deterministic": deterministic,
        "request": {
            "job_id": context.request.job_id,
            "target_role": context.request.target_role,
            "sections": context.request.sections,
            "prompt_version": context.request.prompt_version,
            "source_truth_hash": context.request.source_truth_hash,
            "allowed_evidence_refs": context.allowed_evidence_refs,
            "allowed_item_keys": allowed_item_keys,
        },
        "selected_resume_items": [item.as_dict() for item in selected_items],
        "selected_evidence": [
            _candidate_evidence_provider_payload(item)
            for item in context.selected_evidence
        ],
        "allowed_evidence_refs": context.allowed_evidence_refs,
        "allowed_item_keys": allowed_item_keys,
        "protected_terms": _limited_strings(context.protected_terms, limit=18, max_chars=80),
    }


def resume_wording_context_size(context: ResumeWordingContext) -> int:
    return len(_compact_json(resume_wording_provider_context(context)))


def pack_resume_wording_context(
    context: ResumeWordingContext,
    *,
    max_context_chars: int,
) -> ResumeWordingContext:
    """Shrink resume-wording context without changing selected item identity."""
    if max_context_chars <= 0 or resume_wording_context_size(context) <= max_context_chars:
        return context

    jd_len = len(str(context.job.get("description_text") or ""))
    evidence_len = max((len(item.text) for item in context.selected_evidence), default=0)
    steps = [
        (min(jd_len, 4200), min(evidence_len, 700), 8, 8, 8, 18),
        (min(jd_len, 3200), min(evidence_len, 520), 6, 6, 6, 14),
        (min(jd_len, 2400), min(evidence_len, 400), 5, 5, 5, 12),
        (min(jd_len, 1600), min(evidence_len, 300), 4, 4, 4, 10),
        (min(jd_len, 1000), min(evidence_len, 220), 3, 3, 3, 8),
        (min(jd_len, 600), min(evidence_len, 160), 2, 2, 2, 6),
        (min(jd_len, 300), min(evidence_len, 120), 1, 1, 1, 4),
    ]
    packed = context
    for jd_chars, evidence_chars, requirement_limit, match_limit, reason_limit, protected_limit in steps:
        packed = _repack_resume_wording_context(
            context,
            jd_chars=jd_chars,
            evidence_chars=evidence_chars,
            requirement_limit=requirement_limit,
            match_limit=match_limit,
            reason_limit=reason_limit,
            protected_limit=protected_limit,
        )
        if resume_wording_context_size(packed) <= max_context_chars:
            return packed
    return packed


def _resume_wording_deterministic_payload(context: ResumeWordingContext) -> dict[str, Any]:
    deterministic = _canonicalize_payload_evidence_refs(
        context.deterministic,
        context.selected_evidence,
    )
    analysis = deterministic.get("analysis") or {}
    fit_report = deterministic.get("fit_report") or {}
    recommendation = deterministic.get("resume_recommendation") or {}
    validation = deterministic.get("validation") or {}
    return {
        "resume_decision": recommendation.get("decision"),
        "resume_reasons": _limited_strings(recommendation.get("reasons") or [], limit=5, max_chars=160),
        "resume_validation_status": validation.get("status"),
        "resume_validation_ok": validation.get("ok"),
        "target_role": analysis.get("role_title"),
        "seniority": analysis.get("seniority"),
        "fit_score": fit_report.get("overall_fit_score"),
        "required_skills": _limited_strings(analysis.get("required_skills") or [], limit=12, max_chars=60),
        "preferred_skills": _limited_strings(analysis.get("preferred_skills") or [], limit=10, max_chars=60),
        "responsibilities": _limited_meaningful_strings(
            analysis.get("responsibilities") or [], limit=6, max_chars=120
        ),
        "education_requirements": _limited_strings(
            analysis.get("education_requirements") or [], limit=4, max_chars=100
        ),
        "domain_signals": _limited_strings(analysis.get("domain_signals") or [], limit=6, max_chars=70),
        "location_remote_eligibility": _compact_mapping(analysis.get("location_remote_eligibility") or {}),
        "strongest_matches": _resume_wording_matches(fit_report.get("strongest_matches") or [], limit=5),
        "partial_matches": _resume_wording_matches(fit_report.get("partial_matches") or [], limit=4),
        "missing_requirements": _resume_wording_matches(fit_report.get("missing_requirements") or [], limit=5),
    }


def _resume_wording_matches(matches: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in matches:
        text = str(item.get("text") or "")
        if not _is_meaningful_text(text):
            continue
        out.append(
            {
                "category": item.get("category"),
                "text": _truncate(text, 120),
                "status": item.get("status"),
                "evidence_refs": _dedupe_strings(item.get("evidence_refs") or []),
                "required": item.get("required"),
            }
        )
        if len(out) >= limit:
            break
    return out


def _repack_resume_wording_context(
    context: ResumeWordingContext,
    *,
    jd_chars: int,
    evidence_chars: int,
    requirement_limit: int,
    match_limit: int,
    reason_limit: int,
    protected_limit: int,
) -> ResumeWordingContext:
    deterministic = json.loads(_compact_json(context.deterministic))
    analysis = deterministic.get("analysis") or {}
    analysis["required_skills"] = _limited_strings(
        analysis.get("required_skills") or [], limit=requirement_limit, max_chars=60
    )
    analysis["preferred_skills"] = _limited_strings(
        analysis.get("preferred_skills") or [], limit=max(1, requirement_limit - 2), max_chars=60
    )
    analysis["responsibilities"] = _limited_meaningful_strings(
        analysis.get("responsibilities") or [], limit=requirement_limit, max_chars=100
    )
    analysis["education_requirements"] = _limited_strings(
        analysis.get("education_requirements") or [], limit=max(1, requirement_limit // 2), max_chars=90
    )
    analysis["domain_signals"] = _limited_strings(
        analysis.get("domain_signals") or [], limit=max(1, requirement_limit // 2), max_chars=70
    )
    analysis["requirements"] = [
        {
            **item,
            "text": _truncate(item.get("text"), 100),
        }
        for item in (analysis.get("requirements") or [])[: max(2, requirement_limit)]
        if _is_meaningful_text(str(item.get("text") or ""))
    ]

    fit_report = deterministic.get("fit_report") or {}
    for key in ("strongest_matches", "partial_matches", "missing_requirements"):
        fit_report[key] = [
            {
                **item,
                "text": _truncate(item.get("text"), 110),
            }
            for item in (fit_report.get(key) or [])[:match_limit]
            if _is_meaningful_text(str(item.get("text") or ""))
        ]
    recommendation = deterministic.get("resume_recommendation") or {}
    recommendation["reasons"] = _limited_strings(
        recommendation.get("reasons") or [], limit=reason_limit, max_chars=140
    )

    job = dict(context.job)
    job["description_text"] = _truncate(job.get("description_text"), jd_chars)
    selected_evidence = [
        CandidateEvidenceContext(
            ref=item.ref,
            aliases=[],
            category=item.category,
            key=item.key,
            source=item.source,
            text=_truncate(item.text, evidence_chars),
        )
        for item in context.selected_evidence
    ]
    request = ResumeWordingRequest(
        job_id=context.request.job_id,
        target_role=context.request.target_role,
        sections=context.request.sections,
        allowed_evidence_refs=[item.ref for item in selected_evidence],
        prompt_version=context.request.prompt_version,
        source_truth_hash=context.request.source_truth_hash,
    )
    return ResumeWordingContext(
        job_id=context.job_id,
        purpose=context.purpose,
        prompt_version=context.prompt_version,
        source_truth_hash=context.source_truth_hash,
        request=request,
        job=job,
        master_resume=_master_resume_metadata_only(context.master_resume),
        deterministic=deterministic,
        job_intelligence=None,
        selected_resume_items=context.selected_resume_items,
        selected_evidence=selected_evidence,
        protected_terms=_limited_strings(context.protected_terms, limit=protected_limit, max_chars=80),
    )


def _master_resume_metadata_only(master_resume: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": master_resume.get("id"),
        "identity": master_resume.get("identity"),
        "version": master_resume.get("version"),
        "page_limit": master_resume.get("page_limit"),
        "file_hash": master_resume.get("file_hash"),
    }


def context_fingerprint(context: ApplicationIntelligenceContext, *, model: str) -> str:
    payload = {
        "purpose": context.purpose,
        "prompt_version": context.prompt_version,
        "model": model,
        "job_content_hash": context.job.get("content_hash"),
        "source_truth_hash": context.source_truth_hash,
        "context": context.as_dict(),
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def answer_context_fingerprint(context: ApplicationAnswerContext, *, model: str) -> str:
    payload = {
        "purpose": context.purpose,
        "prompt_version": context.prompt_version,
        "model": model,
        "job_content_hash": context.job.get("content_hash"),
        "source_truth_hash": context.source_truth_hash,
        "questions": [question.as_dict() for question in context.safe_questions],
        "context": context.as_dict(),
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compact_job_payload(job, *, job_id: int, max_jd_chars: int) -> dict[str, Any]:
    return {
        "id": job_id,
        "title": _truncate(job.title, 160),
        "company": _truncate(job.company_name_raw, 160),
        "location_raw": _truncate(job.location_raw, 220),
        "locations": _limited_strings(getattr(job, "locations", []) or [], limit=8, max_chars=80),
        "country": job.country,
        "remote_type": job.remote_type,
        "remote_scope": job.remote_scope,
        "employment_type": job.employment_type,
        "apply_url": _truncate(job.apply_url, 260),
        "description_text": _compact_jd_text_for_provider(
            job.description_text or "",
            title=job.title,
            location=job.location_raw,
            max_chars=max_jd_chars,
        ),
        "content_hash": job.content_hash,
    }


def _compact_deterministic_payload(prepared, filter_payload: dict[str, Any], score_row) -> dict[str, Any]:
    analysis = prepared.analysis
    fit_report = prepared.fit_report
    recommendation = prepared.tailored_resume.recommendation
    return {
        "filter": filter_payload,
        "rank": _rank_payload(score_row),
        "analysis": {
            "role_title": _truncate(analysis.role_title, 160),
            "seniority": analysis.seniority,
            "required_skills": _limited_strings(analysis.required_skills, limit=16, max_chars=60),
            "preferred_skills": _limited_strings(analysis.preferred_skills, limit=16, max_chars=60),
            "responsibilities": _limited_meaningful_strings(analysis.responsibilities, limit=10, max_chars=120),
            "minimum_experience_years": analysis.minimum_experience_years,
            "preferred_experience_years": analysis.preferred_experience_years,
            "domain_signals": _limited_strings(analysis.domain_signals, limit=10, max_chars=80),
            "location_remote_eligibility": _compact_mapping(analysis.location_remote_eligibility),
            "education_requirements": _limited_strings(analysis.education_requirements, limit=8, max_chars=120),
            "hard_blockers": _limited_strings(analysis.hard_blockers, limit=8, max_chars=160),
            "resume_keywords": _limited_strings(analysis.resume_keywords, limit=24, max_chars=50),
            "requirements": _compact_requirements(analysis.requirements, limit=24, max_text_chars=140),
        },
        "fit": {
            "overall_fit_score": fit_report.overall_fit_score,
            "hard_blockers": _limited_strings(fit_report.hard_blockers, limit=8, max_chars=160),
            "evidence_coverage": {
                "requirements_total": (fit_report.evidence_coverage or {}).get("requirements_total"),
                "matched": (fit_report.evidence_coverage or {}).get("matched"),
                "partial": (fit_report.evidence_coverage or {}).get("partial"),
                "unsupported": (fit_report.evidence_coverage or {}).get("unsupported"),
                "evidence_ref_count": (fit_report.evidence_coverage or {}).get("evidence_ref_count"),
                "evidence_refs": _dedupe_strings((fit_report.evidence_coverage or {}).get("evidence_refs") or []),
            },
            "strongest_matches": _compact_matches(fit_report.strongest_matches, limit=6),
            "partial_matches": _compact_matches(fit_report.partial_matches, limit=6),
            "missing_requirements": _compact_matches(fit_report.missing_requirements, limit=8),
        },
        "resume": {
            "decision": recommendation.decision,
            "reasons": _limited_strings(recommendation.reasons, limit=5, max_chars=180),
            "evidence_refs": _dedupe_strings(recommendation.evidence_refs),
            "validation_status": prepared.validation.status,
        },
    }


def _compact_match(match, *, max_text_chars: int = 180) -> dict[str, Any]:
    return {
        "requirement_id": match.requirement_id,
        "category": match.category,
        "text": _truncate(match.text, max_text_chars),
        "status": match.status,
        "evidence_refs": _dedupe_strings(match.evidence_refs),
        "confidence": round(float(match.confidence), 3),
        "required": match.required,
    }


def _compact_matches(matches, *, limit: int, max_text_chars: int = 180) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for match in matches:
        if not _is_meaningful_text(match.text):
            continue
        out.append(_compact_match(match, max_text_chars=max_text_chars))
        if len(out) >= limit:
            break
    return out


def _compact_requirements(requirements, *, limit: int, max_text_chars: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for requirement in requirements:
        if not _is_meaningful_text(requirement.text):
            continue
        out.append(
            {
                "id": requirement.id,
                "category": requirement.category,
                "text": _truncate(requirement.text, max_text_chars),
                "required": requirement.required,
                "minimum_years": requirement.minimum_years,
            }
        )
        if len(out) >= limit:
            break
    return out


def _select_job_relevant_evidence(
    evidence: list[CandidateEvidenceContext],
    relevant_refs: set[str],
    relevance_text: str,
    *,
    max_items: int = MAX_JD_INTELLIGENCE_EVIDENCE_ITEMS,
) -> list[CandidateEvidenceContext]:
    tokens = _keyword_tokens(relevance_text)
    scored: list[tuple[int, int, CandidateEvidenceContext]] = []
    for index, item in enumerate(evidence):
        refs = {item.ref, *item.aliases}
        exact = bool(refs & relevant_refs)
        overlap = len(tokens & _keyword_tokens(f"{item.category} {item.key} {item.text}"))
        if not exact and overlap == 0:
            continue
        score = (1000 if exact else 0) + overlap * 20 + _jd_evidence_priority(item)
        scored.append((-score, index, item))
    scored.sort()
    selected = [item for _score, _index, item in scored[:max_items]]
    return _dedupe_evidence_by_text(selected)


def _job_relevance_text(job, analysis, fit_report) -> str:
    parts = [
        job.title,
        job.company_name_raw,
        job.location_raw,
        job.description_text,
        " ".join(analysis.required_skills),
        " ".join(analysis.preferred_skills),
        " ".join(analysis.responsibilities),
        " ".join(analysis.domain_signals),
        " ".join(analysis.resume_keywords),
    ]
    for match in fit_report.strongest_matches + fit_report.partial_matches + fit_report.missing_requirements:
        parts.append(match.text)
    return " ".join(str(part) for part in parts if part)


def _category_priority(category: str) -> int:
    return {
        "work_experience": 40,
        "projects": 35,
        "skills": 30,
        "evidence": 25,
        "education": 10,
        "personal_profile": 5,
    }.get(category, 0)


def _jd_evidence_priority(item: CandidateEvidenceContext) -> int:
    if item.category == "personal_profile" and item.key in {"current_location", "preferred_locations"}:
        return 900
    if item.category == "skills":
        return 800
    if item.category == "education":
        return 700
    return _category_priority(item.category)


def _candidate_evidence_provider_payload(item: CandidateEvidenceContext) -> dict[str, Any]:
    return {
        "ref": item.ref,
        "category": item.category,
        "key": item.key,
        "source": item.source,
        "text": item.text,
    }


def _canonicalize_payload_evidence_refs(
    payload: dict[str, Any],
    evidence: list[CandidateEvidenceContext],
) -> dict[str, Any]:
    ref_map = _canonical_evidence_ref_map(evidence)
    allowed = set(item.ref for item in evidence)

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        out: dict[str, Any] = {}
        for key, raw in value.items():
            if key == "evidence_refs" and isinstance(raw, list):
                out[key] = _dedupe_strings(
                    [
                        ref_map[ref]
                        for ref in (str(item or "").strip() for item in raw)
                        if ref in ref_map and ref_map[ref] in allowed
                    ]
                )
            else:
                out[key] = visit(raw)
        if "evidence_ref_count" in out and isinstance(out.get("evidence_refs"), list):
            out["evidence_ref_count"] = len(out["evidence_refs"])
        return out

    return visit(payload)


def _canonical_evidence_ref_map(evidence: list[CandidateEvidenceContext]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in evidence:
        mapping[item.ref] = item.ref
        for alias in item.aliases:
            mapping[alias] = item.ref
    return mapping


def _canonical_resume_wording_boundary(
    selected_items: list[SelectedResumeItemContext],
    selected_evidence: list[CandidateEvidenceContext],
    *,
    evidence_chars: int,
) -> tuple[list[SelectedResumeItemContext], list[CandidateEvidenceContext]]:
    ref_map = _canonical_evidence_ref_map(selected_evidence)
    evidence_by_ref = {item.ref: item for item in selected_evidence}
    canonical_items: list[SelectedResumeItemContext] = []
    refs_in_item_order: list[str] = []

    for item in selected_items:
        refs = _dedupe_strings(
            [
                ref_map[ref]
                for ref in (str(raw or "").strip() for raw in item.evidence_refs)
                if ref in ref_map and ref_map[ref] in evidence_by_ref
            ]
        )
        if not refs:
            continue
        refs_in_item_order.extend(refs)
        canonical_items.append(
            SelectedResumeItemContext(
                item_key=item.item_key,
                section=item.section,
                text=item.text,
                evidence_refs=refs,
            )
        )

    final_refs = _dedupe_strings(refs_in_item_order)
    canonical_evidence: list[CandidateEvidenceContext] = []
    for ref in final_refs:
        original = evidence_by_ref.get(ref)
        if original is None:
            continue
        support_text = _selected_item_support_text(ref, canonical_items, fallback=original.text)
        canonical_evidence.append(
            CandidateEvidenceContext(
                ref=ref,
                aliases=[],
                category=original.category,
                key=original.key,
                source=original.source,
                text=_truncate(support_text, evidence_chars),
            )
        )
    return canonical_items, canonical_evidence


def _selected_items_for_evidence(
    selected_items: list[SelectedResumeItemContext],
    selected_evidence: list[CandidateEvidenceContext],
) -> list[SelectedResumeItemContext]:
    allowed = set(item.ref for item in selected_evidence)
    out: list[SelectedResumeItemContext] = []
    for item in selected_items:
        refs = _dedupe_strings([ref for ref in item.evidence_refs if ref in allowed])
        if not refs:
            continue
        out.append(
            SelectedResumeItemContext(
                item_key=item.item_key,
                section=item.section,
                text=item.text,
                evidence_refs=refs,
            )
        )
    return out


def _selected_item_support_text(
    ref: str,
    selected_items: list[SelectedResumeItemContext],
    *,
    fallback: str,
) -> str:
    texts = [
        item.text
        for item in selected_items
        if ref in item.evidence_refs
    ]
    return " ".join(_dedupe_text_units(" ".join(texts))) or fallback


def _dedupe_evidence_by_text(
    evidence: list[CandidateEvidenceContext],
) -> list[CandidateEvidenceContext]:
    out: list[CandidateEvidenceContext] = []
    seen: dict[str, int] = {}
    for item in evidence:
        key = _normalize_for_dedupe(item.text)
        if not key:
            continue
        if key in seen:
            index = seen[key]
            existing = out[index]
            out[index] = CandidateEvidenceContext(
                ref=existing.ref,
                aliases=_dedupe_strings(existing.aliases + [item.ref] + item.aliases),
                category=existing.category,
                key=existing.key,
                source=existing.source,
                text=existing.text,
            )
            continue
        seen[key] = len(out)
        out.append(item)
    return out


def _repack_context(
    context: ApplicationIntelligenceContext,
    *,
    evidence_limit: int,
    evidence_chars: int,
    jd_chars: int,
    match_limit: int,
    match_chars: int,
    keep_rank_text: bool,
) -> ApplicationIntelligenceContext:
    job = dict(context.job)
    job["description_text"] = _truncate(str(job.get("description_text") or ""), jd_chars)
    evidence = [
        CandidateEvidenceContext(
            ref=item.ref,
            aliases=item.aliases,
            category=item.category,
            key=item.key,
            source=item.source,
            text=_truncate(item.text, evidence_chars),
        )
        for item in context.candidate_evidence[:evidence_limit]
        if evidence_chars > 0
    ]
    return ApplicationIntelligenceContext(
        job_id=context.job_id,
        purpose=context.purpose,
        prompt_version=context.prompt_version,
        source_truth_hash=context.source_truth_hash,
        job=job,
        deterministic=_trim_deterministic(
            context.deterministic,
            match_limit=match_limit,
            match_chars=match_chars,
            keep_rank_text=keep_rank_text,
        ),
        candidate_evidence=evidence,
    )


def _trim_deterministic(
    deterministic: dict[str, Any],
    *,
    match_limit: int,
    match_chars: int,
    keep_rank_text: bool,
) -> dict[str, Any]:
    trimmed = json.loads(_compact_json(deterministic))
    rank = trimmed.get("rank") or {}
    if not keep_rank_text:
        rank["flags"] = []
        rank["explanation"] = []
    fit = trimmed.get("fit") or {}
    for key in ("strongest_matches", "partial_matches", "missing_requirements"):
        fit[key] = [
            {
                **item,
                "text": _truncate(item.get("text"), match_chars),
            }
            for item in (fit.get(key) or [])[:match_limit]
        ]
    analysis = trimmed.get("analysis") or {}
    analysis["requirements"] = [
        {
            **item,
            "text": _truncate(item.get("text"), match_chars),
        }
        for item in (analysis.get("requirements") or [])[: max(4, match_limit * 4)]
    ]
    analysis["responsibilities"] = _limited_meaningful_strings(
        analysis.get("responsibilities") or [], limit=max(2, match_limit), max_chars=match_chars
    )
    resume = trimmed.get("resume") or {}
    resume["reasons"] = _limited_strings(resume.get("reasons") or [], limit=max(1, match_limit), max_chars=match_chars)
    return trimmed


def _compact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted((value or {}).keys()):
        raw = value[key]
        if isinstance(raw, str):
            out[key] = _truncate(raw, 180)
        elif isinstance(raw, list):
            out[key] = _limited_strings(raw, limit=8, max_chars=100)
        elif isinstance(raw, dict):
            out[key] = _compact_mapping(raw)
        else:
            out[key] = raw
    return out


def _compact_text(text: str, *, max_chars: int) -> str:
    return _truncate(" ".join(_dedupe_text_units(text)), max_chars)


def _compact_jd_text_for_provider(
    text: str,
    *,
    title: str,
    location: str,
    max_chars: int,
) -> str:
    cleaned = _strip_jd_boilerplate(_compact_text(text, max_chars=max_chars * 2))
    cleaned = _strip_leading_job_page_header(cleaned, title=title, location=location)
    return _truncate(cleaned, max_chars)


def _strip_jd_boilerplate(text: str) -> str:
    cleaned = _remove_navigation_boilerplate(_clean(text))
    if not cleaned:
        return ""
    for marker in ('{"@context"', '"@context"', "var gaCode", "jobs powered by"):
        index = cleaned.lower().find(marker.lower())
        if index != -1:
            cleaned = cleaned[:index]
    return _clean(cleaned)


def _remove_navigation_boilerplate(text: str) -> str:
    cleaned = re.sub(r"\bapply\s+for\s+this\s+job\b", " ", text or "", flags=re.I)
    cleaned = re.sub(r"\bhome\s+page\b", " ", cleaned, flags=re.I)
    return _clean(cleaned)


def _strip_leading_job_page_header(text: str, *, title: str, location: str) -> str:
    cleaned = _clean(text)
    if not cleaned:
        return ""
    lower = cleaned.lower()
    anchors = (
        "what you'll own",
        "what you’ll own",
        "what you will own",
        "what you'll bring",
        "what you’ll bring",
        "responsibilities",
        "requirements",
        "about the role",
        "job description",
    )
    matches = [lower.find(anchor) for anchor in anchors if lower.find(anchor) > 0]
    if not matches:
        return cleaned
    first_anchor = min(matches)
    if first_anchor > 700:
        return cleaned
    prefix = lower[:first_anchor]
    title_tokens = _keyword_tokens(title)
    location_tokens = _keyword_tokens(location)
    metadata_markers = {
        "full time",
        "part time",
        "contract",
        "internship",
        "remote",
        "hybrid",
        "on site",
        "on-site",
        "onsite",
        "engineering",
        "department",
    }
    token_overlap = bool((title_tokens | location_tokens) & _keyword_tokens(prefix))
    if token_overlap or any(marker in prefix for marker in metadata_markers):
        return cleaned[first_anchor:].lstrip(" -/|:")
    return cleaned


def _dedupe_text_units(text: str) -> list[str]:
    cleaned = _clean(text)
    if not cleaned:
        return []
    units = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    if len(units) <= 1:
        units = re.split(r"\s{2,}|;\s+", cleaned)
    out: list[str] = []
    seen: set[str] = set()
    for unit in units:
        value = _clean(unit)
        key = _normalize_for_dedupe(value)
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _keyword_tokens(text: str) -> set[str]:
    stopwords = {
        "and", "are", "but", "for", "from", "have", "into", "that", "the",
        "this", "with", "will", "you", "your", "our", "role", "work",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}", (text or "").lower())
        if token not in stopwords
    }


def _limited_strings(values, *, limit: int, max_chars: int) -> list[str]:
    return [_truncate(value, max_chars) for value in _dedupe_strings(list(values or []))[:limit]]


def _limited_meaningful_strings(values, *, limit: int, max_chars: int) -> list[str]:
    meaningful = [value for value in values or [] if _is_meaningful_text(str(value))]
    return _limited_strings(meaningful, limit=limit, max_chars=max_chars)


def _truncate(value: Any, max_chars: int) -> str:
    text = _remove_navigation_boilerplate(_clean(str(value or "")))
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _is_meaningful_text(text: str) -> bool:
    cleaned = _clean(text)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if "apply for this job" in lowered or "jobs powered by" in lowered:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{2,}", cleaned)
    return len(words) >= 2


def _filter_payload(row) -> dict[str, Any]:
    if row is None:
        return {
            "present": False,
            "passed": False,
            "rules_failed": [],
            "signals": {},
            "notes": [],
        }
    return {
        "present": True,
        "passed": bool(row["passed"]),
        "rules_failed": _loads(row["rules_failed"], []),
        "signals": _compact_filter_signals(_loads(row["signals"], {})),
        "notes": _limited_strings(_loads(row["notes"], []), limit=6, max_chars=140),
        "filters_version": row["filters_version"],
    }


def _rank_payload(row) -> dict[str, Any]:
    if row is None:
        return {
            "present": False,
            "score": None,
            "components": {},
            "flags": [],
            "explanation": [],
        }
    return {
        "present": True,
        "score": float(row["score"]),
        "components": _compact_mapping(_loads(row["components"], {})),
        "flags": _limited_strings(_loads(row["flags"], []), limit=8, max_chars=120),
        "explanation": _limited_strings(_loads(row["explanation"], []), limit=6, max_chars=140),
        "rank_version": row["rank_version"],
    }


def _compact_filter_signals(signals: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "location",
        "title",
        "title_role",
        "seniority",
        "yoe",
        "experience",
        "employment_type",
    }
    return {
        key: _compact_mapping(value) if isinstance(value, dict) else value
        for key, value in (signals or {}).items()
        if key in keep
    }


def _candidate_evidence(fact, *, max_chars: int) -> CandidateEvidenceContext | None:
    if fact.category in EXCLUDED_CONTEXT_CATEGORIES:
        return None
    if fact.category == "personal_profile" and fact.key not in SAFE_PERSONAL_PROFILE_KEYS:
        return None
    if fact.category != "personal_profile" and fact.category not in SAFE_CONTEXT_CATEGORIES:
        return None
    text = _fact_text(fact.value)
    if not text:
        return None
    return CandidateEvidenceContext(
        ref=_fact_ref(fact),
        aliases=_fact_aliases(fact),
        category=fact.category,
        key=fact.key,
        source=fact.source,
        text=text[:max(0, max_chars)],
    )


def _verified_answer_context(answer) -> VerifiedAnswerContext | None:
    if not answer.verified:
        return None
    if answer.human_review_required or is_sensitive_application_question(answer.question_key):
        return None
    text = _clean(answer.answer_text)
    if not text:
        return None
    return VerifiedAnswerContext(
        question_key=answer.question_key,
        category=answer.category,
        answer_text=text[:700],
        evidence_refs=_dedupe_strings(answer.evidence_refs),
        source=answer.source,
    )


def _selected_resume_items(preview, allowed_refs: set[str]) -> list[SelectedResumeItemContext]:
    items: list[SelectedResumeItemContext] = []

    def include(refs: list[str]) -> list[str]:
        return _dedupe_strings([ref for ref in refs if ref in allowed_refs])

    if preview.summary:
        refs = include(preview.summary.evidence_refs)
        if refs:
            items.append(SelectedResumeItemContext("summary", "summary", preview.summary.text, refs))
    for entry_index, entry in enumerate(preview.experience):
        if entry.summary:
            refs = include(entry.summary.evidence_refs)
            if refs:
                items.append(
                    SelectedResumeItemContext(
                        f"experience:{entry_index}:summary",
                        "experience",
                        entry.summary.text,
                        refs,
                    )
                )
        for bullet_index, bullet in enumerate(entry.bullets):
            refs = include(bullet.evidence_refs)
            if refs:
                items.append(
                    SelectedResumeItemContext(
                        f"experience:{entry_index}:bullet:{bullet_index}",
                        "experience",
                        bullet.text,
                        refs,
                    )
                )
    for entry_index, entry in enumerate(preview.projects):
        if entry.summary:
            refs = include(entry.summary.evidence_refs)
            if refs:
                items.append(
                    SelectedResumeItemContext(
                        f"projects:{entry_index}:summary",
                        "projects",
                        entry.summary.text,
                        refs,
                    )
                )
        for bullet_index, bullet in enumerate(entry.bullets):
            refs = include(bullet.evidence_refs)
            if refs:
                items.append(
                    SelectedResumeItemContext(
                        f"projects:{entry_index}:bullet:{bullet_index}",
                        "projects",
                        bullet.text,
                        refs,
                    )
                )
    return items


def _preview_protected_terms(preview) -> list[str]:
    terms: list[str] = []
    for entry in preview.experience:
        terms.extend(str(getattr(entry, key) or "") for key in ("company", "title"))
    for entry in preview.projects:
        terms.extend(str(getattr(entry, key) or "") for key in ("name", "role"))
    for entry in preview.education:
        terms.extend(str(getattr(entry, key) or "") for key in ("institution", "degree", "field_of_study"))
    return _dedupe_strings(terms)


def _application_questions(conn, application_id: int, plan_json: dict[str, Any]) -> list[ApplicationQuestion]:
    questions: list[ApplicationQuestion] = []
    for field in plan_json.get("unanswered_fields") or []:
        questions.append(
            make_application_question(
                key=str(field.get("key") or field.get("label") or ""),
                text=str(field.get("label") or field.get("key") or ""),
                source="application_plan_unanswered",
                required=bool(field.get("human_review_required")),
            )
        )
    for field in plan_json.get("sensitive_fields") or []:
        questions.append(
            make_application_question(
                key=str(field.get("question_key") or field.get("label") or ""),
                text=str(field.get("label") or field.get("question_key") or ""),
                source="application_plan_sensitive",
                required=True,
            )
        )
    runs = ApplicationAutofillRunRepo(conn).list_by_application(application_id)
    if runs:
        latest = runs[0]
        for field in _loads(latest["unresolved_fields"], []):
            questions.append(
                make_application_question(
                    key=str(field.get("key") or field.get("name") or field.get("label") or ""),
                    text=str(field.get("label") or field.get("name") or field.get("key") or ""),
                    source="autofill_unresolved",
                    required=bool(field.get("required")),
                )
            )
        for field in _loads(latest["sensitive_fields"], []):
            questions.append(
                make_application_question(
                    key=str(field.get("key") or field.get("name") or field.get("label") or ""),
                    text=str(field.get("label") or field.get("name") or field.get("key") or ""),
                    source="autofill_sensitive",
                    required=True,
                )
            )
    return _dedupe_questions(questions)


def _dedupe_questions(questions: list[ApplicationQuestion]) -> list[ApplicationQuestion]:
    out: list[ApplicationQuestion] = []
    seen: set[tuple[str, str]] = set()
    for question in questions:
        key = (question.key, question.text.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(question)
    return out


def _latest_valid_job_intelligence(conn, job_id: int) -> dict[str, Any] | None:
    rows = LlmArtifactRepo(conn).list(job_id=job_id, purpose=PURPOSE_JOB_INTELLIGENCE, limit=10)
    for row in rows:
        if row["status"] != "valid":
            continue
        payload = _loads(row["output_json"], {})
        if payload:
            return {
                "artifact_id": row["id"],
                "prompt_version": row["prompt_version"],
                "model": row["model"],
                "context_fingerprint": row["context_fingerprint"],
                "output": payload,
            }
    return None


def _answer_source_truth_hash(
    *,
    evidence: list[CandidateEvidenceContext],
    verified_answers: list[VerifiedAnswerContext],
    questions: list[ApplicationQuestion],
    job_content_hash: str,
    plan_updated_at: str,
    intelligence: dict[str, Any] | None,
) -> str:
    payload = {
        "evidence": [item.as_dict() for item in evidence],
        "verified_answers": [item.as_dict() for item in verified_answers],
        "questions": [item.as_dict() for item in questions],
        "job_content_hash": job_content_hash,
        "plan_updated_at": plan_updated_at,
        "job_intelligence_fingerprint": (intelligence or {}).get("context_fingerprint"),
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fact_ref(fact) -> str:
    return f"fact:{fact.id}" if fact.id is not None else f"{fact.category}:{fact.key}"


def _fact_aliases(fact) -> list[str]:
    selector = f"{fact.category}:{fact.key}"
    if fact.id is None:
        return []
    return [selector]


def _relevant_evidence_refs(fit_payload: dict[str, Any], recommendation_refs: list[str]) -> set[str]:
    refs: set[str] = set(str(ref) for ref in recommendation_refs if ref)
    coverage = fit_payload.get("evidence_coverage") or {}
    refs.update(str(ref) for ref in (coverage.get("evidence_refs") or []) if ref)
    for key in ("strongest_matches", "partial_matches", "missing_requirements"):
        for match in fit_payload.get(key) or []:
            refs.update(str(ref) for ref in (match.get("evidence_refs") or []) if ref)
    return refs


def _filter_relevant_evidence(
    evidence: list[CandidateEvidenceContext],
    relevant_refs: set[str],
) -> list[CandidateEvidenceContext]:
    if not relevant_refs:
        return evidence[:30]
    selected = [
        item
        for item in evidence
        if item.ref in relevant_refs or any(alias in relevant_refs for alias in item.aliases)
    ]
    return selected or evidence[:30]


def _fact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _clean(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return _clean("; ".join(_fact_text(item) for item in value if _fact_text(item)))
    if isinstance(value, dict):
        parts: list[str] = []
        ordered_keys = [
            "name",
            "title",
            "role",
            "company",
            "institution",
            "degree",
            "field_of_study",
            "location",
            "start_date",
            "end_date",
            "summary",
            "skills",
            "achievements",
            "bullet",
            "proficiency",
        ]
        for key in ordered_keys:
            if key in value:
                rendered = _fact_text(value.get(key))
                if rendered:
                    parts.append(f"{key}: {rendered}")
        for key in sorted(k for k in value if k not in ordered_keys):
            rendered = _fact_text(value.get(key))
            if rendered:
                parts.append(f"{key}: {rendered}")
        return _clean("; ".join(parts))
    return _clean(str(value))


def _loads(raw: str | None, fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _clean(value: str) -> str:
    normalized = (value or "").replace("\\n", " ").replace("\\t", " ")
    return " ".join(normalized.split()).strip()


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
