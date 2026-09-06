"""Application planner for attended, human-reviewed applications."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..config.loader import AppConfig
from ..domain.application_planner import (
    ApplicationPlan,
    ApplicationPlanAnswer,
    ApplicationPlanState,
    build_candidate_field_mappings,
    build_known_answer_mappings,
    build_sensitive_fields,
    build_unanswered_answer_fields,
    canonical_question_key,
)
from ..domain.candidate import build_application_profile
from ..persistence.db import parse_dt
from ..persistence.repositories import (
    ApplicationPlanRepo,
    ApplicationRepo,
    ApplicationAnswerDraftRepo,
    CandidateAnswerRepo,
    CandidateFactRepo,
    JobRepo,
    ResumeVariantRepo,
    ResumeWordingArtifactRepo,
)
from . import resume_output as resume_output_service
from .audit import Audit


class ApplicationPlannerError(Exception):
    pass


def plan_application(conn, config: AppConfig, job_id: int) -> dict[str, Any]:
    job_repo = JobRepo(conn)
    job = job_repo.row(job_id)
    if job is None:
        raise ApplicationPlannerError(f"job {job_id} not found")

    app_repo = ApplicationRepo(conn)
    application_id = app_repo.ensure_for_plan(job_id)
    application_row = app_repo.get(application_id)
    plan_repo = ApplicationPlanRepo(conn)
    existing_plan = plan_repo.get_by_application(application_id)

    now = datetime.now(timezone.utc)
    created_at = parse_dt(existing_plan["created_at"]) if existing_plan is not None else now
    blockers: list[str] = []

    if application_row is not None and application_row["status"] != "to_apply":
        blockers.append(f"application_already_recorded: status={application_row['status']}")

    facts = CandidateFactRepo(conn).list()
    profile = build_application_profile(facts)
    candidate_fields, unanswered_fields = build_candidate_field_mappings(profile)

    verified_answers = CandidateAnswerRepo(conn).list(verified=True)
    known_answers, covered_answer_keys = build_known_answer_mappings(verified_answers)
    approved_llm_answers = _approved_llm_answer_mappings(conn, application_id, covered_answer_keys)
    known_answers = _merge_known_answers(known_answers, approved_llm_answers)
    unanswered_fields.extend(build_unanswered_answer_fields(covered_answer_keys))
    sensitive_fields = build_sensitive_fields(known_answers, profile)

    resume_payload: dict[str, Any] | None = None
    try:
        resume_payload = resume_output_service.build_resume_output(conn, config, job_id)
    except resume_output_service.ResumeOutputError as exc:
        blockers.append(f"resume_unavailable: {exc}")

    resume_summary = _resume_summary(resume_payload, blockers)
    deterministic_resume = {**resume_summary, "source": "deterministic"}
    approved_llm_resume = _approved_llm_resume_summary(conn, config, job_id)
    selected_resume_source = "deterministic"
    if approved_llm_resume is not None:
        if approved_llm_resume["usable"] and not blockers:
            resume_summary = {
                "resume_id": approved_llm_resume["resume_id"],
                "resume_path": approved_llm_resume["resume_path"],
                "resume_decision": "llm_wording_approved",
                "resume_page_count": approved_llm_resume["page_count"],
                "resume_page_validation_status": approved_llm_resume["page_validation_status"],
            }
            selected_resume_source = "llm_resume_wording"
        elif not approved_llm_resume["usable"]:
            blockers.append(f"approved_llm_resume_unavailable: {approved_llm_resume['reason']}")
    application_url = str(job["apply_url"] or "").strip()
    if not application_url:
        blockers.append("missing_application_url")

    answer_drafts = _answer_draft_review_summary(conn, application_id)
    review_required = bool(unanswered_fields or sensitive_fields or answer_drafts["pending"])
    state = ApplicationPlanState.BLOCKED.value if blockers else ApplicationPlanState.READY_FOR_REVIEW.value
    plan = ApplicationPlan(
        application_id=application_id,
        job_id=job_id,
        company=job["company_name_raw"],
        role=job["title"],
        application_url=application_url,
        resume_id=resume_summary["resume_id"],
        resume_path=resume_summary["resume_path"],
        resume_decision=resume_summary["resume_decision"],
        resume_page_count=resume_summary["resume_page_count"],
        resume_page_validation_status=resume_summary["resume_page_validation_status"],
        candidate_fields=candidate_fields,
        known_answers=known_answers,
        unanswered_fields=unanswered_fields,
        sensitive_fields=sensitive_fields,
        blockers=blockers,
        review_required=review_required,
        state=state,
        deterministic_resume=deterministic_resume,
        approved_llm_resume=approved_llm_resume,
        selected_resume_source=selected_resume_source,
        answer_drafts=answer_drafts,
        created_at=created_at,
        updated_at=now,
    )
    plan_repo.upsert(plan)
    Audit(conn).human(
        "application_plan_created",
        "application",
        application_id,
        job_id=job_id,
        state=state,
        review_required=review_required,
        blockers=blockers,
        resume_id=plan.resume_id,
    )
    return plan.as_dict()


def get_application_plan(conn, application_id: int) -> dict[str, Any]:
    row = ApplicationPlanRepo(conn).get_by_application(application_id)
    if row is None:
        raise ApplicationPlannerError(f"application plan {application_id} not found")
    return _plan_payload_from_row(row)


def list_application_plans(
    conn,
    *,
    state: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    rows = ApplicationPlanRepo(conn).list(state=state, limit=limit)
    return [_list_payload_from_row(row) for row in rows]


def _resume_summary(
    resume_payload: dict[str, Any] | None,
    blockers: list[str],
) -> dict[str, Any]:
    if resume_payload is None:
        return {
            "resume_id": None,
            "resume_path": None,
            "resume_decision": None,
            "resume_page_count": None,
            "resume_page_validation_status": None,
        }

    resume = resume_payload.get("tailored_resume") or {}
    recommendation = resume.get("recommendation") or {}
    decision = recommendation.get("decision")
    validation = resume_payload.get("validation") or {}
    resume_id = resume_payload.get("resume_id")
    resume_path = resume.get("output_pdf_path")
    page_count = resume.get("page_count")
    page_status = resume.get("page_validation_status")

    if decision == "poor_fit":
        reasons = recommendation.get("reasons") or []
        reason_text = "; ".join(str(reason) for reason in reasons if reason)
        blockers.append("poor_fit_resume_decision" + (f": {reason_text}" if reason_text else ""))
    elif not validation.get("ok"):
        blockers.append("invalid_resume_artifact: resume validation failed")
    elif not resume_path:
        blockers.append("invalid_resume_artifact: missing resume PDF path")
    elif page_count != 1:
        blockers.append(f"invalid_resume_artifact: expected page_count=1, got {page_count}")
    elif page_status != "valid":
        blockers.append(f"invalid_resume_artifact: page_validation_status={page_status}")

    return {
        "resume_id": int(resume_id) if resume_id is not None else None,
        "resume_path": str(resume_path) if resume_path else None,
        "resume_decision": str(decision) if decision else None,
        "resume_page_count": int(page_count) if page_count is not None else None,
        "resume_page_validation_status": str(page_status) if page_status else None,
    }


def _plan_payload_from_row(row) -> dict[str, Any]:
    payload = json.loads(row["plan_json"]) if row["plan_json"] else {}
    payload["application_id"] = row["application_id"]
    payload["job_id"] = row["job_id"]
    payload["state"] = row["state"]
    payload["resume_id"] = row["resume_variant_id"]
    payload["resume_path"] = row["resume_path"]
    payload["review_required"] = bool(row["review_required"])
    payload["created_at"] = row["created_at"]
    payload["updated_at"] = row["updated_at"]
    return payload


def _list_payload_from_row(row) -> dict[str, Any]:
    payload = json.loads(row["plan_json"]) if row["plan_json"] else {}
    return {
        "application_id": row["application_id"],
        "job_id": row["job_id"],
        "company": row["company_name_raw"],
        "role": row["title"],
        "application_url": row["apply_url"],
        "state": row["state"],
        "review_required": bool(row["review_required"]),
        "resume_id": row["resume_variant_id"],
        "resume_path": row["resume_path"],
        "candidate_field_count": len(payload.get("candidate_fields") or []),
        "known_answer_count": len(payload.get("known_answers") or []),
        "unanswered_count": len(payload.get("unanswered_fields") or []),
        "sensitive_count": len(payload.get("sensitive_fields") or []),
        "blockers": payload.get("blockers") or [],
        "selected_resume_source": payload.get("selected_resume_source") or "deterministic",
        "approved_llm_resume": payload.get("approved_llm_resume"),
        "answer_drafts": payload.get("answer_drafts") or {},
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _approved_llm_answer_mappings(
    conn,
    application_id: int,
    covered_answer_keys: set[str],
) -> list[ApplicationPlanAnswer]:
    answers: list[ApplicationPlanAnswer] = []
    for draft in ApplicationAnswerDraftRepo(conn).approved_for_application(application_id):
        key = canonical_question_key(draft.question_key)
        if not key or key in covered_answer_keys:
            continue
        if draft.question_classification != "safe_free_text":
            continue
        if draft.validation_status != "valid" or draft.validation_errors:
            continue
        answers.append(
            ApplicationPlanAnswer(
                question_key=key,
                stored_question_key=draft.question_key,
                category="llm_safe_free_text",
                answer_text=draft.answer_text,
                evidence_refs=list(draft.evidence_refs),
                source=f"llm_answer_draft:{draft.draft_id}",
                human_review_required=False,
                autofill_safe=True,
                draft_id=draft.draft_id,
                review_status=draft.review_status,
            )
        )
        covered_answer_keys.add(key)
    return answers


def _merge_known_answers(
    verified_answers: list[ApplicationPlanAnswer],
    approved_llm_answers: list[ApplicationPlanAnswer],
) -> list[ApplicationPlanAnswer]:
    merged: list[ApplicationPlanAnswer] = []
    seen: set[str] = set()
    for answer in [*verified_answers, *approved_llm_answers]:
        key = canonical_question_key(answer.question_key)
        if key in seen:
            continue
        merged.append(answer)
        seen.add(key)
    return merged


def _approved_llm_resume_summary(conn, config: AppConfig, job_id: int) -> dict[str, Any] | None:
    artifact = ResumeWordingArtifactRepo(conn).approved_for_job(job_id)
    if artifact is None:
        return None
    if artifact.resume_variant_id is None:
        return _llm_resume_summary(artifact, usable=False, reason="missing_resume_variant")
    row = ResumeVariantRepo(conn).get(artifact.resume_variant_id)
    if row is None:
        return _llm_resume_summary(artifact, usable=False, reason=f"resume_variant_not_found:{artifact.resume_variant_id}")
    reason = _resume_variant_unusable_reason(config.root, row)
    return _llm_resume_summary(
        artifact,
        usable=reason is None,
        reason=reason,
        row=row,
    )


def _llm_resume_summary(artifact, *, usable: bool, reason: str | None, row=None) -> dict[str, Any]:
    return {
        "source": "llm_resume_wording",
        "artifact_id": artifact.artifact_id,
        "review_status": artifact.review_status,
        "reviewer_source": artifact.reviewer_source,
        "reviewed_at": artifact.reviewed_at.isoformat() if artifact.reviewed_at else None,
        "resume_id": int(row["id"]) if row is not None else artifact.resume_variant_id,
        "resume_path": row["output_pdf_path"] if row is not None else None,
        "page_count": int(row["page_count"]) if row is not None and row["page_count"] is not None else None,
        "page_validation_status": row["page_validation_status"] if row is not None else None,
        "validation_status": row["validation_status"] if row is not None else None,
        "usable": usable,
        "reason": reason,
        "provider": artifact.provider,
        "model": artifact.model,
        "prompt_version": artifact.prompt_version,
        "cost_inr": artifact.cost_inr,
    }


def _resume_variant_unusable_reason(root, row) -> str | None:
    if row["validation_status"] != "valid":
        return f"validation_status={row['validation_status']}"
    if row["page_validation_status"] != "valid":
        return f"page_validation_status={row['page_validation_status']}"
    if row["page_count"] != 1:
        return f"page_count={row['page_count']}"
    resume_path = str(row["output_pdf_path"] or "").strip()
    if not resume_path:
        return "missing_resume_path"
    absolute_path = json_safe_path(root, resume_path)
    if not absolute_path.exists():
        return f"resume_file_missing:{resume_path}"
    return None


def json_safe_path(root, raw_path: str):
    from pathlib import Path

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _answer_draft_review_summary(conn, application_id: int) -> dict[str, Any]:
    drafts = ApplicationAnswerDraftRepo(conn).list_by_application(application_id)
    by_status = {"pending_review": [], "approved": [], "rejected": []}
    for draft in drafts:
        by_status.setdefault(draft.review_status, [])
        by_status[draft.review_status].append(
            {
                "draft_id": draft.draft_id,
                "question_key": draft.question_key,
                "question_text": draft.question_text,
                "classification": draft.question_classification,
                "validation_status": draft.validation_status,
                "review_status": draft.review_status,
                "reviewer_source": draft.reviewer_source,
                "reviewed_at": draft.reviewed_at.isoformat() if draft.reviewed_at else None,
                "autofill_safe": draft.review_status == "approved"
                and draft.validation_status == "valid"
                and draft.question_classification == "safe_free_text",
                "provider": draft.provider,
                "model": draft.model,
                "prompt_version": draft.prompt_version,
                "cost_inr": draft.cost_inr,
            }
        )
    return {
        "pending": by_status.get("pending_review", []),
        "approved": by_status.get("approved", []),
        "rejected": by_status.get("rejected", []),
        "pending_count": len(by_status.get("pending_review", [])),
        "approved_count": len(by_status.get("approved", [])),
        "rejected_count": len(by_status.get("rejected", [])),
    }
