"""Application planner for attended, human-reviewed applications."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..config.loader import AppConfig
from ..domain.application_planner import (
    ApplicationPlan,
    ApplicationPlanState,
    build_candidate_field_mappings,
    build_known_answer_mappings,
    build_sensitive_fields,
    build_unanswered_answer_fields,
)
from ..domain.candidate import build_application_profile
from ..persistence.db import parse_dt
from ..persistence.repositories import (
    ApplicationPlanRepo,
    ApplicationRepo,
    CandidateAnswerRepo,
    CandidateFactRepo,
    JobRepo,
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
    unanswered_fields.extend(build_unanswered_answer_fields(covered_answer_keys))
    sensitive_fields = build_sensitive_fields(known_answers, profile)

    resume_payload: dict[str, Any] | None = None
    try:
        resume_payload = resume_output_service.build_resume_output(conn, config, job_id)
    except resume_output_service.ResumeOutputError as exc:
        blockers.append(f"resume_unavailable: {exc}")

    resume_summary = _resume_summary(resume_payload, blockers)
    application_url = str(job["apply_url"] or "").strip()
    if not application_url:
        blockers.append("missing_application_url")

    review_required = bool(unanswered_fields or sensitive_fields)
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
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
