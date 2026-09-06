"""Human review and promotion for optional LLM application intelligence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..persistence.repositories import (
    ApplicationAnswerDraftRepo,
    ApplicationPlanRepo,
    ApplicationRepo,
    ResumeVariantRepo,
    ResumeWordingArtifactRepo,
)
from . import application_planner
from .audit import Audit

REVIEW_APPROVED = "approved"
REVIEW_PENDING = "pending_review"
REVIEW_REJECTED = "rejected"
REVIEWER_HUMAN_CLI = "human_cli"


class ApplicationIntelligenceReviewError(Exception):
    pass


def approve_resume_wording_artifact(
    conn,
    config,
    artifact_id: int,
    *,
    reviewer_source: str = REVIEWER_HUMAN_CLI,
    note: str | None = None,
) -> dict[str, Any]:
    repo = ResumeWordingArtifactRepo(conn)
    artifact = repo.get(artifact_id)
    if artifact is None:
        raise ApplicationIntelligenceReviewError(f"resume wording artifact {artifact_id} not found")
    _validate_promotable_resume_artifact(conn, config.root, artifact)

    application_id = ApplicationRepo(conn).ensure_for_plan(artifact.job_id)
    plan_row = ApplicationPlanRepo(conn).get_by_application(application_id)
    blockers = _plan_blockers(plan_row)
    if blockers:
        raise ApplicationIntelligenceReviewError(
            "application plan has blockers; LLM resume promotion cannot override them: " + "; ".join(blockers)
        )

    updated = repo.set_review_status(
        artifact_id,
        review_status=REVIEW_APPROVED,
        reviewer_source=reviewer_source,
        review_note=note,
        promoted_application_id=application_id,
    )
    Audit(conn).human(
        "llm_resume_wording_approved",
        "resume_wording_artifact",
        artifact_id,
        job_id=artifact.job_id,
        resume_variant_id=artifact.resume_variant_id,
        application_id=application_id,
        reviewer_source=reviewer_source,
    )
    plan = application_planner.plan_application(conn, config, artifact.job_id)
    return {
        "status": REVIEW_APPROVED,
        "artifact": updated.as_dict() if updated else None,
        "application_id": application_id,
        "application_plan": plan,
    }


def reject_resume_wording_artifact(
    conn,
    config,
    artifact_id: int,
    *,
    reviewer_source: str = REVIEWER_HUMAN_CLI,
    note: str | None = None,
) -> dict[str, Any]:
    repo = ResumeWordingArtifactRepo(conn)
    artifact = repo.get(artifact_id)
    if artifact is None:
        raise ApplicationIntelligenceReviewError(f"resume wording artifact {artifact_id} not found")
    application_id = ApplicationRepo(conn).ensure_for_plan(artifact.job_id)
    updated = repo.set_review_status(
        artifact_id,
        review_status=REVIEW_REJECTED,
        reviewer_source=reviewer_source,
        review_note=note,
        promoted_application_id=None,
    )
    Audit(conn).human(
        "llm_resume_wording_rejected",
        "resume_wording_artifact",
        artifact_id,
        job_id=artifact.job_id,
        resume_variant_id=artifact.resume_variant_id,
        application_id=application_id,
        reviewer_source=reviewer_source,
    )
    plan = application_planner.plan_application(conn, config, artifact.job_id)
    return {
        "status": REVIEW_REJECTED,
        "artifact": updated.as_dict() if updated else None,
        "application_id": application_id,
        "application_plan": plan,
    }


def approve_answer_draft(
    conn,
    config,
    draft_id: int,
    *,
    reviewer_source: str = REVIEWER_HUMAN_CLI,
    note: str | None = None,
) -> dict[str, Any]:
    repo = ApplicationAnswerDraftRepo(conn)
    draft = repo.get(draft_id)
    if draft is None:
        raise ApplicationIntelligenceReviewError(f"answer draft {draft_id} not found")
    if draft.question_classification != "safe_free_text":
        raise ApplicationIntelligenceReviewError(
            f"only safe_free_text drafts can be approved, got {draft.question_classification}"
        )
    if draft.validation_status != "valid" or draft.validation_errors:
        raise ApplicationIntelligenceReviewError("only valid grounded answer drafts can be approved")
    if not draft.answer_text.strip():
        raise ApplicationIntelligenceReviewError("empty answer drafts cannot be approved")

    updated = repo.set_review_status(
        draft_id,
        review_status=REVIEW_APPROVED,
        reviewer_source=reviewer_source,
        review_note=note,
    )
    Audit(conn).human(
        "llm_application_answer_approved",
        "application_answer_draft",
        draft_id,
        application_id=draft.application_id,
        job_id=draft.job_id,
        question_key=draft.question_key,
        reviewer_source=reviewer_source,
    )
    plan = application_planner.plan_application(conn, config, draft.job_id)
    return {
        "status": REVIEW_APPROVED,
        "draft": updated.as_dict() if updated else None,
        "application_id": draft.application_id,
        "application_plan": plan,
    }


def reject_answer_draft(
    conn,
    config,
    draft_id: int,
    *,
    reviewer_source: str = REVIEWER_HUMAN_CLI,
    note: str | None = None,
) -> dict[str, Any]:
    repo = ApplicationAnswerDraftRepo(conn)
    draft = repo.get(draft_id)
    if draft is None:
        raise ApplicationIntelligenceReviewError(f"answer draft {draft_id} not found")
    updated = repo.set_review_status(
        draft_id,
        review_status=REVIEW_REJECTED,
        reviewer_source=reviewer_source,
        review_note=note,
    )
    Audit(conn).human(
        "llm_application_answer_rejected",
        "application_answer_draft",
        draft_id,
        application_id=draft.application_id,
        job_id=draft.job_id,
        question_key=draft.question_key,
        reviewer_source=reviewer_source,
    )
    plan = application_planner.plan_application(conn, config, draft.job_id)
    return {
        "status": REVIEW_REJECTED,
        "draft": updated.as_dict() if updated else None,
        "application_id": draft.application_id,
        "application_plan": plan,
    }


def list_pending_answer_drafts(
    conn,
    *,
    application_id: int | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return [
        draft.as_dict()
        for draft in ApplicationAnswerDraftRepo(conn).list_by_review_status(
            application_id=application_id,
            review_status=REVIEW_PENDING,
            limit=limit,
        )
    ]


def _validate_promotable_resume_artifact(conn, root: Path, artifact) -> None:
    if artifact.status != "valid" or artifact.validation_errors:
        raise ApplicationIntelligenceReviewError("only valid resume wording artifacts can be promoted")
    if artifact.review_status == REVIEW_REJECTED:
        raise ApplicationIntelligenceReviewError("rejected resume wording artifacts cannot be promoted")
    if artifact.resume_variant_id is None:
        raise ApplicationIntelligenceReviewError("resume wording artifact has no rendered resume variant")
    row = ResumeVariantRepo(conn).get(artifact.resume_variant_id)
    if row is None:
        raise ApplicationIntelligenceReviewError(f"resume variant {artifact.resume_variant_id} not found")
    if row["validation_status"] != "valid":
        raise ApplicationIntelligenceReviewError(f"resume variant validation_status={row['validation_status']}")
    if row["page_validation_status"] != "valid":
        raise ApplicationIntelligenceReviewError(f"resume variant page_validation_status={row['page_validation_status']}")
    if row["page_count"] != 1:
        raise ApplicationIntelligenceReviewError(f"resume variant page_count must be 1, got {row['page_count']}")
    resume_path = str(row["output_pdf_path"] or "").strip()
    if not resume_path:
        raise ApplicationIntelligenceReviewError("resume variant has no output PDF path")
    absolute = Path(resume_path).expanduser()
    if not absolute.is_absolute():
        absolute = root / absolute
    if not absolute.exists():
        raise ApplicationIntelligenceReviewError(f"resume variant PDF is missing: {resume_path}")


def _plan_blockers(row) -> list[str]:
    if row is None or not row["plan_json"]:
        return []
    import json

    try:
        payload = json.loads(row["plan_json"])
    except json.JSONDecodeError:
        return ["stored_application_plan_unreadable"]
    return [str(item) for item in (payload.get("blockers") or []) if item]
