from datetime import datetime, timezone

from jobsearch.domain.application_planner import ApplicationPlan, ApplicationPlanState
from jobsearch.persistence.repositories import ApplicationPlanRepo, ApplicationRepo, JobRepo


def _seed_resume_variant(conn, job_id: int, resume_id: int) -> None:
    conn.execute(
        """
        INSERT INTO resume_variants (
            id, job_id, created_at, fit_score, source_truth_version, source_truth_hash,
            master_resume_identity, master_resume_version, template_identity, page_limit,
            tailoring_decision, tailoring_reasons, tailoring_evidence_refs, content_json,
            validation_status, validation_errors, page_validation_status, output_pdf_path,
            page_count, output_evidence_refs, render_fingerprint
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            resume_id,
            job_id,
            datetime.now(timezone.utc).isoformat(),
            80.0,
            1,
            "truth-hash",
            "master_v1",
            "1",
            "master_v1_template",
            1,
            "tailor",
            "[]",
            "[]",
            "{}",
            "valid",
            "[]",
            "valid",
            "output/pdf/resume.pdf",
            1,
            "[]",
            f"resume-fp-{resume_id}",
        ),
    )


def test_application_plan_repo_persists_refreshable_plan(conn, job_factory):
    job_id, _ = JobRepo(conn).upsert(job_factory(fingerprint="fp-plan-repo"))
    _seed_resume_variant(conn, job_id, resume_id=7)
    application_id = ApplicationRepo(conn).ensure_for_plan(job_id)
    now = datetime.now(timezone.utc)
    plan = ApplicationPlan(
        application_id=application_id,
        job_id=job_id,
        company="Acme",
        role="Backend Engineer",
        application_url="https://example.com/apply",
        resume_id=7,
        resume_path="output/pdf/resume.pdf",
        resume_decision="tailor",
        resume_page_count=1,
        resume_page_validation_status="valid",
        candidate_fields=[],
        known_answers=[],
        unanswered_fields=[],
        sensitive_fields=[],
        blockers=[],
        review_required=False,
        state=ApplicationPlanState.READY_FOR_REVIEW.value,
        created_at=now,
        updated_at=now,
    )

    repo = ApplicationPlanRepo(conn)
    repo.upsert(plan)
    row = repo.get_by_application(application_id)
    listed = repo.list()
    app = ApplicationRepo(conn).get(application_id)

    assert row is not None
    assert row["state"] == "ready_for_review"
    assert row["resume_variant_id"] == 7
    assert app["status"] == "to_apply"
    assert app["applied_at"] is None
    assert len(listed) == 1
    assert listed[0]["application_id"] == application_id

    refreshed = ApplicationPlan(
        **{**plan.__dict__, "state": ApplicationPlanState.BLOCKED.value, "blockers": ["invalid_resume"]}
    )
    repo.upsert(refreshed)

    assert repo.get_by_application(application_id)["state"] == "blocked"
    assert conn.execute("SELECT COUNT(*) c FROM application_plans WHERE state='submitted'").fetchone()["c"] == 0
