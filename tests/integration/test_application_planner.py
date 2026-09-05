from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jobsearch.cli.main import app
from jobsearch.config.loader import AppConfig
from jobsearch.config.schemas import CompaniesFile, Filters, Profile, RankingConfig, Settings, SkillSet
from jobsearch.domain.candidate import make_answer, make_fact
from jobsearch.domain.models import Job
from jobsearch.domain.normalize import normalize_title
from jobsearch.persistence.db import connect, migrate
from jobsearch.persistence.repositories import (
    ApplicationRepo,
    CandidateAnswerRepo,
    CandidateFactRepo,
    JobRepo,
    ScoreRepo,
    SourceRepo,
)
from jobsearch.services import application_planner
from jobsearch.services import ledger as ledger_service
from jobsearch.services import resume_output as resume_output_service


RUNNER = CliRunner()


def _config(root: Path) -> AppConfig:
    return AppConfig(
        root=root,
        settings=Settings(db_path="test.db", digest_dir="digests"),
        profile=Profile(
            experience_stage="early_career",
            target_roles=["Software Engineer", "Backend Engineer"],
            skills=SkillSet(strong=["python", "fastapi"], familiar=["react.js"]),
            preferred_locations=["Bengaluru"],
            summary_for_matching="Early-career backend engineer.",
        ),
        filters=Filters(),
        ranking=RankingConfig(),
        companies=CompaniesFile(),
    )


def _write_config(root: Path) -> None:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "settings.yaml").write_text("db_path: data/test.db\ndigest_dir: data/digests\n", encoding="utf-8")
    (cfg / "profile.yaml").write_text("experience_stage: early_career\n", encoding="utf-8")
    (cfg / "filters.yaml").write_text("version: 2\n", encoding="utf-8")
    (cfg / "ranking.yaml").write_text("version: 2\n", encoding="utf-8")
    (cfg / "companies.yaml").write_text("version: 1\ncompanies: []\n", encoding="utf-8")


def _seed_job(conn, *, fingerprint: str = "fp-plan", apply_url: str = "https://example.com/apply") -> int:
    source_id = SourceRepo(conn).ensure("manual", "manual")
    now = datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc)
    job = Job(
        fingerprint=fingerprint,
        content_hash=f"hash-{fingerprint}",
        source="manual",
        source_id=source_id,
        title="Backend Engineer",
        title_normalized=normalize_title("Backend Engineer"),
        company_name_raw="Acme Labs",
        company_normalized="acme_labs",
        location_raw="Remote India",
        locations=["India"],
        country="IN",
        remote_type="remote",
        remote_scope="india",
        employment_type="full_time",
        description_text="Build Python and FastAPI services.",
        apply_url=apply_url,
        canonical_url=apply_url,
        posted_at=now,
        first_seen_at=now,
    )
    job_id, _ = JobRepo(conn).upsert(job)
    return job_id


def _seed_verified_candidate(conn) -> None:
    fact_repo = CandidateFactRepo(conn)
    for category, key, value in (
        ("personal/profile", "full_name", "Asha Example"),
        ("personal/profile", "email", "asha@example.com"),
        ("personal/profile", "phone", "+91 99999 88888"),
        ("personal/profile", "current_location", "Bengaluru, India"),
        ("links", "linkedin_url", "https://linkedin.com/in/asha-example"),
        ("links", "github_url", "https://github.com/asha-example"),
        ("availability", "notice_period_days", 30),
        ("availability", "earliest_start_date", "2026-10-01"),
    ):
        fact_repo.save(
            make_fact(
                category=category,
                key=key,
                value=value,
                source="manual_verified",
                verified=True,
            )
        )

    answer_repo = CandidateAnswerRepo(conn)
    answer_repo.save(
        make_answer(
            question_key="work_authorization",
            category="eligibility",
            answer_text="Authorized to work in India.",
            source="manual_verified",
            evidence_refs=["work_authorization:summary"],
            verified=True,
            human_review_required=True,
        )
    )
    answer_repo.save(
        make_answer(
            question_key="salary_expectations",
            category="compensation",
            answer_text="Requires human review for each role.",
            source="manual_verified",
            verified=True,
            human_review_required=True,
        )
    )
    answer_repo.save(
        make_answer(
            question_key="sponsorship_required",
            category="eligibility",
            answer_text="No",
            source="draft",
            verified=False,
            human_review_required=True,
        )
    )


def _seed_resume_variant(conn, job_id: int, resume_id: int = 42) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO resume_variants (
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
            82.0,
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
            "output/pdf/resume-job-1.pdf",
            1,
            "[]",
            f"resume-fp-{resume_id}",
        ),
    )


def _resume_payload(
    job_id: int,
    *,
    decision: str = "tailor",
    ok: bool = True,
    page_count: int | None = 1,
    page_status: str = "valid",
    output_pdf_path: str | None = "output/pdf/resume-job-1.pdf",
) -> dict:
    return {
        "resume_id": 42,
        "analysis": {"job_id": job_id, "source": "deterministic"},
        "fit_report": {"overall_fit_score": 82},
        "tailored_resume": {
            "job_id": job_id,
            "recommendation": {
                "decision": decision,
                "reasons": ["Verified evidence supports the role."] if decision != "poor_fit" else ["Missing core skill."],
                "evidence_refs": ["fact:1"],
            },
            "output_pdf_path": output_pdf_path,
            "page_count": page_count,
            "page_validation_status": page_status,
        },
        "validation": {
            "status": "valid" if ok else "invalid",
            "ok": ok,
            "issues": [] if ok else [{"code": "page_overflow", "message": "Expected one page."}],
        },
    }


def _patch_resume(monkeypatch, *, payload: dict | None = None, error: str | None = None) -> None:
    def fake_build_resume_output(conn, config, job_id):
        if error:
            raise resume_output_service.ResumeOutputError(error)
        _seed_resume_variant(conn, job_id)
        return payload or _resume_payload(job_id)

    monkeypatch.setattr(application_planner.resume_output_service, "build_resume_output", fake_build_resume_output)


def test_application_plan_uses_verified_data_answers_and_resume(conn, tmp_path, monkeypatch):
    job_id = _seed_job(conn)
    _seed_verified_candidate(conn)
    _patch_resume(monkeypatch)

    plan = application_planner.plan_application(conn, _config(tmp_path), job_id)
    fields = {field["key"]: field for field in plan["candidate_fields"]}
    answers = {answer["question_key"]: answer for answer in plan["known_answers"]}
    unresolved = {field["key"] for field in plan["unanswered_fields"]}
    sensitive = {field["question_key"]: field for field in plan["sensitive_fields"]}
    app_row = ApplicationRepo(conn).get(plan["application_id"])

    assert plan["state"] == "ready_for_review"
    assert plan["resume_id"] == 42
    assert plan["resume_path"] == "output/pdf/resume-job-1.pdf"
    assert plan["resume_page_count"] == 1
    assert fields["first_name"]["value"] == "Asha"
    assert fields["email"]["value"] == "asha@example.com"
    assert "sponsorship_required" in unresolved
    assert "sponsorship_required" not in answers
    assert answers["work_authorization"]["autofill_safe"] is False
    assert sensitive["work_authorization"]["status"] == "answered_verified_review_required"
    assert app_row["status"] == "to_apply"
    assert app_row["applied_at"] is None
    assert conn.execute("SELECT COUNT(*) c FROM application_plans WHERE state='submitted'").fetchone()["c"] == 0


def test_planned_jobs_remain_rankable_until_manually_recorded(conn, tmp_path, monkeypatch):
    job_id = _seed_job(conn, fingerprint="fp-planned-rank")
    _seed_verified_candidate(conn)
    _patch_resume(monkeypatch)
    conn.execute(
        """INSERT INTO job_scores
           (job_id, stage, score, components, rank_version, matched, flags, explanation, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (job_id, "rank", 80.0, "{}", 1, "{}", "[]", "[]", datetime.now(timezone.utc).isoformat()),
    )

    application_planner.plan_application(conn, _config(tmp_path), job_id)

    assert [row["id"] for row in ScoreRepo(conn).top(10)] == [job_id]
    ledger_service.mark_applied(conn, job_id)
    assert ScoreRepo(conn).top(10) == []


def test_planned_duplicate_cannot_be_recorded_after_duplicate_was_applied(conn):
    first_id = _seed_job(conn, fingerprint="fp-applied-duplicate", apply_url="https://example.com/one")
    duplicate_id = _seed_job(conn, fingerprint="fp-planned-duplicate", apply_url="https://example.com/two")
    JobRepo(conn).add_duplicate(first_id, duplicate_id, "test", 1.0)

    ledger_service.mark_applied(conn, first_id)
    ApplicationRepo(conn).ensure_for_plan(duplicate_id)

    with pytest.raises(ledger_service.LedgerError, match="duplicate"):
        ledger_service.mark_applied(conn, duplicate_id)


def test_poor_fit_decision_blocks_application_plan(conn, tmp_path, monkeypatch):
    job_id = _seed_job(conn, fingerprint="fp-poor-fit")
    _seed_verified_candidate(conn)
    _patch_resume(monkeypatch, payload=_resume_payload(job_id, decision="poor_fit", output_pdf_path=None, page_count=None))

    plan = application_planner.plan_application(conn, _config(tmp_path), job_id)

    assert plan["state"] == "blocked"
    assert plan["resume_path"] is None
    assert any("poor_fit_resume_decision" in blocker for blocker in plan["blockers"])


def test_invalid_resume_artifact_blocks_application_plan(conn, tmp_path, monkeypatch):
    job_id = _seed_job(conn, fingerprint="fp-invalid-resume")
    _seed_verified_candidate(conn)
    _patch_resume(monkeypatch, payload=_resume_payload(job_id, page_count=2, page_status="overflow"))

    plan = application_planner.plan_application(conn, _config(tmp_path), job_id)

    assert plan["state"] == "blocked"
    assert "expected page_count=1" in " ".join(plan["blockers"])


def test_missing_resume_artifact_blocks_application_plan(conn, tmp_path, monkeypatch):
    job_id = _seed_job(conn, fingerprint="fp-missing-resume")
    _seed_verified_candidate(conn)
    _patch_resume(monkeypatch, error="no active master resume is registered")

    plan = application_planner.plan_application(conn, _config(tmp_path), job_id)

    assert plan["state"] == "blocked"
    assert plan["resume_id"] is None
    assert any("resume_unavailable" in blocker for blocker in plan["blockers"])


def test_apply_cli_plan_show_and_list(tmp_path, monkeypatch):
    _write_config(tmp_path)
    conn = connect(tmp_path / "data" / "test.db")
    migrate(conn)
    job_id = _seed_job(conn, fingerprint="fp-cli-plan")
    _seed_verified_candidate(conn)
    conn.close()
    _patch_resume(monkeypatch)
    env = {"JOBSEARCH_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}

    planned = RUNNER.invoke(app, ["apply", "plan", str(job_id), "--json"], env=env)
    assert planned.exit_code == 0, planned.stdout
    payload = json.loads(planned.stdout)
    application_id = payload["application_id"]

    shown = RUNNER.invoke(app, ["apply", "show", str(application_id), "--json"], env=env)
    listed = RUNNER.invoke(app, ["apply", "list", "--json"], env=env)

    assert shown.exit_code == 0, shown.stdout
    assert listed.exit_code == 0, listed.stdout
    assert json.loads(shown.stdout)["application_id"] == application_id
    rows = json.loads(listed.stdout)["applications"]
    assert rows[0]["application_id"] == application_id
    assert rows[0]["state"] == "ready_for_review"
