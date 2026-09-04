from __future__ import annotations

from dataclasses import replace
from datetime import date

from jobsearch.domain.candidate import make_answer, make_fact
from jobsearch.domain.extract import extract_yoe
from jobsearch.domain.resume_intelligence import (
    ResumeContentItem,
    build_fit_report,
    build_job_analysis,
    build_resume_source_model,
    match_requirements,
    tailor_resume,
    validate_tailored_resume,
)

AS_OF = date(2026, 9, 4)


def _facts() -> list:
    return [
        make_fact(
            category="personal/profile",
            key="full_name",
            value="Asha Example",
            source="manual_verified",
            verified=True,
            fact_id=1,
        ),
        make_fact(
            category="personal/profile",
            key="current_location",
            value="Bengaluru, India",
            source="manual_verified",
            verified=True,
            fact_id=2,
        ),
        make_fact(
            category="work_authorization",
            key="summary",
            value="Authorized to work in India. No other work authorization claimed.",
            source="manual_verified",
            verified=True,
            fact_id=3,
        ),
        make_fact(
            category="skills",
            key="strong_skills",
            value=["python", "fastapi", "aws"],
            source="manual_verified",
            verified=True,
            fact_id=4,
        ),
        make_fact(
            category="work experience",
            key="acme_backend_intern",
            value={
                "company": "Acme Labs",
                "title": "Backend Engineering Intern",
                "location": "Bengaluru, India",
                "start_date": "2025-01-06",
                "end_date": "2025-06-30",
                "summary": "Built internal API endpoints.",
                "achievements": [
                    "Built Python and FastAPI APIs used by internal teams.",
                    "Automated status updates for internal backend workflows.",
                ],
            },
            source="resume_2026_09_03",
            verified=True,
            fact_id=5,
        ),
        make_fact(
            category="projects",
            key="ai_jobsearch_agent",
            value={
                "name": "AI Job Search Agent",
                "role": "Builder",
                "summary": "Built a deterministic Python and SQLite job-search pipeline.",
                "skills": ["python", "sqlite", "typer"],
                "achievements": [
                    "Built a resume matching service using Python and FastAPI.",
                ],
            },
            source="manual_verified",
            verified=True,
            fact_id=6,
        ),
        make_fact(
            category="education",
            key="btech_cse",
            value={
                "institution": "Example Institute of Technology",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_date": "2021-08-01",
                "end_date": "2025-05-30",
            },
            source="resume_2026_09_03",
            verified=True,
            fact_id=7,
        ),
        make_fact(
            category="projects",
            key="draft_go_project",
            value={
                "name": "Go Draft",
                "summary": "Built an early draft in Go.",
                "skills": ["go"],
            },
            source="draft_resume",
            verified=False,
            fact_id=8,
        ),
    ]


def _answers() -> list:
    return [
        make_answer(
            question_key="current_location",
            category="profile_field",
            answer_text="Bengaluru, India",
            source="manual_verified",
            evidence_refs=["personal_profile:current_location"],
            verified=True,
            human_review_required=False,
            answer_id=1,
        ),
        make_answer(
            question_key="work_authorization",
            category="attestation",
            answer_text="Authorized to work in India.",
            source="manual_verified",
            evidence_refs=["work_authorization:summary"],
            verified=True,
            human_review_required=True,
            answer_id=2,
        ),
    ]


def _eligible_location() -> dict:
    return {
        "status": "eligible",
        "bucket": "india_remote",
        "reason": "remote within India",
        "cities": ["Bengaluru"],
        "countries": ["IN"],
        "regions": [],
        "country": "IN",
        "remote_type": "remote",
        "remote_scope": "india",
        "multi_location": False,
        "used_description_fallback": False,
        "evidence": ["Remote India"],
    }


def _analysis(job) -> tuple:
    analysis = build_job_analysis(
        job,
        location_signal=_eligible_location(),
        yoe_signal=extract_yoe(job.description_text).as_signal(),
        work_auth_blocker=None,
    )
    facts = _facts()
    answers = _answers()
    source = build_resume_source_model(facts, answers)
    matches = match_requirements(analysis, facts, answers, as_of=AS_OF)
    return analysis, source, matches


def test_job_analysis_and_fit_report_flag_unsupported_go_and_five_years(job_factory):
    job = job_factory(
        title="Backend Engineer",
        description_text=(
            "Requirements: Go, Kubernetes, and 5+ years of experience. "
            "Nice to have Redis. You will build backend APIs and improve platform performance. "
            "Bachelor's degree in Computer Science preferred."
        ),
        location_raw="Remote India",
        remote_type="remote",
        remote_scope="india",
    )

    analysis, source, matches = _analysis(job)
    fit = build_fit_report(analysis, matches, source)
    by_id = {match.requirement_id: match for match in matches}

    assert analysis.required_skills[:2] == ["go", "kubernetes"]
    assert "redis" in analysis.preferred_skills
    assert analysis.minimum_experience_years == 5.0
    assert by_id["required_skill:go"].status == "unsupported"
    assert by_id["required_skill:kubernetes"].status == "unsupported"
    assert by_id["minimum_experience_years"].status == "unsupported"
    assert fit.overall_fit_score <= 35
    assert any("5+ years" in blocker for blocker in fit.hard_blockers)


def test_required_experience_skill_is_partial_when_only_skill_fact_exists(job_factory):
    job = job_factory(
        title="Software Engineer",
        description_text="Required: AWS production experience. You will build backend APIs.",
        location_raw="Remote India",
        remote_type="remote",
        remote_scope="india",
    )

    analysis, _, matches = _analysis(job)
    aws_match = next(match for match in matches if match.requirement_id == "required_skill:aws")

    assert analysis.required_skills == ["aws"]
    assert aws_match.status == "partial"
    assert "skill only" in aws_match.explanation or "skill only" in aws_match.explanation.replace("-", " ")


def test_resume_source_model_excludes_unverified_entries_and_tailoring_stays_valid(job_factory):
    facts = _facts()
    answers = _answers()
    source = build_resume_source_model(facts, answers)

    assert all(project.name != "Go Draft" for project in source.projects)
    assert source.summary is None

    job = job_factory(
        title="Software Engineer",
        description_text="Required: Python and FastAPI. You will build backend APIs.",
        location_raw="Bengaluru, India",
    )
    analysis = build_job_analysis(
        job,
        location_signal={
            **_eligible_location(),
            "bucket": "india_city",
            "reason": "India location stated",
            "remote_type": "onsite",
            "multi_location": False,
        },
        yoe_signal=extract_yoe(job.description_text).as_signal(),
        work_auth_blocker=None,
    )
    matches = match_requirements(analysis, facts, answers, as_of=AS_OF)
    fit = build_fit_report(analysis, matches, source)
    tailored = tailor_resume(analysis, fit, source)
    validation = validate_tailored_resume(tailored, source, as_of=AS_OF)

    assert validation.ok is True
    assert tailored.recommendation.decision == "tailor"
    assert tailored.page_limit == 1
    assert tailored.page_validation_status == "not_rendered"
    assert tailored.template.layout_policy == "preserve_existing_layout"
    assert tailored.preview is not None
    assert tailored.changes
    assert tailored.preview.skills[0].skill == "fastapi" or tailored.preview.skills[0].skill == "python"


def test_tailoring_recommendation_uses_base_when_verified_resume_already_covers_simple_role(job_factory):
    facts = _facts()
    answers = _answers()
    source = build_resume_source_model(facts, answers)
    job = job_factory(
        title="Software Engineer",
        description_text="You will build backend APIs for internal systems.",
        location_raw="Remote India",
        remote_type="remote",
        remote_scope="india",
    )
    analysis = build_job_analysis(
        job,
        location_signal=_eligible_location(),
        yoe_signal=extract_yoe(job.description_text).as_signal(),
        work_auth_blocker=None,
    )
    matches = match_requirements(analysis, facts, answers, as_of=AS_OF)
    fit = build_fit_report(analysis, matches, source)
    tailored = tailor_resume(analysis, fit, source)
    validation = validate_tailored_resume(tailored, source, as_of=AS_OF)

    assert validation.ok is True
    assert tailored.recommendation.decision == "use_base"
    assert tailored.preview is None
    assert tailored.changes == []


def test_validation_rejects_invented_metric_technology_leadership_and_title_mutation(job_factory):
    facts = _facts()
    answers = _answers()
    source = build_resume_source_model(facts, answers)
    job = job_factory(
        title="Software Engineer",
        description_text="Required: Python and FastAPI. You will build backend APIs.",
        location_raw="Remote India",
        remote_type="remote",
        remote_scope="india",
    )
    analysis = build_job_analysis(
        job,
        location_signal=_eligible_location(),
        yoe_signal=extract_yoe(job.description_text).as_signal(),
        work_auth_blocker=None,
    )
    matches = match_requirements(analysis, facts, answers, as_of=AS_OF)
    fit = build_fit_report(analysis, matches, source)
    tailored = tailor_resume(analysis, fit, source)

    assert tailored.preview is not None
    bad_experience = replace(
        tailored.preview.experience[0],
        title="Software Engineer",
        bullets=[
            ResumeContentItem(
                text="Led a team that built Python and FastAPI APIs with Kubernetes and improved quality by 30%.",
                evidence_refs=tailored.preview.experience[0].bullets[0].evidence_refs,
            )
        ],
    )
    invalid = replace(
        tailored,
        preview=replace(
            tailored.preview,
            experience=[bad_experience, *tailored.preview.experience[1:]],
        ),
    )
    validation = validate_tailored_resume(invalid, source, as_of=AS_OF)
    codes = {issue.code for issue in validation.issues}

    assert validation.ok is False
    assert "unsupported_numeric_claim" in codes
    assert "unsupported_technology_claim" in codes
    assert "unsupported_leadership_claim" in codes
    assert "mutated_experience_identity" in codes


def test_unverified_go_fact_never_matches(job_factory):
    job = job_factory(
        title="Software Engineer",
        description_text="Required: Go. You will build backend APIs.",
        location_raw="Remote India",
        remote_type="remote",
        remote_scope="india",
    )

    analysis = build_job_analysis(
        job,
        location_signal=_eligible_location(),
        yoe_signal=extract_yoe(job.description_text).as_signal(),
        work_auth_blocker=None,
    )
    matches = match_requirements(analysis, _facts(), _answers(), as_of=AS_OF)
    go_match = next(match for match in matches if match.requirement_id == "required_skill:go")

    assert go_match.status == "unsupported"


def test_tailoring_recommendation_marks_poor_fit_when_core_requirements_are_unsupported(job_factory):
    analysis, source, matches = _analysis(
        job_factory(
            title="Backend Engineer",
            description_text=(
                "Requirements: Go, Kubernetes, and 5+ years of experience. "
                "You will build backend APIs."
            ),
            location_raw="Remote India",
            remote_type="remote",
            remote_scope="india",
        )
    )
    fit = build_fit_report(analysis, matches, source)
    tailored = tailor_resume(analysis, fit, source)
    validation = validate_tailored_resume(tailored, source, as_of=AS_OF)

    assert validation.ok is True
    assert tailored.recommendation.decision == "poor_fit"
    assert tailored.preview is None
    assert tailored.changes == []
