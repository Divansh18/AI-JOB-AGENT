from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from jobsearch.cli.main import app
from jobsearch.config.loader import AppConfig
from jobsearch.config.schemas import CompaniesFile, Filters, LlmSettings, Profile, RankingConfig, Settings, SkillSet
from jobsearch.domain.application_intelligence import ResumeWordingRequest
from jobsearch.domain.application_planner import ApplicationPlan, ApplicationPlanSensitiveField, ApplicationPlanUnresolved
from jobsearch.domain.candidate import make_answer, make_fact
from jobsearch.domain.filters import evaluate
from jobsearch.domain.models import Job
from jobsearch.domain.normalize import normalize_title
from jobsearch.llm import application_answers as aa
from jobsearch.llm import application_intelligence as ai
from jobsearch.llm import resume_wording as rw
from jobsearch.llm.provider import LLMInvocationError, ProviderResult, Usage, prompt_hash
from jobsearch.llm.schemas import ResumeWordingArtifactExtraction
from jobsearch.services import application_intelligence_context as ctx
from jobsearch.persistence.repositories import (
    ApplicationAnswerDraftRepo,
    ApplicationAutofillRunRepo,
    ApplicationPlanRepo,
    ApplicationRepo,
    CandidateFactRepo,
    CandidateAnswerRepo,
    FilterRepo,
    JobRepo,
    LlmArtifactRepo,
    LlmCallRepo,
    MasterResumeRepo,
    ResumeVariantRepo,
    ResumeWordingArtifactRepo,
    SourceRepo,
)
from jobsearch.services.application_intelligence_context import (
    ApplicationIntelligenceContext,
    CandidateEvidenceContext,
    ResumeWordingContext,
    SelectedResumeItemContext,
    application_intelligence_context_size,
    build_application_answer_context,
    build_application_intelligence_context,
    build_resume_wording_context,
    resume_wording_context_size,
    resume_wording_provider_context,
)
from jobsearch.services import resume_intelligence as resume_service
from jobsearch.services import application_intelligence_review as review_service


RUNNER = CliRunner()


def _config(root: Path, *, llm: LlmSettings | None = None) -> AppConfig:
    return AppConfig(
        root=root,
        settings=Settings(
            db_path="test.db",
            digest_dir="digests",
            llm=llm or LlmSettings(enabled=True),
        ),
        profile=Profile(
            experience_stage="early_career",
            target_roles=["Software Engineer", "Backend Engineer"],
            skills=SkillSet(strong=["python", "fastapi"], familiar=["postgresql"]),
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
    (cfg / "settings.yaml").write_text(
        "db_path: data/test.db\n"
        "digest_dir: data/digests\n"
        "llm:\n"
        "  enabled: false\n",
        encoding="utf-8",
    )
    (cfg / "profile.yaml").write_text("experience_stage: early_career\n", encoding="utf-8")
    (cfg / "filters.yaml").write_text("version: 2\n", encoding="utf-8")
    (cfg / "ranking.yaml").write_text("version: 2\n", encoding="utf-8")
    (cfg / "companies.yaml").write_text("version: 1\ncompanies: []\n", encoding="utf-8")


def _seed_candidate(conn) -> None:
    repo = CandidateFactRepo(conn)
    for fact in (
        make_fact(
            category="personal/profile",
            key="full_name",
            value="Asha Example",
            source="manual_verified",
            verified=True,
        ),
        make_fact(
            category="personal/profile",
            key="current_location",
            value="Bengaluru, India",
            source="manual_verified",
            verified=True,
        ),
        make_fact(
            category="skills",
            key="backend_skills",
            value=["python", "fastapi", "postgresql"],
            source="manual_verified",
            verified=True,
        ),
        make_fact(
            category="work experience",
            key="acme_backend_intern",
            value={
                "company": "Acme Labs",
                "title": "Backend Engineering Intern",
                "location": "Bengaluru, India",
                "start_date": "2026-01-01",
                "end_date": "2026-06-30",
                "summary": "Built internal API endpoints.",
                "achievements": [
                    "Built Python and FastAPI APIs used by internal teams.",
                    "Improved PostgreSQL-backed backend workflows.",
                ],
            },
            source="manual_verified",
            verified=True,
        ),
        make_fact(
            category="projects",
            key="job_agent",
            value={
                "name": "AI Job Search Agent",
                "summary": "Built a deterministic Python and SQLite job-search pipeline.",
                "skills": ["python", "sqlite", "typer"],
                "achievements": ["Built a deterministic matching service using Python."],
            },
            source="manual_verified",
            verified=True,
        ),
        make_fact(
            category="education",
            key="btech_cse",
            value={
                "institution": "Example Institute of Technology",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_date": "2022-08-01",
                "end_date": "2026-05-30",
            },
            source="manual_verified",
            verified=True,
        ),
        make_fact(
            category="skills",
            key="draft_go_skill",
            value=["go"],
            source="draft",
            verified=False,
        ),
        make_fact(
            category="work_authorization",
            key="summary",
            value="Authorized to work in India.",
            source="manual_verified",
            verified=True,
        ),
    ):
        repo.save(fact)


def _seed_job(conn, config: AppConfig, *, description: str | None = None, passed: bool = True) -> int:
    source_id = SourceRepo(conn).ensure("manual", "manual")
    now = datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc)
    jd = description or (
        "Requirements: Python and FastAPI experience. "
        "You will build REST APIs for data integrations and backend workflows. "
        "Preferred: PostgreSQL and SaaS platform experience. "
        "This is a full-time role based in Remote India. "
    ) * 3
    job = Job(
        fingerprint=f"fp-intel-{abs(hash(jd))}",
        content_hash=f"hash-intel-{abs(hash(jd))}",
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
        description_text=jd,
        apply_url="https://jobs.lever.co/acme/123",
        canonical_url="https://jobs.lever.co/acme/123",
        posted_at=now,
        first_seen_at=now,
    )
    job_id, _ = JobRepo(conn).upsert(job)
    result = evaluate(JobRepo(conn).get(job_id), config.filters)
    if not passed:
        result.passed = False
        result.rules_failed.append("title_role")
    FilterRepo(conn).save(job_id, result, config.filters.version)
    conn.execute(
        """
        INSERT INTO job_scores
            (job_id, stage, score, components, rank_version, model, rationale,
             matched, missing, flags, explanation, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            job_id,
            "rank",
            81.0,
            json.dumps({"semantic": 32, "title": 20}),
            config.ranking.version,
            None,
            None,
            json.dumps({"strong": ["python", "fastapi"]}),
            json.dumps([]),
            json.dumps([]),
            json.dumps(["semantic fit 0.80", "remote within India"]),
            now.isoformat(),
        ),
    )
    return job_id


def _seed_master_resume(conn) -> None:
    repo = MasterResumeRepo(conn)
    repo.register(
        identity="master_v1",
        version=1,
        file_path="assets/resume/sample.pdf",
        active=True,
        page_limit=1,
        file_hash="master-resume-hash",
    )
    repo.record_ingestion(
        identity="master_v1",
        version=1,
        file_hash="master-resume-hash",
        extracted_text=(
            "Asha Example\n"
            "Experience\n"
            "Backend Engineering Intern | Acme Labs\n"
            "Built Python and FastAPI APIs used by internal teams.\n"
            "Improved PostgreSQL-backed backend workflows.\n"
            "Projects\n"
            "AI Job Search Agent\n"
            "Built a deterministic matching service using Python."
        ),
        sections={
            "sections_detected": ["summary", "experience", "skills", "projects", "education"],
            "full_name": "Asha Example",
            "experience": [
                {
                    "company": "Acme Labs",
                    "title": "Backend Engineering Intern",
                    "date_text": "Jan 2026 - Jun 2026",
                }
            ],
            "skills": [
                {"group": "Backend", "name": "Python"},
                {"group": "Backend", "name": "FastAPI"},
                {"group": "Data", "name": "PostgreSQL"},
            ],
            "projects": [{"name": "AI Job Search Agent", "raw_header": "AI Job Search Agent"}],
            "education": [
                {
                    "institution": "Example Institute of Technology",
                    "degree": "B.Tech",
                    "raw_text": "Example Institute of Technology - B.Tech - Computer Science",
                }
            ],
        },
    )


def _seed_application_plan(conn, job_id: int, *, safe: bool = True, sensitive: bool = True) -> int:
    application_id = ApplicationRepo(conn).ensure_for_plan(job_id)
    unanswered: list[ApplicationPlanUnresolved] = []
    sensitive_fields: list[ApplicationPlanSensitiveField] = []
    if safe:
        unanswered.append(
            ApplicationPlanUnresolved(
                key="why_interested",
                label="Why are you interested in this role?",
                kind="custom_question",
                reason="custom free-text question",
                human_review_required=True,
            )
        )
    if sensitive:
        sensitive_fields.append(
            ApplicationPlanSensitiveField(
                question_key="sponsorship_required",
                label="Will you require visa sponsorship?",
                status="unresolved_review_required",
                reason="sensitive question requires human review",
            )
        )
    plan = ApplicationPlan(
        application_id=application_id,
        job_id=job_id,
        company="Acme Labs",
        role="Backend Engineer",
        application_url="https://jobs.lever.co/acme/123",
        resume_id=None,
        resume_path="output/resumes/acme.pdf",
        resume_decision="tailor",
        resume_page_count=1,
        resume_page_validation_status="valid",
        candidate_fields=[],
        known_answers=[],
        unanswered_fields=unanswered,
        sensitive_fields=sensitive_fields,
        blockers=[],
        review_required=True,
        state="ready_for_review",
        created_at=datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc),
    )
    ApplicationPlanRepo(conn).upsert(plan)
    return application_id


def _seed_autofill_questions(conn, application_id: int) -> None:
    ApplicationAutofillRunRepo(conn).create(
        {
            "application_id": application_id,
            "ats": "lever",
            "url": "https://jobs.lever.co/acme/123",
            "status": "human_intervention_required",
            "fields_detected": [],
            "fields_filled": [],
            "unresolved_fields": [
                {
                    "key": "relevant_project",
                    "label": "Tell us about a project relevant to this role.",
                    "required": True,
                    "reason": "custom question",
                }
            ],
            "sensitive_fields": [
                {
                    "key": "salary_expectation",
                    "label": "What are your salary expectations?",
                    "required": True,
                    "reason": "sensitive question",
                }
            ],
            "resume_attached": True,
            "human_intervention_required": True,
            "errors": [],
            "submitted": False,
        }
    )


def _valid_output(ref: str = "fact:4") -> dict:
    return {
        "role_summary": "Build backend SaaS integrations and APIs.",
        "must_have_requirements": [
            {
                "text": "Python and FastAPI backend development",
                "category": "skill",
                "candidate_match": "matched",
                "evidence_refs": [ref],
                "rationale": "Verified backend internship evidence names Python and FastAPI APIs.",
            }
        ],
        "preferred_requirements": [
            {
                "text": "PostgreSQL experience",
                "category": "skill",
                "candidate_match": "partial",
                "evidence_refs": [ref],
                "rationale": "Related backend evidence is present.",
            }
        ],
        "role_priorities": ["backend APIs", "data integrations"],
        "grounded_fit_assessment": "Worth applying based on verified backend API evidence.",
        "fit_verdict": "worth_applying",
        "candidate_match_evidence_refs": [ref],
        "uncertainties": ["Production scale expectations are not explicit."],
        "gaps": [],
    }


def _valid_answer_output(ref: str = "fact:4") -> dict:
    return {
        "drafts": [
            {
                "question_key": "why_interested",
                "answer_text": (
                    "I am interested in this backend role because it focuses on Python and FastAPI API work, "
                    "which aligns with verified experience building internal API endpoints."
                ),
                "evidence_refs": [ref],
                "confidence": 0.86,
                "review_required": True,
            }
        ]
    }


def _valid_resume_wording_output(
    *,
    ref: str = "fact:4",
    item_key: str = "experience:0:bullet:0",
    suggested_text: str = "Built Python and FastAPI APIs for internal team workflows.",
    action: str = "rewrite",
) -> dict:
    return {
        "suggestions": [
            {
                "item_key": item_key,
                "action": action,
                "rewritten_text": suggested_text,
                "evidence_refs": [ref],
            }
        ]
    }


def _resume_wording_context_for_item(
    *,
    item_key: str = "experience:0:bullet:0",
    section: str = "experience",
    text: str = "Built Python APIs.",
    refs: list[str] | None = None,
) -> ResumeWordingContext:
    refs = refs or ["fact:1"]
    request = ResumeWordingRequest(
        job_id=1,
        target_role="Backend Engineer",
        sections=[section],
        allowed_evidence_refs=refs,
        prompt_version="resume_wording_v2",
        source_truth_hash="truth",
    )
    return ResumeWordingContext(
        job_id=1,
        purpose="resume_wording",
        prompt_version="resume_wording_v2",
        source_truth_hash="truth",
        request=request,
        job={"id": 1, "title": "Backend Engineer", "description_text": "Build backend APIs."},
        master_resume={"identity": "master_v1", "text": "not exposed"},
        deterministic={
            "resume_recommendation": {"decision": "tailor", "evidence_refs": refs},
            "fit_report": {"strongest_matches": [{"text": "backend APIs", "evidence_refs": refs}]},
            "analysis": {"role_title": "Backend Engineer"},
            "validation": {"status": "valid", "ok": True},
        },
        job_intelligence=None,
        selected_resume_items=[
            SelectedResumeItemContext(
                item_key=item_key,
                section=section,
                text=text,
                evidence_refs=refs,
            )
        ],
        selected_evidence=[
            CandidateEvidenceContext(
                ref=ref,
                aliases=[],
                category="work_experience",
                key="backend",
                source="manual_verified",
                text=text,
            )
            for ref in refs
        ],
    )


def _resume_wording_schema_item(
    *,
    item_key: str = "experience:0:bullet:0",
    action: str = "rewrite",
    rewritten_text: str = "Built Python APIs for backend workflows.",
    refs: list[str] | None = None,
):
    return ResumeWordingArtifactExtraction(
        suggestions=[
            {
                "item_key": item_key,
                "action": action,
                "rewritten_text": rewritten_text,
                "evidence_refs": refs or ["fact:1"],
            }
        ]
    ).suggestions[0]


def _mock_resume_renderer(monkeypatch, *, page_count: int = 1) -> None:
    def write_pdf(_document, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4\n% test\n")

    monkeypatch.setattr(rw.resume_output_service, "write_resume_pdf", write_pdf)
    monkeypatch.setattr(rw.resume_output_service, "count_pdf_pages", lambda _path: page_count)


class FakeProvider:
    name = "claude_cli"
    model = "claude-haiku-4-5"

    def __init__(self, output: dict | None = None, *, error: Exception | None = None, available: bool = True):
        self.output = output or _valid_output()
        self.error = error
        self.is_available = available
        self.calls = 0
        self.last_system = ""
        self.last_user = ""

    def available(self):
        return (self.is_available, "ok" if self.is_available else "not configured")

    def complete(self, *, system, user, schema, purpose=""):
        self.calls += 1
        self.last_system = system
        self.last_user = user
        if self.error:
            raise self.error
        return ProviderResult(
            data=schema(**self.output),
            usage=Usage(input_tokens=100, output_tokens=60, cost_usd=0.001),
            provider=self.name,
            model=self.model,
            prompt_hash=prompt_hash(system, user, self.model),
        )


def test_disabled_llm_returns_skipped_without_provider(conn, tmp_path, monkeypatch):
    def fail_provider(_settings):
        raise AssertionError("provider should not be constructed when llm is disabled")

    monkeypatch.setattr(ai, "get_provider", fail_provider)
    report = ai.analyze_job_intelligence(
        conn,
        _config(tmp_path, llm=LlmSettings(enabled=False)),
        123,
    )

    assert report.status == "skipped"
    assert report.errors == ["llm_disabled"]
    assert report.artifact_id is None


def test_deterministic_system_remains_usable_without_llm(conn, tmp_path):
    config = _config(tmp_path, llm=LlmSettings(enabled=False))
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)

    analysis = resume_service.deterministic_job_analysis(conn, config, job_id)

    assert analysis.source == "deterministic"
    assert analysis.required_skills


def test_context_uses_verified_candidate_evidence_only(conn, tmp_path):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)

    context = build_application_intelligence_context(conn, config, job_id)
    text = json.dumps(context.as_dict(), sort_keys=True)

    assert "Backend Engineering Intern" in text
    assert "draft_go_skill" not in text
    assert "Authorized to work in India" not in text
    assert "Asha Example" not in text
    assert "fact:4" in context.allowed_evidence_refs


def test_jd_intelligence_context_packs_oversized_truth_store(conn, tmp_path):
    config = _config(tmp_path)
    _seed_candidate(conn)
    repo = CandidateFactRepo(conn)
    for index in range(120):
        repo.save(
            make_fact(
                category="projects",
                key=f"irrelevant_archive_{index}",
                value={
                    "name": f"Archive Project {index}",
                    "summary": "Coordinated editorial calendars and vendor review notes. " * 20,
                },
                source="manual_verified",
                verified=True,
            )
        )
    job_id = _seed_job(conn, config)

    context = build_application_intelligence_context(
        conn,
        config,
        job_id,
        max_context_chars=9000,
    )
    text = json.dumps(context.as_dict(), sort_keys=True)

    assert application_intelligence_context_size(context) <= 9000
    assert "Built internal API endpoints" in text
    assert "irrelevant_archive_" not in text


def test_jd_intelligence_context_deduplicates_evidence_text(conn, tmp_path):
    config = _config(tmp_path)
    _seed_candidate(conn)
    duplicate = "Built Python and FastAPI APIs used by internal teams."
    repo = CandidateFactRepo(conn)
    repo.save(
        make_fact(
            category="evidence",
            key="duplicate_backend_1",
            value=duplicate,
            source="manual_verified",
            verified=True,
        )
    )
    repo.save(
        make_fact(
            category="evidence",
            key="duplicate_backend_2",
            value=duplicate,
            source="manual_verified",
            verified=True,
        )
    )
    job_id = _seed_job(conn, config)

    context = build_application_intelligence_context(conn, config, job_id)

    duplicate_items = [item for item in context.candidate_evidence if item.text == duplicate]
    assert len(duplicate_items) == 1
    assert "evidence:duplicate_backend_2" in duplicate_items[0].aliases


def test_jd_intelligence_context_allowed_refs_match_final_packed_evidence(conn, tmp_path):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)

    context = build_application_intelligence_context(conn, config, job_id, max_context_chars=7000)
    payload = context.as_dict()
    packed_refs = [item["ref"] for item in payload["candidate_evidence"]]

    assert payload["allowed_evidence_refs"] == packed_refs
    assert all("aliases" not in item for item in payload["candidate_evidence"])
    deterministic_refs = _payload_evidence_refs(payload["deterministic"])
    assert set(deterministic_refs).issubset(set(packed_refs))


def test_jd_intelligence_context_canonicalizes_personal_profile_refs(conn, tmp_path):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)

    context = build_application_intelligence_context(conn, config, job_id, max_context_chars=9000)
    payload = context.as_dict()
    current_location = [
        item for item in payload["candidate_evidence"]
        if item["category"] == "personal_profile" and item["key"] == "current_location"
    ]

    assert current_location
    assert current_location[0]["ref"].startswith("fact:")
    assert "personal_profile:current_location" not in json.dumps(payload, sort_keys=True)


def test_jd_intelligence_context_removes_stale_refs_after_dedupe_or_truncation():
    context = ApplicationIntelligenceContext(
        job_id=1,
        purpose="job_intelligence",
        prompt_version="test",
        source_truth_hash="truth",
        job={"id": 1, "description_text": "Build REST APIs."},
        deterministic={
            "fit": {
                "evidence_coverage": {
                    "evidence_ref_count": 3,
                    "evidence_refs": ["fact:1", "fact:2", "personal_profile:current_location"],
                },
                "strongest_matches": [
                    {
                        "text": "India location",
                        "evidence_refs": ["personal_profile:current_location"],
                    },
                    {"text": "Dropped fact", "evidence_refs": ["fact:2"]},
                ],
            },
            "resume": {"evidence_refs": ["fact:1", "fact:2"]},
        },
        candidate_evidence=[
            CandidateEvidenceContext(
                ref="fact:1",
                aliases=["personal_profile:current_location"],
                category="personal_profile",
                key="current_location",
                source="manual",
                text="India",
            )
        ],
    )

    payload = context.as_dict()

    assert payload["allowed_evidence_refs"] == ["fact:1"]
    assert set(_payload_evidence_refs(payload["deterministic"])) == {"fact:1"}
    assert payload["deterministic"]["fit"]["evidence_coverage"]["evidence_ref_count"] == 1
    assert "fact:2" not in json.dumps(payload, sort_keys=True)
    assert "personal_profile:current_location" not in json.dumps(payload, sort_keys=True)


def test_jd_intelligence_context_removes_ats_header_metadata_but_keeps_requirements(conn, tmp_path):
    config = _config(tmp_path)
    _seed_candidate(conn)
    jd = (
        "SDE I Bangalore, India Engineering – Backend / Full time / On-site "
        "Apply for this job "
        "What you’ll own as SDE I at Hevo: "
        "Assist in building simple REST API components, focusing on clean implementation and testing. "
        "What you’ll bring to the table: "
        "Foundational knowledge of REST APIs and JSON. "
        "B Tech in Computer Science or equivalent from a reputed college. "
        "Jobs powered by {\"@context\":\"http://schema.org\",\"@type\":\"JobPosting\"}"
    )
    job_id = _seed_job(conn, config, description=jd)

    context = build_application_intelligence_context(conn, config, job_id)
    payload = context.as_dict()
    description = payload["job"]["description_text"]
    analysis = payload["deterministic"]["analysis"]
    analysis_text = json.dumps(analysis, sort_keys=True)

    assert description.startswith("What you’ll own")
    assert "apply for this job" not in json.dumps(payload, sort_keys=True).lower()
    assert "jobs powered by" not in json.dumps(payload, sort_keys=True).lower()
    assert "REST API components" in description
    assert "rest apis" in analysis_text.lower()
    assert "b tech" in analysis_text.lower()


def test_jd_intelligence_context_packs_long_jd_under_budget(conn, tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        llm=LlmSettings(enabled=True, max_input_chars=16000, max_jd_chars=6000),
    )
    _seed_candidate(conn)
    long_jd = (
        "Requirements: Python and FastAPI experience. "
        "You will build REST APIs for data integrations and backend workflows. "
        "Preferred: PostgreSQL and SaaS platform experience. "
        "This is a full-time role based in Remote India. "
    ) * 120
    job_id = _seed_job(conn, config, description=long_jd)
    fake = FakeProvider()
    monkeypatch.setattr(ai, "get_provider", lambda _settings: fake)

    report = ai.analyze_job_intelligence(conn, config, job_id)

    assert report.status == "valid"
    assert fake.calls == 1
    assert len(fake.last_system) + len(fake.last_user) <= config.settings.llm.max_input_chars
    row = LlmArtifactRepo(conn).get(report.artifact_id)
    input_json = json.loads(row["input_json"])
    assert len(input_json["job"]["description_text"]) <= config.settings.llm.max_jd_chars


def test_jd_intelligence_context_retains_relevant_evidence_and_excludes_unverified(conn, tmp_path):
    config = _config(tmp_path)
    _seed_candidate(conn)
    CandidateFactRepo(conn).save(
        make_fact(
            category="projects",
            key="unverified_fastapi_project",
            value="Built unverified FastAPI and Kubernetes services.",
            source="draft",
            verified=False,
        )
    )
    job_id = _seed_job(conn, config)

    context = build_application_intelligence_context(conn, config, job_id, max_context_chars=7000)
    text = json.dumps(context.as_dict(), sort_keys=True)

    assert "fact:4" in context.allowed_evidence_refs
    assert "Built Python and FastAPI APIs used by internal teams" in text
    assert "unverified_fastapi_project" not in text
    assert "Kubernetes services" not in text


def test_poor_fit_job_is_blocked_before_provider(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(
        conn,
        config,
        description=(
            "Requirements: Kubernetes production experience. "
            "You will own infrastructure workflows. "
            "This is a full-time role based in Remote India. "
        ) * 4,
    )
    fake = FakeProvider()
    monkeypatch.setattr(ai, "get_provider", lambda _settings: fake)

    report = ai.analyze_job_intelligence(conn, config, job_id)

    assert report.status == "blocked"
    assert any("poor_fit_resume_decision" in error for error in report.errors)
    assert fake.calls == 0


def test_failed_filter_job_is_blocked_before_provider(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config, passed=False)
    fake = FakeProvider()
    monkeypatch.setattr(ai, "get_provider", lambda _settings: fake)

    report = ai.analyze_job_intelligence(conn, config, job_id)

    assert report.status == "blocked"
    assert any("job_did_not_pass_deterministic_filters" in error for error in report.errors)
    assert fake.calls == 0


def test_valid_provider_output_is_persisted(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    fake = FakeProvider()
    monkeypatch.setattr(ai, "get_provider", lambda _settings: fake)

    report = ai.analyze_job_intelligence(conn, config, job_id)
    row = LlmArtifactRepo(conn).get(report.artifact_id)

    assert report.ok
    assert report.insight.role_summary.startswith("Build backend")
    assert fake.calls == 1
    assert row["status"] == "valid"
    assert json.loads(row["input_json"])["allowed_evidence_refs"]


def test_invalid_evidence_refs_are_rejected_and_persisted(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    fake = FakeProvider(output=_valid_output(ref="fact:999"))
    monkeypatch.setattr(ai, "get_provider", lambda _settings: fake)

    report = ai.analyze_job_intelligence(conn, config, job_id)

    assert report.status == "invalid"
    assert any(issue.code == "invalid_evidence_ref" for issue in report.validation_errors)
    assert LlmArtifactRepo(conn).get(report.artifact_id)["status"] == "invalid"


def test_cached_valid_artifact_does_not_call_provider_again(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    fake = FakeProvider()
    monkeypatch.setattr(ai, "get_provider", lambda _settings: fake)

    first = ai.analyze_job_intelligence(conn, config, job_id)
    second = ai.analyze_job_intelligence(conn, config, job_id)
    calls = conn.execute("SELECT COUNT(*) c FROM llm_calls").fetchone()["c"]

    assert first.ok
    assert second.ok
    assert second.from_cache is True
    assert fake.calls == 1
    assert calls == 1


def test_budget_exhaustion_blocks_before_provider(conn, tmp_path, monkeypatch):
    config = _config(tmp_path, llm=LlmSettings(enabled=True, per_call_cap_inr=0.0001))
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)

    def fail_provider(_settings):
        raise AssertionError("provider should not be constructed when budget is exhausted")

    monkeypatch.setattr(ai, "get_provider", fail_provider)

    report = ai.analyze_job_intelligence(conn, config, job_id)

    assert report.status == "blocked"
    assert any("per-call cap" in error for error in report.errors)


def test_provider_failure_is_persisted_as_failed(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    fake = FakeProvider(error=LLMInvocationError("boom"))
    monkeypatch.setattr(ai, "get_provider", lambda _settings: fake)

    report = ai.analyze_job_intelligence(conn, config, job_id)

    assert report.status == "failed"
    assert report.artifact_id is not None
    assert any("LLMInvocationError" in error for error in report.errors)


def test_intelligence_cli_analyze_disabled_json(tmp_path):
    _write_config(tmp_path)

    result = RUNNER.invoke(
        app,
        ["intelligence", "analyze", "123", "--json"],
        env={"JOBSEARCH_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "skipped"
    assert payload["errors"] == ["llm_disabled"]


def test_answer_context_uses_verified_candidate_evidence_only(conn, tmp_path):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id)

    context = build_application_answer_context(conn, config, application_id)
    text = json.dumps(context.as_dict(), sort_keys=True)

    assert "Backend Engineering Intern" in text
    assert "draft_go_skill" not in text
    assert "Authorized to work in India" not in text
    assert "Asha Example" not in text
    assert [question.key for question in context.safe_questions] == ["why_interested"]
    assert any(question.classification == "sensitive" for question in context.blocked_questions)


def test_safe_question_gets_grounded_answer_draft(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, sensitive=False)
    fake = FakeProvider(output=_valid_answer_output())
    monkeypatch.setattr(aa, "get_provider", lambda _settings: fake)

    result = aa.draft_application_answers(conn, config, application_id)

    assert result.status == "valid"
    assert len(result.drafts) == 1
    assert result.drafts[0].question_key == "why_interested"
    assert result.drafts[0].evidence_refs == ["fact:4"]
    assert result.drafts[0].review_required is True
    assert fake.calls == 1


def test_sensitive_question_is_blocked_before_provider(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, safe=False, sensitive=True)

    def fail_provider(_settings):
        raise AssertionError("provider should not be constructed when only sensitive questions exist")

    monkeypatch.setattr(aa, "get_provider", fail_provider)

    result = aa.draft_application_answers(conn, config, application_id)
    drafts = ApplicationAnswerDraftRepo(conn).list_by_application(application_id)

    assert result.status == "blocked"
    assert result.errors == ["no_llm_draftable_questions"]
    assert len(drafts) == 1
    assert drafts[0].validation_status == "blocked"
    assert drafts[0].question_classification == "sensitive"


def test_mixed_safe_and_sensitive_questions_only_calls_provider_for_safe(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, safe=True, sensitive=True)
    fake = FakeProvider(output=_valid_answer_output())
    monkeypatch.setattr(aa, "get_provider", lambda _settings: fake)

    result = aa.draft_application_answers(conn, config, application_id)
    by_key = {draft.question_key: draft for draft in result.drafts}

    assert result.status == "valid"
    assert result.ok is True
    assert by_key["why_interested"].validation_status == "valid"
    assert by_key["sponsorship_required"].validation_status == "blocked"
    assert fake.calls == 1


def test_autofill_unresolved_safe_question_is_drafted(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, safe=False, sensitive=False)
    _seed_autofill_questions(conn, application_id)
    output = {
        "drafts": [
            {
                "question_key": "relevant_project",
                "answer_text": (
                    "A relevant project is the AI Job Search Agent, where I built a deterministic Python "
                    "and SQLite matching pipeline for job-search workflows."
                ),
                "evidence_refs": ["fact:5"],
                "confidence": 0.83,
                "review_required": True,
            }
        ]
    }
    fake = FakeProvider(output=output)
    monkeypatch.setattr(aa, "get_provider", lambda _settings: fake)

    result = aa.draft_application_answers(conn, config, application_id)
    by_key = {draft.question_key: draft for draft in result.drafts}

    assert result.status == "valid"
    assert by_key["relevant_project"].validation_status == "valid"
    assert by_key["salary_expectation"].validation_status == "blocked"


def test_answer_draft_rejects_unsupported_evidence_refs(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, sensitive=False)
    fake = FakeProvider(output=_valid_answer_output(ref="fact:999"))
    monkeypatch.setattr(aa, "get_provider", lambda _settings: fake)

    result = aa.draft_application_answers(conn, config, application_id)

    assert result.status == "invalid"
    assert any(issue.code == "invalid_evidence_ref" for issue in result.drafts[0].validation_errors)


def test_answer_draft_rejects_unsupported_number_date_and_skill_claim(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, sensitive=False)
    fake = FakeProvider(
        output={
            "drafts": [
                {
                    "question_key": "why_interested",
                    "answer_text": "I bring 5 years of Kubernetes experience to this backend role.",
                    "evidence_refs": ["fact:4"],
                    "confidence": 0.9,
                    "review_required": False,
                }
            ]
        }
    )
    monkeypatch.setattr(aa, "get_provider", lambda _settings: fake)

    result = aa.draft_application_answers(conn, config, application_id)
    codes = {issue.code for issue in result.drafts[0].validation_errors}

    assert result.status == "invalid"
    assert "unsupported_numeric_claim" in codes
    assert "unsupported_skill_claim" in codes


def test_answer_draft_cache_hit_does_not_call_provider_again(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, sensitive=False)
    fake = FakeProvider(output=_valid_answer_output())
    monkeypatch.setattr(aa, "get_provider", lambda _settings: fake)

    first = aa.draft_application_answers(conn, config, application_id)
    second = aa.draft_application_answers(conn, config, application_id)
    calls = conn.execute("SELECT COUNT(*) c FROM llm_calls WHERE purpose='application_answer_drafting'").fetchone()["c"]

    assert first.status == "valid"
    assert second.status == "valid"
    assert second.from_cache is True
    assert fake.calls == 1
    assert calls == 1


def test_answer_budget_exhaustion_blocks_before_provider(conn, tmp_path, monkeypatch):
    config = _config(tmp_path, llm=LlmSettings(enabled=True, per_call_cap_inr=0.0001))
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, sensitive=False)

    def fail_provider(_settings):
        raise AssertionError("provider should not be constructed when budget is exhausted")

    monkeypatch.setattr(aa, "get_provider", fail_provider)

    result = aa.draft_application_answers(conn, config, application_id)

    assert result.status == "blocked"
    assert any("per-call cap" in error for error in result.errors)


def test_answer_provider_failure_is_persisted_as_failed(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, sensitive=False)
    fake = FakeProvider(output=_valid_answer_output(), error=LLMInvocationError("boom"))
    monkeypatch.setattr(aa, "get_provider", lambda _settings: fake)

    result = aa.draft_application_answers(conn, config, application_id)
    drafts = ApplicationAnswerDraftRepo(conn).list_by_application(application_id)

    assert result.status == "failed"
    assert drafts[0].validation_status == "failed"
    assert any("LLMInvocationError" in error for error in result.errors)


def test_answer_drafting_disabled_does_not_modify_candidate_answers(conn, tmp_path, monkeypatch):
    config = _config(tmp_path, llm=LlmSettings(enabled=False))
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, sensitive=False)
    before = len(CandidateAnswerRepo(conn).list())

    def fail_provider(_settings):
        raise AssertionError("provider should not be constructed when disabled")

    monkeypatch.setattr(aa, "get_provider", fail_provider)

    result = aa.draft_application_answers(conn, config, application_id)
    after = len(CandidateAnswerRepo(conn).list())

    assert result.status == "skipped"
    assert result.errors == ["llm_disabled"]
    assert before == after


def test_answer_drafts_remain_separate_from_candidate_answers(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    CandidateAnswerRepo(conn).save(
        make_answer(
            question_key="current_location",
            category="factual",
            answer_text="Bengaluru, India",
            source="manual_verified",
            verified=True,
        )
    )
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, sensitive=False)
    before = len(CandidateAnswerRepo(conn).list())
    fake = FakeProvider(output=_valid_answer_output())
    monkeypatch.setattr(aa, "get_provider", lambda _settings: fake)

    result = aa.draft_application_answers(conn, config, application_id)
    after = len(CandidateAnswerRepo(conn).list())

    assert result.status == "valid"
    assert len(ApplicationAnswerDraftRepo(conn).list_by_application(application_id)) == 1
    assert before == after


def test_intelligence_cli_draft_answers_disabled_json(tmp_path):
    _write_config(tmp_path)

    result = RUNNER.invoke(
        app,
        ["intelligence", "draft-answers", "123", "--json"],
        env={"JOBSEARCH_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "skipped"
    assert payload["errors"] == ["llm_disabled"]


def test_resume_wording_context_exposes_selected_verified_evidence_only(conn, tmp_path):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)

    context, _prepared = build_resume_wording_context(conn, config, job_id)
    payload = resume_wording_provider_context(context)
    text = json.dumps(payload, sort_keys=True)

    assert "Built Python and FastAPI APIs used by internal teams" in text
    assert "draft_go_skill" not in text
    assert "Authorized to work in India" not in text
    assert "Asha Example" not in text
    assert "master_resume" not in payload
    assert "fact:4" in context.allowed_evidence_refs
    assert payload["allowed_evidence_refs"] == [item["ref"] for item in payload["selected_evidence"]]
    assert payload["allowed_item_keys"] == [item["item_key"] for item in payload["selected_resume_items"]]


def test_resume_wording_packs_large_truth_master_and_jd_under_budget(conn, tmp_path, monkeypatch):
    config = _config(tmp_path, llm=LlmSettings(enabled=True, max_input_chars=16000, max_jd_chars=6000))
    _seed_candidate(conn)
    _seed_master_resume(conn)
    repo = CandidateFactRepo(conn)
    for index in range(80):
        repo.save(
            make_fact(
                category="projects",
                key=f"irrelevant_resume_archive_{index}",
                value={
                    "name": f"Archive Project {index}",
                    "summary": "Managed vendor workflows and calendar operations. " * 30,
                },
                source="manual_verified",
                verified=True,
            )
        )
    conn.execute(
        "UPDATE master_resumes SET extracted_text=?",
        ("UNSELECTED_MASTER_TEXT " * 1200,),
    )
    long_jd = (
        "Requirements: Python and FastAPI experience. "
        "You will build REST APIs for data integrations and backend workflows. "
        "Preferred: PostgreSQL and SaaS platform experience. "
        "This is a full-time role based in Remote India. "
    ) * 160
    job_id = _seed_job(conn, config, description=long_jd)
    _mock_resume_renderer(monkeypatch, page_count=1)
    fake = FakeProvider(output=_valid_resume_wording_output())
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    result = rw.improve_resume_wording(conn, config, job_id)
    prompt_payload = json.loads(fake.last_user.split("\n", 1)[1])
    prompt_text = json.dumps(prompt_payload, sort_keys=True)

    assert result.status == "valid"
    assert len(fake.last_system) + len(fake.last_user) <= config.settings.llm.max_input_chars
    assert resume_wording_context_size(build_resume_wording_context(
        conn,
        config,
        job_id,
        max_context_chars=rw._context_payload_budget(config.settings.llm.max_input_chars),
    )[0]) <= rw._context_payload_budget(config.settings.llm.max_input_chars)
    assert "irrelevant_resume_archive_" not in prompt_text
    assert "UNSELECTED_MASTER_TEXT" not in prompt_text
    assert "job_intelligence" not in prompt_payload
    assert "master_resume" not in prompt_payload
    assert prompt_payload["allowed_evidence_refs"] == [
        item["ref"] for item in prompt_payload["selected_evidence"]
    ]
    assert prompt_payload["allowed_item_keys"] == [
        item["item_key"] for item in prompt_payload["selected_resume_items"]
    ]


def test_resume_wording_provider_context_drops_stale_refs_and_item_keys():
    request = ResumeWordingRequest(
        job_id=1,
        target_role="Backend Engineer",
        sections=["experience"],
        allowed_evidence_refs=["fact:1", "fact:2"],
        prompt_version="resume_wording_v2",
        source_truth_hash="truth",
    )
    context = ResumeWordingContext(
        job_id=1,
        purpose="resume_wording",
        prompt_version="resume_wording_v2",
        source_truth_hash="truth",
        request=request,
        job={"id": 1, "title": "Backend Engineer", "description_text": "Build APIs."},
        master_resume={"identity": "master_v1", "text": "Do not send this"},
        deterministic={
            "resume_recommendation": {"decision": "tailor", "evidence_refs": ["fact:1", "fact:2"]},
            "fit_report": {"strongest_matches": [{"text": "API work", "evidence_refs": ["fact:2"]}]},
            "analysis": {"role_title": "Backend Engineer"},
            "validation": {"status": "valid", "ok": True},
        },
        job_intelligence={"artifact_id": 99},
        selected_resume_items=[
            SelectedResumeItemContext(
                item_key="experience:0:bullet:0",
                section="experience",
                text="Built APIs.",
                evidence_refs=["fact:1", "fact:2"],
            ),
            SelectedResumeItemContext(
                item_key="experience:9:bullet:0",
                section="experience",
                text="Unpacked item.",
                evidence_refs=["fact:2"],
            ),
        ],
        selected_evidence=[
            CandidateEvidenceContext(
                ref="fact:1",
                aliases=[],
                category="work_experience",
                key="backend",
                source="manual",
                text="Built APIs.",
            )
        ],
    )

    payload = resume_wording_provider_context(context)
    text = json.dumps(payload, sort_keys=True)

    assert payload["allowed_evidence_refs"] == ["fact:1"]
    assert payload["allowed_item_keys"] == ["experience:0:bullet:0"]
    assert payload["selected_resume_items"][0]["evidence_refs"] == ["fact:1"]
    assert "fact:2" not in text
    assert "experience:9:bullet:0" not in text
    assert "Do not send this" not in text
    assert "artifact_id" not in text


def test_resume_wording_fingerprint_includes_protocol_version(monkeypatch):
    context = _resume_wording_context_for_item()
    first = ctx.resume_wording_context_fingerprint(context, model="test-model")

    assert ctx.resume_wording_provider_context(context)["protocol_version"] == "resume_wording_protocol_v3"

    monkeypatch.setattr(ctx, "RESUME_WORDING_PROTOCOL_VERSION", "resume_wording_protocol_v4")
    second = ctx.resume_wording_context_fingerprint(context, model="test-model")

    assert second != first
    assert ctx.resume_wording_provider_context(context)["protocol_version"] == "resume_wording_protocol_v4"


def test_resume_wording_resolves_original_text_with_unicode_punctuation():
    original = (
        "Next.js → WebSocket → FastAPI → Claude CLI powered Datalyze’s workflows "
        "— shipped as an internal AI platform."
    )
    context = _resume_wording_context_for_item(text=original)
    item = _resume_wording_schema_item(
        rewritten_text=(
            "Built Datalyze’s internal AI workflows with Next.js → WebSocket → "
            "FastAPI → Claude CLI."
        )
    )

    suggestion = rw._validated_suggestion(context, item)

    assert suggestion.ok
    assert suggestion.original_text == original
    assert "Datalyze’s" in suggestion.suggested_text


def test_resume_wording_ignores_model_supplied_original_text():
    original = "Built Datalyze from the ground up — an internal AI platform."
    context = _resume_wording_context_for_item(text=original)
    item = ResumeWordingArtifactExtraction(
        suggestions=[
            {
                "item_key": "experience:0:bullet:0",
                "action": "rewrite",
                "original_text": "Model-supplied source text must not be trusted.",
                "rewritten_text": "Built Datalyze as an internal AI platform.",
                "evidence_refs": ["fact:1"],
            }
        ]
    ).suggestions[0]

    suggestion = rw._validated_suggestion(context, item)

    assert suggestion.ok
    assert suggestion.original_text == original


def test_resume_wording_rejects_unknown_item_key_before_rendering():
    context = _resume_wording_context_for_item()
    item = _resume_wording_schema_item(item_key="experience:99:bullet:0")

    suggestion = rw._validated_suggestion(context, item)

    assert suggestion.validation_status == "invalid"
    assert any(issue.code == "invalid_resume_item_key" for issue in suggestion.validation_errors)


def test_resume_wording_reorder_can_keep_original_text():
    original = "Improved PostgreSQL-backed backend workflows."
    context = _resume_wording_context_for_item(text=original)
    item = _resume_wording_schema_item(action="reorder", rewritten_text=original)

    suggestion = rw._validated_suggestion(context, item)

    assert suggestion.ok
    assert suggestion.action == "reorder"


def test_resume_wording_rejects_cosmetic_summary_name_append():
    context = _resume_wording_context_for_item(
        item_key="projects:0:summary",
        section="projects",
        text="Built APIs for backend workflows.",
    )
    item = _resume_wording_schema_item(
        item_key="projects:0:summary",
        rewritten_text="AI Job Agent: Built APIs for backend workflows.",
    )

    suggestion = rw._validated_suggestion(context, item)

    assert suggestion.validation_status == "invalid"
    assert any(issue.code == "cosmetic_resume_wording_suggestion" for issue in suggestion.validation_errors)


def test_resume_wording_valid_grounded_rewrite_creates_valid_variant(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    _mock_resume_renderer(monkeypatch, page_count=1)
    fake = FakeProvider(output=_valid_resume_wording_output())
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    result = rw.improve_resume_wording(conn, config, job_id)
    variant = ResumeVariantRepo(conn).get(result.resume_variant_id)

    assert result.status == "valid"
    assert result.resume_variant_id is not None
    assert result.suggestions[0].ok
    assert fake.calls == 1
    assert variant["validation_status"] == "valid"
    assert variant["page_validation_status"] == "valid"
    assert "internal team workflows" in json.loads(variant["content_json"])["resume"]["preview"]["experience"][0]["bullets"][0]["text"]


def test_resume_wording_may_omit_already_good_items(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    _mock_resume_renderer(monkeypatch, page_count=1)
    fake = FakeProvider(output=_valid_resume_wording_output(item_key="experience:0:bullet:0"))
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    result = rw.improve_resume_wording(conn, config, job_id)
    prompt_payload = json.loads(fake.last_user.split("\n", 1)[1])

    assert result.status == "valid"
    assert len(prompt_payload["selected_resume_items"]) > len(result.suggestions)
    assert len(result.suggestions) == 1


def test_resume_wording_rejects_no_op_rewrite_as_non_actionable(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    fake = FakeProvider(
        output=_valid_resume_wording_output(
            suggested_text="Built Python and FastAPI APIs used by internal teams."
        )
    )
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    result = rw.improve_resume_wording(conn, config, job_id)
    codes = {issue.code for issue in result.suggestions[0].validation_errors}

    assert result.status == "invalid"
    assert result.resume_variant_id is None
    assert "no_op_resume_wording_suggestion" in codes
    assert any(issue.code == "no_actionable_resume_wording_suggestions" for issue in result.validation_errors)


def test_resume_wording_reorders_selected_bullets(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    _mock_resume_renderer(monkeypatch, page_count=1)
    fake = FakeProvider(
        output=_valid_resume_wording_output(
            item_key="experience:0:bullet:1",
            suggested_text="Improved PostgreSQL-backed backend workflows.",
            action="reorder",
        )
    )
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    result = rw.improve_resume_wording(conn, config, job_id)
    variant = ResumeVariantRepo(conn).get(result.resume_variant_id)
    bullets = json.loads(variant["content_json"])["resume"]["preview"]["experience"][0]["bullets"]

    assert result.status == "valid"
    assert bullets[0]["text"] == "Improved PostgreSQL-backed backend workflows."


def test_resume_wording_rejects_unsupported_skill_without_variant(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    fake = FakeProvider(
        output=_valid_resume_wording_output(suggested_text="Built Kubernetes services for internal teams.")
    )
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    result = rw.improve_resume_wording(conn, config, job_id)

    assert result.status == "invalid"
    assert result.resume_variant_id is None
    assert any(issue.code == "unsupported_skill_claim" for issue in result.suggestions[0].validation_errors)


def test_resume_wording_rejects_unsupported_metric_number(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    fake = FakeProvider(
        output=_valid_resume_wording_output(suggested_text="Improved Python API latency by 40%.")
    )
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    result = rw.improve_resume_wording(conn, config, job_id)

    assert result.status == "invalid"
    assert result.resume_variant_id is None
    assert any(issue.code == "unsupported_numeric_claim" for issue in result.suggestions[0].validation_errors)


def test_resume_wording_rejects_unsupported_yoe_date_and_invalid_ref(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    fake = FakeProvider(
        output=_valid_resume_wording_output(
            ref="fact:999",
            suggested_text="Built Python APIs with 5 years of ownership since 2024.",
        )
    )
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    result = rw.improve_resume_wording(conn, config, job_id)
    codes = {issue.code for issue in result.suggestions[0].validation_errors}

    assert result.status == "invalid"
    assert result.resume_variant_id is None
    assert {"invalid_evidence_ref", "item_evidence_ref_mismatch", "unsupported_years_claim"}.issubset(codes)


def test_resume_wording_rejects_item_key_outside_selected_context(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    fake = FakeProvider(output=_valid_resume_wording_output(item_key="experience:9:bullet:0"))
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    result = rw.improve_resume_wording(conn, config, job_id)

    assert result.status == "invalid"
    assert result.resume_variant_id is None
    assert any(issue.code == "invalid_resume_item_key" for issue in result.suggestions[0].validation_errors)


def test_resume_wording_page_validation_still_enforced(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    _mock_resume_renderer(monkeypatch, page_count=2)
    fake = FakeProvider(output=_valid_resume_wording_output())
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    result = rw.improve_resume_wording(conn, config, job_id)

    assert result.status == "invalid"
    assert result.resume_variant_id is not None
    assert any(issue.code == "page_validation_failed" for issue in result.validation_errors)


def test_resume_wording_cache_hit_does_not_call_provider_again(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    _mock_resume_renderer(monkeypatch, page_count=1)
    fake = FakeProvider(output=_valid_resume_wording_output())
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    first = rw.improve_resume_wording(conn, config, job_id)
    second = rw.improve_resume_wording(conn, config, job_id)
    calls = conn.execute("SELECT COUNT(*) c FROM llm_calls WHERE purpose='resume_wording'").fetchone()["c"]

    assert first.status == "valid"
    assert second.status == "valid"
    assert second.from_cache is True
    assert fake.calls == 1
    assert calls == 1


def test_resume_wording_protocol_version_change_misses_old_cache(conn, tmp_path, monkeypatch):
    config_v1 = _config(
        tmp_path,
        llm=LlmSettings(enabled=True, resume_wording_prompt_version="resume_wording_v1"),
    )
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config_v1)
    _mock_resume_renderer(monkeypatch, page_count=1)
    fake_v1 = FakeProvider(output=_valid_resume_wording_output())
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake_v1)

    first = rw.improve_resume_wording(conn, config_v1, job_id)

    config_v2 = _config(
        tmp_path,
        llm=LlmSettings(enabled=True, resume_wording_prompt_version="resume_wording_v2"),
    )
    fake_v2 = FakeProvider(
        output=_valid_resume_wording_output(
            suggested_text="Built Python and FastAPI APIs for backend team workflows."
        )
    )
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake_v2)

    second = rw.improve_resume_wording(conn, config_v2, job_id)
    third = rw.improve_resume_wording(conn, config_v2, job_id)

    assert first.status == "valid"
    assert first.prompt_version == "resume_wording_v1"
    assert second.status == "valid"
    assert second.prompt_version == "resume_wording_v2"
    assert second.artifact_id != first.artifact_id
    assert second.from_cache is False
    assert third.from_cache is True
    assert third.artifact_id == second.artifact_id
    assert fake_v1.calls == 1
    assert fake_v2.calls == 1


def test_resume_wording_disabled_skips_without_provider(conn, tmp_path, monkeypatch):
    config = _config(tmp_path, llm=LlmSettings(enabled=False))

    def fail_provider(_settings):
        raise AssertionError("provider should not be constructed when disabled")

    monkeypatch.setattr(rw, "get_provider", fail_provider)

    result = rw.improve_resume_wording(conn, config, 123)

    assert result.status == "skipped"
    assert result.errors == ["llm_disabled"]
    assert result.artifact_id is None


def test_resume_wording_provider_failure_is_persisted(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    fake = FakeProvider(output=_valid_resume_wording_output(), error=LLMInvocationError("boom"))
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    result = rw.improve_resume_wording(conn, config, job_id)

    assert result.status == "failed"
    assert result.artifact_id is not None
    assert any("LLMInvocationError" in error for error in result.errors)


def test_resume_wording_budget_exhaustion_blocks_before_provider(conn, tmp_path, monkeypatch):
    config = _config(tmp_path, llm=LlmSettings(enabled=True, per_call_cap_inr=0.0001))
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)

    def fail_provider(_settings):
        raise AssertionError("provider should not be constructed when budget is exhausted")

    monkeypatch.setattr(rw, "get_provider", fail_provider)

    result = rw.improve_resume_wording(conn, config, job_id)

    assert result.status == "blocked"
    assert any("per-call cap" in error for error in result.errors)


def test_resume_wording_does_not_mutate_master_resume(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    before = MasterResumeRepo(conn).get_active().extracted_text
    job_id = _seed_job(conn, config)
    _mock_resume_renderer(monkeypatch, page_count=1)
    fake = FakeProvider(output=_valid_resume_wording_output())
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    result = rw.improve_resume_wording(conn, config, job_id)
    after = MasterResumeRepo(conn).get_active().extracted_text

    assert result.status == "valid"
    assert before == after
    assert len(ResumeWordingArtifactRepo(conn).list_by_job(job_id)) == 1


def test_resume_wording_deterministic_resume_output_still_works_without_llm(conn, tmp_path, monkeypatch):
    config = _config(tmp_path, llm=LlmSettings(enabled=False))
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    _mock_resume_renderer(monkeypatch, page_count=1)

    payload = rw.resume_output_service.build_resume_output(conn, config, job_id)

    assert payload["validation"]["status"] == "valid"
    assert payload["tailored_resume"]["page_validation_status"] == "valid"


def test_intelligence_cli_resume_wording_disabled_json(tmp_path):
    _write_config(tmp_path)

    result = RUNNER.invoke(
        app,
        ["intelligence", "resume-wording", "123", "--json"],
        env={"JOBSEARCH_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "skipped"
    assert payload["errors"] == ["llm_disabled"]


def test_unapproved_resume_wording_artifact_is_not_selected_by_plan(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    _mock_resume_renderer(monkeypatch, page_count=1)
    fake = FakeProvider(output=_valid_resume_wording_output())
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    artifact = rw.improve_resume_wording(conn, config, job_id)
    plan = review_service.application_planner.plan_application(conn, config, job_id)

    assert artifact.status == "valid"
    assert artifact.review_status == "pending_review"
    assert plan["selected_resume_source"] == "deterministic"
    assert plan["resume_id"] != artifact.resume_variant_id


def test_approved_resume_wording_artifact_is_selected_by_plan(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    _mock_resume_renderer(monkeypatch, page_count=1)
    fake = FakeProvider(output=_valid_resume_wording_output())
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    artifact = rw.improve_resume_wording(conn, config, job_id)
    payload = review_service.approve_resume_wording_artifact(conn, _config(tmp_path, llm=LlmSettings(enabled=False)), artifact.artifact_id)
    plan = payload["application_plan"]

    assert payload["status"] == "approved"
    assert payload["artifact"]["review_status"] == "approved"
    assert payload["artifact"]["reviewer_source"] == "human_cli"
    assert plan["selected_resume_source"] == "llm_resume_wording"
    assert plan["resume_id"] == artifact.resume_variant_id
    assert plan["approved_llm_resume"]["artifact_id"] == artifact.artifact_id


def test_rejected_resume_wording_artifact_is_not_selected(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    _mock_resume_renderer(monkeypatch, page_count=1)
    fake = FakeProvider(output=_valid_resume_wording_output())
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    artifact = rw.improve_resume_wording(conn, config, job_id)
    payload = review_service.reject_resume_wording_artifact(conn, config, artifact.artifact_id)

    assert payload["artifact"]["review_status"] == "rejected"
    assert payload["application_plan"]["selected_resume_source"] == "deterministic"


def test_invalid_or_multi_page_resume_wording_artifact_cannot_be_promoted(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    _mock_resume_renderer(monkeypatch, page_count=2)
    fake = FakeProvider(output=_valid_resume_wording_output())
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)

    artifact = rw.improve_resume_wording(conn, config, job_id)

    assert artifact.status == "invalid"
    try:
        review_service.approve_resume_wording_artifact(conn, config, artifact.artifact_id)
    except review_service.ApplicationIntelligenceReviewError as exc:
        assert "only valid resume wording artifacts" in str(exc)
    else:
        raise AssertionError("invalid resume wording artifact should not be promotable")


def test_resume_wording_promotion_cannot_override_plan_blockers(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, safe=False, sensitive=False)
    row = ApplicationPlanRepo(conn).get_by_application(application_id)
    blocked_payload = json.loads(row["plan_json"])
    blocked_payload["blockers"] = ["poor_fit_resume_decision"]
    conn.execute(
        "UPDATE application_plans SET state='blocked', plan_json=? WHERE application_id=?",
        (json.dumps(blocked_payload), application_id),
    )
    _mock_resume_renderer(monkeypatch, page_count=1)
    fake = FakeProvider(output=_valid_resume_wording_output())
    monkeypatch.setattr(rw, "get_provider", lambda _settings: fake)
    artifact = rw.improve_resume_wording(conn, config, job_id)

    try:
        review_service.approve_resume_wording_artifact(conn, config, artifact.artifact_id)
    except review_service.ApplicationIntelligenceReviewError as exc:
        assert "cannot override" in str(exc)
    else:
        raise AssertionError("promotion should not override blocked application plans")


def test_answer_draft_approval_makes_safe_answer_available_to_plan_without_candidate_answer(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, safe=True, sensitive=False)
    before = len(CandidateAnswerRepo(conn).list())
    fake = FakeProvider(output=_valid_answer_output())
    monkeypatch.setattr(aa, "get_provider", lambda _settings: fake)

    drafts = aa.draft_application_answers(conn, config, application_id)
    payload = review_service.approve_answer_draft(conn, _config(tmp_path, llm=LlmSettings(enabled=False)), drafts.drafts[0].draft_id)
    after = len(CandidateAnswerRepo(conn).list())
    answers = {answer["question_key"]: answer for answer in payload["application_plan"]["known_answers"]}

    assert payload["status"] == "approved"
    assert answers["why_interested"]["source"].startswith("llm_answer_draft:")
    assert answers["why_interested"]["review_status"] == "approved"
    assert answers["why_interested"]["autofill_safe"] is True
    assert before == after


def test_rejected_answer_draft_is_not_available_to_plan(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    _seed_master_resume(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, safe=True, sensitive=False)
    fake = FakeProvider(output=_valid_answer_output())
    monkeypatch.setattr(aa, "get_provider", lambda _settings: fake)

    drafts = aa.draft_application_answers(conn, config, application_id)
    payload = review_service.reject_answer_draft(conn, config, drafts.drafts[0].draft_id)

    assert payload["status"] == "rejected"
    assert "why_interested" not in {answer["question_key"] for answer in payload["application_plan"]["known_answers"]}


def test_sensitive_answer_draft_cannot_be_approved(conn, tmp_path):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, safe=False, sensitive=True)
    draft = ApplicationAnswerDraftRepo(conn).save(
        aa.GroundedAnswerDraft(
            application_id=application_id,
            job_id=job_id,
            question_key="sponsorship_required",
            question_text="Will you require visa sponsorship?",
            question_classification="sensitive",
            answer_text="No",
            evidence_refs=[],
            confidence=0.9,
            review_required=True,
            validation_status="valid",
            provider="claude_cli",
            model="claude-haiku-4-5",
            prompt_version="application_answer_drafting_v1",
            context_fingerprint="fp",
            prompt_hash="ph",
            source_truth_hash="truth",
        )
    )

    try:
        review_service.approve_answer_draft(conn, config, draft.draft_id)
    except review_service.ApplicationIntelligenceReviewError as exc:
        assert "safe_free_text" in str(exc)
    else:
        raise AssertionError("sensitive answer draft should not be approvable")


def test_pending_answer_drafts_listed_for_review(conn, tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_candidate(conn)
    job_id = _seed_job(conn, config)
    application_id = _seed_application_plan(conn, job_id, safe=True, sensitive=False)
    fake = FakeProvider(output=_valid_answer_output())
    monkeypatch.setattr(aa, "get_provider", lambda _settings: fake)

    aa.draft_application_answers(conn, config, application_id)
    pending = review_service.list_pending_answer_drafts(conn, application_id=application_id)

    assert len(pending) == 1
    assert pending[0]["review_status"] == "pending_review"


def test_intelligence_review_cli_approve_reject_disabled_paths(tmp_path):
    _write_config(tmp_path)

    result = RUNNER.invoke(
        app,
        ["intelligence", "answers-pending", "--json"],
        env={"JOBSEARCH_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["drafts"] == []


def _payload_evidence_refs(payload) -> list[str]:
    refs: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "evidence_refs" and isinstance(value, list):
                refs.extend(str(item) for item in value)
            refs.extend(_payload_evidence_refs(value))
    elif isinstance(payload, list):
        for value in payload:
            refs.extend(_payload_evidence_refs(value))
    return refs
