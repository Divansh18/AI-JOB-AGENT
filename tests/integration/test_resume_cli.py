from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from jobsearch.cli.main import app
from jobsearch.domain.master_resume import PdfExtraction, PdfLink
from jobsearch.domain.models import Job
from jobsearch.domain.normalize import normalize_title
from jobsearch.persistence.db import connect, migrate
from jobsearch.persistence.repositories import JobRepo, SourceRepo
from jobsearch.render.resume_pdf import ResumePdfDocument, ResumePdfHeader, ResumePdfSection, write_resume_pdf

RUNNER = CliRunner()
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "candidate_import_v1.json"
MASTER_RESUME_TEXT = """Asha Example
asha@example.com | +91 99999 88888 | LinkedIn | GitHub
SUMMARY
Software engineer building Python and FastAPI applications for internal teams.
EXPERIENCE
Backend Engineering Intern | Acme Labs | Jan 2026 - Jun 2026
● Built Python and FastAPI APIs used by internal teams.
● Automated release checks with Playwright.
SKILLS
Languages: Python, TypeScript, SQL
Frontend: Next.js, React.js
Backend: FastAPI, NestJS
Testing: Playwright
PROJECTS
API Copilot — Internal Developer Tool — Python, FastAPI, PostgreSQL | GitHub
● Built a developer tool for API review.
CERTIFICATIONS
PCEP - Certified Entry-Level Python Programmer (Python Institute)
EDUCATION
Example University | Bachelor of Computer Applications (BCA) — Graduated: June 2026
"""


def _write_config(root: Path) -> None:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "settings.yaml").write_text("db_path: data/test.db\ndigest_dir: data/digests\n", encoding="utf-8")
    (cfg / "profile.yaml").write_text("experience_stage: early_career\n", encoding="utf-8")
    (cfg / "filters.yaml").write_text("version: 2\n", encoding="utf-8")
    (cfg / "ranking.yaml").write_text("version: 2\n", encoding="utf-8")
    (cfg / "companies.yaml").write_text("version: 1\ncompanies: []\n", encoding="utf-8")


def _seed_job(
    root: Path,
    *,
    fingerprint: str = "fp-phase-1b-cli",
    title: str = "Backend Engineer",
    location_raw: str = "Remote India",
    remote_type: str = "remote",
    remote_scope: str = "india",
    description_text: str | None = None,
) -> int:
    conn = connect(root / "data" / "test.db")
    migrate(conn)
    source_id = SourceRepo(conn).ensure("manual", "manual")
    now = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)
    job = Job(
        fingerprint=fingerprint,
        content_hash=f"hash-{fingerprint}",
        source="manual",
        source_id=source_id,
        title=title,
        title_normalized=normalize_title(title),
        company_name_raw="Acme Labs",
        company_normalized="acme",
        location_raw=location_raw,
        locations=["Bengaluru"],
        country="IN",
        remote_type=remote_type,
        remote_scope=remote_scope,
        employment_type="full_time",
        description_text=description_text
        or (
            "Required: Python and FastAPI. Nice to have Redis. "
            "You will build backend APIs and improve platform performance. "
            "Bachelor's degree in Computer Science preferred."
        ),
        apply_url="https://example.com/jobs/phase-1b",
        canonical_url="https://example.com/jobs/phase-1b",
        posted_at=now,
        first_seen_at=now,
    )
    job_id, _ = JobRepo(conn).upsert(job)
    conn.close()
    return job_id


def _write_master_resume(root: Path) -> Path:
    path = root / "assets" / "resume" / "sample.pdf"
    write_resume_pdf(
        ResumePdfDocument(
            header=ResumePdfHeader(full_name="Asha Example", contact_items=["asha@example.com", "+91 99999 88888"]),
            sections=[ResumePdfSection(title="Summary", lines=["Sample master resume."])],
        ),
        path,
    )
    return path


def _sample_extraction(_: Path) -> PdfExtraction:
    return PdfExtraction(
        text=MASTER_RESUME_TEXT,
        page_count=1,
        links=[
            PdfLink(label="LinkedIn", url="https://linkedin.com/in/asha-example", page=1, top=742.0),
            PdfLink(label="GitHub", url="https://github.com/asha-example", page=1, top=742.0),
        ],
    )


def _register_and_ingest_master(root: Path, env: dict[str, str], monkeypatch) -> None:
    _write_master_resume(root)
    monkeypatch.setattr("jobsearch.services.master_resume.extract_pdf_resume", _sample_extraction)
    registered = RUNNER.invoke(app, ["resume", "master", "register", "assets/resume/sample.pdf", "--json"], env=env)
    assert registered.exit_code == 0, registered.stdout
    ingested = RUNNER.invoke(app, ["resume", "master", "ingest", "--json"], env=env)
    assert ingested.exit_code == 0, ingested.stdout


def test_resume_cli_analyze_fit_tailor_show_and_list(tmp_path):
    _write_config(tmp_path)
    job_id = _seed_job(tmp_path)
    env = {"JOBSEARCH_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}

    imported = RUNNER.invoke(app, ["candidate", "import-json", str(FIXTURE)], env=env)
    assert imported.exit_code == 0, imported.stdout

    analyzed = RUNNER.invoke(app, ["jobs", "analyze", str(job_id), "--json"], env=env)
    assert analyzed.exit_code == 0, analyzed.stdout
    analysis = json.loads(analyzed.stdout)["analysis"]
    assert analysis["role_title"] == "Backend Engineer"
    assert analysis["source"] == "deterministic"
    assert analysis["provider"] is None
    assert analysis["model"] is None
    assert "python" in analysis["required_skills"]

    fit = RUNNER.invoke(app, ["jobs", "fit", str(job_id), "--json"], env=env)
    assert fit.exit_code == 0, fit.stdout
    fit_payload = json.loads(fit.stdout)
    assert fit_payload["analysis"]["source"] == "deterministic"
    assert fit_payload["fit_report"]["overall_fit_score"] >= 50

    tailored = RUNNER.invoke(app, ["resume", "tailor", str(job_id), "--json"], env=env)
    assert tailored.exit_code == 0, tailored.stdout
    tailored_payload = json.loads(tailored.stdout)
    assert tailored_payload["validation"]["status"] == "valid"
    assert tailored_payload["tailored_resume"]["recommendation"]["decision"] == "tailor"
    assert tailored_payload["tailored_resume"]["page_limit"] == 1
    assert tailored_payload["tailored_resume"]["page_validation_status"] == "not_rendered"
    assert tailored_payload["tailored_resume"]["preview"] is not None
    resume_id = tailored_payload["resume_id"]

    listed = RUNNER.invoke(app, ["resume", "list", "--json"], env=env)
    assert listed.exit_code == 0, listed.stdout
    variants = json.loads(listed.stdout)["resume_variants"]
    assert len(variants) == 1
    assert variants[0]["id"] == resume_id
    assert variants[0]["tailoring_decision"] == "tailor"
    assert variants[0]["master_resume_identity"] == "canonical_master_resume"
    assert variants[0]["template_identity"] == "canonical_master_resume_template"
    assert variants[0]["page_limit"] == 1

    shown = RUNNER.invoke(app, ["resume", "show", str(resume_id), "--json"], env=env)
    assert shown.exit_code == 0, shown.stdout
    shown_payload = json.loads(shown.stdout)
    assert shown_payload["id"] == resume_id
    assert shown_payload["tailoring_decision"] == "tailor"
    assert shown_payload["content_json"]["analysis"]["job_id"] == job_id
    assert shown_payload["content_json"]["resume"]["page_limit"] == 1
    assert shown_payload["content_json"]["resume"]["master_resume"]["identity"] == "canonical_master_resume"
    assert shown_payload["content_json"]["source_model"]["experience"]


def test_resume_build_tailor_creates_one_page_pdf_and_reuses_existing_variant(tmp_path, monkeypatch):
    _write_config(tmp_path)
    job_id = _seed_job(tmp_path)
    env = {"JOBSEARCH_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}

    imported = RUNNER.invoke(app, ["candidate", "import-json", str(FIXTURE)], env=env)
    assert imported.exit_code == 0, imported.stdout
    _register_and_ingest_master(tmp_path, env, monkeypatch)

    built = RUNNER.invoke(app, ["resume", "build", str(job_id), "--json"], env=env)
    assert built.exit_code == 0, built.stdout
    payload = json.loads(built.stdout)
    assert payload["tailored_resume"]["recommendation"]["decision"] == "tailor"
    assert payload["tailored_resume"]["page_validation_status"] == "valid"
    assert payload["tailored_resume"]["page_count"] == 1
    assert payload["tailored_resume"]["output_pdf_path"].startswith("output/pdf/resume-job-")
    assert (tmp_path / payload["tailored_resume"]["output_pdf_path"]).exists()
    assert payload["tailored_resume"]["evidence_refs_used"]
    assert payload["reused_existing"] is False

    validated = RUNNER.invoke(app, ["resume", "validate", str(payload["resume_id"]), "--json"], env=env)
    assert validated.exit_code == 0, validated.stdout
    validated_payload = json.loads(validated.stdout)
    assert validated_payload["validation"]["status"] == "valid"
    assert validated_payload["tailored_resume"]["page_validation_status"] == "valid"
    assert validated_payload["tailored_resume"]["page_count"] == 1

    built_again = RUNNER.invoke(app, ["resume", "build", str(job_id), "--json"], env=env)
    assert built_again.exit_code == 0, built_again.stdout
    reused_payload = json.loads(built_again.stdout)
    assert reused_payload["resume_id"] == payload["resume_id"]
    assert reused_payload["reused_existing"] is True


def test_resume_build_use_base_points_to_master_resume_without_copying(tmp_path, monkeypatch):
    _write_config(tmp_path)
    job_id = _seed_job(
        tmp_path,
        fingerprint="fp-phase-1d-use-base",
        title="Software Engineer",
        description_text="You will build backend APIs for internal systems.",
    )
    env = {"JOBSEARCH_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}

    imported = RUNNER.invoke(app, ["candidate", "import-json", str(FIXTURE)], env=env)
    assert imported.exit_code == 0, imported.stdout
    _register_and_ingest_master(tmp_path, env, monkeypatch)

    built = RUNNER.invoke(app, ["resume", "build", str(job_id), "--json"], env=env)
    assert built.exit_code == 0, built.stdout
    payload = json.loads(built.stdout)
    assert payload["tailored_resume"]["recommendation"]["decision"] == "use_base"
    assert payload["tailored_resume"]["page_validation_status"] == "valid"
    assert payload["tailored_resume"]["page_count"] == 1
    assert payload["tailored_resume"]["output_pdf_path"] == "assets/resume/sample.pdf"
    assert payload["tailored_resume"]["preview"] is None
    assert payload["reused_existing"] is False


def test_resume_build_returns_poor_fit_without_output_pdf(tmp_path, monkeypatch):
    _write_config(tmp_path)
    job_id = _seed_job(
        tmp_path,
        fingerprint="fp-phase-1d-poor-fit",
        title="Backend Engineer",
        description_text="Requirements: Go, Kubernetes, and 5+ years of experience. You will build backend APIs.",
    )
    env = {"JOBSEARCH_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}

    imported = RUNNER.invoke(app, ["candidate", "import-json", str(FIXTURE)], env=env)
    assert imported.exit_code == 0, imported.stdout
    _register_and_ingest_master(tmp_path, env, monkeypatch)

    built = RUNNER.invoke(app, ["resume", "build", str(job_id), "--json"], env=env)
    assert built.exit_code == 0, built.stdout
    payload = json.loads(built.stdout)
    assert payload["tailored_resume"]["recommendation"]["decision"] == "poor_fit"
    assert payload["tailored_resume"]["page_validation_status"] == "not_applicable"
    assert payload["tailored_resume"]["output_pdf_path"] is None
    assert payload["tailored_resume"]["page_count"] is None


def test_resume_build_fails_when_no_master_resume_is_registered(tmp_path):
    _write_config(tmp_path)
    job_id = _seed_job(tmp_path, fingerprint="fp-phase-1d-no-master")
    env = {"JOBSEARCH_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}

    imported = RUNNER.invoke(app, ["candidate", "import-json", str(FIXTURE)], env=env)
    assert imported.exit_code == 0, imported.stdout

    built = RUNNER.invoke(app, ["resume", "build", str(job_id)], env=env)

    assert built.exit_code == 1
