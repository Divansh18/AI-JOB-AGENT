from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from jobsearch.cli.main import app
from jobsearch.domain.candidate import make_fact
from jobsearch.domain.master_resume import PdfExtraction, PdfLink
from jobsearch.persistence.db import connect, migrate
from jobsearch.persistence.repositories import CandidateFactRepo

RUNNER = CliRunner()

SAMPLE_RESUME_TEXT = """Asha Example
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


def _write_resume_file(root: Path) -> Path:
    path = root / "assets" / "resume" / "sample.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n%deterministic test stub\n")
    return path


def _sample_extraction(_: Path) -> PdfExtraction:
    return PdfExtraction(
        text=SAMPLE_RESUME_TEXT,
        page_count=1,
        links=[
            PdfLink(label="LinkedIn", url="https://linkedin.com/in/asha-example", page=1, top=742.0),
            PdfLink(label="GitHub", url="https://github.com/asha-example", page=1, top=742.0),
        ],
    )


def test_resume_master_cli_register_show_and_ingest_are_deterministic(tmp_path, monkeypatch):
    _write_config(tmp_path)
    _write_resume_file(tmp_path)
    env = {"JOBSEARCH_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}
    monkeypatch.setattr("jobsearch.services.master_resume.extract_pdf_resume", _sample_extraction)

    registered = RUNNER.invoke(app, ["resume", "master", "register", "assets/resume/sample.pdf", "--json"], env=env)
    assert registered.exit_code == 0, registered.stdout
    registered_payload = json.loads(registered.stdout)
    assert registered_payload["master_resume"]["identity"] == "master_v1"
    assert registered_payload["master_resume"]["version"] == 1
    assert registered_payload["master_resume"]["active"] is True
    assert registered_payload["master_resume"]["page_limit"] == 1

    shown = RUNNER.invoke(app, ["resume", "master", "show", "--json"], env=env)
    assert shown.exit_code == 0, shown.stdout
    shown_payload = json.loads(shown.stdout)
    assert shown_payload["master_resume"]["file_path"] == "assets/resume/sample.pdf"

    ingested = RUNNER.invoke(app, ["resume", "master", "ingest", "--json"], env=env)
    assert ingested.exit_code == 0, ingested.stdout
    ingest_payload = json.loads(ingested.stdout)
    assert ingest_payload["detected_sections"] == [
        "summary",
        "experience",
        "skills",
        "projects",
        "certifications",
        "education",
    ]
    assert ingest_payload["imported_count"] > 0
    assert ingest_payload["deleted_count"] == 0
    assert ingest_payload["conflicts"] == []
    assert ingest_payload["validation_errors"] == []
    assert ingest_payload["candidate_validation"]["ok"] is True

    validated = RUNNER.invoke(app, ["candidate", "validate", "--json"], env=env)
    assert validated.exit_code == 0, validated.stdout
    assert json.loads(validated.stdout)["ok"] is True

    ingested_again = RUNNER.invoke(app, ["resume", "master", "ingest", "--json"], env=env)
    assert ingested_again.exit_code == 0, ingested_again.stdout
    ingest_again_payload = json.loads(ingested_again.stdout)
    assert ingest_again_payload["deleted_count"] == 0
    assert ingest_again_payload["created_count"] == 0
    assert ingest_again_payload["updated_count"] == 0
    assert ingest_again_payload["unchanged_count"] >= ingest_payload["imported_count"]


def test_resume_master_ingest_surfaces_conflicts_without_overwriting_verified_facts(tmp_path, monkeypatch):
    _write_config(tmp_path)
    _write_resume_file(tmp_path)
    env = {"JOBSEARCH_ROOT": str(tmp_path)}
    monkeypatch.setattr("jobsearch.services.master_resume.extract_pdf_resume", _sample_extraction)

    conn = connect(tmp_path / "data" / "test.db")
    migrate(conn)
    CandidateFactRepo(conn).save(
        make_fact(
            category="personal/profile",
            key="summary",
            value="Different verified summary already stored.",
            source="manual_verified",
            verified=True,
        )
    )
    conn.close()

    registered = RUNNER.invoke(app, ["resume", "master", "register", "assets/resume/sample.pdf"], env=env)
    assert registered.exit_code == 0, registered.stdout

    ingested = RUNNER.invoke(app, ["resume", "master", "ingest", "--json"], env=env)
    assert ingested.exit_code == 0, ingested.stdout
    payload = json.loads(ingested.stdout)

    assert len(payload["conflicts"]) == 1
    assert payload["conflicts"][0]["category"] == "personal_profile"
    assert payload["conflicts"][0]["key"] == "summary"

    conn = connect(tmp_path / "data" / "test.db")
    migrate(conn)
    summary = CandidateFactRepo(conn).get("personal_profile", "summary")
    conn.close()
    assert summary is not None
    assert summary.value == "Different verified summary already stored."
    assert summary.source == "manual_verified"


def test_resume_master_register_fails_when_pdf_is_missing(tmp_path):
    _write_config(tmp_path)
    env = {"JOBSEARCH_ROOT": str(tmp_path)}

    registered = RUNNER.invoke(app, ["resume", "master", "register", "assets/resume/missing.pdf"], env=env)

    assert registered.exit_code == 1
