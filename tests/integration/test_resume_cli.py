from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from jobsearch.cli.main import app
from jobsearch.domain.models import Job
from jobsearch.domain.normalize import normalize_title
from jobsearch.persistence.db import connect, migrate
from jobsearch.persistence.repositories import JobRepo, SourceRepo

RUNNER = CliRunner()
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "candidate_import_v1.json"


def _write_config(root: Path) -> None:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "settings.yaml").write_text("db_path: data/test.db\ndigest_dir: data/digests\n", encoding="utf-8")
    (cfg / "profile.yaml").write_text("experience_stage: early_career\n", encoding="utf-8")
    (cfg / "filters.yaml").write_text("version: 2\n", encoding="utf-8")
    (cfg / "ranking.yaml").write_text("version: 2\n", encoding="utf-8")
    (cfg / "companies.yaml").write_text("version: 1\ncompanies: []\n", encoding="utf-8")


def _seed_job(root: Path) -> int:
    conn = connect(root / "data" / "test.db")
    migrate(conn)
    source_id = SourceRepo(conn).ensure("manual", "manual")
    now = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)
    job = Job(
        fingerprint="fp-phase-1b-cli",
        content_hash="hash-phase-1b-cli",
        source="manual",
        source_id=source_id,
        title="Backend Engineer",
        title_normalized=normalize_title("Backend Engineer"),
        company_name_raw="Acme Labs",
        company_normalized="acme",
        location_raw="Remote India",
        locations=["Bengaluru"],
        country="IN",
        remote_type="remote",
        remote_scope="india",
        employment_type="full_time",
        description_text=(
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
