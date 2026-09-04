import json

from jobsearch.persistence.repositories import ResumeVariantRepo


def test_resume_variant_repo_round_trips_content_and_errors(conn):
    conn.execute(
        """
        INSERT INTO jobs (
            fingerprint, content_hash, source_id, source_name, company_id, company_name_raw,
            company_normalized, external_id, title, title_normalized, location_raw, locations,
            country, remote_type, remote_scope, employment_type, description_text, description_chars,
            apply_url, canonical_url, posted_at, first_seen_at, last_seen_at, raw
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "fp-resume-1",
            "hash-resume-1",
            None,
            "manual",
            None,
            "Acme",
            "acme",
            "job-1",
            "Software Engineer",
            "software engineer",
            "Bengaluru, India",
            "[]",
            "IN",
            "onsite",
            "unknown",
            "full_time",
            "Build software.",
            len("Build software."),
            "https://example.com/jobs/1",
            "https://example.com/jobs/1",
            "2026-09-04T00:00:00+00:00",
            "2026-09-04T00:00:00+00:00",
            "2026-09-04T00:00:00+00:00",
            "{}",
        ),
    )
    job_id = conn.execute("SELECT id FROM jobs WHERE fingerprint='fp-resume-1'").fetchone()["id"]

    repo = ResumeVariantRepo(conn)
    resume_id = repo.create(
        job_id=job_id,
        fit_score=78.0,
        source_truth_version=1,
        source_truth_hash="abc123",
        master_resume_identity="canonical_master_resume",
        master_resume_version="v1",
        template_identity="canonical_master_resume_template",
        page_limit=1,
        tailoring_decision="tailor",
        tailoring_reasons=["Verified project evidence can better surface python and fastapi."],
        tailoring_evidence_refs=["fact:5", "fact:6"],
        content={"resume": {"skills": ["python"]}},
        validation_status="valid",
        validation_errors=[],
        page_validation_status="valid",
        output_pdf_path="output/pdf/resume-job-1.pdf",
        page_count=1,
        output_evidence_refs=["fact:5", "fact:6"],
        render_fingerprint="resume-fp-1",
    )

    row = repo.get(resume_id)
    listed = repo.list(limit=5)
    by_fingerprint = repo.find_by_render_fingerprint("resume-fp-1")

    assert row is not None
    assert row["source_truth_hash"] == "abc123"
    assert row["master_resume_identity"] == "canonical_master_resume"
    assert row["template_identity"] == "canonical_master_resume_template"
    assert row["page_limit"] == 1
    assert row["tailoring_decision"] == "tailor"
    assert json.loads(row["tailoring_reasons"]) == [
        "Verified project evidence can better surface python and fastapi."
    ]
    assert json.loads(row["tailoring_evidence_refs"]) == ["fact:5", "fact:6"]
    assert json.loads(row["content_json"])["resume"]["skills"] == ["python"]
    assert json.loads(row["validation_errors"]) == []
    assert row["page_validation_status"] == "valid"
    assert row["output_pdf_path"] == "output/pdf/resume-job-1.pdf"
    assert row["page_count"] == 1
    assert json.loads(row["output_evidence_refs"]) == ["fact:5", "fact:6"]
    assert row["render_fingerprint"] == "resume-fp-1"
    assert len(listed) == 1
    assert listed[0]["id"] == resume_id
    assert by_fingerprint is not None
    assert by_fingerprint["id"] == resume_id

    repo.update_rendered_output(
        resume_id,
        content={"resume": {"skills": ["python", "fastapi"]}},
        validation_status="invalid",
        validation_errors=[{"code": "overflow"}],
        page_validation_status="overflow",
        output_pdf_path=None,
        page_count=2,
        output_evidence_refs=["fact:6"],
    )
    updated = repo.get(resume_id)

    assert updated is not None
    assert updated["validation_status"] == "invalid"
    assert json.loads(updated["validation_errors"]) == [{"code": "overflow"}]
    assert updated["page_validation_status"] == "overflow"
    assert updated["output_pdf_path"] is None
    assert updated["page_count"] == 2
    assert json.loads(updated["output_evidence_refs"]) == ["fact:6"]
