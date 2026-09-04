from __future__ import annotations

from jobsearch.persistence.repositories import MasterResumeRepo


def test_master_resume_repo_registers_active_resume_and_records_ingestion(conn):
    repo = MasterResumeRepo(conn)

    registered = repo.register(
        identity="master_v1",
        version=1,
        file_path="assets/resume/sample.pdf",
        active=True,
        page_limit=1,
        file_hash="abc123",
    )
    ingested = repo.record_ingestion(
        identity="master_v1",
        version=1,
        file_hash="abc123",
        extracted_text="SUMMARY\nExample summary",
        sections={"sections_detected": ["summary"], "section_counts": {"summary": 1}},
    )

    active = repo.get_active()

    assert registered.identity == "master_v1"
    assert active is not None
    assert active.id == registered.id
    assert ingested.last_ingested_at is not None
    assert ingested.sections["sections_detected"] == ["summary"]
