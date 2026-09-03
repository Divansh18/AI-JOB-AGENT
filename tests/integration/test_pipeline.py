"""End-to-end pipeline behaviour against a real SQLite database.

No network: postings are injected directly, which is what the discover stage
would have produced.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jobsearch.config.loader import AppConfig
from jobsearch.config.schemas import (
    CompaniesFile, Filters, Profile, RankingConfig, Settings, SkillSet,
)
from jobsearch.persistence.repositories import (
    ApplicationRepo, FilterRepo, JobRepo, ScoreRepo, SourceRepo,
)
from jobsearch.services import ledger as ledger_service
from jobsearch.services import pipeline as pipeline_service
from jobsearch.services.normalization import normalize_posting
from jobsearch.domain.models import RawPosting

NOW = datetime.now(timezone.utc)
GOOD_JD = ("We are hiring a software engineer to build REST APIs with Python and "
           "FastAPI, a React frontend and PostgreSQL. Docker and Linux experience "
           "is useful. 0-2 years of experience. Based in Bengaluru, India. ") * 3
SENIOR_JD = ("Senior role. You must have at least 8 years of experience leading "
             "distributed systems teams. ") * 6


@pytest.fixture
def config(tmp_path) -> AppConfig:
    return AppConfig(
        root=tmp_path,
        settings=Settings(db_path="test.db", digest_dir="digests"),
        profile=Profile(
            experience_stage="early_career",
            target_roles=["Software Engineer", "Backend Engineer"],
            skills=SkillSet(strong=["python", "fastapi", "react.js", "postgresql",
                                    "docker", "linux"],
                            familiar=["aws", "redis"]),
            preferred_locations=["Bengaluru", "Delhi NCR"],
            summary_for_matching="Early-career full-stack engineer using React and FastAPI.",
        ),
        filters=Filters(),
        ranking=RankingConfig(),
        companies=CompaniesFile(),
    )


def _posting(**kw) -> RawPosting:
    d = dict(source="greenhouse", external_id="gh:1", title="Software Engineer",
             company_name="Acme Technologies", location_raw="Bengaluru, India",
             description_text=GOOD_JD, apply_url="https://acme.example/jobs/1",
             canonical_url="https://acme.example/jobs/1", posted_at=NOW)
    d.update(kw)
    return RawPosting(**d)


def _ingest(conn, postings):
    source_id = SourceRepo(conn).ensure("greenhouse", "ats")
    ids = []
    for p in postings:
        job = normalize_posting(p)
        job.source_id = source_id
        job_id, _ = JobRepo(conn).upsert(job)
        ids.append(job_id)
    return ids


def test_full_pipeline_filters_ranks_and_orders(conn, config):
    # Distinct URLs: identical canonical URLs would (correctly) be deduped,
    # which is not what this test is exercising.
    _ingest(conn, [
        _posting(external_id="gh:1", title="Software Engineer",
                 apply_url="https://acme.example/jobs/1",
                 canonical_url="https://acme.example/jobs/1"),
        _posting(external_id="gh:2", title="Senior Staff Engineer",
                 description_text=SENIOR_JD,
                 apply_url="https://acme.example/jobs/2",
                 canonical_url="https://acme.example/jobs/2"),
        _posting(external_id="gh:3", title="QA Automation Engineer",
                 apply_url="https://acme.example/jobs/3",
                 canonical_url="https://acme.example/jobs/3"),
        _posting(external_id="gh:4", title="Backend Engineer",
                 apply_url="https://acme.example/jobs/4",
                 canonical_url="https://acme.example/jobs/4"),
    ])
    pipeline_service.dedupe(conn)
    f = pipeline_service.apply_filters(conn, config)
    r = pipeline_service.rank(conn, config)

    assert f.passed_filters == 2, f"expected SWE + Backend to pass, got {f.passed_filters}"
    assert f.filtered_out == 2
    assert r.ranked == 2

    top = ScoreRepo(conn).top(10)
    titles = [row["title"] for row in top]
    assert "Software Engineer" in titles and "Backend Engineer" in titles
    assert all(0 <= row["score"] <= 100 for row in top)
    assert top == sorted(top, key=lambda r: -r["score"])


def test_pipeline_is_idempotent(conn, config):
    """A1/A7: re-running must not duplicate rows or reset first_seen_at."""
    postings = [_posting(external_id=f"gh:{i}", title="Software Engineer",
                         apply_url=f"https://acme.example/jobs/{i}",
                         canonical_url=f"https://acme.example/jobs/{i}")
                for i in range(1, 6)]
    _ingest(conn, postings)
    pipeline_service.run_all(conn, config, skip_discover=True)

    first_seen = {r["id"]: r["first_seen_at"]
                  for r in conn.execute("SELECT id, first_seen_at FROM jobs")}
    count_before = JobRepo(conn).count()
    scores_before = {r["id"]: r["score"] for r in ScoreRepo(conn).top(50)}

    _ingest(conn, postings)  # same payload again
    pipeline_service.run_all(conn, config, skip_discover=True)

    assert JobRepo(conn).count() == count_before, "re-run created duplicate jobs"
    after = {r["id"]: r["first_seen_at"]
             for r in conn.execute("SELECT id, first_seen_at FROM jobs")}
    assert after == first_seen, "first_seen_at was overwritten"
    assert {r["id"]: r["score"] for r in ScoreRepo(conn).top(50)} == scores_before


def test_cross_source_duplicates_collapse_to_one_card(conn, config):
    """A4: the same role from three sources must appear once."""
    shared = dict(title="Backend Engineer", company_name="Acme Technologies",
                  location_raw="Bengaluru, India", description_text=GOOD_JD)
    _ingest(conn, [
        _posting(source="greenhouse", external_id="gh:77",
                 canonical_url="https://acme.example/jobs/77",
                 apply_url="https://acme.example/jobs/77", **shared),
        _posting(source="adzuna", external_id="az:88",
                 canonical_url="https://adzuna.example/88",
                 apply_url="https://adzuna.example/88", **shared),
        _posting(source="manual", external_id=None,
                 canonical_url="https://manual.example/99",
                 apply_url="https://manual.example/99", **shared),
    ])
    assert JobRepo(conn).count() == 3
    linked = pipeline_service.dedupe(conn)
    assert linked == 2, f"expected 2 duplicate links, got {linked}"

    pipeline_service.apply_filters(conn, config)
    pipeline_service.rank(conn, config)
    top = ScoreRepo(conn).top(10)
    assert len(top) == 1, f"duplicate cards leaked into the digest: {[r['title'] for r in top]}"


def test_rerun_recomputes_without_refetching(conn, config):
    _ingest(conn, [_posting()])
    pipeline_service.run_all(conn, config, skip_discover=True)
    before = ScoreRepo(conn).top(1)[0]["score"]

    config.ranking.weights.freshness = 0.0
    pipeline_service.run_all(conn, config, skip_discover=True, rerun=True)
    after = ScoreRepo(conn).top(1)[0]["score"]
    assert after != before, "rerun did not apply the new weights"
    assert JobRepo(conn).count() == 1


def test_rerun_reassesses_location_from_text_not_stale_stored_fields(conn, config):
    bad_desc = ("We need deep experience in distributed systems and public cloud platforms. "
                "3+ years is preferred. ") * 8
    ids = _ingest(conn, [
        _posting(
            external_id="gh:sf",
            title="Software Engineer, Platform",
            location_raw="San Francisco, CA; New York, NY",
            description_text=bad_desc,
            apply_url="https://acme.example/jobs/sf",
            canonical_url="https://acme.example/jobs/sf",
        ),
        _posting(
            external_id="gh:in",
            title="Backend Engineer",
            location_raw="Bengaluru, India",
            apply_url="https://acme.example/jobs/in",
            canonical_url="https://acme.example/jobs/in",
        ),
    ])
    foreign_id, india_id = ids

    conn.execute(
        "UPDATE jobs SET country='US', remote_type='remote', remote_scope='unknown' WHERE id=?",
        (foreign_id,),
    )

    pipeline_service.run_all(conn, config, skip_discover=True, rerun=True)

    filtered = FilterRepo(conn).get(foreign_id)
    assert "country" in json.loads(filtered["rules_failed"])

    top = ScoreRepo(conn).top(10)
    assert [row["id"] for row in top] == [india_id]


def test_pipeline_excludes_clear_business_support_titles_but_keeps_unusual_engineering(conn, config):
    _ingest(conn, [
        _posting(
            external_id="gh:ops",
            title="Strategy and Operations Associate",
            apply_url="https://acme.example/jobs/ops",
            canonical_url="https://acme.example/jobs/ops",
        ),
        _posting(
            external_id="gh:support",
            title="Intermediate Support Engineer",
            apply_url="https://acme.example/jobs/support",
            canonical_url="https://acme.example/jobs/support",
        ),
        _posting(
            external_id="gh:money",
            title="Associate - Monetisation",
            apply_url="https://acme.example/jobs/money",
            canonical_url="https://acme.example/jobs/money",
        ),
        _posting(
            external_id="gh:ap",
            title="Accounts Payable, Spend Management Coordinator",
            location_raw="Remote, India",
            apply_url="https://acme.example/jobs/ap",
            canonical_url="https://acme.example/jobs/ap",
        ),
        _posting(
            external_id="gh:sdk",
            title="SDK Engineer - JavaScript",
            apply_url="https://acme.example/jobs/sdk",
            canonical_url="https://acme.example/jobs/sdk",
        ),
        _posting(
            external_id="gh:infra",
            title="Associate Infrastructure Engineer",
            apply_url="https://acme.example/jobs/infra",
            canonical_url="https://acme.example/jobs/infra",
        ),
    ])

    pipeline_service.run_all(conn, config, skip_discover=True)

    failed_titles = {
        row["title"]: json.loads(FilterRepo(conn).get(row["id"])["rules_failed"])
        for row in conn.execute("SELECT id, title FROM jobs")
    }
    for title in (
        "Strategy and Operations Associate",
        "Intermediate Support Engineer",
        "Associate - Monetisation",
        "Accounts Payable, Spend Management Coordinator",
    ):
        assert "title_role" in failed_titles[title], title

    top_titles = [row["title"] for row in ScoreRepo(conn).top(10)]
    assert "SDK Engineer - JavaScript" in top_titles
    assert "Associate Infrastructure Engineer" in top_titles
    assert "Strategy and Operations Associate" not in top_titles


def test_filter_explanations_are_recorded(conn, config):
    """A5: every excluded job must be explainable."""
    _ingest(conn, [
        _posting(external_id="gh:s", title="Senior Software Engineer",
                 description_text=SENIOR_JD),
        _posting(external_id="gh:q", title="QA Engineer"),
    ])
    pipeline_service.apply_filters(conn, config)
    repo = FilterRepo(conn)
    for row in conn.execute("SELECT job_id FROM filter_results WHERE passed=0"):
        rules = json.loads(repo.get(row["job_id"])["rules_failed"])
        assert rules, f"job {row['job_id']} excluded with no rule recorded"


def test_ledger_prevents_duplicate_applications(conn, config):
    job_ids = _ingest(conn, [_posting()])
    app_id = ledger_service.mark_applied(conn, job_ids[0], channel="greenhouse")
    assert app_id
    with pytest.raises(ledger_service.LedgerError, match="already recorded"):
        ledger_service.mark_applied(conn, job_ids[0])


def test_ledger_blocks_applying_to_a_known_duplicate(conn, config):
    shared = dict(title="Backend Engineer", company_name="Acme Technologies",
                  location_raw="Bengaluru, India", description_text=GOOD_JD)
    ids = _ingest(conn, [
        _posting(source="greenhouse", external_id="gh:5",
                 canonical_url="https://a.example/5", apply_url="https://a.example/5", **shared),
        _posting(source="adzuna", external_id="az:5",
                 canonical_url="https://b.example/5", apply_url="https://b.example/5", **shared),
    ])
    pipeline_service.dedupe(conn)
    ledger_service.mark_applied(conn, ids[0])
    with pytest.raises(ledger_service.LedgerError, match="duplicate"):
        ledger_service.mark_applied(conn, ids[1])


def test_application_state_machine_is_enforced(conn, config):
    job_ids = _ingest(conn, [_posting()])
    app_id = ledger_service.mark_applied(conn, job_ids[0])
    ledger_service.update_status(conn, app_id, "recruiter_reply")
    ledger_service.update_status(conn, app_id, "screen_scheduled")
    with pytest.raises(ledger_service.LedgerError, match="cannot move"):
        ledger_service.update_status(conn, app_id, "applied")  # backwards
    ledger_service.update_status(conn, app_id, "interviewing")
    ledger_service.update_status(conn, app_id, "offer")
    with pytest.raises(ledger_service.LedgerError):
        ledger_service.update_status(conn, app_id, "interviewing")

    events = conn.execute(
        "SELECT to_status FROM application_events WHERE application_id=? ORDER BY id",
        (app_id,)).fetchall()
    assert [e["to_status"] for e in events] == [
        "applied", "recruiter_reply", "screen_scheduled", "interviewing", "offer"]


def test_audit_trail_records_decisions(conn, config):
    job_ids = _ingest(conn, [_posting()])
    pipeline_service.run_all(conn, config, skip_discover=True)
    ledger_service.mark_applied(conn, job_ids[0])
    actions = [r["action"] for r in conn.execute("SELECT action FROM events")]
    assert "filter_complete" in actions
    assert "rank_complete" in actions
    assert "application_recorded" in actions


def test_applied_jobs_leave_the_digest(conn, config):
    job_ids = _ingest(conn, [_posting(external_id="gh:1"),
                             _posting(external_id="gh:2", title="Backend Engineer",
                                      apply_url="https://a.example/2",
                                      canonical_url="https://a.example/2")])
    pipeline_service.run_all(conn, config, skip_discover=True)
    assert len(ScoreRepo(conn).top(10)) == 2
    ledger_service.mark_applied(conn, job_ids[0])
    assert len(ScoreRepo(conn).top(10)) == 1


def test_dismissed_jobs_leave_the_digest(conn, config):
    job_ids = _ingest(conn, [_posting()])
    pipeline_service.run_all(conn, config, skip_discover=True)
    ledger_service.dismiss(conn, job_ids[0], reason="not interested")
    assert ScoreRepo(conn).top(10) == []
