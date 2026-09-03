"""Digest assembly and rendering."""

import time
from datetime import datetime, timedelta, timezone

import pytest

from jobsearch.render.digest import render_html, write_digest
from jobsearch.services import digest as digest_service
from jobsearch.services import ledger as ledger_service
from jobsearch.services import pipeline as pipeline_service
from jobsearch.services.normalization import normalize_posting
from jobsearch.domain.models import RawPosting
from jobsearch.persistence.repositories import SourceRepo, JobRepo

from .test_pipeline import GOOD_JD, SENIOR_JD, config, _posting, _ingest  # noqa: F401


def test_digest_builds_and_renders(conn, config):
    _ingest(conn, [
        _posting(external_id=f"gh:{i}", title=t,
                 apply_url=f"https://acme.example/jobs/{i}",
                 canonical_url=f"https://acme.example/jobs/{i}")
        for i, t in enumerate(["Software Engineer", "Backend Engineer",
                               "Full Stack Developer", "AI Engineer"], start=1)
    ])
    pipeline_service.run_all(conn, config, skip_discover=True)
    data = digest_service.build(conn, config)

    assert data.counts["passed_filters"] == 4
    assert len(data.new_today) == 4
    assert data.new_today == sorted(data.new_today, key=lambda c: -c.score)
    for card in data.new_today:
        assert card.job_id and card.title and card.apply_url
        assert 0 <= card.score <= 100
        assert abs(sum(card.components.values()) - card.score) < 0.6
        assert card.age_hours >= 0

    html = render_html(data)
    assert "<!doctype html>" in html.lower()
    assert "Software Engineer" in html
    assert "jobsearch apply" in html


def test_digest_is_self_contained_and_offline(conn, config, tmp_path):
    """A6: no external requests, so the file opens with no network."""
    _ingest(conn, [_posting()])
    pipeline_service.run_all(conn, config, skip_discover=True)
    html = render_html(digest_service.build(conn, config))

    for bad in ("<script src=", "<link rel=\"stylesheet\"", "cdn.", "googleapis",
                "http://cdn", "unpkg", "jsdelivr"):
        assert bad not in html, f"digest pulls an external resource: {bad}"
    assert "<style>" in html, "styles must be inlined"


def test_digest_renders_quickly(conn, config):
    """A6: under 5 seconds."""
    _ingest(conn, [
        _posting(external_id=f"gh:{i}", title="Software Engineer",
                 apply_url=f"https://acme.example/jobs/{i}",
                 canonical_url=f"https://acme.example/jobs/{i}")
        for i in range(60)
    ])
    pipeline_service.run_all(conn, config, skip_discover=True)
    start = time.monotonic()
    render_html(digest_service.build(conn, config))
    assert time.monotonic() - start < 5.0


def test_digest_escapes_untrusted_job_text(conn, config):
    """Job text comes from third parties and must never render as markup."""
    _ingest(conn, [_posting(title="Engineer <script>alert(1)</script>",
                            company_name="Acme <img src=x onerror=alert(1)>")])
    pipeline_service.run_all(conn, config, skip_discover=True)
    html = render_html(digest_service.build(conn, config))
    # What matters is that no attacker-controlled angle bracket survives as
    # markup. The payload may remain as inert escaped text.
    assert "<script" not in html.lower()
    assert "<img" not in html.lower()
    assert "&lt;script&gt;" in html, "payload should be present but escaped"
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_digest_excludes_applied_and_dismissed(conn, config):
    # Distinct titles: three identical roles at one company would be treated
    # as duplicates, which is correct but not what this test covers.
    ids = _ingest(conn, [
        _posting(external_id=f"gh:{i}", title=t,
                 apply_url=f"https://acme.example/jobs/{i}",
                 canonical_url=f"https://acme.example/jobs/{i}")
        for i, t in enumerate(["Software Engineer", "Backend Engineer",
                               "Frontend Engineer"], start=1)
    ])
    pipeline_service.run_all(conn, config, skip_discover=True)
    assert len(digest_service.build(conn, config).new_today) == 3
    ledger_service.mark_applied(conn, ids[0])
    ledger_service.dismiss(conn, ids[1], reason="wrong stack")
    assert len(digest_service.build(conn, config).new_today) == 1


def test_digest_surfaces_followups_and_rejected_sample(conn, config):
    ids = _ingest(conn, [
        _posting(external_id="gh:1", apply_url="https://a.example/1",
                 canonical_url="https://a.example/1"),
        _posting(external_id="gh:2", title="QA Engineer",
                 apply_url="https://a.example/2", canonical_url="https://a.example/2"),
    ])
    pipeline_service.run_all(conn, config, skip_discover=True)
    app_id = ledger_service.mark_applied(conn, ids[0])
    old = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    conn.execute("UPDATE applications SET updated_at=? WHERE id=?", (old, app_id))

    data = digest_service.build(conn, config)
    assert any(a["application_id"] == app_id for a in data.needs_action)
    assert data.rejected_sample, "filtered-out spot check must not be empty"
    assert all(r["rules"] for r in data.rejected_sample)
    assert "title_role" in data.filter_histogram


def test_digest_written_to_disk(conn, config, tmp_path):
    _ingest(conn, [_posting()])
    pipeline_service.run_all(conn, config, skip_discover=True)
    path = write_digest(digest_service.build(conn, config), tmp_path / "digests")
    assert path.exists() and path.suffix == ".html"
    assert path.read_text(encoding="utf-8").lower().startswith("<!doctype html>")
