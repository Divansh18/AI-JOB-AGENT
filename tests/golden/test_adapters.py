"""Golden-file parsing tests against recorded API payloads.

No network. If an ATS changes its response shape, these fail with a diff
rather than silently producing empty or wrong jobs in production.
"""

import json
from pathlib import Path

import pytest

from jobsearch.services.normalization import normalize_posting
from jobsearch.sources import ashby, greenhouse, lever

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


CASES = [
    ("greenhouse_board.json", greenhouse.parse_board, "greenhouse", "gh:"),
    ("lever_board.json", lever.parse_board, "lever", "lv:"),
    ("ashby_board.json", ashby.parse_board, "ashby", "ab:"),
]


@pytest.mark.parametrize("fixture,parser,source,prefix", CASES, ids=[c[2] for c in CASES])
def test_adapter_parses_recorded_payload(fixture, parser, source, prefix):
    postings = list(parser(_load(fixture), "Test Company", company_id=7))
    assert postings, f"{source} produced no postings"
    for p in postings:
        assert p.source == source
        assert p.external_id and p.external_id.startswith(prefix)
        assert p.title.strip(), "title must not be empty"
        assert p.company_name == "Test Company"
        assert p.company_id == 7
        assert p.apply_url.startswith("http"), f"bad apply_url: {p.apply_url!r}"


@pytest.mark.parametrize("fixture,parser,source,prefix", CASES, ids=[c[2] for c in CASES])
def test_normalization_produces_complete_jobs(fixture, parser, source, prefix):
    for posting in parser(_load(fixture), "Test Company", company_id=7):
        job = normalize_posting(posting)
        assert job.fingerprint and len(job.fingerprint) == 64
        assert job.content_hash and len(job.content_hash) == 64
        assert job.title_normalized == job.title_normalized.lower()
        assert job.company_normalized == "test"  # "Company" is a stripped suffix
        assert job.remote_type in ("onsite", "hybrid", "remote", "unknown")
        assert job.employment_type in (
            "full_time", "internship", "contract", "part_time", "unknown")
        assert job.first_seen_at is not None
        assert isinstance(job.locations, list)


@pytest.mark.parametrize("fixture,parser,source,prefix", CASES, ids=[c[2] for c in CASES])
def test_parsing_is_stable_across_runs(fixture, parser, source, prefix):
    """Same payload must always yield the same fingerprints."""
    a = [normalize_posting(p).fingerprint
         for p in parser(_load(fixture), "Test Company", 7)]
    b = [normalize_posting(p).fingerprint
         for p in parser(_load(fixture), "Test Company", 7)]
    assert a == b
    assert len(set(a)) == len(a), "fingerprints must be unique within a board"


def test_html_descriptions_become_plain_text():
    for posting in greenhouse.parse_board(_load("greenhouse_board.json"), "X", 1):
        job = normalize_posting(posting)
        assert "<div" not in job.description_text
        assert "&lt;" not in job.description_text
        assert "&nbsp;" not in job.description_text


def test_adapter_survives_empty_and_malformed_payloads():
    assert list(greenhouse.parse_board({}, "X", 1)) == []
    assert list(greenhouse.parse_board({"jobs": None}, "X", 1)) == []
    assert list(lever.parse_board([], "X", 1)) == []
    assert list(ashby.parse_board({"jobs": []}, "X", 1)) == []
