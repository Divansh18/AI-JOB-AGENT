"""Golden regressions for deterministic location eligibility."""

import json
from pathlib import Path

import pytest

from jobsearch.config.schemas import Filters
from jobsearch.domain.filters import evaluate
from jobsearch.domain.models import RawPosting
from jobsearch.services.normalization import normalize_posting

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "location_cases.json"


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text(encoding="utf-8")), ids=lambda c: c["name"])
def test_location_case_golden(case):
    posting = RawPosting(
        source="greenhouse",
        external_id=f"golden:{case['name']}",
        title=case["title"],
        company_name="Golden Co",
        location_raw=case["location_raw"],
        description_text=(case["description_text"] + " ") * 8,
        apply_url=f"https://example.com/{case['name']}",
        canonical_url=f"https://example.com/{case['name']}",
    )
    job = normalize_posting(posting)
    result = evaluate(job, Filters())
    expected = case["expected"]

    assert job.country == expected["country"]
    assert job.remote_type == expected["remote_type"]
    assert job.remote_scope == expected["remote_scope"]
    assert result.passed is expected["passed"]
    assert result.signals["location"]["bucket"] == expected["bucket"]
    if expected["rule"] is None:
        assert result.rules_failed == []
    else:
        assert expected["rule"] in result.rules_failed
