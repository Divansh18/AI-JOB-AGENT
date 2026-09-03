"""Golden regressions for deterministic title-role eligibility."""

import json
from pathlib import Path

import pytest

from jobsearch.config.schemas import Filters
from jobsearch.domain.filters import evaluate
from jobsearch.domain.models import RawPosting
from jobsearch.services.normalization import normalize_posting

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "title_cases.json"


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text(encoding="utf-8")), ids=lambda c: c["name"])
def test_title_case_golden(case):
    posting = RawPosting(
        source="greenhouse",
        external_id=f"golden:{case['name']}",
        title=case["title"],
        company_name="Golden Co",
        location_raw="Bengaluru, India",
        description_text=("We build Python and React products with FastAPI. " * 8),
        apply_url=f"https://example.com/{case['name']}",
        canonical_url=f"https://example.com/{case['name']}",
    )
    job = normalize_posting(posting)
    result = evaluate(job, Filters())
    expected = case["expected"]

    assert result.signals["title_tier"] == expected["tier"]
    assert result.passed is expected["passed"]
    if expected["rule"] is None:
        assert result.rules_failed == []
    else:
        assert expected["rule"] in result.rules_failed
