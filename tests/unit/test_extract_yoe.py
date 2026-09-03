"""YOE grading is the most correctness-sensitive rule in the system."""

from pathlib import Path

import pytest
import yaml

from jobsearch.domain.extract import YoeVerdict, extract_yoe

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "yoe_snippets.yaml"
CASES = yaml.safe_load(FIXTURES.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["text"][:45])
def test_yoe_verdicts(case):
    result = extract_yoe(case["text"])
    assert result.verdict.value == case["verdict"], (
        f"{case['text']!r}\n  expected {case['verdict']}, got {result.verdict.value} "
        f"(required_min={result.required_min}, preferred_min={result.preferred_min})"
    )


def test_hard_high_is_the_only_excluding_verdict():
    """Policy: only 4+ *required* is filtered; everything else is kept."""
    from jobsearch.config.schemas import Filters

    assert Filters().experience.exclude_verdicts == ["hard_high"]
    kept = {YoeVerdict.STRONG, YoeVerdict.OK, YoeVerdict.PENALTY,
            YoeVerdict.SOFT_HIGH, YoeVerdict.UNKNOWN}
    assert YoeVerdict.HARD_HIGH not in kept


def test_fresher_cue_overrides_high_requirement():
    text = ("We are hiring freshers. Some senior openings require 8+ years of experience.")
    assert extract_yoe(text).verdict == YoeVerdict.PENALTY


def test_range_uses_lower_bound_as_entry_bar():
    r = extract_yoe("2-6 years of experience required")
    assert r.required_min == 2
    assert r.verdict == YoeVerdict.STRONG


def test_empty_and_none_are_unknown():
    assert extract_yoe("").verdict == YoeVerdict.UNKNOWN
    assert extract_yoe(None).verdict == YoeVerdict.UNKNOWN
