"""Every rule must fire on a positive case and stay silent on a negative one."""

import pytest

from jobsearch.domain.filters import RULE_DESCRIPTIONS, evaluate
from tests.conftest import make_job

LONG = "We build backend services with Python and FastAPI. " * 10


def test_clean_job_passes(filters):
    assert evaluate(make_job(description_text=LONG), filters).passed


@pytest.mark.parametrize("rule,job_kwargs", [
    ("title_role", dict(title="QA Automation Engineer")),
    ("title_seniority", dict(title="Senior Software Engineer")),
    ("employment_type", dict(title="Software Engineer (Contract)",
                             employment_type="contract")),
    ("remote_scope", dict(remote_scope="us_only", remote_type="remote")),
    ("work_auth", dict(description_text=LONG + " Must be authorized to work in the United States.")),
    ("company_blocked", dict(company_normalized="acme staffing",
                             company_name_raw="Acme Staffing")),
    ("too_short", dict(description_text="short")),
    ("country", dict(country="US", locations=[], remote_type="onsite",
                     location_raw="Austin, TX")),
])
def test_rule_fires(filters, rule, job_kwargs):
    kwargs = {"description_text": LONG, **job_kwargs}
    result = evaluate(make_job(**kwargs), filters)
    assert not result.passed
    assert rule in result.rules_failed, f"expected {rule}, got {result.rules_failed}"


def test_yoe_hard_high_excludes_but_penalty_does_not(filters):
    """The graded policy: only 4+ required is excluded."""
    excluded = make_job(description_text=LONG + " Requires 6+ years of experience.")
    assert not evaluate(excluded, filters).passed
    assert "yoe_hard_high" in evaluate(excluded, filters).rules_failed

    for text in [" Minimum 3 years of experience required.",
                 " 5+ years of experience is a plus.",
                 " 0-2 years of experience."]:
        job = make_job(description_text=LONG + text)
        assert evaluate(job, filters).passed, f"{text!r} should be kept"


def test_internship_is_kept_not_excluded(filters):
    """Policy: internships are deprioritized in ranking, never filtered out."""
    job = make_job(title="Software Engineer Intern", employment_type="internship",
                   description_text=LONG)
    result = evaluate(job, filters)
    assert result.passed
    assert any("internship" in n for n in result.notes)


def test_ambiguous_signals_are_kept(filters):
    """Missing signals must never exclude - optimising against false negatives."""
    job = make_job(title="Platform Engineer", country=None, locations=[],
                   location_raw="", remote_type="unknown", remote_scope="unknown",
                   employment_type="unknown", description_text=LONG)
    assert evaluate(job, filters).passed


def test_india_remote_and_global_remote_are_eligible(filters):
    for scope in ("india", "global", "apac"):
        job = make_job(country=None, remote_type="remote", remote_scope=scope,
                       location_raw="Remote", locations=[], description_text=LONG)
        assert evaluate(job, filters).passed, scope


def test_every_failed_rule_has_a_description(filters):
    job = make_job(title="Senior QA Engineer", employment_type="contract",
                   remote_scope="us_only", description_text="tiny")
    for rule in evaluate(job, filters).rules_failed:
        assert rule in RULE_DESCRIPTIONS, f"rule {rule} has no description"


def test_signals_are_recorded_for_explainability(filters):
    result = evaluate(make_job(description_text=LONG + " 3+ years required."), filters)
    assert result.signals["title_tier"] == "core_swe"
    assert result.signals["yoe"]["verdict"] == "penalty"
    assert "employment_type" in result.signals
