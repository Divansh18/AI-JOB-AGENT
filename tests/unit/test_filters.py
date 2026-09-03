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
    ("remote_scope", dict(location_raw="Remote (US)", remote_type="remote")),
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


@pytest.mark.parametrize("title", [
    "Strategy and Operations Associate",
    "Intermediate Support Engineer",
    "Associate - Monetisation",
    "Accounts Payable, Spend Management Coordinator",
    "Analyst (Supply Analytics, Bangkok-based, Relocation provided)",
    "PPSL- Product Management- Devices",
    "Deal Desk",
    "Intern - Admin and Operations",
    "Product Research Specialist (W from Groww)",
    "Language Expert - Taiwan (Bangkok Based)",
    "Creative Sourcer",
    "growth and business - max and wallet",
    "Workplace & Engagement Coordinator",
    "Developer Relations Engineer",
    "Customer Experience Engineer, L1",
    "AI Field Engineer - Enterprise",
    "Value Solutions Engineer (Inside Presales - America Region)",
    "Solution Engineering",
    "Solution Engineer - Insurance & Asset Management",
    "Service Desk Specialist",
])
def test_clear_non_target_titles_are_excluded(filters, title):
    result = evaluate(make_job(title=title, description_text=LONG), filters)
    assert not result.passed
    assert "title_role" in result.rules_failed


@pytest.mark.parametrize("title", [
    "SDK Engineer - JavaScript",
    "Performance Engineer - Benchmarking",
    "Associate Infrastructure Engineer",
    "Product Engineer - Manufacturing Operations",
])
def test_unusual_engineering_titles_are_kept(filters, title):
    result = evaluate(make_job(title=title, description_text=LONG), filters)
    assert result.passed
    assert "title_role" not in result.rules_failed


def test_remote_description_location_snippet_can_resolve_restrictions(filters):
    job = make_job(
        title="Associate Infrastructure Engineer",
        location_raw="CA Remote (BC & ON only); U.S. Remote",
        locations=[],
        country=None,
        description_text=LONG + "Location: Remote-first (United States; BC & ON, Canada)",
    )
    result = evaluate(job, filters)
    assert not result.passed
    assert "remote_scope" in result.rules_failed
    assert result.signals["location"]["used_description_fallback"] is True
    assert result.signals["location"]["countries"] == ["US", "CA"]
    assert result.signals["location"]["remote_scope"] == "us_only"


@pytest.mark.parametrize(
    "title,location_raw,expected_country,expected_countries",
    [
        ("Deployed Engineer - LATAM", "Mexico City", "MX", ["MX"]),
        ("Software Engineer (Backend), Enterprise", "Budapest, Hungary", "HU", ["HU"]),
        ("Data Engineer", "Mountain View", "US", ["US"]),
        ("Software Engineer, Robotics", "Argentina; Uruguay", None, ["AR", "UY"]),
    ],
)
def test_explicit_foreign_offices_are_excluded(filters, title, location_raw, expected_country, expected_countries):
    result = evaluate(make_job(title=title, location_raw=location_raw, description_text=LONG), filters)
    assert not result.passed
    assert "country" in result.rules_failed
    assert result.signals["location"]["bucket"] == "foreign_office"
    assert result.signals["location"]["country"] == expected_country
    assert result.signals["location"]["countries"] == expected_countries


@pytest.mark.parametrize("title,location_raw", [
    ("Machine Learning Engineer", "Austria (remote)"),
    ("Industrial Compute", "US - Remote"),
])
def test_restricted_remote_markers_are_excluded(filters, title, location_raw):
    result = evaluate(make_job(title=title, location_raw=location_raw, description_text=LONG), filters)
    assert not result.passed
    assert "remote_scope" in result.rules_failed
    assert result.signals["location"]["bucket"] == "remote_restricted"


def test_bangkok_is_excluded_as_foreign_office(filters):
    result = evaluate(
        make_job(
            title="Agoda Tech - Software Engineering & Engineering Management roles (Hywel)",
            location_raw="Bangkok",
            description_text=LONG,
        ),
        filters,
    )
    assert not result.passed
    assert "country" in result.rules_failed
    assert result.signals["location"]["country"] == "TH"
    assert result.signals["location"]["bucket"] == "foreign_office"


def test_explicit_foreign_country_remote_overrides_apac_scope(filters):
    result = evaluate(
        make_job(
            title="Customer Engineer, APAC",
            location_raw="Australia (Remote)",
            description_text=LONG,
        ),
        filters,
    )
    assert not result.passed
    assert "remote_scope" in result.rules_failed
    assert result.signals["location"]["country"] == "AU"
    assert result.signals["location"]["bucket"] == "remote_restricted"


def test_india_remote_and_global_remote_are_eligible(filters):
    for scope in ("india", "global", "apac"):
        job = make_job(country=None, remote_type="remote", remote_scope=scope,
                       location_raw="Remote", locations=[], description_text=LONG)
        assert evaluate(job, filters).passed, scope


def test_description_mentions_distributed_do_not_imply_remote(filters):
    job = make_job(
        title="Software Engineer, Platform",
        location_raw="San Francisco, CA; New York, NY",
        country="US",
        remote_type="remote",
        description_text=LONG + "Deep understanding of distributed systems is required.",
    )
    result = evaluate(job, filters)
    assert not result.passed
    assert "country" in result.rules_failed
    assert result.signals["location"]["remote_type"] == "unknown"


def test_foreign_remote_unknown_scope_is_excluded_when_country_is_explicit(filters):
    job = make_job(
        title="Machine Learning Engineer",
        location_raw="Remote - Ireland",
        country="IE",
        remote_type="remote",
        description_text=LONG + "Our dedication to remote-first work supports the team globally.",
    )
    result = evaluate(job, filters)
    assert not result.passed
    assert "remote_scope" in result.rules_failed
    assert result.signals["location"]["reason"] == "remote restricted to Ireland"


def test_remote_emea_restriction_in_title_is_excluded(filters):
    job = make_job(
        title="Intermediate Backend Engineer, EMEA",
        location_raw="Remote, Poland",
        country="PL",
        remote_type="remote",
        description_text=LONG,
    )
    result = evaluate(job, filters)
    assert not result.passed
    assert "remote_scope" in result.rules_failed
    assert result.signals["location"]["remote_scope"] == "emea_only"


def test_unknown_remote_is_kept_but_flagged_unknown(filters):
    job = make_job(
        title="Software Engineer",
        location_raw="Remote",
        country=None,
        locations=[],
        remote_type="remote",
        description_text=LONG,
    )
    result = evaluate(job, filters)
    assert result.passed
    assert result.signals["location"]["status"] == "unknown"
    assert "remote eligibility unclear" in result.notes


def test_india_in_multi_location_is_kept(filters):
    job = make_job(
        title="Backend Engineer",
        location_raw="Bengaluru, India; San Francisco, CA",
        locations=["Bengaluru"],
        country="IN",
        description_text=LONG,
    )
    result = evaluate(job, filters)
    assert result.passed
    assert result.signals["location"]["bucket"] == "india_multi_location"


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
