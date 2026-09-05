import pytest

from jobsearch.domain.extract import RemoteScope, RemoteType, extract_location_facts
from jobsearch.domain.models import RawPosting
from jobsearch.services.normalization import normalize_posting


@pytest.mark.parametrize(
    ("description", "expected_city", "expected_remote_type", "expected_remote_scope"),
    (
        (
            "SDE I Bangalore, India Engineering - Backend / Full time / On-site apply for this job.",
            "Bengaluru",
            RemoteType.ONSITE,
            RemoteScope.UNKNOWN,
        ),
        (
            "Backend Engineer Bengaluru, India Engineering / Full time / Hybrid apply for this job.",
            "Bengaluru",
            RemoteType.HYBRID,
            RemoteScope.UNKNOWN,
        ),
        (
            "Software Engineer Pune, India Engineering / Full time / On-site apply for this job.",
            "Pune",
            RemoteType.ONSITE,
            RemoteScope.UNKNOWN,
        ),
        (
            "Frontend Engineer Hyderabad, India Engineering / Full time / On-site apply for this job.",
            "Hyderabad",
            RemoteType.ONSITE,
            RemoteScope.UNKNOWN,
        ),
        (
            "Backend Engineer Remote India Engineering / Full time / Remote apply for this job.",
            None,
            RemoteType.REMOTE,
            RemoteScope.INDIA,
        ),
    ),
)
def test_ats_header_locations_are_extracted_without_location_raw(
    description,
    expected_city,
    expected_remote_type,
    expected_remote_scope,
):
    facts = extract_location_facts("Software Engineer", "", description)

    if expected_city:
        assert expected_city in facts.cities
    assert facts.countries == ["IN"]
    assert facts.country == "IN"
    assert facts.remote_type == expected_remote_type
    assert facts.remote_scope == expected_remote_scope
    assert facts.used_description_fallback is True
    assert facts.evidence


def test_description_fallback_does_not_override_explicit_foreign_location():
    facts = extract_location_facts(
        "Enterprise Solutions Engineer, Israel",
        "Tel Aviv District, Israel",
        "Candidates are based in Bangalore, India. Employees work from the office.",
    )

    assert facts.country == "IL"
    assert facts.countries == ["IL"]
    assert facts.cities == []
    assert facts.used_description_fallback is False


def test_normalization_populates_display_location_from_explicit_india_header():
    job = normalize_posting(
        RawPosting(
            source="manual",
            external_id=None,
            title="SDE I",
            company_name="Hevo Data",
            location_raw="",
            description_text="SDE I Bangalore, India Engineering - Backend / Full time / On-site apply for this job.",
            apply_url="https://jobs.lever.co/hevodata/abc123",
            canonical_url="https://jobs.lever.co/hevodata/abc123",
        )
    )

    assert job.location_raw == "Bengaluru, India"
    assert job.locations == ["Bengaluru"]
    assert job.country == "IN"
    assert job.remote_type == "onsite"
