from pathlib import Path

import pytest

from jobsearch.services.ashby_autofill import detect_ashby_fields
from jobsearch.services.autofill import ATS_ASHBY, ATS_GREENHOUSE, ATS_LEVER, ATS_UNSUPPORTED, detect_ats
from jobsearch.services.greenhouse_autofill import (
    GreenhouseAutofillAdapter,
    SubmissionGuardError,
    detect_greenhouse_fields,
    detect_human_intervention_required,
)
from jobsearch.services.lever_autofill import detect_lever_fields


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_detect_ats_recognizes_supported_hosted_urls_only():
    assert detect_ats("https://job-boards.greenhouse.io/gitlab/jobs/8736877002") == ATS_GREENHOUSE
    assert detect_ats("https://boards.greenhouse.io/example/jobs/123") == ATS_GREENHOUSE
    assert detect_ats("https://jobs.lever.co/example/123") == ATS_LEVER
    assert detect_ats("https://apply.lever.co/example/123") == ATS_LEVER
    assert detect_ats("https://jobs.ashbyhq.com/example/123") == ATS_ASHBY
    assert detect_ats("https://example.com/jobs/123") == ATS_UNSUPPORTED
    assert detect_ats("https://clever.co/example/123") == ATS_UNSUPPORTED


def test_greenhouse_field_detection_maps_common_resume_custom_and_sensitive_fields():
    html = (FIXTURES / "greenhouse_application_form.html").read_text(encoding="utf-8")

    fields = detect_greenhouse_fields(html)
    by_key = {field.key for field in fields}
    sensitive = [field for field in fields if field.sensitive]

    assert {"first_name", "last_name", "email", "phone", "linkedin_url", "github_url", "current_location", "resume"} <= by_key
    assert any(field.key == "sponsorship_required" for field in sensitive)
    assert any(field.custom for field in fields if field.key not in {"resume", "first_name", "last_name", "email", "phone"})
    assert "Submit Application" not in {field.label for field in fields}


def test_captcha_detection_flags_human_intervention():
    html = (FIXTURES / "greenhouse_captcha_form.html").read_text(encoding="utf-8")

    assert detect_human_intervention_required(html) is True


def test_submission_guard_blocks_submit_actions():
    with pytest.raises(SubmissionGuardError):
        GreenhouseAutofillAdapter.assert_not_submission_action("Submit Application")


def test_lever_field_detection_maps_common_resume_custom_and_sensitive_fields():
    html = (FIXTURES / "lever_application_form.html").read_text(encoding="utf-8")

    fields = detect_lever_fields(html)
    by_key = {field.key for field in fields}
    sensitive = [field for field in fields if field.sensitive]

    assert {"full_name", "email", "phone", "linkedin_url", "github_url", "portfolio_url", "current_location", "resume"} <= by_key
    assert any(field.key == "sponsorship_required" for field in sensitive)
    assert any(field.key == "why_are_you_interested_in_this_role" for field in fields)
    assert "Submit application" not in {field.label for field in fields}


def test_ashby_field_detection_maps_common_resume_custom_and_sensitive_fields():
    html = (FIXTURES / "ashby_application_form.html").read_text(encoding="utf-8")

    fields = detect_ashby_fields(html)
    by_key = {field.key for field in fields}
    sensitive = [field for field in fields if field.sensitive]

    assert {"full_name", "email", "phone", "linkedin_url", "github_url", "portfolio_url", "current_location", "resume"} <= by_key
    assert any(field.key == "work_authorization" for field in sensitive)
    assert any(field.key == "short_experience_summary" for field in fields)
    assert "Submit Application" not in {field.label for field in fields}


def test_field_detection_uses_valid_selector_for_digit_started_ids():
    fields = detect_ashby_fields(
        '<label for="6e7d79f2-136a-41eb-b2ff-b05c46ac6403">Github Profile</label>'
        '<input id="6e7d79f2-136a-41eb-b2ff-b05c46ac6403" name="github">'
    )

    assert fields[0].key == "github_url"
    assert fields[0].selector == '[id="6e7d79f2-136a-41eb-b2ff-b05c46ac6403"]'
