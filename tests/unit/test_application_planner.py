from jobsearch.domain.application_planner import (
    build_candidate_field_mappings,
    build_known_answer_mappings,
    build_sensitive_fields,
    canonical_question_key,
    is_sensitive_application_question,
)
from jobsearch.domain.candidate import ApplicationProfile, make_answer


def test_candidate_field_mapping_derives_names_only_from_verified_full_name():
    profile = ApplicationProfile(
        full_name="Asha Example",
        email="asha@example.com",
        provenance={
            "full_name": {"fact_id": 1, "selector": "personal_profile:full_name", "source": "manual"},
            "email": {"fact_id": 2, "selector": "personal_profile:email", "source": "manual"},
        },
    )

    fields, unresolved = build_candidate_field_mappings(profile)
    values = {field.key: field for field in fields}
    missing = {field.key for field in unresolved}

    assert values["first_name"].value == "Asha"
    assert values["last_name"].value == "Example"
    assert values["first_name"].evidence_ref == "fact:1"
    assert values["email"].source == "manual"
    assert "phone" in missing


def test_single_part_name_leaves_last_name_unresolved():
    fields, unresolved = build_candidate_field_mappings(ApplicationProfile(full_name="Asha"))

    assert {field.key: field.value for field in fields}["first_name"] == "Asha"
    assert any(field.key == "last_name" for field in unresolved)


def test_answer_mapping_uses_verified_answers_only_and_marks_sensitive_review():
    answers = [
        make_answer(
            question_key="salary_expectations",
            answer_text="Human review required.",
            verified=True,
            human_review_required=True,
            source="manual",
        ),
        make_answer(
            question_key="sponsorship_required",
            answer_text="No",
            verified=False,
            human_review_required=True,
            source="manual",
        ),
    ]

    known, covered = build_known_answer_mappings(answers)

    assert canonical_question_key("salary_expectations") == "salary_expectation"
    assert is_sensitive_application_question("visa_sponsorship") is True
    assert [answer.question_key for answer in known] == ["salary_expectation"]
    assert known[0].autofill_safe is False
    assert covered == {"salary_expectation"}


def test_sensitive_notice_period_is_not_required_when_explicitly_verified():
    known, _ = build_known_answer_mappings([])
    profile = ApplicationProfile(notice_period_days=30)

    sensitive = build_sensitive_fields(known, profile)

    assert "notice_period" not in {field.question_key for field in sensitive}
    assert "work_authorization" in {field.question_key for field in sensitive}
