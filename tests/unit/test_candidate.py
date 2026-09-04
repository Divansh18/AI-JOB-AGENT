from jobsearch.domain.candidate import (
    build_application_profile,
    make_answer,
    make_fact,
    validate_fact,
    validate_store,
)


def test_application_profile_uses_verified_facts_only():
    facts = [
        make_fact(
            category="personal/profile",
            key="full_name",
            value="Asha Example",
            source="manual_verified",
            verified=True,
        ),
        make_fact(
            category="personal/profile",
            key="phone",
            value="+91 99999 88888",
            source="resume_draft",
            verified=False,
        ),
        make_fact(
            category="links",
            key="portfolio_url",
            value="https://asha.example.dev",
            source="resume_draft",
            verified=False,
        ),
    ]

    profile = build_application_profile(facts)

    assert profile.full_name == "Asha Example"
    assert profile.phone is None
    assert profile.portfolio_url is None


def test_validate_store_rejects_duplicate_verified_singletons():
    facts = [
        make_fact(
            category="personal/profile",
            key="email",
            value="asha@example.com",
            source="manual_verified",
            verified=True,
        ),
        make_fact(
            category="personal_profile",
            key="email",
            value="asha+dup@example.com",
            source="manual_verified",
            verified=True,
        ),
    ]

    report = validate_store(facts, [])

    assert not report.ok
    assert any(issue.code == "duplicate_singleton_field" for issue in report.issues)


def test_sensitive_answers_always_require_human_review():
    answer = make_answer(
        question_key="salary expectations",
        category="attestation",
        answer_text="Prefer to discuss after understanding scope.",
        source="manual",
        verified=True,
        human_review_required=False,
    )

    assert answer.human_review_required is True


def test_structured_work_experience_requires_company_and_title():
    fact = make_fact(
        category="work experience",
        key="broken_record",
        value={"start_date": "2026-01-01"},
        source="resume_draft",
        verified=False,
    )

    issues = validate_fact(fact)

    assert any(issue.code == "missing_experience_company" for issue in issues)
    assert any(issue.code == "missing_experience_title" for issue in issues)
