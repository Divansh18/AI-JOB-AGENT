from jobsearch.domain.candidate import make_answer, make_fact
from jobsearch.persistence.repositories import CandidateAnswerRepo, CandidateFactRepo


def test_candidate_fact_repo_upserts_by_category_and_key(conn):
    repo = CandidateFactRepo(conn)
    saved = repo.save(
        make_fact(
            category="personal/profile",
            key="email",
            value="asha@example.com",
            source="manual_verified",
            verified=True,
        )
    )
    updated = repo.save(
        make_fact(
            category="personal/profile",
            key="email",
            value="asha+updated@example.com",
            source="manual_verified",
            verified=True,
        )
    )

    facts = repo.list()

    assert saved.id == updated.id
    assert len(facts) == 1
    assert facts[0].value == "asha+updated@example.com"


def test_candidate_answer_repo_round_trips_evidence_refs(conn):
    repo = CandidateAnswerRepo(conn)
    saved = repo.save(
        make_answer(
            question_key="current_location",
            category="profile_field",
            answer_text="Bengaluru, India",
            source="manual_verified",
            evidence_refs=["personal_profile:current_location"],
            verified=True,
            human_review_required=False,
        )
    )

    answers = repo.list()

    assert len(answers) == 1
    assert answers[0].id == saved.id
    assert answers[0].evidence_refs == ["personal_profile:current_location"]
