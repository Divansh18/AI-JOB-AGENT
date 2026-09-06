from jobsearch.domain.application_intelligence import (
    GroundedAnswerDraft,
    LlmJobInsight,
    make_resume_wording_suggestion,
    classify_application_question,
    make_requirement,
    validate_grounded_answer_draft,
    validate_llm_job_insight,
    validate_resume_wording_suggestion,
)
from jobsearch.llm.schemas import GroundedAnswerDraftSetExtraction, LlmJobInsightExtraction, ResumeWordingArtifactExtraction


def test_structured_output_schema_accepts_grounded_jd_intelligence():
    parsed = LlmJobInsightExtraction(
        role_summary="Build backend SaaS integrations.",
        must_have_requirements=[
            {
                "text": "Python backend development",
                "category": "skill",
                "candidate_match": "matched",
                "evidence_refs": ["fact:4"],
                "rationale": "Verified API work names Python.",
            }
        ],
        preferred_requirements=[],
        role_priorities=["backend APIs", "data integrations"],
        grounded_fit_assessment="Good evidence overlap on backend API work.",
        fit_verdict="worth_applying",
        candidate_match_evidence_refs=["fact:4"],
        uncertainties=["Scale expectations are not explicit."],
        gaps=[],
    )

    assert parsed.fit_verdict == "worth_applying"
    assert parsed.must_have_requirements[0].candidate_match == "matched"


def test_llm_insight_validation_rejects_invalid_evidence_refs():
    insight = LlmJobInsight(
        job_id=1,
        role_summary="Build backend APIs.",
        must_have_requirements=[
            make_requirement(
                text="Python backend development",
                category="skill",
                required=True,
                candidate_match="matched",
                evidence_refs=["fact:999"],
            )
        ],
        preferred_requirements=[],
        role_priorities=[],
        grounded_fit_assessment="Candidate matches Python.",
        fit_verdict="worth_applying",
        candidate_match_evidence_refs=["fact:999"],
    )

    issues = validate_llm_job_insight(insight, allowed_evidence_refs={"fact:4"})

    assert {issue.code for issue in issues} == {"invalid_evidence_ref", "invalid_candidate_match_evidence_ref"}


def test_llm_insight_validation_rejects_unsupported_candidate_match_claims():
    insight = LlmJobInsight(
        job_id=1,
        role_summary="Build backend APIs.",
        must_have_requirements=[
            make_requirement(
                text="Python backend development",
                category="skill",
                required=True,
                candidate_match="matched",
                evidence_refs=[],
            )
        ],
        preferred_requirements=[],
        role_priorities=[],
        grounded_fit_assessment="Candidate matches Python.",
        fit_verdict="worth_applying",
    )

    issues = validate_llm_job_insight(insight, allowed_evidence_refs={"fact:4"})

    assert [issue.code for issue in issues] == ["candidate_claim_missing_evidence_refs"]


def test_llm_insight_validation_keeps_sensitive_topics_deterministic_only():
    insight = LlmJobInsight(
        job_id=1,
        role_summary="Build backend APIs.",
        must_have_requirements=[
            make_requirement(
                text="Must be authorized to work in the US",
                category="work_authorization",
                required=True,
                candidate_match="matched",
                evidence_refs=["fact:4"],
            )
        ],
        preferred_requirements=[],
        role_priorities=[],
        grounded_fit_assessment="Candidate is authorized.",
        fit_verdict="worth_applying",
    )

    issues = validate_llm_job_insight(insight, allowed_evidence_refs={"fact:4"})

    assert {issue.code for issue in issues} == {
        "sensitive_evidence_ref_forbidden",
        "sensitive_match_claim_forbidden",
    }


def test_question_classifier_allows_only_safe_free_text_for_llm():
    assert classify_application_question("why_interested", "Why are you interested in this role?") == "safe_free_text"
    assert classify_application_question("relevant_project", "Tell us about a project relevant to this role.") == "safe_free_text"
    assert classify_application_question("salary_expectation", "What are your salary expectations?") == "sensitive"
    assert classify_application_question("sponsorship", "Will you require visa sponsorship?") == "sensitive"
    assert classify_application_question("legal", "I certify this information is accurate.") == "legal_attestation"
    assert classify_application_question("gender", "What is your gender?") == "self_identification"
    assert classify_application_question("email", "Email address") == "factual_candidate_field"
    assert classify_application_question("custom", "Share anything we should know") == "unknown"


def test_answer_draft_schema_accepts_structured_grounded_output():
    parsed = GroundedAnswerDraftSetExtraction(
        drafts=[
            {
                "question_key": "why_interested",
                "answer_text": "I am interested in the backend API focus.",
                "evidence_refs": ["fact:4"],
                "confidence": 0.84,
                "review_required": True,
            }
        ]
    )

    assert parsed.drafts[0].question_key == "why_interested"
    assert parsed.drafts[0].evidence_refs == ["fact:4"]


def test_grounded_answer_validation_rejects_unsupported_claims():
    draft = GroundedAnswerDraft(
        application_id=1,
        job_id=2,
        question_key="why_interested",
        question_text="Why are you interested in this role?",
        question_classification="safe_free_text",
        answer_text="I have 5 years of Kubernetes experience and admire your recent funding.",
        evidence_refs=["fact:4"],
        confidence=0.9,
        review_required=False,
        validation_status="valid",
    )

    issues = validate_grounded_answer_draft(
        draft,
        allowed_evidence_refs={"fact:4"},
        evidence_text_by_ref={"fact:4": "Built Python and FastAPI APIs."},
        job_context_text="Backend APIs for data integrations.",
    )

    assert {
        "unsupported_numeric_claim",
        "unsupported_skill_claim",
        "unsupported_company_claim",
    }.issubset({issue.code for issue in issues})


def test_grounded_answer_validation_blocks_sensitive_draft_text():
    draft = GroundedAnswerDraft(
        application_id=1,
        job_id=2,
        question_key="salary_expectation",
        question_text="What are your salary expectations?",
        question_classification="sensitive",
        answer_text="I expect market compensation.",
        evidence_refs=[],
        confidence=0.8,
        review_required=True,
        validation_status="valid",
    )

    issues = validate_grounded_answer_draft(
        draft,
        allowed_evidence_refs={"fact:4"},
        evidence_text_by_ref={"fact:4": "Built Python APIs."},
    )

    assert [issue.code for issue in issues] == ["blocked_question_answered"]


def test_resume_wording_schema_accepts_structured_suggestions():
    parsed = ResumeWordingArtifactExtraction(
        suggestions=[
            {
                "item_key": "experience:0:bullet:0",
                "action": "rewrite",
                "rewritten_text": "Built Python and FastAPI APIs for internal team workflows.",
                "evidence_refs": ["fact:4"],
            }
        ]
    )

    assert parsed.suggestions[0].item_key == "experience:0:bullet:0"
    assert parsed.suggestions[0].rewritten_text.startswith("Built Python")
    assert parsed.suggestions[0].evidence_refs == ["fact:4"]


def test_resume_wording_validation_accepts_grounded_rewrite():
    suggestion = make_resume_wording_suggestion(
        section="experience",
        item_key="experience:0:bullet:0",
        original_text="Built Python and FastAPI APIs used by internal teams.",
        suggested_text="Built Python and FastAPI APIs for internal team workflows.",
        evidence_refs=["fact:4"],
    )

    issues = validate_resume_wording_suggestion(
        suggestion,
        allowed_evidence_refs={"fact:4"},
        evidence_text_by_ref={"fact:4": "Built Python and FastAPI APIs used by internal teams."},
        protected_terms=["Acme Labs", "Backend Engineering Intern"],
    )

    assert issues == []


def test_resume_wording_validation_accepts_equivalent_k_and_comma_numbers():
    suggestion = make_resume_wording_suggestion(
        section="projects",
        item_key="projects:0:bullet:0",
        original_text="Processed 13K+ records through verified backend workflows.",
        suggested_text="Processed over 13,000 records through verified backend workflows.",
        evidence_refs=["fact:4"],
    )

    issues = validate_resume_wording_suggestion(
        suggestion,
        allowed_evidence_refs={"fact:4"},
        evidence_text_by_ref={"fact:4": "Processed 13K+ records through verified backend workflows."},
    )

    assert issues == []


def test_resume_wording_validation_rejects_unsupported_nearby_number():
    suggestion = make_resume_wording_suggestion(
        section="projects",
        item_key="projects:0:bullet:0",
        original_text="Processed 13K+ records through verified backend workflows.",
        suggested_text="Processed over 13,500 records through verified backend workflows.",
        evidence_refs=["fact:4"],
    )

    issues = validate_resume_wording_suggestion(
        suggestion,
        allowed_evidence_refs={"fact:4"},
        evidence_text_by_ref={"fact:4": "Processed 13K+ records through verified backend workflows."},
    )

    assert any(issue.code == "unsupported_numeric_claim" for issue in issues)
    assert any("13,500" in issue.message for issue in issues)


def test_resume_wording_validation_accepts_supported_duration():
    suggestion = make_resume_wording_suggestion(
        section="projects",
        item_key="projects:0:bullet:0",
        original_text="Reduced sync runtime to 60 seconds.",
        suggested_text="Kept sync runtime at 60 seconds.",
        evidence_refs=["fact:4"],
    )

    issues = validate_resume_wording_suggestion(
        suggestion,
        allowed_evidence_refs={"fact:4"},
        evidence_text_by_ref={"fact:4": "Reduced sync runtime to 60 seconds."},
    )

    assert issues == []


def test_resume_wording_validation_rejects_unsupported_duration_change():
    suggestion = make_resume_wording_suggestion(
        section="projects",
        item_key="projects:0:bullet:0",
        original_text="Reduced sync runtime to 60 seconds.",
        suggested_text="Kept sync runtime at 30 seconds.",
        evidence_refs=["fact:4"],
    )

    issues = validate_resume_wording_suggestion(
        suggestion,
        allowed_evidence_refs={"fact:4"},
        evidence_text_by_ref={"fact:4": "Reduced sync runtime to 60 seconds."},
    )

    assert any(issue.code == "unsupported_numeric_claim" for issue in issues)
    assert any("30 seconds" in issue.message for issue in issues)


def test_resume_wording_validation_rejects_unsupported_skill_and_metric():
    suggestion = make_resume_wording_suggestion(
        section="experience",
        item_key="experience:0:bullet:0",
        original_text="Built Python and FastAPI APIs used by internal teams.",
        suggested_text="Built Kubernetes services that improved latency by 40%.",
        evidence_refs=["fact:4"],
    )

    issues = validate_resume_wording_suggestion(
        suggestion,
        allowed_evidence_refs={"fact:4"},
        evidence_text_by_ref={"fact:4": "Built Python and FastAPI APIs used by internal teams."},
    )

    assert {"unsupported_skill_claim", "unsupported_numeric_claim"}.issubset({issue.code for issue in issues})


def test_resume_wording_validation_still_rejects_unsupported_percentages_and_dates():
    suggestion = make_resume_wording_suggestion(
        section="projects",
        item_key="projects:0:bullet:0",
        original_text="Improved backend reliability for internal workflows.",
        suggested_text="Improved backend reliability by 40% in 2024.",
        evidence_refs=["fact:4"],
    )

    issues = validate_resume_wording_suggestion(
        suggestion,
        allowed_evidence_refs={"fact:4"},
        evidence_text_by_ref={"fact:4": "Improved backend reliability for internal workflows."},
    )
    messages = [issue.message for issue in issues if issue.code == "unsupported_numeric_claim"]

    assert any("40%" in message for message in messages)
    assert any("2024" in message for message in messages)


def test_resume_wording_validation_rejects_unsupported_yoe_date_and_ref():
    suggestion = make_resume_wording_suggestion(
        section="experience",
        item_key="experience:0:bullet:0",
        original_text="Built Python APIs.",
        suggested_text="Built Python APIs with 5 years of ownership since 2024.",
        evidence_refs=["fact:999"],
    )

    issues = validate_resume_wording_suggestion(
        suggestion,
        allowed_evidence_refs={"fact:4"},
        evidence_text_by_ref={"fact:4": "Built Python APIs."},
    )

    assert {"invalid_evidence_ref", "unsupported_numeric_claim", "unsupported_years_claim"}.issubset(
        {issue.code for issue in issues}
    )
