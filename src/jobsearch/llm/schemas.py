"""Future-facing structured output schemas.

Phase 1B runs deterministically with no provider configured. These models are
kept only as protocol definitions for a later provider-backed path.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TriageResult(BaseModel):
    """Second-opinion assessment of one job against the candidate profile."""

    fit_score: int = Field(ge=0, le=100)
    verdict: str
    matched_requirements: list[str] = Field(default_factory=list, max_length=12)
    missing_requirements: list[str] = Field(default_factory=list, max_length=12)
    seniority_assessment: str = "unknown"
    red_flags: list[str] = Field(default_factory=list, max_length=8)
    one_line_rationale: str = ""

    @field_validator("verdict")
    @classmethod
    def _verdict(cls, v: str) -> str:
        allowed = {"strong", "worth_applying", "stretch", "poor_fit"}
        v = (v or "").strip().lower().replace(" ", "_").replace("-", "_")
        if v not in allowed:
            raise ValueError(f"verdict must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("seniority_assessment")
    @classmethod
    def _seniority(cls, v: str) -> str:
        allowed = {"early_career", "mid", "senior", "unknown"}
        v = (v or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
        return v if v in allowed else "unknown"

    @field_validator("one_line_rationale")
    @classmethod
    def _rationale(cls, v: str) -> str:
        return (v or "").strip()[:400]


TRIAGE_JSON_SPEC = """{
  "fit_score": <integer 0-100>,
  "verdict": "strong" | "worth_applying" | "stretch" | "poor_fit",
  "matched_requirements": [<up to 8 short strings>],
  "missing_requirements": [<up to 8 short strings>],
  "seniority_assessment": <seniority the ROLE demands: "early_career"|"mid"|"senior"|"unknown">,
  "red_flags": [<up to 5 short strings>],
  "one_line_rationale": "<one sentence, max 200 chars>"
}"""


class JobAnalysisExtraction(BaseModel):
    """Validated structured extraction for explicit job requirements."""

    role_title: str = ""
    seniority: str = "unknown"
    required_skills: list[str] = Field(default_factory=list, max_length=20)
    preferred_skills: list[str] = Field(default_factory=list, max_length=20)
    responsibilities: list[str] = Field(default_factory=list, max_length=10)
    minimum_experience_years: float | None = Field(default=None, ge=0, le=50)
    preferred_experience_years: float | None = Field(default=None, ge=0, le=50)
    domain_signals: list[str] = Field(default_factory=list, max_length=10)
    education_requirements: list[str] = Field(default_factory=list, max_length=8)
    resume_keywords: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("role_title")
    @classmethod
    def _role_title(cls, v: str) -> str:
        return (v or "").strip()[:200]

    @field_validator("seniority")
    @classmethod
    def _seniority(cls, v: str) -> str:
        allowed = {"early_career", "mid", "senior", "unknown"}
        cleaned = (v or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
        return cleaned if cleaned in allowed else "unknown"

    @field_validator(
        "required_skills",
        "preferred_skills",
        "domain_signals",
        "education_requirements",
        "resume_keywords",
    )
    @classmethod
    def _clean_lists(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in values or []:
            cleaned = (item or "").strip().lower()
            if cleaned and cleaned not in seen:
                out.append(cleaned[:120])
                seen.add(cleaned)
        return out

    @field_validator("responsibilities")
    @classmethod
    def _responsibilities(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in values or []:
            cleaned = " ".join((item or "").split()).strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                out.append(cleaned[:260])
                seen.add(key)
        return out


JOB_ANALYSIS_JSON_SPEC = """{
  "role_title": "<short role title>",
  "seniority": "early_career" | "mid" | "senior" | "unknown",
  "required_skills": [<explicitly required skills only>],
  "preferred_skills": [<explicitly preferred / nice-to-have skills only>],
  "responsibilities": [<up to 8 explicit responsibility statements>],
  "minimum_experience_years": <number or null>,
  "preferred_experience_years": <number or null>,
  "domain_signals": [<explicit industry or product signals>],
  "education_requirements": [<explicit education requirements>],
  "resume_keywords": [<keywords worth reflecting in a tailored resume>]
}"""


class LlmRequirementExtraction(BaseModel):
    """One grounded requirement from an LLM JD-intelligence response."""

    text: str
    category: str = "other"
    candidate_match: str = "unknown"
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)
    rationale: str = ""

    @field_validator("text", "rationale")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        return " ".join((value or "").split()).strip()[:300]

    @field_validator("category")
    @classmethod
    def _category(cls, value: str) -> str:
        allowed = {
            "skill",
            "experience",
            "responsibility",
            "domain",
            "education",
            "location",
            "work_authorization",
            "sponsorship",
            "salary",
            "relocation",
            "notice_period",
            "legal",
            "demographic",
            "other",
        }
        cleaned = (value or "other").strip().lower().replace(" ", "_").replace("-", "_")
        return cleaned if cleaned in allowed else "other"

    @field_validator("candidate_match")
    @classmethod
    def _candidate_match(cls, value: str) -> str:
        allowed = {"matched", "partial", "unsupported", "unknown"}
        cleaned = (value or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "supported": "matched",
            "match": "matched",
            "partially_matched": "partial",
            "partial_match": "partial",
            "missing": "unsupported",
            "not_supported": "unsupported",
            "unclear": "unknown",
        }
        cleaned = aliases.get(cleaned, cleaned)
        return cleaned if cleaned in allowed else "unknown"

    @field_validator("evidence_refs")
    @classmethod
    def _evidence_refs(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in values or []:
            ref = str(item or "").strip()
            if ref and ref not in seen:
                out.append(ref[:120])
                seen.add(ref)
        return out


class LlmJobInsightExtraction(BaseModel):
    """Structured output for Phase 2C.1 JD intelligence."""

    role_summary: str = ""
    must_have_requirements: list[LlmRequirementExtraction] = Field(default_factory=list, max_length=20)
    preferred_requirements: list[LlmRequirementExtraction] = Field(default_factory=list, max_length=20)
    role_priorities: list[str] = Field(default_factory=list, max_length=12)
    grounded_fit_assessment: str = ""
    fit_verdict: str = "unknown"
    candidate_match_evidence_refs: list[str] = Field(default_factory=list, max_length=20)
    uncertainties: list[str] = Field(default_factory=list, max_length=12)
    gaps: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("role_summary", "grounded_fit_assessment")
    @classmethod
    def _short_text(cls, value: str) -> str:
        return " ".join((value or "").split()).strip()[:700]

    @field_validator("fit_verdict")
    @classmethod
    def _fit_verdict(cls, value: str) -> str:
        allowed = {"strong_match", "worth_applying", "stretch", "poor_fit", "unknown"}
        cleaned = (value or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "strong": "strong_match",
            "good_fit": "worth_applying",
            "worth_applying": "worth_applying",
            "poor": "poor_fit",
            "not_a_fit": "poor_fit",
        }
        cleaned = aliases.get(cleaned, cleaned)
        return cleaned if cleaned in allowed else "unknown"

    @field_validator("role_priorities", "candidate_match_evidence_refs", "uncertainties", "gaps")
    @classmethod
    def _clean_lists(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in values or []:
            cleaned = " ".join(str(item or "").split()).strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                out.append(cleaned[:200])
                seen.add(key)
        return out


APPLICATION_INTELLIGENCE_JSON_SPEC = """{
  "role_summary": "<brief summary of what this job actually does>",
  "must_have_requirements": [
    {
      "text": "<explicit must-have requirement>",
      "category": "skill" | "experience" | "responsibility" | "domain" | "education" | "location" | "work_authorization" | "sponsorship" | "salary" | "relocation" | "notice_period" | "legal" | "demographic" | "other",
      "candidate_match": "matched" | "partial" | "unsupported" | "unknown",
      "evidence_refs": ["<only refs from allowed_evidence_refs; empty if unsupported/unknown>"],
      "rationale": "<short grounded reason>"
    }
  ],
  "preferred_requirements": [<same shape as must_have_requirements>],
  "role_priorities": [<up to 8 role priorities inferred from the JD>],
  "grounded_fit_assessment": "<fit opinion grounded only in supplied deterministic signals and evidence>",
  "fit_verdict": "strong_match" | "worth_applying" | "stretch" | "poor_fit" | "unknown",
  "candidate_match_evidence_refs": ["<only refs from allowed_evidence_refs>"],
  "uncertainties": [<things unclear in the JD or candidate evidence>],
  "gaps": [<important unsupported requirements>]
}"""


class GroundedAnswerDraftExtraction(BaseModel):
    """One provider-drafted answer for a safe free-text application question."""

    question_key: str
    answer_text: str
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = Field(default=0.0, ge=0, le=1)
    review_required: bool = True

    @field_validator("question_key")
    @classmethod
    def _question_key(cls, value: str) -> str:
        return (value or "").strip().lower().replace(" ", "_").replace("-", "_")[:120]

    @field_validator("answer_text")
    @classmethod
    def _answer_text(cls, value: str) -> str:
        return " ".join((value or "").split()).strip()[:1600]

    @field_validator("evidence_refs")
    @classmethod
    def _evidence_refs(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in values or []:
            ref = str(item or "").strip()
            if ref and ref not in seen:
                out.append(ref[:120])
                seen.add(ref)
        return out


class GroundedAnswerDraftSetExtraction(BaseModel):
    """Structured output for Phase 2C.2 grounded answer drafting."""

    drafts: list[GroundedAnswerDraftExtraction] = Field(default_factory=list, max_length=12)


APPLICATION_ANSWER_DRAFT_JSON_SPEC = """{
  "drafts": [
    {
      "question_key": "<one of the supplied safe question keys>",
      "answer_text": "<concise grounded answer, using only the supplied job context and candidate evidence>",
      "evidence_refs": ["<only refs from allowed_evidence_refs>"],
      "confidence": <number from 0 to 1>,
      "review_required": <true if evidence is weak, answer is subjective, or human review is prudent>
    }
  ]
}"""


class ResumeWordingSuggestionExtraction(BaseModel):
    """One provider-suggested rewrite/reorder for selected resume content."""

    item_key: str
    action: str = "rewrite"
    rewritten_text: str
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("item_key", "action")
    @classmethod
    def _short_id(cls, value: str) -> str:
        return " ".join((value or "").split()).strip()[:160]

    @field_validator("rewritten_text")
    @classmethod
    def _text(cls, value: str) -> str:
        return " ".join((value or "").split()).strip()[:1200]

    @field_validator("evidence_refs")
    @classmethod
    def _evidence_refs(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in values or []:
            ref = str(item or "").strip()
            if ref and ref not in seen:
                out.append(ref[:120])
                seen.add(ref)
        return out


class ResumeWordingArtifactExtraction(BaseModel):
    """Structured output for Phase 2C.3 resume wording suggestions."""

    suggestions: list[ResumeWordingSuggestionExtraction] = Field(default_factory=list, max_length=20)


RESUME_WORDING_JSON_SPEC = """{
  "suggestions": [
    {
      "item_key": "<exact item_key supplied in selected_resume_items>",
      "action": "rewrite" | "reorder" | "shorten",
      "rewritten_text": "<improved wording using only the same selected evidence>",
      "evidence_refs": ["<only refs attached to this selected item>"]
    }
  ]
}"""
