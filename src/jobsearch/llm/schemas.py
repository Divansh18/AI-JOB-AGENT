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
