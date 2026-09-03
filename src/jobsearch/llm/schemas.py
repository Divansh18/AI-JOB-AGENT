"""Structured output schemas.

Both providers must return objects validated against these exact models.
Validation is centralised in provider.validate_into() so the two paths cannot
drift apart.
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
