"""Verified candidate facts, answers, and application profile assembly.

This module stays pure: it normalizes and validates truth-store data, and it
builds the downstream application profile strictly from verified facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CandidateCategory(str, Enum):
    PERSONAL_PROFILE = "personal_profile"
    WORK_EXPERIENCE = "work_experience"
    PROJECTS = "projects"
    EDUCATION = "education"
    SKILLS = "skills"
    LINKS = "links"
    AVAILABILITY = "availability"
    WORK_AUTHORIZATION = "work_authorization"
    LOCATION_PREFERENCES = "location_preferences"
    EVIDENCE = "evidence"


SENSITIVE_QUESTION_KEYS = {
    "salary_expectations",
    "work_authorization",
    "willingness_to_relocate",
    "relocation_willingness",
    "notice_period",
    "earliest_start_date",
    "right_to_work",
    "visa_sponsorship",
    "why_are_you_interested_in_this_role",
}

PROFILE_FIELD_SELECTORS: dict[str, list[tuple[str, str]]] = {
    "full_name": [("personal_profile", "full_name")],
    "email": [("personal_profile", "email")],
    "phone": [("personal_profile", "phone")],
    "current_location": [("personal_profile", "current_location")],
    "linkedin_url": [("links", "linkedin_url")],
    "github_url": [("links", "github_url")],
    "portfolio_url": [("links", "portfolio_url")],
    "notice_period_days": [("availability", "notice_period_days")],
    "notice_period_text": [("availability", "notice_period_text")],
    "earliest_start_date": [("availability", "earliest_start_date")],
    "years_experience": [("personal_profile", "years_experience")],
    "experience_stage": [("personal_profile", "experience_stage")],
    "work_authorization": [("work_authorization", "summary")],
    "relocation_willing": [("location_preferences", "relocation_willing")],
    "preferred_locations": [("location_preferences", "preferred_locations")],
}

SINGLETON_SELECTORS = {
    f"{category}:{key}"
    for selectors in PROFILE_FIELD_SELECTORS.values()
    for category, key in selectors
}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+?[0-9()\-\s]{7,20}$")
URL_RE = re.compile(r"^https?://[^\s]+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_CATEGORY_ALIASES = {
    "personal": CandidateCategory.PERSONAL_PROFILE.value,
    "profile": CandidateCategory.PERSONAL_PROFILE.value,
    "personal_profile": CandidateCategory.PERSONAL_PROFILE.value,
    "work_experience": CandidateCategory.WORK_EXPERIENCE.value,
    "experience": CandidateCategory.WORK_EXPERIENCE.value,
    "projects": CandidateCategory.PROJECTS.value,
    "project": CandidateCategory.PROJECTS.value,
    "education": CandidateCategory.EDUCATION.value,
    "skills": CandidateCategory.SKILLS.value,
    "links": CandidateCategory.LINKS.value,
    "availability": CandidateCategory.AVAILABILITY.value,
    "work_authorization": CandidateCategory.WORK_AUTHORIZATION.value,
    "location_preferences": CandidateCategory.LOCATION_PREFERENCES.value,
    "relocation": CandidateCategory.LOCATION_PREFERENCES.value,
    "evidence": CandidateCategory.EVIDENCE.value,
    "achievements": CandidateCategory.EVIDENCE.value,
    "evidence_bullets": CandidateCategory.EVIDENCE.value,
    "reusable_achievements_evidence_bullets": CandidateCategory.EVIDENCE.value,
}


@dataclass(frozen=True)
class CandidateFact:
    category: str
    key: str
    value: Any
    source: str
    verified: bool
    confidence: float | None = None
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def selector(self) -> str:
        return f"{self.category}:{self.key}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "verified": self.verified,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass(frozen=True)
class VerifiedAnswer:
    question_key: str
    answer_text: str
    category: str = "general"
    source: str = "manual"
    evidence_refs: list[str] = field(default_factory=list)
    verified: bool = False
    human_review_required: bool = False
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question_key": self.question_key,
            "category": self.category,
            "answer_text": self.answer_text,
            "source": self.source,
            "evidence_refs": self.evidence_refs,
            "verified": self.verified,
            "human_review_required": self.human_review_required,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass(frozen=True)
class ApplicationProfile:
    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    current_location: str | None = None
    linkedin_url: str | None = None
    github_url: str | None = None
    portfolio_url: str | None = None
    notice_period_days: int | None = None
    notice_period_text: str | None = None
    earliest_start_date: str | None = None
    years_experience: float | None = None
    experience_stage: str | None = None
    work_authorization: str | None = None
    relocation_willing: bool | None = None
    preferred_locations: list[str] | None = None
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "current_location": self.current_location,
            "linkedin_url": self.linkedin_url,
            "github_url": self.github_url,
            "portfolio_url": self.portfolio_url,
            "notice_period_days": self.notice_period_days,
            "notice_period_text": self.notice_period_text,
            "earliest_start_date": self.earliest_start_date,
            "years_experience": self.years_experience,
            "experience_stage": self.experience_stage,
            "work_authorization": self.work_authorization,
            "relocation_willing": self.relocation_willing,
            "preferred_locations": self.preferred_locations,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"level": self.level, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class CandidateValidationReport:
    issues: list[ValidationIssue]
    fact_count: int
    verified_fact_count: int
    answer_count: int
    verified_answer_count: int
    profile: ApplicationProfile

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "fact_count": self.fact_count,
            "verified_fact_count": self.verified_fact_count,
            "answer_count": self.answer_count,
            "verified_answer_count": self.verified_answer_count,
            "issues": [issue.as_dict() for issue in self.issues],
            "profile": self.profile.as_dict(),
        }


@dataclass(frozen=True)
class CandidateImportBundle:
    version: int
    facts: list[CandidateFact] = field(default_factory=list)
    answers: list[VerifiedAnswer] = field(default_factory=list)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")


def normalize_category(value: str) -> str:
    slug = _slug(value)
    if not slug:
        return ""
    return _CATEGORY_ALIASES.get(slug, slug)


def normalize_key(value: str) -> str:
    return _slug(value)


def is_sensitive_question(question_key: str) -> bool:
    return normalize_key(question_key) in SENSITIVE_QUESTION_KEYS


def make_fact(
    *,
    category: str,
    key: str,
    value: Any,
    source: str,
    verified: bool,
    confidence: float | None = None,
    fact_id: int | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> CandidateFact:
    return CandidateFact(
        category=normalize_category(category),
        key=normalize_key(key),
        value=value,
        source=(source or "").strip(),
        verified=bool(verified),
        confidence=confidence,
        id=fact_id,
        created_at=created_at,
        updated_at=updated_at,
    )


def make_answer(
    *,
    question_key: str,
    answer_text: str,
    category: str = "general",
    source: str = "manual",
    evidence_refs: list[str] | None = None,
    verified: bool = False,
    human_review_required: bool = False,
    answer_id: int | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> VerifiedAnswer:
    key = normalize_key(question_key)
    review_required = bool(human_review_required) or is_sensitive_question(key)
    refs = [str(ref).strip() for ref in (evidence_refs or []) if str(ref).strip()]
    return VerifiedAnswer(
        question_key=key,
        answer_text=(answer_text or "").strip(),
        category=normalize_key(category or "general") or "general",
        source=(source or "").strip(),
        evidence_refs=refs,
        verified=bool(verified),
        human_review_required=review_required,
        id=answer_id,
        created_at=created_at,
        updated_at=updated_at,
    )


def bundle_from_payload(payload: dict[str, Any]) -> CandidateImportBundle:
    version = int(payload.get("version", 0))
    facts = [
        make_fact(
            category=item.get("category", ""),
            key=item.get("key") or item.get("type", ""),
            value=item["value"] if "value" in item else item.get("content"),
            source=item.get("source", "import_json"),
            verified=item.get("verified", False),
            confidence=item.get("confidence"),
        )
        for item in payload.get("facts", [])
    ]
    answers = [
        make_answer(
            question_key=item.get("question_key") or item.get("key", ""),
            category=item.get("category", "general"),
            answer_text=item.get("answer_text") or item.get("answer", ""),
            source=item.get("source", "import_json"),
            evidence_refs=item.get("evidence_refs") or item.get("source_refs") or [],
            verified=item.get("verified", False),
            human_review_required=item.get("human_review_required", False),
        )
        for item in payload.get("answers", [])
    ]
    return CandidateImportBundle(version=version, facts=facts, answers=answers)


def _valid_date(value: str) -> bool:
    if not DATE_RE.fullmatch(value):
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def _list_of_strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def _issue(level: str, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(level=level, code=code, message=message)


def _validate_scalar(selector: str, value: Any) -> list[ValidationIssue]:
    if selector == "personal_profile:full_name":
        return [] if isinstance(value, str) and value.strip() else [
            _issue("error", "invalid_full_name", "full_name must be a non-empty string")
        ]
    if selector == "personal_profile:email":
        return [] if isinstance(value, str) and EMAIL_RE.fullmatch(value.strip()) else [
            _issue("error", "invalid_email", "email must be a valid email address")
        ]
    if selector == "personal_profile:phone":
        return [] if isinstance(value, str) and PHONE_RE.fullmatch(value.strip()) else [
            _issue("error", "invalid_phone", "phone must look like a phone number")
        ]
    if selector in {
        "personal_profile:current_location",
        "availability:notice_period_text",
        "personal_profile:experience_stage",
        "work_authorization:summary",
    }:
        return [] if isinstance(value, str) and value.strip() else [
            _issue("error", "invalid_text_value", f"{selector} must be a non-empty string")
        ]
    if selector in {"links:linkedin_url", "links:github_url", "links:portfolio_url"}:
        return [] if isinstance(value, str) and URL_RE.fullmatch(value.strip()) else [
            _issue("error", "invalid_url", f"{selector} must be an http(s) URL")
        ]
    if selector == "availability:notice_period_days":
        return [] if isinstance(value, int) and value >= 0 else [
            _issue("error", "invalid_notice_period", "notice_period_days must be a non-negative integer")
        ]
    if selector == "availability:earliest_start_date":
        return [] if isinstance(value, str) and _valid_date(value.strip()) else [
            _issue("error", "invalid_start_date", "earliest_start_date must be YYYY-MM-DD")
        ]
    if selector == "personal_profile:years_experience":
        return [] if isinstance(value, (int, float)) and 0 <= float(value) <= 50 else [
            _issue("error", "invalid_years_experience", "years_experience must be between 0 and 50")
        ]
    if selector == "location_preferences:relocation_willing":
        return [] if isinstance(value, bool) else [
            _issue("error", "invalid_relocation_flag", "relocation_willing must be true or false")
        ]
    if selector == "location_preferences:preferred_locations":
        return [] if _list_of_strings(value) else [
            _issue("error", "invalid_preferred_locations", "preferred_locations must be a list of strings")
        ]
    return []


def _validate_work_experience(value: Any) -> list[ValidationIssue]:
    if not isinstance(value, dict):
        return [_issue("error", "invalid_work_experience", "work_experience value must be an object")]
    issues: list[ValidationIssue] = []
    if not isinstance(value.get("company"), str) or not value["company"].strip():
        issues.append(_issue("error", "missing_experience_company", "work_experience requires company"))
    title = value.get("title") or value.get("role")
    if not isinstance(title, str) or not title.strip():
        issues.append(_issue("error", "missing_experience_title", "work_experience requires title or role"))
    for date_key in ("start_date", "end_date"):
        if value.get(date_key) is not None:
            if not isinstance(value[date_key], str) or not _valid_date(value[date_key].strip()):
                issues.append(_issue("error", "invalid_experience_date", f"{date_key} must be YYYY-MM-DD"))
    if value.get("achievements") is not None and not _list_of_strings(value["achievements"]):
        issues.append(_issue("error", "invalid_experience_achievements", "achievements must be a list of strings"))
    return issues


def _validate_project(value: Any) -> list[ValidationIssue]:
    if not isinstance(value, dict):
        return [_issue("error", "invalid_project", "project value must be an object")]
    issues: list[ValidationIssue] = []
    if not isinstance(value.get("name"), str) or not value["name"].strip():
        issues.append(_issue("error", "missing_project_name", "project requires name"))
    if value.get("link") is not None:
        link = value["link"]
        if not isinstance(link, str) or not URL_RE.fullmatch(link.strip()):
            issues.append(_issue("error", "invalid_project_link", "project link must be an http(s) URL"))
    if value.get("skills") is not None and not _list_of_strings(value["skills"]):
        issues.append(_issue("error", "invalid_project_skills", "project skills must be a list of strings"))
    return issues


def _validate_education(value: Any) -> list[ValidationIssue]:
    if not isinstance(value, dict):
        return [_issue("error", "invalid_education", "education value must be an object")]
    issues: list[ValidationIssue] = []
    if not isinstance(value.get("institution"), str) or not value["institution"].strip():
        issues.append(_issue("error", "missing_education_institution", "education requires institution"))
    degree = value.get("degree") or value.get("program")
    if not isinstance(degree, str) or not degree.strip():
        issues.append(_issue("error", "missing_education_degree", "education requires degree or program"))
    for date_key in ("start_date", "end_date"):
        if value.get(date_key) is not None:
            if not isinstance(value[date_key], str) or not _valid_date(value[date_key].strip()):
                issues.append(_issue("error", "invalid_education_date", f"{date_key} must be YYYY-MM-DD"))
    return issues


def _validate_skills(value: Any) -> list[ValidationIssue]:
    if isinstance(value, str) and value.strip():
        return []
    if _list_of_strings(value):
        return []
    if isinstance(value, dict) and isinstance(value.get("name"), str) and value["name"].strip():
        return []
    return [_issue("error", "invalid_skills_value", "skills value must be a string, list of strings, or object with name")]


def _validate_evidence(value: Any) -> list[ValidationIssue]:
    if isinstance(value, str) and value.strip():
        return []
    if isinstance(value, dict) and isinstance(value.get("bullet"), str) and value["bullet"].strip():
        return []
    return [_issue("error", "invalid_evidence_value", "evidence value must be a string or object with bullet")]


def validate_fact(fact: CandidateFact) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if fact.category not in {c.value for c in CandidateCategory}:
        issues.append(_issue("error", "invalid_category", f"unknown category {fact.category!r}"))
    if not fact.key:
        issues.append(_issue("error", "missing_key", "fact key is required"))
    if not fact.source:
        issues.append(_issue("error", "missing_source", "fact source is required"))
    if fact.confidence is not None and not (0.0 <= fact.confidence <= 1.0):
        issues.append(_issue("error", "invalid_confidence", "confidence must be between 0.0 and 1.0"))
    if fact.value is None:
        issues.append(_issue("error", "missing_value", "fact value is required"))

    issues.extend(_validate_scalar(fact.selector, fact.value))

    if fact.category == CandidateCategory.WORK_EXPERIENCE.value:
        issues.extend(_validate_work_experience(fact.value))
    elif fact.category == CandidateCategory.PROJECTS.value:
        issues.extend(_validate_project(fact.value))
    elif fact.category == CandidateCategory.EDUCATION.value:
        issues.extend(_validate_education(fact.value))
    elif fact.category == CandidateCategory.SKILLS.value:
        issues.extend(_validate_skills(fact.value))
    elif fact.category == CandidateCategory.EVIDENCE.value:
        issues.extend(_validate_evidence(fact.value))

    return issues


def validate_answer(answer: VerifiedAnswer) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not answer.question_key:
        issues.append(_issue("error", "missing_question_key", "question_key is required"))
    if not answer.category:
        issues.append(_issue("error", "missing_answer_category", "answer category is required"))
    if not answer.answer_text:
        issues.append(_issue("error", "missing_answer_text", "answer_text is required"))
    if not answer.source:
        issues.append(_issue("error", "missing_answer_source", "answer source is required"))
    if any(not isinstance(ref, str) or not ref.strip() for ref in answer.evidence_refs):
        issues.append(_issue("error", "invalid_evidence_refs", "evidence_refs must contain non-empty strings"))
    if is_sensitive_question(answer.question_key) and not answer.human_review_required:
        issues.append(_issue("error", "sensitive_answer_requires_review",
                             f"{answer.question_key} must always require human review"))
    return issues


def build_application_profile(facts: list[CandidateFact]) -> ApplicationProfile:
    verified = {fact.selector: fact for fact in facts if fact.verified}
    values: dict[str, Any] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for field_name, selectors in PROFILE_FIELD_SELECTORS.items():
        chosen: CandidateFact | None = None
        for category, key in selectors:
            chosen = verified.get(f"{category}:{key}")
            if chosen is not None:
                break
        values[field_name] = chosen.value if chosen is not None else None
        if chosen is not None:
            provenance[field_name] = {
                "fact_id": chosen.id,
                "selector": chosen.selector,
                "source": chosen.source,
            }
    values["provenance"] = provenance
    return ApplicationProfile(**values)


def validate_store(facts: list[CandidateFact], answers: list[VerifiedAnswer]) -> CandidateValidationReport:
    issues: list[ValidationIssue] = []
    for fact in facts:
        issues.extend(validate_fact(fact))
    for answer in answers:
        issues.extend(validate_answer(answer))

    verified_counts: dict[str, int] = {}
    for fact in facts:
        if fact.verified and fact.selector in SINGLETON_SELECTORS:
            verified_counts[fact.selector] = verified_counts.get(fact.selector, 0) + 1
    for selector, count in verified_counts.items():
        if count > 1:
            issues.append(_issue("error", "duplicate_singleton_field",
                                 f"{selector} has {count} verified records; expected at most one"))

    profile = build_application_profile(facts)
    verified_selectors = {fact.selector for fact in facts if fact.verified}
    for field_name, selectors in PROFILE_FIELD_SELECTORS.items():
        chosen = getattr(profile, field_name)
        if not any(f"{category}:{key}" in verified_selectors for category, key in selectors):
            if chosen is not None:
                issues.append(_issue("error", "inferred_profile_field",
                                     f"{field_name} was populated without verified source data"))

    for field_name, prov in profile.provenance.items():
        selector = prov.get("selector", "")
        if selector not in verified_selectors:
            issues.append(_issue("error", "unverified_profile_leak",
                                 f"{field_name} references unverified selector {selector}"))

    return CandidateValidationReport(
        issues=issues,
        fact_count=len(facts),
        verified_fact_count=sum(1 for fact in facts if fact.verified),
        answer_count=len(answers),
        verified_answer_count=sum(1 for answer in answers if answer.verified),
        profile=profile,
    )
