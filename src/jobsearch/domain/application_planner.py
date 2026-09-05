"""Deterministic application planning primitives.

The planner consumes only verified candidate facts and answers. It does not
detect live web forms, infer sensitive answers, or submit applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .candidate import ApplicationProfile, VerifiedAnswer, normalize_key


class ApplicationPlanState(str, Enum):
    PLANNED = "planned"
    READY_FOR_REVIEW = "ready_for_review"
    SUBMITTED = "submitted"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class ApplicationPlanField:
    key: str
    label: str
    value: Any
    evidence_ref: str | None = None
    source: str | None = None
    autofill_safe: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "evidence_ref": self.evidence_ref,
            "source": self.source,
            "autofill_safe": self.autofill_safe,
        }


@dataclass(frozen=True)
class ApplicationPlanAnswer:
    question_key: str
    stored_question_key: str
    category: str
    answer_text: str
    evidence_refs: list[str] = field(default_factory=list)
    source: str | None = None
    human_review_required: bool = False
    autofill_safe: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_key": self.question_key,
            "stored_question_key": self.stored_question_key,
            "category": self.category,
            "answer_text": self.answer_text,
            "evidence_refs": self.evidence_refs,
            "source": self.source,
            "human_review_required": self.human_review_required,
            "autofill_safe": self.autofill_safe,
        }


@dataclass(frozen=True)
class ApplicationPlanUnresolved:
    key: str
    label: str
    kind: str
    reason: str
    human_review_required: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "reason": self.reason,
            "human_review_required": self.human_review_required,
        }


@dataclass(frozen=True)
class ApplicationPlanSensitiveField:
    question_key: str
    label: str
    status: str
    reason: str
    answer_text: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    human_review_required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_key": self.question_key,
            "label": self.label,
            "status": self.status,
            "reason": self.reason,
            "answer_text": self.answer_text,
            "evidence_refs": self.evidence_refs,
            "human_review_required": self.human_review_required,
        }


@dataclass(frozen=True)
class ApplicationPlan:
    application_id: int | None
    job_id: int
    company: str
    role: str
    application_url: str
    resume_id: int | None
    resume_path: str | None
    resume_decision: str | None
    resume_page_count: int | None
    resume_page_validation_status: str | None
    candidate_fields: list[ApplicationPlanField]
    known_answers: list[ApplicationPlanAnswer]
    unanswered_fields: list[ApplicationPlanUnresolved]
    sensitive_fields: list[ApplicationPlanSensitiveField]
    blockers: list[str]
    review_required: bool
    state: str
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "application_id": self.application_id,
            "job_id": self.job_id,
            "company": self.company,
            "role": self.role,
            "application_url": self.application_url,
            "resume_id": self.resume_id,
            "resume_path": self.resume_path,
            "resume_decision": self.resume_decision,
            "resume_page_count": self.resume_page_count,
            "resume_page_validation_status": self.resume_page_validation_status,
            "candidate_fields": [field.as_dict() for field in self.candidate_fields],
            "known_answers": [answer.as_dict() for answer in self.known_answers],
            "unanswered_fields": [field.as_dict() for field in self.unanswered_fields],
            "sensitive_fields": [field.as_dict() for field in self.sensitive_fields],
            "blockers": self.blockers,
            "review_required": self.review_required,
            "state": self.state,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


ANSWER_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "work_authorization": ("work_authorization", "right_to_work"),
    "sponsorship_required": ("sponsorship_required", "visa_sponsorship", "sponsorship"),
    "relocation": ("relocation", "willingness_to_relocate", "relocation_willingness"),
    "notice_period": ("notice_period",),
    "salary_expectation": ("salary_expectation", "salary_expectations", "compensation"),
    "earliest_start_date": ("earliest_start_date",),
    "disability": ("disability", "disability_status"),
    "veteran_status": ("veteran_status",),
    "gender_self_identification": ("gender", "gender_self_identification", "self_identification"),
    "criminal_history": ("criminal_history", "background_check"),
    "legal_attestation": ("legal_attestation", "legal_declaration", "attestation"),
}

REQUIRED_ANSWER_KEYS = (
    "work_authorization",
    "sponsorship_required",
    "relocation",
    "notice_period",
    "salary_expectation",
    "earliest_start_date",
)

SENSITIVE_REVIEW_KEYS = (
    "salary_expectation",
    "work_authorization",
    "sponsorship_required",
    "relocation",
    "disability",
    "veteran_status",
    "gender_self_identification",
    "criminal_history",
    "legal_attestation",
    "notice_period",
)

_ANSWER_LABELS = {
    "work_authorization": "Work authorization",
    "sponsorship_required": "Sponsorship required",
    "relocation": "Relocation",
    "notice_period": "Notice period",
    "salary_expectation": "Salary expectation",
    "earliest_start_date": "Earliest start date",
    "disability": "Disability status",
    "veteran_status": "Veteran status",
    "gender_self_identification": "Gender / self-identification",
    "criminal_history": "Criminal history",
    "legal_attestation": "Legal declaration / attestation",
}


def canonical_question_key(question_key: str) -> str:
    key = normalize_key(question_key)
    for canonical, aliases in ANSWER_KEY_ALIASES.items():
        if key in aliases:
            return canonical
    return key


def is_sensitive_application_question(question_key: str) -> bool:
    return canonical_question_key(question_key) in SENSITIVE_REVIEW_KEYS


def build_candidate_field_mappings(
    profile: ApplicationProfile,
) -> tuple[list[ApplicationPlanField], list[ApplicationPlanUnresolved]]:
    fields: list[ApplicationPlanField] = []
    unresolved: list[ApplicationPlanUnresolved] = []

    full_name_ref = _source_ref(profile.provenance.get("full_name"))
    full_name_source = _source_name(profile.provenance.get("full_name"))
    if profile.full_name:
        first_name, last_name = split_full_name(profile.full_name)
        fields.append(_field("full_name", "Full name", profile.full_name, full_name_ref, full_name_source))
        if first_name:
            fields.append(_field("first_name", "First name", first_name, full_name_ref, full_name_source))
        else:
            unresolved.append(_unresolved("first_name", "First name", "verified full_name is missing"))
        if last_name:
            fields.append(_field("last_name", "Last name", last_name, full_name_ref, full_name_source))
        else:
            unresolved.append(_unresolved("last_name", "Last name", "verified full_name has no separate last name"))
    else:
        for key, label in (
            ("full_name", "Full name"),
            ("first_name", "First name"),
            ("last_name", "Last name"),
        ):
            unresolved.append(_unresolved(key, label, "verified full_name is missing"))

    _add_profile_field(fields, unresolved, profile, "email", "Email", required=True)
    _add_profile_field(fields, unresolved, profile, "phone", "Phone", required=True)
    _add_profile_field(fields, unresolved, profile, "linkedin_url", "LinkedIn", required=True)
    _add_profile_field(fields, unresolved, profile, "github_url", "GitHub", required=True)
    _add_profile_field(fields, unresolved, profile, "portfolio_url", "Portfolio", required=False)
    _add_profile_field(fields, unresolved, profile, "current_location", "Current location", required=True)
    _add_notice_period(fields, unresolved, profile)
    _add_profile_field(fields, unresolved, profile, "earliest_start_date", "Earliest start date", required=True)
    return fields, unresolved


def build_known_answer_mappings(
    answers: list[VerifiedAnswer],
) -> tuple[list[ApplicationPlanAnswer], set[str]]:
    known: list[ApplicationPlanAnswer] = []
    covered: set[str] = set()
    for answer in sorted(answers, key=lambda item: (canonical_question_key(item.question_key), item.category)):
        if not answer.verified:
            continue
        canonical = canonical_question_key(answer.question_key)
        review_required = answer.human_review_required or is_sensitive_application_question(canonical)
        known.append(
            ApplicationPlanAnswer(
                question_key=canonical,
                stored_question_key=answer.question_key,
                category=answer.category,
                answer_text=answer.answer_text,
                evidence_refs=list(answer.evidence_refs),
                source=answer.source,
                human_review_required=review_required,
                autofill_safe=not review_required,
            )
        )
        covered.add(canonical)
    return known, covered


def build_unanswered_answer_fields(covered_answer_keys: set[str]) -> list[ApplicationPlanUnresolved]:
    return [
        ApplicationPlanUnresolved(
            key=key,
            label=_ANSWER_LABELS[key],
            kind="application_answer",
            reason="no verified candidate answer is stored",
            human_review_required=is_sensitive_application_question(key),
        )
        for key in REQUIRED_ANSWER_KEYS
        if key not in covered_answer_keys
    ]


def build_sensitive_fields(
    known_answers: list[ApplicationPlanAnswer],
    profile: ApplicationProfile,
) -> list[ApplicationPlanSensitiveField]:
    first_by_key: dict[str, ApplicationPlanAnswer] = {}
    for answer in known_answers:
        first_by_key.setdefault(answer.question_key, answer)

    fields: list[ApplicationPlanSensitiveField] = []
    for key in SENSITIVE_REVIEW_KEYS:
        answer = first_by_key.get(key)
        if answer is not None:
            fields.append(
                ApplicationPlanSensitiveField(
                    question_key=key,
                    label=_ANSWER_LABELS[key],
                    status="answered_verified_review_required",
                    reason="sensitive application question requires human review",
                    answer_text=answer.answer_text,
                    evidence_refs=answer.evidence_refs,
                )
            )
            continue
        if key == "notice_period" and (profile.notice_period_text or profile.notice_period_days is not None):
            continue
        fields.append(
            ApplicationPlanSensitiveField(
                question_key=key,
                label=_ANSWER_LABELS[key],
                status="unresolved_review_required",
                reason="sensitive application question cannot be inferred",
            )
        )
    return fields


def split_full_name(full_name: str) -> tuple[str | None, str | None]:
    parts = [part for part in full_name.strip().split() if part]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[-1]


def _add_notice_period(
    fields: list[ApplicationPlanField],
    unresolved: list[ApplicationPlanUnresolved],
    profile: ApplicationProfile,
) -> None:
    if profile.notice_period_text:
        prov = profile.provenance.get("notice_period_text")
        fields.append(
            _field("notice_period", "Notice period", profile.notice_period_text, _source_ref(prov), _source_name(prov))
        )
        return
    if profile.notice_period_days is not None:
        prov = profile.provenance.get("notice_period_days")
        fields.append(
            _field(
                "notice_period",
                "Notice period",
                f"{profile.notice_period_days} days",
                _source_ref(prov),
                _source_name(prov),
            )
        )
        return
    unresolved.append(
        ApplicationPlanUnresolved(
            key="notice_period",
            label="Notice period",
            kind="candidate_field",
            reason="verified notice period is missing",
            human_review_required=True,
        )
    )


def _add_profile_field(
    fields: list[ApplicationPlanField],
    unresolved: list[ApplicationPlanUnresolved],
    profile: ApplicationProfile,
    attr: str,
    label: str,
    *,
    required: bool,
) -> None:
    value = getattr(profile, attr)
    if value is None or value == "":
        if required:
            unresolved.append(_unresolved(attr, label, f"verified {attr} is missing"))
        return
    prov = profile.provenance.get(attr)
    fields.append(_field(attr, label, value, _source_ref(prov), _source_name(prov)))


def _field(
    key: str,
    label: str,
    value: Any,
    evidence_ref: str | None,
    source: str | None,
) -> ApplicationPlanField:
    return ApplicationPlanField(
        key=key,
        label=label,
        value=value,
        evidence_ref=evidence_ref,
        source=source,
        autofill_safe=True,
    )


def _unresolved(key: str, label: str, reason: str) -> ApplicationPlanUnresolved:
    return ApplicationPlanUnresolved(
        key=key,
        label=label,
        kind="candidate_field",
        reason=reason,
        human_review_required=False,
    )


def _source_ref(provenance: dict[str, Any] | None) -> str | None:
    if not provenance:
        return None
    fact_id = provenance.get("fact_id")
    if fact_id is not None:
        return f"fact:{fact_id}"
    selector = provenance.get("selector")
    return str(selector) if selector else None


def _source_name(provenance: dict[str, Any] | None) -> str | None:
    if not provenance:
        return None
    source = provenance.get("source")
    return str(source) if source else None
