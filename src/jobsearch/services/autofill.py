"""Attended browser autofill abstraction.

Concrete browser control stays behind this boundary so planning remains
deterministic and browser-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

from ..domain.application_planner import ApplicationPlan


ATS_GREENHOUSE = "greenhouse"
ATS_LEVER = "lever"
ATS_ASHBY = "ashby"
ATS_UNSUPPORTED = "unsupported_ats"
AUTOFILL_STATUS_FILLED_FOR_REVIEW = "filled_for_review"
AUTOFILL_STATUS_HUMAN_INTERVENTION_REQUIRED = "human_intervention_required"
AUTOFILL_STATUS_UNSUPPORTED_ATS = "unsupported_ats"
AUTOFILL_STATUS_FAILED = "failed"


@dataclass(frozen=True)
class DetectedApplicationField:
    name: str
    key: str | None = None
    label: str | None = None
    selector: str | None = None
    field_type: str = "text"
    required: bool = False
    sensitive: bool = False
    custom: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "key": self.key,
            "label": self.label,
            "selector": self.selector,
            "field_type": self.field_type,
            "required": self.required,
            "sensitive": self.sensitive,
            "custom": self.custom,
        }


@dataclass(frozen=True)
class AutofillResult:
    application_id: int | None = None
    ats: str = ATS_UNSUPPORTED
    url: str = ""
    fields_detected: list[dict] = field(default_factory=list)
    filled_fields: list[str] = field(default_factory=list)
    fields_filled: list[str] = field(default_factory=list)
    unresolved_fields: list[dict] = field(default_factory=list)
    sensitive_fields: list[dict] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    resume_attached: bool = False
    human_intervention_required: bool = False
    errors: list[str] = field(default_factory=list)
    status: str = AUTOFILL_STATUS_FAILED
    submitted: bool = False

    def as_dict(self) -> dict:
        fields_filled = self.fields_filled or self.filled_fields
        return {
            "application_id": self.application_id,
            "ats": self.ats,
            "url": self.url,
            "fields_detected": self.fields_detected,
            "fields_filled": fields_filled,
            "filled_fields": fields_filled,
            "resume_attached": self.resume_attached,
            "unresolved_fields": self.unresolved_fields,
            "sensitive_fields": self.sensitive_fields,
            "human_intervention_required": self.human_intervention_required,
            "errors": self.errors,
            "blockers": self.blockers,
            "status": self.status,
            "submitted": False,
        }


class AttendedAutofillSession(Protocol):
    def open_application_url(self, application_url: str) -> None:
        """Open the application URL for a human-attended session."""

    def detect_fields(self) -> list[DetectedApplicationField]:
        """Return detected fields without filling or submitting."""

    def fill_known_safe_fields(self, plan: ApplicationPlan) -> AutofillResult:
        """Fill only non-sensitive, verified fields from the plan."""

    def attach_resume(self, resume_path: str) -> AutofillResult:
        """Attach the selected one-page resume artifact."""

    def unresolved_fields(self, plan: ApplicationPlan) -> list[DetectedApplicationField]:
        """Report fields that need human input or review."""


def detect_ats(application_url: str) -> str:
    host = urlparse(application_url or "").netloc.lower()
    if _matches_host(host, "greenhouse.io"):
        return ATS_GREENHOUSE
    if _matches_host(host, "lever.co"):
        return ATS_LEVER
    if _matches_host(host, "ashbyhq.com"):
        return ATS_ASHBY
    return ATS_UNSUPPORTED


def _matches_host(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")
