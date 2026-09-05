"""Attended browser autofill abstraction.

Phase 2A defines the boundary only. Concrete browser control belongs in a
later phase and must remain outside deterministic planner logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..domain.application_planner import ApplicationPlan


@dataclass(frozen=True)
class DetectedApplicationField:
    name: str
    label: str | None = None
    field_type: str = "text"
    required: bool = False
    sensitive: bool = False


@dataclass(frozen=True)
class AutofillResult:
    filled_fields: list[str] = field(default_factory=list)
    unresolved_fields: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    resume_attached: bool = False
    submitted: bool = False


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
