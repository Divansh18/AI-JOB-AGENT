"""Greenhouse attended autofill adapter."""

from __future__ import annotations

from .ats_autofill import (
    AutofillRuntimeError,
    BrowserAutofillAdapter,
    SubmissionGuardError,
    detect_application_fields,
    detect_human_intervention_required,
)
from .autofill import ATS_GREENHOUSE


class GreenhouseAutofillAdapter(BrowserAutofillAdapter):
    ats = ATS_GREENHOUSE
    form_not_detected_error = "greenhouse_form_not_detected"


def detect_greenhouse_fields(html: str):
    return detect_application_fields(html)


__all__ = [
    "AutofillRuntimeError",
    "GreenhouseAutofillAdapter",
    "SubmissionGuardError",
    "detect_greenhouse_fields",
    "detect_human_intervention_required",
]
