"""Ashby attended autofill adapter."""

from __future__ import annotations

from .ats_autofill import (
    AutofillRuntimeError,
    BrowserAutofillAdapter,
    SubmissionGuardError,
    detect_application_fields,
    detect_human_intervention_required,
)
from .autofill import ATS_ASHBY


class AshbyAutofillAdapter(BrowserAutofillAdapter):
    ats = ATS_ASHBY
    form_not_detected_error = "ashby_form_not_detected"


def detect_ashby_fields(html: str):
    return detect_application_fields(html)


__all__ = [
    "AshbyAutofillAdapter",
    "AutofillRuntimeError",
    "SubmissionGuardError",
    "detect_ashby_fields",
    "detect_human_intervention_required",
]
