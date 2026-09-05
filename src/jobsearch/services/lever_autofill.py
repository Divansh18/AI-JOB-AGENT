"""Lever attended autofill adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .ats_autofill import (
    AutofillRuntimeError,
    BrowserAutofillAdapter,
    SubmissionGuardError,
    detect_application_fields,
    detect_human_intervention_required,
)
from .autofill import ATS_LEVER, AUTOFILL_STATUS_FAILED, AutofillResult


class LeverAutofillAdapter(BrowserAutofillAdapter):
    ats = ATS_LEVER
    form_not_detected_error = "lever_form_not_detected"

    def fill_page(
        self,
        page,
        plan: dict[str, Any],
        resume_path: Path,
        *,
        open_url: bool = False,
    ) -> AutofillResult:
        if open_url:
            page.goto(str(plan.get("application_url") or ""), wait_until="domcontentloaded", timeout=self.timeout_ms)
            try:
                page.wait_for_selector("form, input, textarea, select, a, button", timeout=self.timeout_ms)
            except Exception:
                pass

        html = page.content()
        state = detect_lever_page_state(html)
        should_open_form = state == "job_detail" or (state == "unknown" and not detect_human_intervention_required(html))
        if should_open_form:
            if not self._open_application_form(page):
                return self._failure_result(plan, ["lever_apply_button_not_detected"])
            try:
                page.wait_for_selector("form, input, textarea, select", timeout=self.timeout_ms)
            except Exception:
                pass
            if detect_lever_page_state(page.content()) != "application_form":
                return self._failure_result(plan, ["lever_application_form_not_detected_after_apply"])
        return super().fill_page(page, plan, resume_path, open_url=False)

    def _open_application_form(self, page) -> bool:
        for locator in _apply_action_locators(page):
            try:
                if locator.count() <= 0:
                    continue
                target = locator.first
                self.assert_not_submission_action(_LEVER_APPLY_CTA_LABEL)
                target.click()
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=min(self.timeout_ms, 5000))
                except Exception:
                    pass
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def assert_not_submission_action(label: str) -> None:
        if _FINAL_SUBMIT_RE.search(label or ""):
            raise SubmissionGuardError("lever autofill adapter is not allowed to invoke submission actions")

    def _failure_result(self, plan: dict[str, Any], errors: list[str]) -> AutofillResult:
        application_id = plan.get("application_id")
        return AutofillResult(
            application_id=int(application_id) if application_id is not None else None,
            ats=ATS_LEVER,
            url=str(plan.get("application_url") or ""),
            fields_detected=[],
            fields_filled=[],
            filled_fields=[],
            unresolved_fields=[],
            sensitive_fields=[],
            resume_attached=False,
            human_intervention_required=True,
            errors=errors,
            status=AUTOFILL_STATUS_FAILED,
            submitted=False,
        )


def detect_lever_fields(html: str):
    return detect_application_fields(html)


def detect_lever_page_state(html: str) -> str:
    if detect_lever_fields(html):
        return "application_form"
    if _has_apply_for_this_job_action(html):
        return "job_detail"
    return "unknown"


def _has_apply_for_this_job_action(html: str) -> bool:
    labels = re.findall(r"<(?:a|button)\b[^>]*>(.*?)</(?:a|button)>", html or "", re.I | re.S)
    for label in labels:
        text = re.sub(r"<[^>]+>", " ", label)
        if _LEVER_APPLY_CTA_RE.search(re.sub(r"\s+", " ", text).strip()):
            return True
    return False


def _apply_action_locators(page) -> list[Any]:
    locators: list[Any] = []
    for role in ("link", "button"):
        try:
            locators.append(page.get_by_role(role, name=_LEVER_APPLY_CTA_RE))
        except Exception:
            pass
    try:
        locators.append(page.locator("a, button").filter(has_text=_LEVER_APPLY_CTA_RE))
    except Exception:
        pass
    try:
        locators.append(page.locator('a[href*="apply"], a[href*="#"], button').filter(has_text=_LEVER_APPLY_CTA_RE))
    except Exception:
        pass
    return locators


_LEVER_APPLY_CTA_LABEL = "Apply for this job"
_LEVER_APPLY_CTA_RE = re.compile(r"^\s*apply\s+(?:for\s+this\s+(?:job|position|role)|now)\s*$", re.I)
_FINAL_SUBMIT_RE = re.compile(r"\b(submit|submit application|send application|final|confirm)\b", re.I)


__all__ = [
    "AutofillRuntimeError",
    "LeverAutofillAdapter",
    "SubmissionGuardError",
    "detect_lever_page_state",
    "detect_human_intervention_required",
    "detect_lever_fields",
]
