"""Shared deterministic browser autofill helpers for supported ATS adapters."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from selectolax.parser import HTMLParser

from ..domain.application_planner import canonical_question_key, is_sensitive_application_question
from .autofill import (
    ATS_UNSUPPORTED,
    AUTOFILL_STATUS_FAILED,
    AUTOFILL_STATUS_FILLED_FOR_REVIEW,
    AUTOFILL_STATUS_HUMAN_INTERVENTION_REQUIRED,
    AutofillResult,
    DetectedApplicationField,
)


SAFE_CANDIDATE_FIELD_KEYS = {
    "first_name",
    "last_name",
    "full_name",
    "email",
    "phone",
    "linkedin_url",
    "github_url",
    "portfolio_url",
    "current_location",
    "location",
}

_CAPTCHA_PATTERNS = (
    "g-recaptcha",
    "h-captcha",
    "hcaptcha",
    "cf-turnstile",
    "turnstile",
    "captcha",
    "bot check",
    "verify you are human",
)

_SUBMIT_TEXT_RE = re.compile(r"\b(submit application|submit|apply|send application)\b", re.I)
_NON_PERSON_NAME_RE = re.compile(r"\b(company|employer|institution|school|university|referrer)\s+name\b")


class AutofillRuntimeError(Exception):
    pass


class SubmissionGuardError(AutofillRuntimeError):
    pass


class BrowserAutofillAdapter:
    ats = ATS_UNSUPPORTED
    form_not_detected_error = "application_form_not_detected"

    def __init__(
        self,
        *,
        headless: bool = False,
        wait_for_review: bool = True,
        timeout_ms: int = 30000,
        slow_mo_ms: int = 50,
    ):
        self.headless = headless
        self.wait_for_review = wait_for_review
        self.timeout_ms = timeout_ms
        self.slow_mo_ms = slow_mo_ms

    def run(self, plan: dict[str, Any], resume_path: Path) -> AutofillResult:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise AutofillRuntimeError(
                "Playwright is not installed. Install project dependencies and run "
                "`playwright install chromium` before attended autofill."
            ) from exc

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=self.headless, slow_mo=self.slow_mo_ms)
            except Exception as exc:  # pragma: no cover - depends on local env
                raise AutofillRuntimeError(
                    "Playwright could not launch Chromium. Run `playwright install chromium`."
                ) from exc
            page = browser.new_page()
            try:
                result = self.fill_page(page, plan, resume_path, open_url=True)
                if self.wait_for_review:
                    input("Autofill stopped before submission. Review the browser, then press Enter to close it.")
                return result
            finally:
                if not self.wait_for_review:
                    browser.close()

    def detect_fields(self, html: str) -> list[DetectedApplicationField]:
        return detect_application_fields(html)

    def fill_page(
        self,
        page,
        plan: dict[str, Any],
        resume_path: Path,
        *,
        open_url: bool = False,
    ) -> AutofillResult:
        url = str(plan.get("application_url") or "")
        application_id = plan.get("application_id")
        if open_url:
            page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            try:
                page.wait_for_selector("form, input, textarea, select", timeout=self.timeout_ms)
            except Exception:
                pass

        html = page.content()
        fields = self.detect_fields(html)
        fields_by_selector = {field.selector: field for field in fields if field.selector}
        filled: list[str] = []
        unresolved: list[dict[str, Any]] = []
        sensitive: list[dict[str, Any]] = []
        errors: list[str] = []
        resume_attached = False
        human_intervention_required = False

        if not fields:
            errors.append(self.form_not_detected_error)
            human_intervention_required = True

        if detect_human_intervention_required(html):
            errors.append("captcha_or_bot_check_detected")
            human_intervention_required = True

        safe_values = candidate_values(plan)
        answer_values = answer_values_from_plan(plan)
        seen_unresolved: set[tuple[str, str]] = set()

        for field in fields:
            if not field.selector:
                continue
            if field.key == "resume" or field.field_type == "file":
                continue
            if field.sensitive:
                append_sensitive(sensitive, field, "sensitive question requires human review")
                continue
            if field.field_type in {"checkbox", "radio"}:
                append_unresolved(seen_unresolved, unresolved, field, "field type requires human review")
                continue
            value: Any | None = None
            if field.key in SAFE_CANDIDATE_FIELD_KEYS:
                value = safe_values.get(field.key)
            elif field.key in answer_values:
                value = answer_values[field.key]
            if value is None or value == "":
                append_unresolved(seen_unresolved, unresolved, field, "no verified autofill value supplied")
                continue
            try:
                self.fill_field(page, field, str(value))
            except Exception as exc:
                errors.append(f"fill_failed:{field.key or field.name}:{exc}")
                append_unresolved(seen_unresolved, unresolved, field, "field could not be filled automatically")
                continue
            filled.append(field.key or field.name)

        resume_fields = [field for field in fields if field.key == "resume" or field.field_type == "file"]
        if not resume_fields:
            errors.append("resume_upload_field_not_detected")
            human_intervention_required = True
        else:
            resume_field = resume_fields[0]
            try:
                self.attach_resume(page, resume_field, resume_path)
                resume_attached = True
            except Exception as exc:
                errors.append(f"resume_upload_failed:{exc}")
                human_intervention_required = True

        status = (
            AUTOFILL_STATUS_HUMAN_INTERVENTION_REQUIRED
            if human_intervention_required
            else AUTOFILL_STATUS_FILLED_FOR_REVIEW
        )
        if errors and not filled and not resume_attached:
            status = AUTOFILL_STATUS_FAILED

        return AutofillResult(
            application_id=int(application_id) if application_id is not None else None,
            ats=self.ats,
            url=url,
            fields_detected=[field.as_dict() for field in fields_by_selector.values()],
            fields_filled=filled,
            filled_fields=filled,
            unresolved_fields=unresolved,
            sensitive_fields=sensitive,
            resume_attached=resume_attached,
            human_intervention_required=human_intervention_required or bool(sensitive or unresolved),
            errors=errors,
            status=status,
            submitted=False,
        )

    def fill_field(self, page, field: DetectedApplicationField, value: str) -> None:
        self.assert_not_submission_action(field.label or field.name)
        locator = page.locator(field.selector)
        if field.field_type == "select":
            try:
                locator.select_option(label=value)
                return
            except Exception:
                locator.select_option(value=value)
                return
        locator.fill(value)

    def attach_resume(self, page, field: DetectedApplicationField, resume_path: Path) -> None:
        if not resume_path.exists():
            raise AutofillRuntimeError(f"resume file not found: {resume_path}")
        if not field.selector:
            raise AutofillRuntimeError("resume field has no selector")
        page.locator(field.selector).set_input_files(str(resume_path))

    @staticmethod
    def assert_not_submission_action(label: str) -> None:
        if _SUBMIT_TEXT_RE.search(label or ""):
            raise SubmissionGuardError("autofill adapter is not allowed to invoke submission actions")


def detect_application_fields(html: str) -> list[DetectedApplicationField]:
    tree = HTMLParser(html or "")
    labels = labels_by_target(tree)
    fields: list[DetectedApplicationField] = []
    seen: set[str] = set()
    for node in tree.css("input, textarea, select"):
        attrs = node.attributes
        tag = node.tag or ""
        field_type = field_type_for(tag, attrs)
        if field_type in {"hidden", "submit", "button", "reset"}:
            continue
        selector = selector_for(attrs)
        if not selector or selector in seen:
            continue
        seen.add(selector)
        name = str(attrs.get("name") or attrs.get("id") or selector)
        label = label_for(node, attrs, labels)
        probe = " ".join(str(part or "") for part in (name, attrs.get("id"), label, attrs.get("placeholder")))
        key = field_key(probe, field_type)
        sensitive = is_sensitive_probe(probe, key)
        fields.append(
            DetectedApplicationField(
                name=name,
                key=key,
                label=label,
                selector=selector,
                field_type=field_type,
                required=("required" in attrs or str(attrs.get("aria-required", "")).lower() == "true"),
                sensitive=sensitive,
                custom=key not in SAFE_CANDIDATE_FIELD_KEYS and key != "resume",
            )
        )
    return fields


def detect_human_intervention_required(html: str) -> bool:
    lowered = (html or "").lower()
    return any(pattern in lowered for pattern in _CAPTCHA_PATTERNS)


def labels_by_target(tree: HTMLParser) -> dict[str, str]:
    labels: dict[str, str] = {}
    for label in tree.css("label"):
        target = label.attributes.get("for")
        text = clean_text(label.text(separator=" "))
        if target and text:
            labels[str(target)] = text
    return labels


def label_for(node, attrs: dict[str, Any], labels: dict[str, str]) -> str | None:
    node_id = attrs.get("id")
    if node_id and str(node_id) in labels:
        return labels[str(node_id)]
    aria = attrs.get("aria-label")
    if aria:
        return clean_text(str(aria))
    placeholder = attrs.get("placeholder")
    if placeholder:
        return clean_text(str(placeholder))
    parent = node.parent
    if parent is not None and parent.tag == "label":
        text = clean_text(parent.text(separator=" "))
        if text:
            return text
    return None


def field_type_for(tag: str, attrs: dict[str, Any]) -> str:
    if tag == "textarea":
        return "textarea"
    if tag == "select":
        return "select"
    return str(attrs.get("type") or "text").strip().lower()


def selector_for(attrs: dict[str, Any]) -> str | None:
    node_id = attrs.get("id")
    if node_id:
        node_id = str(node_id)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*$", node_id):
            return f"#{css_escape(node_id)}"
        return f'[id="{css_attr_escape(node_id)}"]'
    name = attrs.get("name")
    if name:
        return f'[name="{css_attr_escape(str(name))}"]'
    return None


def field_key(probe: str, field_type: str) -> str | None:
    text = normalize_probe(probe)
    if field_type == "file" and re.search(r"\b(resume|cv|curriculum vitae)\b", text):
        return "resume"
    patterns = (
        ("first_name", r"\b(first name|given name|firstname|first_name)\b"),
        ("last_name", r"\b(last name|family name|surname|lastname|last_name)\b"),
        ("email", r"\b(email|e-mail)\b"),
        ("phone", r"\b(phone|mobile|telephone)\b"),
        ("linkedin_url", r"\blinkedin\b"),
        ("github_url", r"\bgithub\b"),
        ("portfolio_url", r"\b(portfolio|personal website|website|url)\b"),
        ("current_location", r"\b(current location|location|city|where are you located)\b"),
        ("work_authorization", r"\b(work authorization|authorized to work|right to work|right-to-work)\b"),
        ("sponsorship_required", r"\b(sponsorship|visa)\b"),
        ("salary_expectation", r"\b(salary|compensation|expected ctc|ctc|pay)\b"),
        ("relocation", r"\b(relocat|move to)\b"),
        ("notice_period", r"\b(notice period|available to start|availability)\b"),
        ("earliest_start_date", r"\b(earliest start|start date|when can you start)\b"),
        ("disability", r"\bdisability\b"),
        ("veteran_status", r"\bveteran\b"),
        ("gender_self_identification", r"\b(gender|self-identification|self identification)\b"),
        ("criminal_history", r"\b(criminal|conviction|background check)\b"),
        ("legal_attestation", r"\b(attest|certify|legal|declaration)\b"),
        (
            "why_are_you_interested_in_this_role",
            r"\b(why are you interested|why.*this role|why.*this company|interest in this role)\b",
        ),
        ("short_experience_summary", r"\b(briefly describe your experience|experience summary|short summary)\b"),
    )
    for key, pattern in patterns:
        if re.search(pattern, text):
            return key
    if not _NON_PERSON_NAME_RE.search(text) and re.search(r"\b(full name|legal name|name)\b", text):
        return "full_name"
    return None


def is_sensitive_probe(probe: str, key: str | None) -> bool:
    if key and is_sensitive_application_question(key):
        return True
    text = normalize_probe(probe)
    return any(
        token in text
        for token in (
            "salary",
            "compensation",
            "visa",
            "sponsorship",
            "relocat",
            "notice period",
            "disability",
            "veteran",
            "gender",
            "criminal",
            "attest",
            "legal",
        )
    )


def candidate_values(plan: dict[str, Any]) -> dict[str, Any]:
    values = {
        field.get("key"): field.get("value")
        for field in (plan.get("candidate_fields") or [])
        if field.get("autofill_safe") and field.get("key")
    }
    if "current_location" in values:
        values["location"] = values["current_location"]
    return values


def answer_values_from_plan(plan: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for answer in plan.get("known_answers") or []:
        key = canonical_question_key(str(answer.get("question_key") or ""))
        if not key or answer.get("human_review_required") or not answer.get("autofill_safe", True):
            continue
        values[key] = str(answer.get("answer_text") or "")
    return values


def append_unresolved(
    seen: set[tuple[str, str]],
    unresolved: list[dict[str, Any]],
    field: DetectedApplicationField,
    reason: str,
) -> None:
    key = (field.key or field.name, field.selector or field.name)
    if key in seen:
        return
    seen.add(key)
    unresolved.append(
        {
            "key": field.key,
            "name": field.name,
            "label": field.label,
            "selector": field.selector,
            "reason": reason,
            "human_review_required": field.sensitive,
        }
    )


def append_sensitive(
    sensitive: list[dict[str, Any]],
    field: DetectedApplicationField,
    reason: str,
) -> None:
    sensitive.append(
        {
            "key": field.key,
            "name": field.name,
            "label": field.label,
            "selector": field.selector,
            "reason": reason,
            "human_review_required": True,
        }
    )


def normalize_probe(value: str) -> str:
    return re.sub(r"[_\-\[\]\(\):]+", " ", (value or "").lower())


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def css_escape(value: str) -> str:
    return re.sub(r"([^a-zA-Z0-9_-])", lambda match: "\\" + match.group(1), value)


def css_attr_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
