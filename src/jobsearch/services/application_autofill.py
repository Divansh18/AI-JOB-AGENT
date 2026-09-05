"""Orchestrate attended application autofill runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config.loader import AppConfig
from ..persistence.repositories import ApplicationAutofillRunRepo
from . import application_planner
from .audit import Audit
from .autofill import (
    ATS_ASHBY,
    ATS_GREENHOUSE,
    ATS_LEVER,
    ATS_UNSUPPORTED,
    AUTOFILL_STATUS_FAILED,
    AUTOFILL_STATUS_UNSUPPORTED_ATS,
    AutofillResult,
    detect_ats,
)
from .ashby_autofill import AshbyAutofillAdapter
from .greenhouse_autofill import GreenhouseAutofillAdapter
from .lever_autofill import LeverAutofillAdapter


class ApplicationAutofillError(Exception):
    pass


def run_attended_autofill(
    conn,
    config: AppConfig,
    application_id: int,
    *,
    headless: bool = False,
    wait_for_review: bool = True,
    timeout_ms: int = 30000,
) -> dict[str, Any]:
    plan = application_planner.get_application_plan(conn, application_id)
    if plan.get("blockers"):
        return _save_result(
            conn,
            AutofillResult(
                application_id=application_id,
                ats=detect_ats(plan["application_url"]),
                url=plan["application_url"],
                status=AUTOFILL_STATUS_FAILED,
                human_intervention_required=True,
                errors=[f"plan_blocked:{blocker}" for blocker in plan["blockers"]],
                submitted=False,
            ),
        )

    ats = detect_ats(plan["application_url"])
    adapter_cls = _adapter_for_ats(ats)
    if adapter_cls is None:
        return _save_result(
            conn,
            AutofillResult(
                application_id=application_id,
                ats=ATS_UNSUPPORTED,
                url=plan["application_url"],
                status=AUTOFILL_STATUS_UNSUPPORTED_ATS,
                human_intervention_required=True,
                errors=[f"unsupported_ats:{ats}"],
                submitted=False,
            ),
        )

    resume_error = _validate_resume_plan(config.root, plan)
    if resume_error:
        return _save_result(
            conn,
            AutofillResult(
                application_id=application_id,
                ats=ats,
                url=plan["application_url"],
                status=AUTOFILL_STATUS_FAILED,
                human_intervention_required=True,
                errors=[resume_error],
                submitted=False,
            ),
        )

    adapter = adapter_cls(
        headless=headless,
        wait_for_review=wait_for_review,
        timeout_ms=timeout_ms,
    )
    try:
        result = adapter.run(plan, _resolve_resume_path(config.root, plan["resume_path"]))
    except Exception as exc:
        result = AutofillResult(
            application_id=application_id,
            ats=ats,
            url=plan["application_url"],
            status=AUTOFILL_STATUS_FAILED,
            human_intervention_required=True,
            errors=[str(exc)],
            submitted=False,
        )
    return _save_result(conn, result)


def _adapter_for_ats(ats: str):
    if ats == ATS_GREENHOUSE:
        return GreenhouseAutofillAdapter
    if ats == ATS_LEVER:
        return LeverAutofillAdapter
    if ats == ATS_ASHBY:
        return AshbyAutofillAdapter
    return None


def _validate_resume_plan(root: Path, plan: dict[str, Any]) -> str | None:
    if plan.get("resume_page_validation_status") != "valid":
        return f"invalid_resume: page_validation_status={plan.get('resume_page_validation_status')}"
    if plan.get("resume_page_count") != 1:
        return f"invalid_resume: expected page_count=1, got {plan.get('resume_page_count')}"
    resume_path = plan.get("resume_path")
    if not resume_path:
        return "invalid_resume: missing resume_path"
    if not _resolve_resume_path(root, resume_path).exists():
        return f"invalid_resume: file not found: {resume_path}"
    return None


def _resolve_resume_path(root: Path, resume_path: str) -> Path:
    path = Path(resume_path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _save_result(conn, result: AutofillResult) -> dict[str, Any]:
    payload = result.as_dict()
    run_id = ApplicationAutofillRunRepo(conn).create(payload)
    payload["autofill_run_id"] = run_id
    Audit(conn).human(
        "application_autofill_run",
        "application",
        result.application_id or 0,
        ats=payload["ats"],
        status=payload["status"],
        resume_attached=payload["resume_attached"],
        human_intervention_required=payload["human_intervention_required"],
        submitted=False,
    )
    return payload
