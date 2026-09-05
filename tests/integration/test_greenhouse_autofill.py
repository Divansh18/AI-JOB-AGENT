from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jobsearch.cli.main import app
from jobsearch.config.loader import AppConfig
from jobsearch.config.schemas import CompaniesFile, Filters, Profile, RankingConfig, Settings
from jobsearch.domain.application_planner import ApplicationPlan, ApplicationPlanField, ApplicationPlanState
from jobsearch.domain.models import Job
from jobsearch.domain.normalize import normalize_title
from jobsearch.persistence.db import connect, migrate
from jobsearch.persistence.repositories import (
    ApplicationAutofillRunRepo,
    ApplicationPlanRepo,
    ApplicationRepo,
    JobRepo,
    SourceRepo,
)
from jobsearch.services import application_autofill
from jobsearch.services.autofill import (
    ATS_ASHBY,
    ATS_GREENHOUSE,
    ATS_LEVER,
    AUTOFILL_STATUS_FILLED_FOR_REVIEW,
    AUTOFILL_STATUS_FAILED,
    AUTOFILL_STATUS_UNSUPPORTED_ATS,
    AutofillResult,
)
from jobsearch.services.ashby_autofill import AshbyAutofillAdapter
from jobsearch.services.greenhouse_autofill import GreenhouseAutofillAdapter
from jobsearch.services.lever_autofill import LeverAutofillAdapter, detect_lever_page_state


RUNNER = CliRunner()
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class FakeLocator:
    def __init__(self, page: FakePage, selector: str):
        self.page = page
        self.selector = selector

    def fill(self, value: str) -> None:
        self.page.filled[self.selector] = value

    def select_option(self, *, label=None, value=None) -> None:
        self.page.selected[self.selector] = label or value

    def set_input_files(self, path: str) -> None:
        self.page.uploaded[self.selector] = path


class FakePage:
    def __init__(self, html: str):
        self.html = html
        self.goto_url = None
        self.filled: dict[str, str] = {}
        self.selected: dict[str, str] = {}
        self.uploaded: dict[str, str] = {}

    def goto(self, url: str, **kwargs) -> None:
        self.goto_url = url

    def wait_for_selector(self, selector: str, **kwargs) -> None:
        return None

    def content(self) -> str:
        return self.html

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(self, selector)


class EmptyActionLocator:
    @property
    def first(self):
        return self

    def count(self) -> int:
        return 0

    def click(self) -> None:
        raise AssertionError("empty action locator should not be clicked")


class FakeApplyActionLocator:
    def __init__(self, page: FakeLeverPage):
        self.page = page

    @property
    def first(self):
        return self

    def count(self) -> int:
        return 1

    def click(self) -> None:
        self.page.clicked_apply += 1
        self.page.html = self.page.form_html

    def filter(self, **kwargs):
        return self


class FakeLeverPage(FakePage):
    def __init__(self, html: str, form_html: str, *, has_apply_button: bool = True):
        super().__init__(html)
        self.form_html = form_html
        self.has_apply_button = has_apply_button
        self.clicked_apply = 0
        self.clicked_submit = 0
        self.load_state_waits: list[str] = []

    def get_by_role(self, role: str, **kwargs):
        if self.has_apply_button and role == "link":
            return FakeApplyActionLocator(self)
        return EmptyActionLocator()

    def locator(self, selector: str):
        if selector in {"a, button", 'a[href*="apply"], a[href*="#"], button'}:
            return FakeApplyActionLocator(self) if self.has_apply_button else EmptyActionLocator()
        if "submit" in selector.lower():
            self.clicked_submit += 1
        return FakeLocator(self, selector)

    def wait_for_load_state(self, state: str, **kwargs) -> None:
        self.load_state_waits.append(state)


def _config(root: Path) -> AppConfig:
    return AppConfig(
        root=root,
        settings=Settings(db_path="test.db", digest_dir="digests"),
        profile=Profile(),
        filters=Filters(),
        ranking=RankingConfig(),
        companies=CompaniesFile(),
    )


def _write_config(root: Path) -> None:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "settings.yaml").write_text("db_path: data/test.db\ndigest_dir: data/digests\n", encoding="utf-8")
    (cfg / "profile.yaml").write_text("experience_stage: early_career\n", encoding="utf-8")
    (cfg / "filters.yaml").write_text("version: 2\n", encoding="utf-8")
    (cfg / "ranking.yaml").write_text("version: 2\n", encoding="utf-8")
    (cfg / "companies.yaml").write_text("version: 1\ncompanies: []\n", encoding="utf-8")


def _seed_job(conn, *, apply_url: str = "https://job-boards.greenhouse.io/acme/jobs/123") -> int:
    source_id = SourceRepo(conn).ensure("manual", "manual")
    now = datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc)
    job = Job(
        fingerprint="fp-autofill",
        content_hash="hash-autofill",
        source="manual",
        source_id=source_id,
        title="Backend Engineer",
        title_normalized=normalize_title("Backend Engineer"),
        company_name_raw="Acme Labs",
        company_normalized="acme_labs",
        location_raw="Remote India",
        locations=["India"],
        country="IN",
        remote_type="remote",
        remote_scope="india",
        employment_type="full_time",
        description_text="Build Python services.",
        apply_url=apply_url,
        canonical_url=apply_url,
        posted_at=now,
        first_seen_at=now,
    )
    return JobRepo(conn).upsert(job)[0]


def _candidate_fields() -> list[ApplicationPlanField]:
    return [
        ApplicationPlanField(key="full_name", label="Full name", value="Asha Example", evidence_ref="fact:1"),
        ApplicationPlanField(key="first_name", label="First name", value="Asha", evidence_ref="fact:1"),
        ApplicationPlanField(key="last_name", label="Last name", value="Example", evidence_ref="fact:1"),
        ApplicationPlanField(key="email", label="Email", value="asha@example.com", evidence_ref="fact:2"),
        ApplicationPlanField(key="phone", label="Phone", value="+91 99999 88888", evidence_ref="fact:3"),
        ApplicationPlanField(key="linkedin_url", label="LinkedIn", value="https://linkedin.com/in/asha-example", evidence_ref="fact:4"),
        ApplicationPlanField(key="github_url", label="GitHub", value="https://github.com/asha-example", evidence_ref="fact:5"),
        ApplicationPlanField(key="portfolio_url", label="Portfolio", value="https://asha.dev", evidence_ref="fact:7"),
        ApplicationPlanField(key="current_location", label="Current location", value="Bengaluru, India", evidence_ref="fact:6"),
    ]


def _seed_plan(
    conn,
    root: Path,
    *,
    apply_url: str = "https://job-boards.greenhouse.io/acme/jobs/123",
    resume_status: str = "valid",
    page_count: int | None = 1,
    resume_exists: bool = True,
    blockers: list[str] | None = None,
) -> int:
    job_id = _seed_job(conn, apply_url=apply_url)
    application_id = ApplicationRepo(conn).ensure_for_plan(job_id)
    resume_path = root / "output" / "pdf" / "resume.pdf"
    if resume_exists:
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        resume_path.write_bytes(b"%PDF-1.4\n%stub\n")
    now = datetime.now(timezone.utc)
    plan = ApplicationPlan(
        application_id=application_id,
        job_id=job_id,
        company="Acme Labs",
        role="Backend Engineer",
        application_url=apply_url,
        resume_id=None,
        resume_path="output/pdf/resume.pdf",
        resume_decision="tailor",
        resume_page_count=page_count,
        resume_page_validation_status=resume_status,
        candidate_fields=_candidate_fields(),
        known_answers=[],
        unanswered_fields=[],
        sensitive_fields=[],
        blockers=blockers or [],
        review_required=True,
        state=ApplicationPlanState.BLOCKED.value if blockers else ApplicationPlanState.READY_FOR_REVIEW.value,
        created_at=now,
        updated_at=now,
    )
    ApplicationPlanRepo(conn).upsert(plan)
    return application_id


def test_greenhouse_adapter_fills_safe_fields_attaches_resume_and_stops_before_submit(tmp_path):
    html = (FIXTURES / "greenhouse_application_form.html").read_text(encoding="utf-8")
    page = FakePage(html)
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF-1.4\n%stub\n")
    plan = {
        "application_id": 7,
        "application_url": "https://job-boards.greenhouse.io/acme/jobs/123",
        "candidate_fields": [field.as_dict() for field in _candidate_fields()],
        "known_answers": [],
    }

    result = GreenhouseAutofillAdapter(wait_for_review=False).fill_page(page, plan, resume_path, open_url=True)
    filled = result.as_dict()["fields_filled"]

    assert page.goto_url == plan["application_url"]
    assert page.filled["#first_name"] == "Asha"
    assert page.filled["#last_name"] == "Example"
    assert page.filled["#email"] == "asha@example.com"
    assert page.filled["#phone"] == "+91 99999 88888"
    assert page.filled["#job_application_answers_attributes_0_text_value"] == "https://linkedin.com/in/asha-example"
    assert page.filled["#job_application_answers_attributes_1_text_value"] == "https://github.com/asha-example"
    assert page.filled["#job_application_answers_attributes_2_text_value"] == "Bengaluru, India"
    assert page.uploaded["#resume"] == str(resume_path)
    assert "sponsorship_required" not in filled
    assert result.submitted is False
    assert result.resume_attached is True


def test_greenhouse_adapter_leaves_missing_custom_and_sensitive_questions_unresolved(tmp_path):
    html = (FIXTURES / "greenhouse_application_form.html").read_text(encoding="utf-8")
    page = FakePage(html)
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF-1.4\n%stub\n")
    plan = {
        "application_id": 7,
        "application_url": "https://job-boards.greenhouse.io/acme/jobs/123",
        "candidate_fields": [field.as_dict() for field in _candidate_fields() if field.key != "github_url"],
        "known_answers": [
            {
                "question_key": "sponsorship_required",
                "answer_text": "No",
                "human_review_required": True,
                "autofill_safe": False,
            }
        ],
    }

    result = GreenhouseAutofillAdapter(wait_for_review=False).fill_page(page, plan, resume_path)
    payload = result.as_dict()

    assert "#job_application_answers_attributes_1_text_value" not in page.filled
    assert any(field["key"] == "github_url" for field in payload["unresolved_fields"])
    assert any(field["key"] == "sponsorship_required" for field in payload["sensitive_fields"])
    assert payload["human_intervention_required"] is True
    assert payload["submitted"] is False


def test_greenhouse_adapter_detects_captcha(tmp_path):
    html = (FIXTURES / "greenhouse_captcha_form.html").read_text(encoding="utf-8")
    page = FakePage(html)
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF-1.4\n%stub\n")

    result = GreenhouseAutofillAdapter(wait_for_review=False).fill_page(
        page,
        {
            "application_id": 7,
            "application_url": "https://job-boards.greenhouse.io/acme/jobs/123",
            "candidate_fields": [field.as_dict() for field in _candidate_fields()],
            "known_answers": [],
        },
        resume_path,
    )

    assert result.human_intervention_required is True
    assert "captcha_or_bot_check_detected" in result.errors


def test_unsupported_ats_is_persisted_without_browser(conn, tmp_path):
    app_id = _seed_plan(conn, tmp_path, apply_url="https://jobs.example.com/acme/123")

    result = application_autofill.run_attended_autofill(conn, _config(tmp_path), app_id, wait_for_review=False)
    runs = ApplicationAutofillRunRepo(conn).list_by_application(app_id)

    assert result["status"] == AUTOFILL_STATUS_UNSUPPORTED_ATS
    assert result["submitted"] is False
    assert len(runs) == 1
    assert json.loads(runs[0]["result_json"])["status"] == AUTOFILL_STATUS_UNSUPPORTED_ATS


def test_lever_adapter_fills_safe_fields_attaches_resume_and_stops_before_submit(tmp_path):
    html = (FIXTURES / "lever_application_form.html").read_text(encoding="utf-8")
    page = FakePage(html)
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF-1.4\n%stub\n")
    plan = {
        "application_id": 8,
        "application_url": "https://jobs.lever.co/acme/123",
        "candidate_fields": [field.as_dict() for field in _candidate_fields()],
        "known_answers": [],
    }

    result = LeverAutofillAdapter(wait_for_review=False).fill_page(page, plan, resume_path, open_url=True)
    payload = result.as_dict()

    assert page.goto_url == plan["application_url"]
    assert page.filled["#name"] == "Asha Example"
    assert page.filled["#email"] == "asha@example.com"
    assert page.filled["#phone"] == "+91 99999 88888"
    assert page.filled["#urls_LinkedIn"] == "https://linkedin.com/in/asha-example"
    assert page.filled["#urls_GitHub"] == "https://github.com/asha-example"
    assert page.filled["#urls_Portfolio"] == "https://asha.dev"
    assert page.filled["#location"] == "Bengaluru, India"
    assert page.uploaded["#resume"] == str(resume_path)
    assert "#question_1" not in page.filled
    assert "#question_2" not in page.selected
    assert any(field["key"] == "why_are_you_interested_in_this_role" for field in payload["unresolved_fields"])
    assert any(field["key"] == "sponsorship_required" for field in payload["sensitive_fields"])
    assert payload["submitted"] is False
    assert payload["resume_attached"] is True


def test_lever_adapter_clicks_job_detail_apply_then_fills_form(tmp_path):
    detail_html = (FIXTURES / "lever_job_detail_page.html").read_text(encoding="utf-8")
    form_html = (FIXTURES / "lever_application_form.html").read_text(encoding="utf-8")
    page = FakeLeverPage(detail_html, form_html)
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF-1.4\n%stub\n")
    plan = {
        "application_id": 8,
        "application_url": "https://jobs.lever.co/hevodata/6cbbe304-e065-4711-bf3e-756795d2bc2a",
        "candidate_fields": [field.as_dict() for field in _candidate_fields()],
        "known_answers": [],
    }

    result = LeverAutofillAdapter(wait_for_review=False).fill_page(page, plan, resume_path, open_url=True)
    payload = result.as_dict()

    assert detect_lever_page_state(detail_html) == "job_detail"
    assert page.goto_url == plan["application_url"]
    assert page.clicked_apply == 1
    assert page.clicked_submit == 0
    assert page.filled["#name"] == "Asha Example"
    assert page.filled["#email"] == "asha@example.com"
    assert page.uploaded["#resume"] == str(resume_path)
    assert len(payload["fields_detected"]) > 0
    assert len(payload["fields_filled"]) > 0
    assert payload["submitted"] is False
    assert payload["resume_attached"] is True


def test_lever_adapter_already_on_form_does_not_click_apply(tmp_path):
    form_html = (FIXTURES / "lever_application_form.html").read_text(encoding="utf-8")
    page = FakeLeverPage(form_html, form_html)
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF-1.4\n%stub\n")
    plan = {
        "application_id": 8,
        "application_url": "https://jobs.lever.co/acme/123/apply",
        "candidate_fields": [field.as_dict() for field in _candidate_fields()],
        "known_answers": [],
    }

    result = LeverAutofillAdapter(wait_for_review=False).fill_page(page, plan, resume_path, open_url=True)

    assert detect_lever_page_state(form_html) == "application_form"
    assert page.clicked_apply == 0
    assert page.clicked_submit == 0
    assert result.resume_attached is True
    assert result.submitted is False


def test_lever_adapter_missing_apply_button_fails_safely(tmp_path):
    page = FakeLeverPage(
        "<html><body><h1>SDE I</h1><p>No application form is visible.</p></body></html>",
        (FIXTURES / "lever_application_form.html").read_text(encoding="utf-8"),
        has_apply_button=False,
    )
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF-1.4\n%stub\n")
    plan = {
        "application_id": 8,
        "application_url": "https://jobs.lever.co/acme/123",
        "candidate_fields": [field.as_dict() for field in _candidate_fields()],
        "known_answers": [],
    }

    result = LeverAutofillAdapter(wait_for_review=False).fill_page(page, plan, resume_path, open_url=True)
    payload = result.as_dict()

    assert page.clicked_apply == 0
    assert page.clicked_submit == 0
    assert payload["status"] == AUTOFILL_STATUS_FAILED
    assert payload["errors"] == ["lever_apply_button_not_detected"]
    assert payload["human_intervention_required"] is True
    assert payload["submitted"] is False


def test_ashby_adapter_fills_safe_fields_safe_custom_answer_and_stops_before_submit(tmp_path):
    html = (FIXTURES / "ashby_application_form.html").read_text(encoding="utf-8")
    page = FakePage(html)
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF-1.4\n%stub\n")
    plan = {
        "application_id": 9,
        "application_url": "https://jobs.ashbyhq.com/acme/123",
        "candidate_fields": [field.as_dict() for field in _candidate_fields()],
        "known_answers": [
            {
                "question_key": "short_experience_summary",
                "answer_text": "Backend engineer focused on Python services.",
                "human_review_required": False,
                "autofill_safe": True,
            }
        ],
    }

    result = AshbyAutofillAdapter(wait_for_review=False).fill_page(page, plan, resume_path, open_url=True)
    payload = result.as_dict()

    assert page.goto_url == plan["application_url"]
    assert page.filled["#ashby_name"] == "Asha Example"
    assert page.filled["#ashby_email"] == "asha@example.com"
    assert page.filled["#ashby_phone"] == "+91 99999 88888"
    assert page.filled["#ashby_linkedin"] == "https://linkedin.com/in/asha-example"
    assert page.filled["#ashby_github"] == "https://github.com/asha-example"
    assert page.filled["#ashby_website"] == "https://asha.dev"
    assert page.filled["#ashby_location"] == "Bengaluru, India"
    assert page.filled["#ashby_summary"] == "Backend engineer focused on Python services."
    assert page.uploaded["#ashby_resume"] == str(resume_path)
    assert "#ashby_authorized" not in page.selected
    assert "short_experience_summary" in payload["fields_filled"]
    assert any(field["key"] == "work_authorization" for field in payload["sensitive_fields"])
    assert payload["submitted"] is False
    assert payload["resume_attached"] is True


@pytest.mark.parametrize(
    ("adapter_cls", "url", "fixture_name"),
    (
        (LeverAutofillAdapter, "https://jobs.lever.co/acme/123", "lever_application_form.html"),
        (AshbyAutofillAdapter, "https://jobs.ashbyhq.com/acme/123", "ashby_application_form.html"),
    ),
)
def test_supported_adapters_detect_captcha_or_bot_check(adapter_cls, url, fixture_name, tmp_path):
    html = (FIXTURES / fixture_name).read_text(encoding="utf-8") + '<div class="h-captcha"></div>'
    page = FakePage(html)
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF-1.4\n%stub\n")

    result = adapter_cls(wait_for_review=False).fill_page(
        page,
        {
            "application_id": 10,
            "application_url": url,
            "candidate_fields": [field.as_dict() for field in _candidate_fields()],
            "known_answers": [],
        },
        resume_path,
    )

    assert result.human_intervention_required is True
    assert "captcha_or_bot_check_detected" in result.errors


@pytest.mark.parametrize(
    ("ats", "url", "adapter_attr"),
    (
        (ATS_LEVER, "https://jobs.lever.co/acme/123", "LeverAutofillAdapter"),
        (ATS_ASHBY, "https://jobs.ashbyhq.com/acme/123", "AshbyAutofillAdapter"),
    ),
)
def test_supported_non_greenhouse_results_are_persisted(conn, tmp_path, monkeypatch, ats, url, adapter_attr):
    app_id = _seed_plan(conn, tmp_path, apply_url=url)

    class FakeAdapter:
        def __init__(self, **kwargs):
            pass

        def run(self, plan, resume_path):
            return AutofillResult(
                application_id=plan["application_id"],
                ats=ats,
                url=plan["application_url"],
                fields_detected=[{"key": "email"}],
                fields_filled=["email"],
                resume_attached=True,
                unresolved_fields=[],
                sensitive_fields=[],
                human_intervention_required=False,
                errors=[],
                status=AUTOFILL_STATUS_FILLED_FOR_REVIEW,
                submitted=False,
            )

    monkeypatch.setattr(application_autofill, adapter_attr, FakeAdapter)

    result = application_autofill.run_attended_autofill(conn, _config(tmp_path), app_id, wait_for_review=False)
    runs = ApplicationAutofillRunRepo(conn).list_by_application(app_id)

    assert result["ats"] == ats
    assert result["status"] == AUTOFILL_STATUS_FILLED_FOR_REVIEW
    assert result["submitted"] is False
    assert len(runs) == 1
    assert runs[0]["submitted"] == 0


def test_missing_or_invalid_resume_blocks_before_browser(conn, tmp_path):
    app_id = _seed_plan(conn, tmp_path, resume_exists=False)

    result = application_autofill.run_attended_autofill(conn, _config(tmp_path), app_id, wait_for_review=False)

    assert result["status"] == AUTOFILL_STATUS_FAILED
    assert "file not found" in result["errors"][0]
    assert result["submitted"] is False


def test_invalid_page_validation_blocks_before_browser(conn, tmp_path):
    app_id = _seed_plan(conn, tmp_path, resume_status="overflow", page_count=2)

    result = application_autofill.run_attended_autofill(conn, _config(tmp_path), app_id, wait_for_review=False)

    assert result["status"] == AUTOFILL_STATUS_FAILED
    assert "page_validation_status=overflow" in result["errors"][0]
    assert result["submitted"] is False


def test_blocked_plan_stops_before_browser(conn, tmp_path):
    app_id = _seed_plan(conn, tmp_path, blockers=["poor_fit_resume_decision"])

    result = application_autofill.run_attended_autofill(conn, _config(tmp_path), app_id, wait_for_review=False)

    assert result["status"] == AUTOFILL_STATUS_FAILED
    assert result["errors"] == ["plan_blocked:poor_fit_resume_decision"]
    assert result["submitted"] is False


def test_greenhouse_result_is_persisted(conn, tmp_path, monkeypatch):
    app_id = _seed_plan(conn, tmp_path)

    class FakeAdapter:
        def __init__(self, **kwargs):
            pass

        def run(self, plan, resume_path):
            return AutofillResult(
                application_id=plan["application_id"],
                ats=ATS_GREENHOUSE,
                url=plan["application_url"],
                fields_detected=[{"key": "email"}],
                fields_filled=["email"],
                resume_attached=True,
                unresolved_fields=[],
                sensitive_fields=[],
                human_intervention_required=False,
                errors=[],
                status=AUTOFILL_STATUS_FILLED_FOR_REVIEW,
                submitted=False,
            )

    monkeypatch.setattr(application_autofill, "GreenhouseAutofillAdapter", FakeAdapter)

    result = application_autofill.run_attended_autofill(conn, _config(tmp_path), app_id, wait_for_review=False)
    runs = ApplicationAutofillRunRepo(conn).list_by_application(app_id)

    assert result["status"] == AUTOFILL_STATUS_FILLED_FOR_REVIEW
    assert result["resume_attached"] is True
    assert len(runs) == 1
    assert runs[0]["submitted"] == 0


def test_apply_autofill_cli_invokes_service(tmp_path, monkeypatch):
    _write_config(tmp_path)
    conn = connect(tmp_path / "data" / "test.db")
    migrate(conn)
    app_id = _seed_plan(conn, tmp_path)
    conn.close()

    def fake_run(conn, config, application_id, **kwargs):
        return {
            "autofill_run_id": 99,
            "application_id": application_id,
            "ats": "greenhouse",
            "url": "https://job-boards.greenhouse.io/acme/jobs/123",
            "fields_detected": [{"key": "email"}],
            "fields_filled": ["email"],
            "resume_attached": True,
            "unresolved_fields": [],
            "sensitive_fields": [],
            "human_intervention_required": False,
            "errors": [],
            "status": "filled_for_review",
            "submitted": False,
        }

    monkeypatch.setattr(application_autofill, "run_attended_autofill", fake_run)

    result = RUNNER.invoke(
        app,
        ["apply", "autofill", str(app_id), "--headless", "--no-wait", "--json"],
        env={"JOBSEARCH_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["application_id"] == app_id
    assert payload["submitted"] is False
