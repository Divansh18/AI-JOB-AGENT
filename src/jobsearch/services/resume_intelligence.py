"""Phase 1B resume intelligence services."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from ..domain.filters import evaluate
from ..domain.resume_intelligence import (
    FitReport,
    JobAnalysis,
    ResumeSourceModel,
    ResumeValidationReport,
    TailoredResume,
    build_fit_report,
    build_job_analysis,
    build_resume_source_model,
    match_requirements,
    tailor_resume,
    truth_store_hash,
    validate_tailored_resume,
)
from ..persistence.repositories import (
    CandidateAnswerRepo,
    CandidateFactRepo,
    FilterRepo,
    JobRepo,
    ResumeVariantRepo,
)
from .audit import Audit

TRUTH_STORE_VERSION = 1


class ResumeIntelligenceError(Exception):
    pass


@dataclass(frozen=True)
class FitBundle:
    analysis: JobAnalysis
    source_model: ResumeSourceModel
    fit_report: FitReport
    source_truth_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis.as_dict(),
            "source_model": self.source_model.as_dict(),
            "fit_report": self.fit_report.as_dict(),
            "source_truth_hash": self.source_truth_hash,
        }


@dataclass(frozen=True)
class ResumeVariantResult:
    resume_id: int
    analysis: JobAnalysis
    source_model: ResumeSourceModel
    fit_report: FitReport
    tailored_resume: TailoredResume
    validation: ResumeValidationReport
    source_truth_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "resume_id": self.resume_id,
            "analysis": self.analysis.as_dict(),
            "source_model": self.source_model.as_dict(),
            "fit_report": self.fit_report.as_dict(),
            "tailored_resume": self.tailored_resume.as_dict(),
            "validation": self.validation.as_dict(),
            "source_truth_hash": self.source_truth_hash,
        }


def _job_or_raise(conn, job_id: int):
    job = JobRepo(conn).get(job_id)
    if job is None:
        raise ResumeIntelligenceError(f"job {job_id} not found")
    return job


def _truth(conn) -> tuple[list, list]:
    facts = CandidateFactRepo(conn).list(verified=True)
    answers = CandidateAnswerRepo(conn).list(verified=True)
    return facts, answers


def _signals(conn, config, job) -> dict[str, Any]:
    stored = FilterRepo(conn).get(job.id or 0)
    if stored and stored["signals"]:
        try:
            return json.loads(stored["signals"])
        except json.JSONDecodeError:
            pass
    return evaluate(job, config.filters).signals


def deterministic_job_analysis(conn, config, job_id: int) -> JobAnalysis:
    job = _job_or_raise(conn, job_id)
    signals = _signals(conn, config, job)
    return build_job_analysis(
        job,
        location_signal=signals.get("location") or {},
        yoe_signal=signals.get("yoe") or {},
        work_auth_blocker=signals.get("work_auth_blocker"),
    )


def fit_job(
    conn,
    config,
    job_id: int,
    *,
    analysis: JobAnalysis | None = None,
    as_of: date | None = None,
) -> FitBundle:
    facts, answers = _truth(conn)
    active_analysis = analysis or deterministic_job_analysis(conn, config, job_id)
    source_model = build_resume_source_model(facts, answers)
    matches = match_requirements(active_analysis, facts, answers, as_of=as_of)
    fit_report = build_fit_report(active_analysis, matches, source_model)
    return FitBundle(
        analysis=active_analysis,
        source_model=source_model,
        fit_report=fit_report,
        source_truth_hash=truth_store_hash(facts, answers),
    )


def create_resume_variant(
    conn,
    config,
    job_id: int,
    *,
    analysis: JobAnalysis | None = None,
    as_of: date | None = None,
) -> ResumeVariantResult:
    bundle = fit_job(conn, config, job_id, analysis=analysis, as_of=as_of)
    tailored = tailor_resume(bundle.analysis, bundle.fit_report, bundle.source_model)
    validation = validate_tailored_resume(tailored, bundle.source_model, as_of=as_of)

    payload = {
        "analysis": bundle.analysis.as_dict(),
        "fit_report": bundle.fit_report.as_dict(),
        "resume": tailored.as_dict(),
    }
    resume_id = ResumeVariantRepo(conn).create(
        job_id=job_id,
        fit_score=float(bundle.fit_report.overall_fit_score),
        source_truth_version=TRUTH_STORE_VERSION,
        source_truth_hash=bundle.source_truth_hash,
        master_resume_identity=tailored.master_resume.identity,
        master_resume_version=tailored.master_resume.version,
        template_identity=tailored.template.identity,
        page_limit=tailored.page_limit,
        tailoring_decision=tailored.recommendation.decision,
        tailoring_reasons=tailored.recommendation.reasons,
        tailoring_evidence_refs=tailored.recommendation.evidence_refs,
        content=payload,
        validation_status=validation.status,
        validation_errors=[issue.as_dict() for issue in validation.issues],
    )

    Audit(conn).human(
        "resume_variant_created",
        "resume_variant",
        resume_id,
        job_id=job_id,
        fit_score=bundle.fit_report.overall_fit_score,
        validation_status=validation.status,
        tailoring_decision=tailored.recommendation.decision,
    )
    return ResumeVariantResult(
        resume_id=resume_id,
        analysis=bundle.analysis,
        source_model=bundle.source_model,
        fit_report=bundle.fit_report,
        tailored_resume=tailored,
        validation=validation,
        source_truth_hash=bundle.source_truth_hash,
    )


def list_resume_variants(conn, *, job_id: int | None = None, limit: int = 20) -> list[dict[str, Any]]:
    rows = ResumeVariantRepo(conn).list(job_id=job_id, limit=limit)
    return [_resume_row_to_dict(row) for row in rows]


def get_resume_variant(conn, resume_id: int) -> dict[str, Any]:
    row = ResumeVariantRepo(conn).get(resume_id)
    if row is None:
        raise ResumeIntelligenceError(f"resume variant {resume_id} not found")
    return _resume_row_to_dict(row)


def _resume_row_to_dict(row) -> dict[str, Any]:
    content = json.loads(row["content_json"]) if row["content_json"] else {}
    errors = json.loads(row["validation_errors"]) if row["validation_errors"] else []
    reasons = json.loads(row["tailoring_reasons"]) if row["tailoring_reasons"] else []
    evidence = json.loads(row["tailoring_evidence_refs"]) if row["tailoring_evidence_refs"] else []
    base = dict(row)
    base["content_json"] = content
    base["validation_errors"] = errors
    base["tailoring_reasons"] = reasons
    base["tailoring_evidence_refs"] = evidence
    return base
