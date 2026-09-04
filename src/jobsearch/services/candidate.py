"""Candidate Truth Store services."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.candidate import (
    CandidateFact,
    CandidateImportBundle,
    CandidateValidationReport,
    VerifiedAnswer,
    build_application_profile,
    bundle_from_payload,
    make_answer,
    make_fact,
    validate_store,
)
from ..persistence.repositories import CandidateAnswerRepo, CandidateFactRepo
from .audit import Audit


class CandidateError(Exception):
    pass


@dataclass(frozen=True)
class CandidateImportResult:
    facts_upserted: int
    answers_upserted: int
    report: CandidateValidationReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "facts_upserted": self.facts_upserted,
            "answers_upserted": self.answers_upserted,
            "report": self.report.as_dict(),
        }


def _errors(report: CandidateValidationReport) -> list[str]:
    return [f"{issue.code}: {issue.message}" for issue in report.issues if issue.level == "error"]


def _ensure_valid(bundle: CandidateImportBundle) -> None:
    if bundle.version != 1:
        raise CandidateError(f"unsupported candidate import version: {bundle.version}")
    report = validate_store(bundle.facts, bundle.answers)
    errors = _errors(report)
    if errors:
        raise CandidateError("; ".join(errors))


def add_fact(
    conn,
    *,
    category: str,
    key: str,
    value: Any,
    source: str,
    verified: bool,
    confidence: float | None = None,
) -> CandidateFact:
    fact = make_fact(
        category=category,
        key=key,
        value=value,
        source=source,
        verified=verified,
        confidence=confidence,
    )
    _ensure_valid(CandidateImportBundle(version=1, facts=[fact], answers=[]))
    saved = CandidateFactRepo(conn).save(fact)
    Audit(conn).human("candidate_fact_saved", "candidate_fact", saved.id or 0,
                      category=saved.category, key=saved.key, verified=saved.verified)
    return saved


def list_facts(conn, *, category: str | None = None, verified_only: bool = False) -> list[CandidateFact]:
    return CandidateFactRepo(conn).list(
        category=category,
        verified=True if verified_only else None,
    )


def add_answer(
    conn,
    *,
    question_key: str,
    answer_text: str,
    category: str,
    source: str,
    evidence_refs: list[str] | None = None,
    verified: bool = False,
    human_review_required: bool = False,
) -> VerifiedAnswer:
    answer = make_answer(
        question_key=question_key,
        answer_text=answer_text,
        category=category,
        source=source,
        evidence_refs=evidence_refs,
        verified=verified,
        human_review_required=human_review_required,
    )
    _ensure_valid(CandidateImportBundle(version=1, facts=[], answers=[answer]))
    saved = CandidateAnswerRepo(conn).save(answer)
    Audit(conn).human("candidate_answer_saved", "candidate_answer", saved.id or 0,
                      question_key=saved.question_key, verified=saved.verified)
    return saved


def list_answers(conn, *, question_key: str | None = None, verified_only: bool = False) -> list[VerifiedAnswer]:
    return CandidateAnswerRepo(conn).list(
        question_key=question_key,
        verified=True if verified_only else None,
    )


def build_profile(conn):
    return build_application_profile(CandidateFactRepo(conn).list())


def validate(conn) -> CandidateValidationReport:
    facts = CandidateFactRepo(conn).list()
    answers = CandidateAnswerRepo(conn).list()
    return validate_store(facts, answers)


def import_json(conn, path: Path) -> CandidateImportResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bundle = bundle_from_payload(payload)
    _ensure_valid(bundle)

    fact_repo = CandidateFactRepo(conn)
    answer_repo = CandidateAnswerRepo(conn)
    facts_upserted = 0
    answers_upserted = 0
    for fact in bundle.facts:
        fact_repo.save(fact)
        facts_upserted += 1
    for answer in bundle.answers:
        answer_repo.save(answer)
        answers_upserted += 1

    report = validate(conn)
    errors = _errors(report)
    if errors:
        raise CandidateError("; ".join(errors))

    Audit(conn).human("candidate_imported_json", payload_path=str(path),
                      facts_upserted=facts_upserted, answers_upserted=answers_upserted)
    return CandidateImportResult(
        facts_upserted=facts_upserted,
        answers_upserted=answers_upserted,
        report=report,
    )
