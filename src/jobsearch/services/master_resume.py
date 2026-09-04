"""Master resume registration and deterministic PDF ingestion."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from ..domain.candidate import CandidateFact, CandidateValidationReport
from ..domain.master_resume import (
    MasterResumeConflict,
    MasterResumeRecord,
    ParsedMasterResume,
    PdfExtraction,
    PdfLink,
    build_master_resume_facts,
    fact_semantic_identity,
    facts_equivalent,
    parse_master_resume_text,
    section_counts,
)
from ..persistence.repositories import CandidateFactRepo, MasterResumeRepo
from . import candidate as candidate_service
from .audit import Audit

DEFAULT_MASTER_IDENTITY = "master_v1"
DEFAULT_MASTER_VERSION = 1
DEFAULT_PAGE_LIMIT = 1


class MasterResumeError(Exception):
    pass


@dataclass(frozen=True)
class MasterResumeIngestResult:
    master_resume: MasterResumeRecord
    parsed_resume: ParsedMasterResume
    deleted_count: int
    created_count: int
    updated_count: int
    unchanged_count: int
    conflicts: list[MasterResumeConflict]
    validation_report: CandidateValidationReport
    validation_errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "master_resume": self.master_resume.as_dict(),
            "parsed_resume": self.parsed_resume.as_dict(),
            "detected_sections": self.parsed_resume.sections_detected,
            "section_counts": section_counts(self.parsed_resume),
            "deleted_count": self.deleted_count,
            "created_count": self.created_count,
            "updated_count": self.updated_count,
            "unchanged_count": self.unchanged_count,
            "imported_count": self.created_count + self.updated_count,
            "conflicts": [conflict.as_dict() for conflict in self.conflicts],
            "validation_errors": self.validation_errors,
            "candidate_validation": self.validation_report.as_dict(),
        }


def register_master_resume(
    conn,
    config,
    *,
    file_path: str,
    identity: str = DEFAULT_MASTER_IDENTITY,
    version: int = DEFAULT_MASTER_VERSION,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    active: bool = True,
) -> MasterResumeRecord:
    absolute_path, stored_path = _resolve_user_path(config.root, file_path)
    if absolute_path.suffix.lower() != ".pdf":
        raise MasterResumeError(f"master resume must be a PDF: {file_path}")

    record = MasterResumeRepo(conn).register(
        identity=identity,
        version=version,
        file_path=stored_path,
        active=active,
        page_limit=page_limit,
        file_hash=_file_hash(absolute_path),
    )
    Audit(conn).human(
        "master_resume_registered",
        "master_resume",
        record.id or 0,
        identity=record.identity,
        version=record.version,
        file_path=record.file_path,
        active=record.active,
    )
    return record


def show_active_master_resume(conn) -> MasterResumeRecord:
    record = MasterResumeRepo(conn).get_active()
    if record is None:
        raise MasterResumeError("no active master resume is registered")
    return record


def ingest_active_master_resume(
    conn,
    config,
    *,
    extractor: Callable[[Path], PdfExtraction] | None = None,
) -> MasterResumeIngestResult:
    repo = MasterResumeRepo(conn)
    record = repo.get_active()
    if record is None:
        raise MasterResumeError("no active master resume is registered")

    absolute_path, _ = _resolve_registered_path(config.root, record.file_path)
    extraction = (extractor or extract_pdf_resume)(absolute_path)
    parsed = parse_master_resume_text(extraction.text, page_count=extraction.page_count, links=extraction.links)
    validation_errors: list[str] = []
    if extraction.page_count != record.page_limit:
        validation_errors.append(
            f"page_count_mismatch: expected {record.page_limit} page(s), found {extraction.page_count}"
        )

    source = _source_name(record.version)
    facts = build_master_resume_facts(parsed, source=source)
    deleted = CandidateFactRepo(conn).delete_missing_from_source(
        source,
        {(fact.category, fact.key) for fact in facts},
    )
    created, updated, unchanged, conflicts = _merge_imported_facts(conn, facts)
    stored = repo.record_ingestion(
        identity=record.identity,
        version=record.version,
        file_hash=_file_hash(absolute_path),
        extracted_text=extraction.text,
        sections=parsed.as_dict(),
    )
    validation = candidate_service.validate(conn)

    Audit(conn).human(
        "master_resume_ingested",
        "master_resume",
        stored.id or 0,
        identity=stored.identity,
        version=stored.version,
        deleted_count=deleted,
        created_count=created,
        updated_count=updated,
        unchanged_count=unchanged,
        conflict_count=len(conflicts),
        validation_errors=validation_errors,
    )
    return MasterResumeIngestResult(
        master_resume=stored,
        parsed_resume=parsed,
        deleted_count=deleted,
        created_count=created,
        updated_count=updated,
        unchanged_count=unchanged,
        conflicts=conflicts,
        validation_report=validation,
        validation_errors=validation_errors,
    )


def extract_pdf_resume(path: Path) -> PdfExtraction:
    absolute_path = path.expanduser().resolve()
    if not absolute_path.exists():
        raise MasterResumeError(f"master resume not found: {path}")
    try:
        return _extract_pdf_with_pypdf(absolute_path)
    except MasterResumeError:
        raise
    except ModuleNotFoundError as exc:
        raise MasterResumeError(
            "deterministic PDF extraction is unavailable; install pypdf or set JOBSEARCH_PDF_SITE_PACKAGES"
        ) from exc


def _extract_pdf_with_pypdf(path: Path) -> PdfExtraction:
    PdfReader = _load_pypdf_reader()

    reader = PdfReader(str(path))
    pages_text: list[str] = []
    links: list[PdfLink] = []
    for page_index, page in enumerate(reader.pages, start=1):
        pages_text.append(page.extract_text() or "")
        for annot_ref in page.get("/Annots") or []:
            annot = annot_ref.get_object()
            action = annot.get("/A") or {}
            url = action.get("/URI")
            label = annot.get("/Contents")
            if not isinstance(url, str):
                continue
            rect = annot.get("/Rect") or []
            top = None
            if isinstance(rect, (list, tuple)) and len(rect) >= 4:
                try:
                    top = float(max(rect[1], rect[3]))
                except (TypeError, ValueError):
                    top = None
            links.append(
                PdfLink(
                    label=str(label or ""),
                    url=url.strip(),
                    page=page_index,
                    top=top,
                )
            )
    return PdfExtraction(text="\n".join(pages_text).strip(), page_count=len(reader.pages), links=links)


def _load_pypdf_reader():
    try:
        from pypdf import PdfReader

        return PdfReader
    except ModuleNotFoundError:
        pass

    importlib.invalidate_caches()
    for candidate in _pdf_site_packages_candidates():
        if not candidate.exists():
            continue
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            from pypdf import PdfReader

            return PdfReader
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError("pypdf")


def _pdf_site_packages_candidates() -> list[Path]:
    candidates = [
        os.environ.get("JOBSEARCH_PDF_SITE_PACKAGES"),
    ]
    candidates.extend(
        str(path)
        for path in sorted(
            (Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib").glob(
                "python*/site-packages"
            )
        )
    )
    seen: set[str] = set()
    out: list[Path] = []
    for item in candidates:
        if not item:
            continue
        candidate = Path(item).expanduser()
        text = str(candidate)
        if text in seen:
            continue
        if not candidate.exists():
            continue
        out.append(candidate)
        seen.add(text)
    return out


def _resolve_user_path(root: Path, file_path: str) -> tuple[Path, str]:
    raw_path = Path(file_path).expanduser()
    absolute_path = raw_path if raw_path.is_absolute() else (root / raw_path)
    absolute_path = absolute_path.resolve()
    if not absolute_path.exists() or not absolute_path.is_file():
        raise MasterResumeError(f"master resume not found: {file_path}")
    try:
        stored_path = absolute_path.relative_to(root).as_posix()
    except ValueError:
        stored_path = str(absolute_path)
    return absolute_path, stored_path


def _resolve_registered_path(root: Path, file_path: str) -> tuple[Path, str]:
    path = Path(file_path).expanduser()
    absolute_path = path if path.is_absolute() else (root / path)
    absolute_path = absolute_path.resolve()
    if not absolute_path.exists() or not absolute_path.is_file():
        raise MasterResumeError(f"registered master resume file is missing: {file_path}")
    try:
        stored_path = absolute_path.relative_to(root).as_posix()
    except ValueError:
        stored_path = str(absolute_path)
    return absolute_path, stored_path


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_name(version: int) -> str:
    return f"master_resume_v{version}"


def _facts_equal(existing: CandidateFact, incoming: CandidateFact) -> bool:
    return (
        existing.category == incoming.category
        and existing.key == incoming.key
        and existing.source == incoming.source
        and existing.verified == incoming.verified
        and existing.confidence == incoming.confidence
        and json.dumps(existing.value, ensure_ascii=True, sort_keys=True)
        == json.dumps(incoming.value, ensure_ascii=True, sort_keys=True)
    )


def _merge_imported_facts(conn, incoming_facts: list[CandidateFact]) -> tuple[int, int, int, list[MasterResumeConflict]]:
    repo = CandidateFactRepo(conn)
    existing = repo.list()
    by_category: dict[str, list[CandidateFact]] = {}
    for fact in existing:
        by_category.setdefault(fact.category, []).append(fact)

    created = 0
    updated = 0
    unchanged = 0
    conflicts: list[MasterResumeConflict] = []
    for incoming in incoming_facts:
        direct = repo.get(incoming.category, incoming.key)
        if direct is not None:
            if _facts_equal(direct, incoming):
                unchanged += 1
                continue
            if direct.source == incoming.source or not direct.verified:
                repo.save(incoming)
                _refresh_category_cache(by_category, incoming.category, repo)
                updated += 1
                continue
            if facts_equivalent(direct, incoming):
                unchanged += 1
                continue
            conflicts.append(
                MasterResumeConflict(
                    category=incoming.category,
                    key=incoming.key,
                    reason="existing verified fact has a different value",
                    existing_source=direct.source,
                    existing_value=direct.value,
                    incoming_value=incoming.value,
                )
            )
            continue

        semantic = _find_semantic_match(by_category.get(incoming.category, []), incoming)
        if semantic is not None:
            if semantic.source == incoming.source or not semantic.verified:
                repo.save(replace(incoming, key=semantic.key))
                _refresh_category_cache(by_category, incoming.category, repo)
                updated += 1
                continue
            if facts_equivalent(semantic, incoming):
                unchanged += 1
                continue
            conflicts.append(
                MasterResumeConflict(
                    category=incoming.category,
                    key=incoming.key,
                    reason="existing verified fact with the same semantic identity differs",
                    existing_source=semantic.source,
                    existing_value=semantic.value,
                    incoming_value=incoming.value,
                )
            )
            continue

        repo.save(incoming)
        _refresh_category_cache(by_category, incoming.category, repo)
        created += 1
    return created, updated, unchanged, conflicts


def _find_semantic_match(existing_facts: list[CandidateFact], incoming: CandidateFact) -> CandidateFact | None:
    target = fact_semantic_identity(incoming)
    if target is None:
        return None
    for fact in existing_facts:
        if fact.key == incoming.key and fact.category == incoming.category:
            continue
        if fact_semantic_identity(fact) == target:
            return fact
    return None


def _refresh_category_cache(index: dict[str, list[CandidateFact]], category: str, repo: CandidateFactRepo) -> None:
    index[category] = repo.list(category=category)
