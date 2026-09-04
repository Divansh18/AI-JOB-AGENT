"""Build and validate actual resume PDF outputs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

from ..domain.candidate import CandidateFact, build_application_profile
from ..domain.resume_intelligence import (
    MasterResume,
    PAGE_STATUS_MISSING_OUTPUT,
    PAGE_STATUS_NOT_APPLICABLE,
    PAGE_STATUS_OVERFLOW,
    PAGE_STATUS_VALID,
    ResumeChange,
    ResumeContentItem,
    ResumeEducationEntry,
    ResumeExperienceEntry,
    ResumeLinkItem,
    ResumeProjectEntry,
    ResumeSkillItem,
    ResumeSourceModel,
    ResumeTemplate,
    TailoredResume,
    TailoringRecommendation,
    build_resume_source_model,
    validate_tailored_resume,
)
from ..persistence.repositories import CandidateAnswerRepo, CandidateFactRepo, MasterResumeRepo, ResumeVariantRepo
from ..render.resume_pdf import ResumePdfDocument, ResumePdfEntry, ResumePdfHeader, ResumePdfSection, count_pdf_pages, write_resume_pdf
from . import resume_intelligence as resume_service
from .audit import Audit

DEFAULT_SECTION_ORDER = ("summary", "experience", "skills", "projects", "certifications", "education")
MAX_RENDER_ATTEMPTS = 16


class ResumeOutputError(Exception):
    pass


@dataclass(frozen=True)
class OutputLine:
    text: str
    evidence_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MasterLayoutContext:
    full_name: str | None
    section_order: list[str]
    experience_dates: dict[tuple[str, str], str]
    project_headers: dict[str, str]
    skill_groups: list[tuple[str, list[tuple[str, str]]]]
    education_lines: dict[tuple[str, str], str]
    certifications: list[OutputLine]


@dataclass(frozen=True)
class ResumeOutputState:
    full_name: str
    contact_items: list[OutputLine]
    section_order: list[str]
    summary: ResumeContentItem | None
    skills: list[ResumeSkillItem]
    experience: list[ResumeExperienceEntry]
    projects: list[ResumeProjectEntry]
    certifications: list[OutputLine]
    education: list[ResumeEducationEntry]


def build_resume_output(
    conn,
    config,
    job_id: int,
    *,
    analysis=None,
    as_of: date | None = None,
) -> dict[str, Any]:
    master_record = MasterResumeRepo(conn).get_active()
    if master_record is None:
        raise ResumeOutputError("no active master resume is registered")

    prepared = resume_service.prepare_resume_variant(conn, config, job_id, analysis=analysis, as_of=as_of)
    render_fingerprint = _render_fingerprint(prepared, master_record.file_hash)
    repo = ResumeVariantRepo(conn)
    existing = repo.find_by_render_fingerprint(render_fingerprint)
    if (
        existing is not None
        and existing["validation_status"] == "valid"
        and _stored_output_is_available(existing, config.root)
    ):
        return _payload_from_row(existing, reused_existing=True)

    facts = CandidateFactRepo(conn).list(verified=True)
    if prepared.tailored_resume.recommendation.decision == "use_base":
        rendered_resume = _build_use_base_resume(config.root, master_record.file_path, master_record.version, facts, prepared.tailored_resume)
    elif prepared.tailored_resume.recommendation.decision == "poor_fit":
        rendered_resume = _build_poor_fit_resume(prepared.tailored_resume)
    else:
        rendered_resume = _build_tailored_resume(
            config.root,
            master_record,
            facts,
            prepared.tailored_resume,
            render_fingerprint,
        )

    validation = validate_tailored_resume(rendered_resume, prepared.source_model, as_of=as_of)
    content = {
        "analysis": prepared.analysis.as_dict(),
        "source_model": prepared.source_model.as_dict(),
        "fit_report": prepared.fit_report.as_dict(),
        "resume": rendered_resume.as_dict(),
    }

    if existing is None:
        resume_id = repo.create(
            job_id=job_id,
            fit_score=float(prepared.fit_report.overall_fit_score),
            source_truth_version=resume_service.TRUTH_STORE_VERSION,
            source_truth_hash=prepared.source_truth_hash,
            master_resume_identity=rendered_resume.master_resume.identity,
            master_resume_version=rendered_resume.master_resume.version,
            template_identity=rendered_resume.template.identity,
            page_limit=rendered_resume.page_limit,
            tailoring_decision=rendered_resume.recommendation.decision,
            tailoring_reasons=rendered_resume.recommendation.reasons,
            tailoring_evidence_refs=rendered_resume.recommendation.evidence_refs,
            content=content,
            validation_status=validation.status,
            validation_errors=[issue.as_dict() for issue in validation.issues],
            page_validation_status=rendered_resume.page_validation_status,
            output_pdf_path=rendered_resume.output_pdf_path,
            page_count=rendered_resume.page_count,
            output_evidence_refs=rendered_resume.evidence_refs_used,
            render_fingerprint=render_fingerprint,
        )
    else:
        resume_id = int(existing["id"])
        repo.update_rendered_output(
            resume_id,
            content=content,
            validation_status=validation.status,
            validation_errors=[issue.as_dict() for issue in validation.issues],
            page_validation_status=rendered_resume.page_validation_status,
            output_pdf_path=rendered_resume.output_pdf_path,
            page_count=rendered_resume.page_count,
            output_evidence_refs=rendered_resume.evidence_refs_used,
            render_fingerprint=render_fingerprint,
        )

    Audit(conn).human(
        "resume_output_built",
        "resume_variant",
        resume_id,
        job_id=job_id,
        fit_score=prepared.fit_report.overall_fit_score,
        validation_status=validation.status,
        tailoring_decision=rendered_resume.recommendation.decision,
        page_validation_status=rendered_resume.page_validation_status,
        output_pdf_path=rendered_resume.output_pdf_path,
    )
    row = repo.get(resume_id)
    if row is None:
        raise ResumeOutputError(f"resume variant {resume_id} was not persisted")
    return _payload_from_row(row, reused_existing=False)


def validate_resume_output(
    conn,
    config,
    resume_id: int,
    *,
    as_of: date | None = None,
) -> dict[str, Any]:
    repo = ResumeVariantRepo(conn)
    row = repo.get(resume_id)
    if row is None:
        raise ResumeOutputError(f"resume variant {resume_id} not found")

    content = json.loads(row["content_json"]) if row["content_json"] else {}
    source_model = _source_model_from_payload(content.get("source_model")) or _current_source_model(conn)
    resume = _tailored_resume_from_payload(content.get("resume") or {})

    updated = _revalidate_rendered_resume(config.root, resume)
    validation = validate_tailored_resume(updated, source_model, as_of=as_of)
    content["source_model"] = source_model.as_dict()
    content["resume"] = updated.as_dict()
    repo.update_rendered_output(
        resume_id,
        content=content,
        validation_status=validation.status,
        validation_errors=[issue.as_dict() for issue in validation.issues],
        page_validation_status=updated.page_validation_status,
        output_pdf_path=updated.output_pdf_path,
        page_count=updated.page_count,
        output_evidence_refs=updated.evidence_refs_used,
        render_fingerprint=row["render_fingerprint"],
    )

    Audit(conn).human(
        "resume_output_validated",
        "resume_variant",
        resume_id,
        validation_status=validation.status,
        page_validation_status=updated.page_validation_status,
        output_pdf_path=updated.output_pdf_path,
        page_count=updated.page_count,
    )
    refreshed = repo.get(resume_id)
    if refreshed is None:
        raise ResumeOutputError(f"resume variant {resume_id} disappeared during validation")
    return _payload_from_row(refreshed, reused_existing=False)


def _build_use_base_resume(
    root: Path,
    stored_path: str,
    version: int,
    facts: list[CandidateFact],
    tailored: TailoredResume,
) -> TailoredResume:
    absolute_path, normalized_path = _resolve_stored_path(root, stored_path)
    if not absolute_path.exists():
        raise ResumeOutputError(f"registered master resume file is missing: {stored_path}")
    page_count = count_pdf_pages(absolute_path)
    page_status = PAGE_STATUS_VALID if page_count == tailored.page_limit else PAGE_STATUS_OVERFLOW
    evidence_refs = _master_resume_evidence_refs(facts, version)
    return replace(
        tailored,
        template=replace(tailored.template, render_validation_status=page_status),
        page_validation_status=page_status,
        output_pdf_path=normalized_path,
        page_count=page_count,
        evidence_refs_used=evidence_refs,
    )


def _build_poor_fit_resume(tailored: TailoredResume) -> TailoredResume:
    return replace(
        tailored,
        template=replace(tailored.template, render_validation_status=PAGE_STATUS_NOT_APPLICABLE),
        page_validation_status=PAGE_STATUS_NOT_APPLICABLE,
        output_pdf_path=None,
        page_count=None,
        evidence_refs_used=[],
    )


def _build_tailored_resume(
    root: Path,
    master_record,
    facts: list[CandidateFact],
    tailored: TailoredResume,
    render_fingerprint: str,
) -> TailoredResume:
    if tailored.preview is None:
        raise ResumeOutputError("tailor decision is missing a preview model")

    context = _master_layout_context(master_record, facts)
    state = _build_output_state(facts, tailored.preview, context)
    base_name = f"resume-job-{tailored.job_id}-{render_fingerprint[:12]}"
    tmp_dir = root / "tmp" / "pdfs"
    final_relative = Path("output") / "pdf" / f"{base_name}.pdf"
    final_path = root / final_relative
    attempts: list[Path] = []
    current = state
    last_page_count: int | None = None

    for attempt in range(1, MAX_RENDER_ATTEMPTS + 1):
        attempt_path = tmp_dir / f"{base_name}-attempt{attempt:02d}.pdf"
        attempts.append(attempt_path)
        write_resume_pdf(_pdf_document(current, context), attempt_path)
        last_page_count = count_pdf_pages(attempt_path)
        if last_page_count == tailored.page_limit:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                final_path.unlink()
            attempt_path.replace(final_path)
            _cleanup_attempts(path for path in attempts if path != final_path)
            return replace(
                tailored,
                template=replace(tailored.template, render_validation_status=PAGE_STATUS_VALID),
                page_validation_status=PAGE_STATUS_VALID,
                preview=_preview_from_state(tailored.preview, current),
                output_pdf_path=final_relative.as_posix(),
                page_count=last_page_count,
                evidence_refs_used=_output_state_evidence_refs(current),
            )
        next_state = _trim_output_state(current)
        if next_state is None:
            _cleanup_attempts(attempts)
            return replace(
                tailored,
                template=replace(tailored.template, render_validation_status=PAGE_STATUS_OVERFLOW),
                page_validation_status=PAGE_STATUS_OVERFLOW,
                preview=_preview_from_state(tailored.preview, current),
                output_pdf_path=None,
                page_count=last_page_count,
                evidence_refs_used=_output_state_evidence_refs(current),
            )
        current = next_state

    _cleanup_attempts(attempts)
    return replace(
        tailored,
        template=replace(tailored.template, render_validation_status=PAGE_STATUS_OVERFLOW),
        page_validation_status=PAGE_STATUS_OVERFLOW,
        preview=_preview_from_state(tailored.preview, current),
        output_pdf_path=None,
        page_count=last_page_count,
        evidence_refs_used=_output_state_evidence_refs(current),
    )


def _build_output_state(
    facts: list[CandidateFact],
    preview: ResumeSourceModel,
    context: MasterLayoutContext,
) -> ResumeOutputState:
    profile = build_application_profile(facts)
    full_name = profile.full_name or context.full_name
    if not full_name:
        raise ResumeOutputError("verified full_name is missing from the truth store")
    return ResumeOutputState(
        full_name=full_name,
        contact_items=_contact_items(profile, preview.links),
        section_order=_ordered_sections(context.section_order, preview),
        summary=preview.summary,
        skills=list(preview.skills),
        experience=list(preview.experience),
        projects=list(preview.projects),
        certifications=list(context.certifications),
        education=list(preview.education),
    )


def _pdf_document(state: ResumeOutputState, context: MasterLayoutContext) -> ResumePdfDocument:
    sections: list[ResumePdfSection] = []
    for name in state.section_order:
        if name == "summary" and state.summary:
            sections.append(ResumePdfSection(title="Summary", lines=[_ascii_text(state.summary.text)]))
        elif name == "experience" and state.experience:
            sections.append(
                ResumePdfSection(
                    title="Experience",
                    entries=[
                        ResumePdfEntry(
                            heading=_ascii_text(f"{entry.title}  |  {entry.company}"),
                            meta_right=_ascii_text(_experience_date(entry, context)),
                            summary=_ascii_text(entry.summary.text) if entry.summary else None,
                            bullets=[_ascii_text(item.text) for item in entry.bullets],
                        )
                        for entry in state.experience
                    ],
                )
            )
        elif name == "skills" and state.skills:
            sections.append(
                ResumePdfSection(
                    title="Skills",
                    lines=[_ascii_text(line.text) for line in _skill_lines(state.skills, context)],
                )
            )
        elif name == "projects" and state.projects:
            sections.append(
                ResumePdfSection(
                    title="Projects",
                    entries=[
                        ResumePdfEntry(
                            heading=_ascii_text(_project_heading(entry, context)),
                            summary=_ascii_text(entry.summary.text) if entry.summary else None,
                            bullets=[_ascii_text(item.text) for item in entry.bullets],
                        )
                        for entry in state.projects
                    ],
                )
            )
        elif name == "certifications" and state.certifications:
            sections.append(
                ResumePdfSection(
                    title="Certifications",
                    lines=[_ascii_text(line.text) for line in state.certifications],
                )
            )
        elif name == "education" and state.education:
            sections.append(
                ResumePdfSection(
                    title="Education",
                    lines=[_ascii_text(_education_line(entry, context)) for entry in state.education],
                )
            )
    return ResumePdfDocument(
        header=ResumePdfHeader(
            full_name=_ascii_text(state.full_name),
            contact_items=[_ascii_text(item.text) for item in state.contact_items],
        ),
        sections=sections,
    )


def _trim_output_state(state: ResumeOutputState) -> ResumeOutputState | None:
    for index in range(len(state.projects) - 1, -1, -1):
        entry = state.projects[index]
        if len(entry.bullets) > 1:
            trimmed = replace(entry, bullets=entry.bullets[:-1])
            return replace(state, projects=[*state.projects[:index], trimmed, *state.projects[index + 1 :]])
    if len(state.skills) > 14:
        return replace(state, skills=state.skills[:-1])
    if len(state.projects) > 3:
        return replace(state, projects=state.projects[:-1])
    for index in range(len(state.projects) - 1, -1, -1):
        entry = state.projects[index]
        if entry.summary is not None:
            trimmed = replace(entry, summary=None)
            return replace(state, projects=[*state.projects[:index], trimmed, *state.projects[index + 1 :]])
    for index in range(len(state.experience) - 1, -1, -1):
        entry = state.experience[index]
        minimum = 2 if index == 0 else 1
        if len(entry.bullets) > minimum:
            trimmed = replace(entry, bullets=entry.bullets[:-1])
            return replace(state, experience=[*state.experience[:index], trimmed, *state.experience[index + 1 :]])
    if state.certifications:
        return replace(state, certifications=state.certifications[:-1])
    trimmed_summary = _trim_summary(state.summary)
    if trimmed_summary != state.summary:
        return replace(state, summary=trimmed_summary)
    if len(state.projects) > 2:
        return replace(state, projects=state.projects[:-1])
    if len(state.skills) > 8:
        return replace(state, skills=state.skills[:-1])
    return None


def _trim_summary(summary: ResumeContentItem | None) -> ResumeContentItem | None:
    if summary is None:
        return None
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", summary.text) if part.strip()]
    if len(sentences) > 2:
        return replace(summary, text=" ".join(sentences[:2]))
    if len(sentences) == 2:
        return replace(summary, text=sentences[0])
    if len(sentences) == 1:
        return None
    return summary


def _preview_from_state(preview: ResumeSourceModel, state: ResumeOutputState) -> ResumeSourceModel:
    return replace(
        preview,
        summary=state.summary,
        skills=state.skills,
        experience=state.experience,
        projects=state.projects,
    )


def _output_state_evidence_refs(state: ResumeOutputState) -> list[str]:
    refs: list[str] = []
    refs.extend(ref for item in state.contact_items for ref in item.evidence_refs)
    if state.summary:
        refs.extend(state.summary.evidence_refs)
    refs.extend(ref for item in state.skills for ref in item.evidence_refs)
    for entry in state.experience:
        refs.extend(entry.evidence_refs)
        if entry.summary:
            refs.extend(entry.summary.evidence_refs)
        refs.extend(ref for bullet in entry.bullets for ref in bullet.evidence_refs)
    for entry in state.projects:
        refs.extend(entry.evidence_refs)
        if entry.summary:
            refs.extend(entry.summary.evidence_refs)
        refs.extend(ref for bullet in entry.bullets for ref in bullet.evidence_refs)
        refs.extend(ref for skill in entry.skills for ref in skill.evidence_refs)
    refs.extend(ref for item in state.certifications for ref in item.evidence_refs)
    refs.extend(ref for item in state.education for ref in item.evidence_refs)
    return _dedupe(refs)


def _contact_items(profile, links: list[ResumeLinkItem]) -> list[OutputLine]:
    items: list[OutputLine] = []
    if profile.email:
        items.append(OutputLine(text=profile.email, evidence_refs=[_provenance_ref(profile.provenance.get("email"))]))
    if profile.phone:
        items.append(OutputLine(text=profile.phone, evidence_refs=[_provenance_ref(profile.provenance.get("phone"))]))
    for link in links:
        items.append(OutputLine(text=link.label, evidence_refs=link.evidence_refs))
    out: list[OutputLine] = []
    seen: set[str] = set()
    for item in items:
        text = item.text.strip()
        if not text or text in seen:
            continue
        out.append(OutputLine(text=text, evidence_refs=[ref for ref in item.evidence_refs if ref]))
        seen.add(text)
    return out


def _ordered_sections(section_order: list[str], preview: ResumeSourceModel) -> list[str]:
    present = {
        "summary": preview.summary is not None,
        "experience": bool(preview.experience),
        "skills": bool(preview.skills),
        "projects": bool(preview.projects),
        "certifications": True,
        "education": bool(preview.education),
    }
    ordered = [name for name in section_order if present.get(name)]
    for name in DEFAULT_SECTION_ORDER:
        if name not in ordered and present.get(name):
            ordered.append(name)
    return ordered


def _experience_date(entry: ResumeExperienceEntry, context: MasterLayoutContext) -> str:
    key = (_normalize(entry.company), _normalize(entry.title))
    if key in context.experience_dates:
        return context.experience_dates[key]
    return _format_date_range(entry.start_date, entry.end_date)


def _format_date_range(start_date: str | None, end_date: str | None) -> str:
    start = (start_date or "").strip()
    end = (end_date or "").strip()
    if start and end:
        return f"{start} - {end}"
    return start or end or ""


def _project_heading(entry: ResumeProjectEntry, context: MasterLayoutContext) -> str:
    header = context.project_headers.get(_normalize(entry.name))
    if header:
        return header
    parts = [entry.name]
    if entry.role:
        parts.append(entry.role)
    skills = ", ".join(_display_skill_name(item.skill) for item in entry.skills[:5])
    if skills:
        parts.append(skills)
    return " - ".join(part for part in parts if part)


def _education_line(entry: ResumeEducationEntry, context: MasterLayoutContext) -> str:
    key = (_normalize(entry.institution), _normalize(entry.degree))
    if key in context.education_lines:
        return context.education_lines[key]
    parts = [entry.institution, entry.degree]
    if entry.field_of_study:
        parts.append(entry.field_of_study)
    return " - ".join(part for part in parts if part)


def _skill_lines(skills: list[ResumeSkillItem], context: MasterLayoutContext) -> list[OutputLine]:
    ordered_skills = []
    refs_by_skill: dict[str, list[str]] = {}
    seen: set[str] = set()
    for item in skills:
        refs_by_skill.setdefault(item.skill, [])
        refs_by_skill[item.skill].extend(item.evidence_refs)
        if item.skill not in seen:
            ordered_skills.append(item.skill)
            seen.add(item.skill)

    display_names: dict[str, str] = {}
    lines: list[OutputLine] = []
    grouped: set[str] = set()
    for group, members in context.skill_groups:
        members_by_skill = {skill: display for skill, display in members}
        display_names.update(members_by_skill)
        present = [skill for skill in ordered_skills if skill in members_by_skill]
        if not present:
            continue
        grouped.update(present)
        lines.append(
            OutputLine(
                text=f"{group}: {', '.join(members_by_skill.get(skill, _display_skill_name(skill)) for skill in present)}",
                evidence_refs=_dedupe(ref for skill in present for ref in refs_by_skill.get(skill, [])),
            )
        )

    extras = [skill for skill in ordered_skills if skill not in grouped]
    if extras:
        lines.append(
            OutputLine(
                text="Additional: " + ", ".join(display_names.get(skill, _display_skill_name(skill)) for skill in extras),
                evidence_refs=_dedupe(ref for skill in extras for ref in refs_by_skill.get(skill, [])),
            )
        )
    if lines:
        return lines
    if not ordered_skills:
        return []
    chunk_size = 6
    fallback: list[OutputLine] = []
    for start in range(0, len(ordered_skills), chunk_size):
        chunk = ordered_skills[start : start + chunk_size]
        fallback.append(
            OutputLine(
                text=", ".join(_display_skill_name(skill) for skill in chunk),
                evidence_refs=_dedupe(ref for skill in chunk for ref in refs_by_skill.get(skill, [])),
            )
        )
    return fallback


def _master_layout_context(master_record, facts: list[CandidateFact]) -> MasterLayoutContext:
    sections = master_record.sections or {}
    section_order = [name for name in sections.get("sections_detected", []) if name in DEFAULT_SECTION_ORDER]
    if not section_order:
        section_order = list(DEFAULT_SECTION_ORDER)

    experience_dates: dict[tuple[str, str], str] = {}
    for entry in sections.get("experience") or []:
        experience_dates[(_normalize(entry.get("company")), _normalize(entry.get("title")))] = str(entry.get("date_text") or "")

    project_headers = {
        _normalize(entry.get("name")): str(entry.get("raw_header") or entry.get("name") or "")
        for entry in (sections.get("projects") or [])
        if entry.get("name")
    }

    skill_groups: list[tuple[str, list[tuple[str, str]]]] = []
    grouped: dict[str, list[tuple[str, str]]] = {}
    for item in sections.get("skills") or []:
        group = str(item.get("group") or "").strip()
        name = str(item.get("name") or "").strip()
        if not group or not name:
            continue
        grouped.setdefault(group, [])
        grouped[group].append((_canonicalize_skill(name), name))
    for group, items in grouped.items():
        skill_groups.append((group, items))

    education_lines = {
        (_normalize(entry.get("institution")), _normalize(entry.get("degree"))): str(entry.get("raw_text") or "")
        for entry in (sections.get("education") or [])
        if entry.get("institution") and entry.get("degree")
    }

    certification_refs = _certification_refs(facts, master_record.version)
    certifications = [
        OutputLine(
            text=str(entry.get("raw_text") or entry.get("name") or "").strip(),
            evidence_refs=certification_refs.get(str(entry.get("raw_text") or entry.get("name") or "").strip(), []),
        )
        for entry in (sections.get("certifications") or [])
        if str(entry.get("raw_text") or entry.get("name") or "").strip()
    ]

    return MasterLayoutContext(
        full_name=sections.get("full_name"),
        section_order=section_order,
        experience_dates=experience_dates,
        project_headers=project_headers,
        skill_groups=skill_groups,
        education_lines=education_lines,
        certifications=certifications,
    )


def _certification_refs(facts: list[CandidateFact], version: int) -> dict[str, list[str]]:
    expected_source = f"master_resume_v{version}"
    refs: dict[str, list[str]] = {}
    for fact in facts:
        if fact.source != expected_source or fact.category != "evidence" or not isinstance(fact.value, dict):
            continue
        if fact.value.get("kind") != "certification":
            continue
        text = str(fact.value.get("bullet") or "").strip()
        if not text:
            continue
        refs[text] = [_fact_ref(fact)]
    return refs


def _master_resume_evidence_refs(facts: list[CandidateFact], version: int) -> list[str]:
    source = f"master_resume_v{version}"
    return _dedupe(_fact_ref(fact) for fact in facts if fact.source == source and fact.verified)


def _revalidate_rendered_resume(root: Path, resume: TailoredResume) -> TailoredResume:
    if resume.recommendation.decision == "poor_fit":
        return replace(
            resume,
            template=replace(resume.template, render_validation_status=PAGE_STATUS_NOT_APPLICABLE),
            page_validation_status=PAGE_STATUS_NOT_APPLICABLE,
            output_pdf_path=None,
            page_count=None,
            evidence_refs_used=resume.evidence_refs_used or [],
        )

    if not resume.output_pdf_path:
        return replace(
            resume,
            template=replace(resume.template, render_validation_status=PAGE_STATUS_MISSING_OUTPUT),
            page_validation_status=PAGE_STATUS_MISSING_OUTPUT,
            output_pdf_path=None,
            page_count=None,
        )

    absolute_path, normalized_path = _resolve_path_no_check(root, resume.output_pdf_path)
    if not absolute_path.exists():
        return replace(
            resume,
            template=replace(resume.template, render_validation_status=PAGE_STATUS_MISSING_OUTPUT),
            page_validation_status=PAGE_STATUS_MISSING_OUTPUT,
            output_pdf_path=normalized_path,
            page_count=None,
        )

    page_count = count_pdf_pages(absolute_path)
    page_status = PAGE_STATUS_VALID if page_count == resume.page_limit else PAGE_STATUS_OVERFLOW
    return replace(
        resume,
        template=replace(resume.template, render_validation_status=page_status),
        page_validation_status=page_status,
        output_pdf_path=normalized_path,
        page_count=page_count,
    )


def _current_source_model(conn) -> ResumeSourceModel:
    facts = CandidateFactRepo(conn).list(verified=True)
    answers = CandidateAnswerRepo(conn).list(verified=True)
    return build_resume_source_model(facts, answers)


def _payload_from_row(row, *, reused_existing: bool) -> dict[str, Any]:
    content = json.loads(row["content_json"]) if row["content_json"] else {}
    errors = json.loads(row["validation_errors"]) if row["validation_errors"] else []
    output_evidence_refs = json.loads(row["output_evidence_refs"]) if row["output_evidence_refs"] else []
    resume = dict(content.get("resume") or {})
    resume["output_pdf_path"] = row["output_pdf_path"]
    resume["page_count"] = row["page_count"]
    resume["page_validation_status"] = row["page_validation_status"]
    resume["evidence_refs_used"] = output_evidence_refs
    if isinstance(resume.get("template"), dict):
        resume["template"]["render_validation_status"] = row["page_validation_status"]
    return {
        "resume_id": row["id"],
        "analysis": content.get("analysis") or {},
        "source_model": content.get("source_model") or {},
        "fit_report": content.get("fit_report") or {},
        "tailored_resume": resume,
        "validation": {
            "status": row["validation_status"],
            "ok": row["validation_status"] == "valid" and not errors,
            "issues": errors,
        },
        "source_truth_hash": row["source_truth_hash"],
        "reused_existing": reused_existing,
    }


def _render_fingerprint(prepared, master_file_hash: str) -> str:
    payload = {
        "analysis": prepared.analysis.as_dict(),
        "source_model": prepared.source_model.as_dict(),
        "fit_report": prepared.fit_report.as_dict(),
        "resume": prepared.tailored_resume.as_dict(),
        "master_file_hash": master_file_hash,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def _stored_output_is_available(row, root: Path) -> bool:
    decision = row["tailoring_decision"]
    if decision == "poor_fit":
        return True
    path = row["output_pdf_path"]
    if not path:
        return False
    absolute_path, _ = _resolve_path_no_check(root, path)
    return absolute_path.exists()


def _resolve_stored_path(root: Path, stored_path: str) -> tuple[Path, str]:
    absolute_path, normalized_path = _resolve_path_no_check(root, stored_path)
    if not absolute_path.exists():
        raise ResumeOutputError(f"resume PDF is missing: {stored_path}")
    return absolute_path, normalized_path


def _resolve_path_no_check(root: Path, stored_path: str) -> tuple[Path, str]:
    raw_path = Path(stored_path).expanduser()
    absolute_path = raw_path if raw_path.is_absolute() else (root / raw_path)
    absolute_path = absolute_path.resolve()
    try:
        normalized = absolute_path.relative_to(root).as_posix()
    except ValueError:
        normalized = str(absolute_path)
    return absolute_path, normalized


def _cleanup_attempts(paths) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def _tailored_resume_from_payload(payload: dict[str, Any]) -> TailoredResume:
    return TailoredResume(
        job_id=int(payload.get("job_id") or 0),
        target_role=str(payload.get("target_role") or ""),
        master_resume=MasterResume(**(payload.get("master_resume") or {})),
        template=ResumeTemplate(**(payload.get("template") or {})),
        page_limit=int(payload.get("page_limit") or 1),
        page_validation_status=str(payload.get("page_validation_status") or PAGE_STATUS_MISSING_OUTPUT),
        recommendation=TailoringRecommendation(**(payload.get("recommendation") or {})),
        changes=[ResumeChange(**item) for item in (payload.get("changes") or [])],
        preview=_source_model_from_payload(payload.get("preview")),
        output_pdf_path=payload.get("output_pdf_path"),
        page_count=payload.get("page_count"),
        evidence_refs_used=list(payload.get("evidence_refs_used") or []),
    )


def _source_model_from_payload(payload: dict[str, Any] | None) -> ResumeSourceModel | None:
    if not payload:
        return None
    return ResumeSourceModel(
        summary=_content_item_from_payload(payload.get("summary")),
        skills=[ResumeSkillItem(**item) for item in (payload.get("skills") or [])],
        experience=[_experience_entry_from_payload(item) for item in (payload.get("experience") or [])],
        projects=[_project_entry_from_payload(item) for item in (payload.get("projects") or [])],
        education=[ResumeEducationEntry(**item) for item in (payload.get("education") or [])],
        links=[ResumeLinkItem(**item) for item in (payload.get("links") or [])],
        evidence_bullets=[ResumeContentItem(**item) for item in (payload.get("evidence_bullets") or [])],
    )


def _content_item_from_payload(payload: dict[str, Any] | None) -> ResumeContentItem | None:
    if not payload:
        return None
    return ResumeContentItem(**payload)


def _experience_entry_from_payload(payload: dict[str, Any]) -> ResumeExperienceEntry:
    return ResumeExperienceEntry(
        company=str(payload.get("company") or ""),
        title=str(payload.get("title") or ""),
        location=payload.get("location"),
        start_date=payload.get("start_date"),
        end_date=payload.get("end_date"),
        summary=_content_item_from_payload(payload.get("summary")),
        bullets=[ResumeContentItem(**item) for item in (payload.get("bullets") or [])],
        evidence_refs=list(payload.get("evidence_refs") or []),
    )


def _project_entry_from_payload(payload: dict[str, Any]) -> ResumeProjectEntry:
    return ResumeProjectEntry(
        name=str(payload.get("name") or ""),
        role=payload.get("role"),
        link=payload.get("link"),
        summary=_content_item_from_payload(payload.get("summary")),
        bullets=[ResumeContentItem(**item) for item in (payload.get("bullets") or [])],
        skills=[ResumeSkillItem(**item) for item in (payload.get("skills") or [])],
        evidence_refs=list(payload.get("evidence_refs") or []),
    )


def _provenance_ref(provenance: dict[str, Any] | None) -> str | None:
    if not provenance:
        return None
    fact_id = provenance.get("fact_id")
    selector = provenance.get("selector")
    if fact_id is not None:
        return f"fact:{fact_id}"
    return selector


def _fact_ref(fact: CandidateFact) -> str:
    return f"fact:{fact.id}" if fact.id is not None else fact.selector


def _display_skill_name(skill: str) -> str:
    display_names = {
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "node.js": "Node.js",
        "react.js": "React.js",
        "next.js": "Next.js",
        "express.js": "Express.js",
        "rest apis": "REST APIs",
        "ci/cd": "CI/CD",
        "llm orchestration": "LLM Orchestration",
        "aws": "AWS",
        "gcp": "GCP",
        "sql": "SQL",
    }
    return display_names.get(skill, skill.title())


def _canonicalize_skill(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _ascii_text(value: str) -> str:
    return (
        (value or "")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2011", "-")
        .replace("\xa0", " ")
        .strip()
    )


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _dedupe(items) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out
