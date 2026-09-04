"""Deterministic local PDF rendering for resume variants."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

LETTER_PAGE_SIZE = (612.0, 792.0)


@dataclass(frozen=True)
class ResumePdfHeader:
    full_name: str
    contact_items: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResumePdfEntry:
    heading: str
    meta_right: str | None = None
    summary: str | None = None
    bullets: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResumePdfSection:
    title: str
    lines: list[str] = field(default_factory=list)
    entries: list[ResumePdfEntry] = field(default_factory=list)


@dataclass(frozen=True)
class ResumePdfDocument:
    header: ResumePdfHeader
    sections: list[ResumePdfSection]
    page_size: tuple[float, float] = LETTER_PAGE_SIZE


def write_resume_pdf(document: ResumePdfDocument, output_path: Path) -> None:
    _ensure_pdf_runtime()

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=document.page_size,
        leftMargin=34,
        rightMargin=34,
        topMargin=26,
        bottomMargin=24,
    )

    stylesheet = getSampleStyleSheet()
    name_style = ParagraphStyle(
        "resume_name",
        parent=stylesheet["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    contact_style = ParagraphStyle(
        "resume_contact",
        parent=stylesheet["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=10.2,
        alignment=TA_CENTER,
        spaceAfter=5,
    )
    section_style = ParagraphStyle(
        "resume_section",
        parent=stylesheet["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.6,
        leading=11,
        alignment=TA_LEFT,
        spaceAfter=1,
    )
    body_style = ParagraphStyle(
        "resume_body",
        parent=stylesheet["Normal"],
        fontName="Helvetica",
        fontSize=8.4,
        leading=9.8,
        alignment=TA_LEFT,
        spaceAfter=1.5,
    )
    entry_left_style = ParagraphStyle(
        "resume_entry_left",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=8.7,
        leading=10.1,
        spaceAfter=0,
    )
    entry_right_style = ParagraphStyle(
        "resume_entry_right",
        parent=body_style,
        fontName="Helvetica",
        fontSize=8.3,
        leading=9.8,
        alignment=TA_RIGHT,
        spaceAfter=0,
    )
    bullet_style = ParagraphStyle(
        "resume_bullet",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=0,
        spaceAfter=0.8,
    )

    story = [
        Paragraph(escape(document.header.full_name), name_style),
    ]
    if document.header.contact_items:
        story.append(Paragraph(escape("  |  ".join(document.header.contact_items)), contact_style))

    for index, section in enumerate(document.sections):
        if index > 0:
            story.append(Spacer(1, 3))
        story.append(Paragraph(escape(section.title.upper()), section_style))
        story.append(HRFlowable(width="100%", thickness=0.75, color=colors.black, spaceBefore=0, spaceAfter=3))
        for line in section.lines:
            story.append(Paragraph(_escape_multiline(line), body_style))
        for entry in section.entries:
            story.extend(
                _entry_story(
                    entry,
                    doc.width,
                    Paragraph,
                    Spacer,
                    Table,
                    TableStyle,
                    entry_left_style,
                    entry_right_style,
                    body_style,
                    bullet_style,
                )
            )

    doc.build(story)


def count_pdf_pages(path: Path) -> int:
    _ensure_pdf_runtime()
    from pypdf import PdfReader

    return len(PdfReader(str(path)).pages)


def _entry_story(
    entry: ResumePdfEntry,
    width: float,
    paragraph_cls,
    spacer_cls,
    table_cls,
    table_style_cls,
    entry_left_style,
    entry_right_style,
    body_style,
    bullet_style,
) -> list:
    story: list = []
    if entry.meta_right:
        table = table_cls(
            [
                [
                    paragraph_cls(escape(entry.heading), entry_left_style),
                    paragraph_cls(escape(entry.meta_right), entry_right_style),
                ]
            ],
            colWidths=[max(120, width - 104), 104],
        )
        table.setStyle(
            table_style_cls(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)
    else:
        story.append(paragraph_cls(escape(entry.heading), entry_left_style))
    if entry.summary:
        story.append(paragraph_cls(_escape_multiline(entry.summary), body_style))
    for bullet in entry.bullets:
        story.append(paragraph_cls(_escape_multiline(f"- {bullet}"), bullet_style))
    story.append(spacer_cls(1, 1.5))
    return story


def _escape_multiline(text: str) -> str:
    return "<br/>".join(escape(part) for part in text.splitlines())


def _ensure_pdf_runtime() -> None:
    try:
        import reportlab  # noqa: F401
        import pypdf  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    importlib.invalidate_caches()
    for candidate in _pdf_site_packages_candidates():
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)
    import reportlab  # noqa: F401
    import pypdf  # noqa: F401


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
    out: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        if not item:
            continue
        candidate = Path(item).expanduser()
        text = str(candidate)
        if text in seen or not candidate.exists():
            continue
        out.append(candidate)
        seen.add(text)
    return out
