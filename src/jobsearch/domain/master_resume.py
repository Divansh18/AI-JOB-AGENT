"""Deterministic master-resume parsing and truth-store mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .candidate import CandidateFact, make_fact
from .taxonomy import canonical_skill

SECTION_HEADERS = (
    "SUMMARY",
    "EXPERIENCE",
    "SKILLS",
    "PROJECTS",
    "CERTIFICATIONS",
    "EDUCATION",
)
PROFILE_LINK_KEYS = {
    "linkedin": "linkedin_url",
    "github": "github_url",
    "portfolio": "portfolio_url",
}
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
PHONE_RE = re.compile(r"\+?[0-9][0-9()\-\s]{7,20}[0-9]")


@dataclass(frozen=True)
class MasterResumeRecord:
    identity: str
    version: int
    file_path: str
    active: bool
    page_limit: int
    file_hash: str
    sections: dict[str, Any] = field(default_factory=dict)
    extracted_text: str | None = None
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_ingested_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "identity": self.identity,
            "version": self.version,
            "file_path": self.file_path,
            "active": self.active,
            "page_limit": self.page_limit,
            "file_hash": self.file_hash,
            "sections": self.sections,
            "has_extracted_text": bool(self.extracted_text),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_ingested_at": self.last_ingested_at.isoformat() if self.last_ingested_at else None,
        }


@dataclass(frozen=True)
class PdfLink:
    label: str
    url: str
    page: int
    top: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "url": self.url,
            "page": self.page,
            "top": self.top,
        }


@dataclass(frozen=True)
class PdfExtraction:
    text: str
    page_count: int
    links: list[PdfLink] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "page_count": self.page_count,
            "links": [link.as_dict() for link in self.links],
        }


@dataclass(frozen=True)
class ParsedSkillItem:
    group: str
    name: str
    raw_line: str

    def as_dict(self) -> dict[str, Any]:
        return {"group": self.group, "name": self.name, "raw_line": self.raw_line}


@dataclass(frozen=True)
class ParsedExperienceEntry:
    title: str
    company: str
    date_text: str | None
    raw_header: str
    bullets: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "company": self.company,
            "date_text": self.date_text,
            "raw_header": self.raw_header,
            "bullets": self.bullets,
        }


@dataclass(frozen=True)
class ParsedProjectEntry:
    name: str
    tagline: str | None
    tech_stack: list[str]
    link_labels: list[str]
    raw_header: str
    bullets: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tagline": self.tagline,
            "tech_stack": self.tech_stack,
            "link_labels": self.link_labels,
            "raw_header": self.raw_header,
            "bullets": self.bullets,
        }


@dataclass(frozen=True)
class ParsedCertificationEntry:
    name: str
    raw_text: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "raw_text": self.raw_text}


@dataclass(frozen=True)
class ParsedEducationEntry:
    institution: str
    degree: str
    graduation_text: str | None
    raw_text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "institution": self.institution,
            "degree": self.degree,
            "graduation_text": self.graduation_text,
            "raw_text": self.raw_text,
        }


@dataclass(frozen=True)
class ParsedMasterResume:
    page_count: int
    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    profile_links: dict[str, str] = field(default_factory=dict)
    summary: str | None = None
    experience: list[ParsedExperienceEntry] = field(default_factory=list)
    skills: list[ParsedSkillItem] = field(default_factory=list)
    projects: list[ParsedProjectEntry] = field(default_factory=list)
    certifications: list[ParsedCertificationEntry] = field(default_factory=list)
    education: list[ParsedEducationEntry] = field(default_factory=list)
    sections_detected: list[str] = field(default_factory=list)
    raw_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_count": self.page_count,
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "profile_links": self.profile_links,
            "summary": self.summary,
            "experience": [entry.as_dict() for entry in self.experience],
            "skills": [entry.as_dict() for entry in self.skills],
            "projects": [entry.as_dict() for entry in self.projects],
            "certifications": [entry.as_dict() for entry in self.certifications],
            "education": [entry.as_dict() for entry in self.education],
            "sections_detected": self.sections_detected,
            "section_counts": section_counts(self),
        }


@dataclass(frozen=True)
class MasterResumeConflict:
    category: str
    key: str
    reason: str
    existing_source: str
    existing_value: Any
    incoming_value: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "key": self.key,
            "reason": self.reason,
            "existing_source": self.existing_source,
            "existing_value": self.existing_value,
            "incoming_value": self.incoming_value,
        }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")


def _normalize_space(value: str) -> str:
    return " ".join((value or "").replace("\xa0", " ").split()).strip()


def _normalize_text(value: str) -> str:
    return _normalize_space(value).lower()


def _normalize_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in (text or "").splitlines():
        line = _normalize_space(raw)
        if line:
            lines.append(line)
    return lines


def _is_section_header(line: str) -> bool:
    return _normalize_space(line).upper() in SECTION_HEADERS


def _split_sections(text: str) -> tuple[list[str], dict[str, list[str]]]:
    header_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in _normalize_lines(text):
        if _is_section_header(line):
            current = line.lower()
            sections.setdefault(current, [])
            continue
        if current is None:
            header_lines.append(line)
        else:
            sections[current].append(line)
    return header_lines, sections


def _strip_bullet(line: str) -> str:
    return re.sub(r"^[\u2022\u25cf•\-]+\s*", "", line).strip()


def _join_lines(lines: list[str]) -> str | None:
    joined = " ".join(_normalize_space(line) for line in lines if _normalize_space(line))
    return joined or None


def _parse_summary(lines: list[str]) -> str | None:
    return _join_lines(lines)


def _looks_like_experience_header(line: str) -> bool:
    return line.count("|") >= 2 and not line.startswith(("●", "•", "-"))


def _parse_experience(lines: list[str]) -> list[ParsedExperienceEntry]:
    entries: list[ParsedExperienceEntry] = []
    current_header: str | None = None
    current_bullets: list[str] = []
    for line in lines:
        if _looks_like_experience_header(line):
            if current_header is not None:
                entries.append(_experience_entry(current_header, current_bullets))
            current_header = line
            current_bullets = []
            continue
        if line.startswith(("●", "•", "-")):
            if current_header is None:
                continue
            current_bullets.append(_strip_bullet(line))
            continue
        if current_bullets:
            current_bullets[-1] = _normalize_space(f"{current_bullets[-1]} {line}")
        elif current_header is not None:
            current_header = _normalize_space(f"{current_header} {line}")
    if current_header is not None:
        entries.append(_experience_entry(current_header, current_bullets))
    return entries


def _experience_entry(raw_header: str, bullets: list[str]) -> ParsedExperienceEntry:
    parts = [_normalize_space(part) for part in raw_header.split("|")]
    parts = [part for part in parts if part]
    title = parts[0] if parts else ""
    company = parts[1] if len(parts) > 1 else ""
    date_text = parts[2] if len(parts) > 2 else None
    return ParsedExperienceEntry(
        title=title,
        company=company,
        date_text=date_text,
        raw_header=raw_header,
        bullets=[_normalize_space(item) for item in bullets if _normalize_space(item)],
    )


def _parse_skills(lines: list[str]) -> list[ParsedSkillItem]:
    grouped: list[str] = []
    current: str | None = None
    for line in lines:
        if ":" in line:
            if current:
                grouped.append(current)
            current = line
        elif current:
            current = _normalize_space(f"{current} {line}")
    if current:
        grouped.append(current)

    items: list[ParsedSkillItem] = []
    for raw_line in grouped:
        group, values = raw_line.split(":", 1)
        for value in values.split(","):
            name = _normalize_space(value)
            if name:
                items.append(ParsedSkillItem(group=_normalize_space(group), name=name, raw_line=raw_line))
    return items


def _parse_projects(lines: list[str]) -> list[ParsedProjectEntry]:
    entries: list[ParsedProjectEntry] = []
    current_header: str | None = None
    current_bullets: list[str] = []
    for line in lines:
        if line.startswith(("●", "•", "-")):
            if current_header is None:
                continue
            current_bullets.append(_strip_bullet(line))
            continue
        if current_header is None:
            current_header = line
            continue
        if current_bullets and _looks_like_project_header(line):
            entries.append(_project_entry(current_header, current_bullets))
            current_header = line
            current_bullets = []
        elif current_bullets:
            current_bullets[-1] = _normalize_space(f"{current_bullets[-1]} {line}")
        else:
            current_header = _normalize_space(f"{current_header} {line}")
    if current_header is not None:
        entries.append(_project_entry(current_header, current_bullets))
    return entries


def _looks_like_project_header(line: str) -> bool:
    return "|" in line


def _project_entry(raw_header: str, bullets: list[str]) -> ParsedProjectEntry:
    parts = [part.strip() for part in re.split(r"\s+[—–]\s+", raw_header) if part.strip()]
    name = parts[0] if parts else raw_header
    tagline: str | None = None
    stack_segment = ""
    if len(parts) >= 3:
        tagline = parts[1]
        stack_segment = parts[2]
    elif len(parts) == 2:
        stack_segment = parts[1]

    pipe_parts = [_normalize_space(part) for part in stack_segment.split("|") if _normalize_space(part)]
    tech_stack = [_normalize_space(item) for item in pipe_parts[0].split(",")] if pipe_parts else []
    tech_stack = [item for item in tech_stack if item]
    link_labels = pipe_parts[1:] if len(pipe_parts) > 1 else []
    return ParsedProjectEntry(
        name=name,
        tagline=tagline,
        tech_stack=tech_stack,
        link_labels=link_labels,
        raw_header=raw_header,
        bullets=[_normalize_space(item) for item in bullets if _normalize_space(item)],
    )


def _parse_certifications(lines: list[str]) -> list[ParsedCertificationEntry]:
    items: list[str] = []
    current: str | None = None
    for line in lines:
        if current is None:
            current = line
            continue
        if "|" in line or _is_section_header(line):
            items.append(current)
            current = line
            continue
        current = _normalize_space(f"{current} {line}")
    if current:
        items.append(current)
    return [
        ParsedCertificationEntry(name=item, raw_text=item)
        for item in items
        if _normalize_space(item)
    ]


def _parse_education(lines: list[str]) -> list[ParsedEducationEntry]:
    items: list[str] = []
    current: str | None = None
    for line in lines:
        if current is None:
            current = line
            continue
        if "|" in line and current:
            items.append(current)
            current = line
            continue
        current = _normalize_space(f"{current} {line}")
    if current:
        items.append(current)

    parsed: list[ParsedEducationEntry] = []
    for raw_text in items:
        institution = raw_text
        degree = ""
        graduation_text: str | None = None
        if "|" in raw_text:
            left, right = raw_text.split("|", 1)
            institution = _normalize_space(left)
            right = _normalize_space(right)
            match = re.search(r"\s+[—–-]\s+Graduated:\s*", right)
            if match:
                degree = _normalize_space(right[:match.start()])
                graduation_text = f"Graduated: {right[match.end():].strip()}"
            else:
                degree = right
        parsed.append(
            ParsedEducationEntry(
                institution=institution,
                degree=degree,
                graduation_text=graduation_text,
                raw_text=raw_text,
            )
        )
    return parsed


def _detect_profile_links(links: list[PdfLink]) -> dict[str, str]:
    detected: dict[str, str] = {}
    for link in sorted(links, key=lambda item: (item.page, -(item.top or 0.0), item.label.lower(), item.url)):
        label = _slug(link.label)
        key = PROFILE_LINK_KEYS.get(label)
        if key is None:
            continue
        if link.top is not None and link.top < 700:
            continue
        if key == "linkedin_url" and "linkedin.com" not in link.url.lower():
            continue
        if key == "github_url" and "github.com" not in link.url.lower():
            continue
        if key == "portfolio_url" and "github.com" in link.url.lower():
            continue
        detected.setdefault(key, link.url.strip())
    return detected


def parse_master_resume_text(
    text: str,
    *,
    page_count: int,
    links: list[PdfLink] | None = None,
) -> ParsedMasterResume:
    header_lines, sections = _split_sections(text)
    full_name = header_lines[0] if header_lines else None
    contact_text = " ".join(header_lines[1:]) if len(header_lines) > 1 else ""
    email_match = EMAIL_RE.search(contact_text)
    phone_match = PHONE_RE.search(contact_text)

    summary = _parse_summary(sections.get("summary", []))
    experience = _parse_experience(sections.get("experience", []))
    skills = _parse_skills(sections.get("skills", []))
    projects = _parse_projects(sections.get("projects", []))
    certifications = _parse_certifications(sections.get("certifications", []))
    education = _parse_education(sections.get("education", []))

    detected = [
        name
        for name, value in (
            ("summary", summary),
            ("experience", experience),
            ("skills", skills),
            ("projects", projects),
            ("certifications", certifications),
            ("education", education),
        )
        if value
    ]

    return ParsedMasterResume(
        page_count=page_count,
        full_name=full_name,
        email=email_match.group(0).strip() if email_match else None,
        phone=_normalize_space(phone_match.group(0)) if phone_match else None,
        profile_links=_detect_profile_links(links or []),
        summary=summary,
        experience=experience,
        skills=skills,
        projects=projects,
        certifications=certifications,
        education=education,
        sections_detected=detected,
        raw_text=text,
    )


def section_counts(parsed: ParsedMasterResume | dict[str, Any]) -> dict[str, int]:
    if isinstance(parsed, dict):
        return {
            "summary": 1 if parsed.get("summary") else 0,
            "experience": len(parsed.get("experience") or []),
            "skills": len(parsed.get("skills") or []),
            "projects": len(parsed.get("projects") or []),
            "certifications": len(parsed.get("certifications") or []),
            "education": len(parsed.get("education") or []),
        }
    return {
        "summary": 1 if parsed.summary else 0,
        "experience": len(parsed.experience),
        "skills": len(parsed.skills),
        "projects": len(parsed.projects),
        "certifications": len(parsed.certifications),
        "education": len(parsed.education),
    }


def _skill_key(name: str) -> str:
    normalized = canonical_skill(_normalize_text(name))
    return _slug(normalized or name)


def _text_mentions_skill(text: str, raw_name: str, normalized_name: str) -> bool:
    haystack = _normalize_text(text)
    haystack_compact = re.sub(r"[^a-z0-9]+", "", haystack)
    for candidate in {raw_name, normalized_name}:
        cleaned = _normalize_text(candidate)
        if not cleaned:
            continue
        if re.search(rf"\b{re.escape(cleaned)}\b", haystack):
            return True
        compact = re.sub(r"[^a-z0-9]+", "", cleaned)
        if compact and compact in haystack_compact:
            return True
    return False


def _derive_skill_facts(parsed: ParsedMasterResume, source: str) -> list[CandidateFact]:
    catalog: dict[str, dict[str, Any]] = {}

    def ensure(raw_name: str, *, group: str | None = None, raw_line: str | None = None, context: str | None = None) -> None:
        name = _normalize_space(raw_name)
        if not name:
            return
        normalized_name = canonical_skill(name.lower())
        key = _skill_key(name)
        item = catalog.setdefault(
            key,
            {
                "name": name,
                "normalized_name": normalized_name,
                "groups": [],
                "raw_lines": [],
                "contexts": set(),
            },
        )
        if group and group not in item["groups"]:
            item["groups"].append(group)
        if raw_line and raw_line not in item["raw_lines"]:
            item["raw_lines"].append(raw_line)
        if context:
            item["contexts"].add(context)

    for skill in parsed.skills:
        ensure(skill.name, group=skill.group, raw_line=skill.raw_line, context="skills")
    for project in parsed.projects:
        for skill in project.tech_stack:
            ensure(skill, group="Project Stack", raw_line=project.raw_header, context="project")

    experience_texts = [" ".join([entry.raw_header, *entry.bullets]) for entry in parsed.experience]
    project_texts = [" ".join([entry.raw_header, *(entry.bullets or [])]) for entry in parsed.projects]
    for item in catalog.values():
        if any(_text_mentions_skill(text, item["name"], item["normalized_name"]) for text in experience_texts):
            item["contexts"].add("work_experience")
        if any(_text_mentions_skill(text, item["name"], item["normalized_name"]) for text in project_texts):
            item["contexts"].add("project")

    facts: list[CandidateFact] = []
    for key in sorted(catalog):
        item = catalog[key]
        contexts = sorted(item["contexts"])
        proficiency = None
        if "work_experience" in item["contexts"]:
            proficiency = "professional"
        elif "project" in item["contexts"]:
            proficiency = "project"
        facts.append(
            make_fact(
                category="skills",
                key=f"skill_{key}",
                value={
                    "name": item["name"],
                    "normalized_name": item["normalized_name"],
                    "groups": item["groups"],
                    "raw_lines": item["raw_lines"],
                    "source_contexts": contexts,
                    "proficiency": proficiency,
                },
                source=source,
                verified=True,
            )
        )
    return facts


def build_master_resume_facts(parsed: ParsedMasterResume, *, source: str) -> list[CandidateFact]:
    facts: list[CandidateFact] = []
    if parsed.full_name:
        facts.append(
            make_fact(
                category="personal/profile",
                key="full_name",
                value=parsed.full_name,
                source=source,
                verified=True,
            )
        )
    if parsed.email:
        facts.append(
            make_fact(
                category="personal/profile",
                key="email",
                value=parsed.email,
                source=source,
                verified=True,
            )
        )
    if parsed.phone:
        facts.append(
            make_fact(
                category="personal/profile",
                key="phone",
                value=parsed.phone,
                source=source,
                verified=True,
            )
        )
    if parsed.summary:
        facts.append(
            make_fact(
                category="personal/profile",
                key="summary",
                value=parsed.summary,
                source=source,
                verified=True,
            )
        )

    for key, url in sorted(parsed.profile_links.items()):
        facts.append(
            make_fact(
                category="links",
                key=key,
                value=url,
                source=source,
                verified=True,
            )
        )

    facts.extend(_derive_skill_facts(parsed, source))

    for idx, entry in enumerate(parsed.experience, start=1):
        experience_key = f"{idx:02d}_{_slug(entry.company)}_{_slug(entry.title)}"
        facts.append(
            make_fact(
                category="work_experience",
                key=experience_key,
                value={
                    "company": entry.company,
                    "title": entry.title,
                    "date_text": entry.date_text,
                    "raw_header": entry.raw_header,
                    "achievements": entry.bullets,
                    "sort_index": idx,
                },
                source=source,
                verified=True,
            )
        )
        for bullet_idx, bullet in enumerate(entry.bullets, start=1):
            facts.append(
                make_fact(
                    category="evidence",
                    key=f"experience_{idx:02d}_bullet_{bullet_idx:02d}_{_slug(bullet)[:40]}",
                    value={
                        "bullet": bullet,
                        "origin": "work_experience",
                        "parent_key": experience_key,
                        "proficiency": "professional",
                    },
                    source=source,
                    verified=True,
                )
            )

    for idx, entry in enumerate(parsed.projects, start=1):
        project_key = f"{idx:02d}_{_slug(entry.name)}"
        value: dict[str, Any] = {
            "name": entry.name,
            "skills": entry.tech_stack,
            "achievements": entry.bullets,
            "raw_header": entry.raw_header,
            "link_labels": entry.link_labels,
            "sort_index": idx,
        }
        if entry.tagline:
            value["role"] = entry.tagline
            value["summary"] = entry.tagline
        facts.append(
            make_fact(
                category="projects",
                key=project_key,
                value=value,
                source=source,
                verified=True,
            )
        )
        for bullet_idx, bullet in enumerate(entry.bullets, start=1):
            facts.append(
                make_fact(
                    category="evidence",
                    key=f"project_{idx:02d}_bullet_{bullet_idx:02d}_{_slug(bullet)[:40]}",
                    value={
                        "bullet": bullet,
                        "origin": "project",
                        "parent_key": project_key,
                        "proficiency": "project",
                    },
                    source=source,
                    verified=True,
                )
            )

    for idx, entry in enumerate(parsed.certifications, start=1):
        facts.append(
            make_fact(
                category="evidence",
                key=f"certification_{idx:02d}_{_slug(entry.name)}",
                value={
                    "bullet": entry.raw_text,
                    "kind": "certification",
                },
                source=source,
                verified=True,
            )
        )

    for idx, entry in enumerate(parsed.education, start=1):
        value = {
            "institution": entry.institution,
            "degree": entry.degree,
            "raw_text": entry.raw_text,
            "sort_index": idx,
        }
        if entry.graduation_text:
            value["graduation_text"] = entry.graduation_text
        facts.append(
            make_fact(
                category="education",
                key=f"{idx:02d}_{_slug(entry.institution)}_{_slug(entry.degree)}",
                value=value,
                source=source,
                verified=True,
            )
        )
    return facts


def fact_semantic_identity(fact: CandidateFact) -> tuple[Any, ...] | None:
    if fact.category == "work_experience" and isinstance(fact.value, dict):
        title = str(fact.value.get("title") or fact.value.get("role") or "")
        company = str(fact.value.get("company") or "")
        if title and company:
            return ("work_experience", _slug(company), _slug(title))
    if fact.category == "projects" and isinstance(fact.value, dict):
        name = str(fact.value.get("name") or "")
        if name:
            return ("projects", _slug(name))
    if fact.category == "education" and isinstance(fact.value, dict):
        institution = str(fact.value.get("institution") or "")
        degree = str(fact.value.get("degree") or fact.value.get("program") or "")
        if institution and degree:
            return ("education", _slug(institution), _slug(degree))
    if fact.category == "skills" and isinstance(fact.value, dict) and isinstance(fact.value.get("name"), str):
        return ("skills", _skill_key(fact.value["name"]))
    if fact.category == "evidence":
        if isinstance(fact.value, str) and fact.value.strip():
            return ("evidence", "bullet", _slug(fact.value))
        if isinstance(fact.value, dict) and isinstance(fact.value.get("bullet"), str):
            return ("evidence", str(fact.value.get("kind") or "bullet"), _slug(fact.value["bullet"]))
    if fact.category in {"personal_profile", "links"}:
        return (fact.category, fact.key)
    return None


def facts_equivalent(existing: CandidateFact, incoming: CandidateFact) -> bool:
    if fact_semantic_identity(existing) != fact_semantic_identity(incoming):
        return False

    if existing.category in {"personal_profile", "links"}:
        return _normalize_text(str(existing.value)) == _normalize_text(str(incoming.value))

    if existing.category == "skills":
        return True

    if existing.category == "education":
        return True

    if existing.category == "evidence":
        existing_text = existing.value if isinstance(existing.value, str) else (existing.value or {}).get("bullet", "")
        incoming_text = incoming.value if isinstance(incoming.value, str) else (incoming.value or {}).get("bullet", "")
        return _normalize_text(str(existing_text)) == _normalize_text(str(incoming_text))

    if existing.category == "work_experience":
        existing_bullets = {
            _normalize_text(item)
            for item in (existing.value or {}).get("achievements", [])
            if isinstance(item, str) and item.strip()
        }
        incoming_bullets = {
            _normalize_text(item)
            for item in (incoming.value or {}).get("achievements", [])
            if isinstance(item, str) and item.strip()
        }
        if not existing_bullets or not incoming_bullets:
            return True
        return existing_bullets <= incoming_bullets or incoming_bullets <= existing_bullets

    if existing.category == "projects":
        existing_bullets = {
            _normalize_text(item)
            for item in (existing.value or {}).get("achievements", [])
            if isinstance(item, str) and item.strip()
        }
        incoming_bullets = {
            _normalize_text(item)
            for item in (incoming.value or {}).get("achievements", [])
            if isinstance(item, str) and item.strip()
        }
        existing_skills = {
            _normalize_text(item)
            for item in (existing.value or {}).get("skills", [])
            if isinstance(item, str) and item.strip()
        }
        incoming_skills = {
            _normalize_text(item)
            for item in (incoming.value or {}).get("skills", [])
            if isinstance(item, str) and item.strip()
        }
        bullet_match = not existing_bullets or not incoming_bullets or existing_bullets <= incoming_bullets or incoming_bullets <= existing_bullets
        skill_match = not existing_skills or not incoming_skills or existing_skills <= incoming_skills or incoming_skills <= existing_skills
        return bullet_match and skill_match

    return existing.value == incoming.value
