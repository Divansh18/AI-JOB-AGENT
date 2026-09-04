"""Deterministic job analysis, evidence matching, and safe resume tailoring."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable

from .candidate import CandidateFact, VerifiedAnswer, build_application_profile
from .models import Job
from .normalize import normalize_title
from .taxonomy import canonical_skill, classify_title

EXPLICIT_SUMMARY_KEYS = {
    "personal_profile:summary",
    "personal_profile:professional_summary",
    "personal_profile:headline",
}
SUMMARY_ANSWER_KEYS = {
    "short_experience_summary",
    "experience_summary",
    "profile_summary",
}
REQUIRED_CUES = (
    "required",
    "requirements",
    "must have",
    "must possess",
    "need",
    "needs",
    "you have",
    "qualification",
    "we require",
)
PREFERRED_CUES = (
    "preferred",
    "nice to have",
    "nice-to-have",
    "bonus",
    "plus",
    "good to have",
    "ideally",
)
EXPERIENCE_CUES = (
    "experience with",
    "hands-on",
    "production experience",
    "professional experience",
    "real-world experience",
    "practical experience",
    "in production",
)
RESPONSIBILITY_VERBS = (
    "build",
    "design",
    "develop",
    "implement",
    "maintain",
    "optimize",
    "ship",
    "scale",
    "deploy",
    "own",
    "lead",
    "collaborate",
    "improve",
    "create",
    "deliver",
)
LEADERSHIP_PATTERNS = (
    r"\bled\b",
    r"\bmanaged\b",
    r"\bmentored\b",
    r"\bowned\b",
    r"\bdrove\b",
    r"\bheaded\b",
)
EDUCATION_PATTERNS = (
    r"\bbachelor'?s\b",
    r"\bb\.?tech\b",
    r"\bb\.?e\b",
    r"\bbs\b",
    r"\bmaster'?s\b",
    r"\bm\.?tech\b",
    r"\bms\b",
    r"\bphd\b",
    r"\bcomputer science\b",
    r"\bcomputer engineering\b",
    r"\brelated field\b",
)
DOMAIN_SIGNALS = {
    "ai": (r"\bartificial intelligence\b", r"\bai\b", r"\bllm\b", r"\bgenai\b", r"\bmachine learning\b"),
    "developer_tools": (r"\bdeveloper tools?\b", r"\btooling\b", r"\bsdk\b", r"\bplatform\b"),
    "fintech": (r"\bfintech\b", r"\bpayments?\b", r"\bbanking\b", r"\bwallet\b"),
    "data": (r"\bdata platform\b", r"\bdata pipeline\b", r"\betl\b", r"\banalytics platform\b"),
    "consumer": (r"\bconsumer\b", r"\bend user\b", r"\bmarketplace\b"),
    "enterprise": (r"\benterprise\b", r"\bb2b\b", r"\bsaas\b"),
    "infrastructure": (r"\binfrastructure\b", r"\bdistributed systems?\b", r"\bperformance\b"),
}
RESPONSIBILITY_KEYWORDS = {
    "apis": (r"\bapi\b", r"\bapis\b", r"\bbackend\b"),
    "frontend": (r"\bfront.?end\b", r"\bui\b"),
    "full_stack": (r"\bfull.?stack\b",),
    "data_pipelines": (r"\bdata pipelines?\b", r"\betl\b"),
    "machine_learning": (r"\bmachine learning\b", r"\bml\b"),
    "distributed_systems": (r"\bdistributed systems?\b",),
    "infrastructure": (r"\binfrastructure\b", r"\bplatform\b"),
    "sdk": (r"\bsdk\b",),
    "performance": (r"\bperformance\b", r"\blatency\b", r"\bthroughput\b"),
    "testing": (r"\btesting\b", r"\bquality\b"),
}
SKILL_PATTERNS = {
    "python": (r"\bpython\b",),
    "fastapi": (r"\bfastapi\b", r"\bfast api\b"),
    "django": (r"\bdjango\b",),
    "flask": (r"\bflask\b",),
    "java": (r"\bjava\b",),
    "go": (r"\bgo\b", r"\bgolang\b"),
    "javascript": (r"\bjavascript\b", r"\bjs\b"),
    "typescript": (r"\btypescript\b", r"\bts\b"),
    "react.js": (r"\breact\b", r"\breact\.js\b", r"\breactjs\b"),
    "next.js": (r"\bnext\b", r"\bnext\.js\b", r"\bnextjs\b"),
    "node.js": (r"\bnode\b", r"\bnode\.js\b", r"\bnodejs\b"),
    "express.js": (r"\bexpress\b", r"\bexpress\.js\b", r"\bexpressjs\b"),
    "nestjs": (r"\bnest\b", r"\bnestjs\b", r"\bnest\.js\b"),
    "playwright": (r"\bplaywright\b",),
    "rest apis": (r"\brest api\b", r"\brest apis\b", r"\brestful api\b", r"\brestful apis\b"),
    "graphql": (r"\bgraphql\b",),
    "sql": (r"\bsql\b",),
    "postgresql": (r"\bpostgresql\b", r"\bpostgres\b", r"\bpsql\b"),
    "mysql": (r"\bmysql\b",),
    "sqlite": (r"\bsqlite\b",),
    "mongodb": (r"\bmongodb\b", r"\bmongo\b"),
    "redis": (r"\bredis\b",),
    "docker": (r"\bdocker\b", r"\bcontainers?\b"),
    "kubernetes": (r"\bkubernetes\b", r"\bk8s\b"),
    "aws": (r"\baws\b", r"\bamazon web services\b"),
    "gcp": (r"\bgcp\b", r"\bgoogle cloud\b"),
    "azure": (r"\bazure\b",),
    "linux": (r"\blinux\b", r"\bunix\b"),
    "git": (r"\bgit\b",),
    "github": (r"\bgithub\b", r"\bgit hub\b"),
    "ci/cd": (r"\bci/cd\b", r"\bci cd\b", r"\bcontinuous integration\b"),
    "machine learning": (r"\bmachine learning\b", r"\bml\b"),
    "pytorch": (r"\bpytorch\b",),
    "tensorflow": (r"\btensorflow\b",),
    "numpy": (r"\bnumpy\b",),
    "pandas": (r"\bpandas\b",),
    "airflow": (r"\bairflow\b",),
    "spark": (r"\bspark\b",),
    "llm orchestration": (
        r"\bllm\b",
        r"\bllms\b",
        r"\blarge language models?\b",
        r"\blangchain\b",
        r"\bllamaindex\b",
        r"\brag\b",
    ),
}
DEFAULT_MASTER_RESUME_IDENTITY = "canonical_master_resume"
DEFAULT_MASTER_RESUME_VERSION = "v1"
DEFAULT_TEMPLATE_IDENTITY = "canonical_master_resume_template"
RESUME_PAGE_LIMIT = 1
PAGE_STATUS_NOT_RENDERED = "not_rendered"
PAGE_STATUS_VALID = "valid"
PAGE_STATUS_OVERFLOW = "overflow"
PAGE_STATUS_NOT_APPLICABLE = "not_applicable"
PAGE_STATUS_MISSING_OUTPUT = "missing_output"


@dataclass(frozen=True)
class JobRequirement:
    id: str
    category: str
    text: str
    required: bool
    anchors: list[str] = field(default_factory=list)
    evidence_mode: str = "skill"
    minimum_years: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "text": self.text,
            "required": self.required,
            "anchors": self.anchors,
            "evidence_mode": self.evidence_mode,
            "minimum_years": self.minimum_years,
        }


@dataclass(frozen=True)
class JobAnalysis:
    job_id: int
    role_title: str
    seniority: str
    required_skills: list[str]
    preferred_skills: list[str]
    responsibilities: list[str]
    minimum_experience_years: float | None
    preferred_experience_years: float | None
    domain_signals: list[str]
    location_remote_eligibility: dict[str, Any]
    education_requirements: list[str]
    hard_blockers: list[str]
    resume_keywords: list[str]
    requirements: list[JobRequirement] = field(default_factory=list)
    source: str = "deterministic"
    provider: str | None = None
    model: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "role_title": self.role_title,
            "seniority": self.seniority,
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "responsibilities": self.responsibilities,
            "minimum_experience_years": self.minimum_experience_years,
            "preferred_experience_years": self.preferred_experience_years,
            "domain_signals": self.domain_signals,
            "location_remote_eligibility": self.location_remote_eligibility,
            "education_requirements": self.education_requirements,
            "hard_blockers": self.hard_blockers,
            "resume_keywords": self.resume_keywords,
            "requirements": [requirement.as_dict() for requirement in self.requirements],
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RequirementMatch:
    requirement_id: str
    category: str
    text: str
    status: str
    evidence_refs: list[str]
    confidence: float
    explanation: str
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "category": self.category,
            "text": self.text,
            "status": self.status,
            "evidence_refs": self.evidence_refs,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "required": self.required,
        }


@dataclass(frozen=True)
class FitReport:
    job_id: int
    overall_fit_score: int
    strongest_matches: list[RequirementMatch]
    partial_matches: list[RequirementMatch]
    missing_requirements: list[RequirementMatch]
    hard_blockers: list[str]
    evidence_coverage: dict[str, Any]
    tailoring_recommendations: list[str]
    all_matches: list[RequirementMatch] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "overall_fit_score": self.overall_fit_score,
            "strongest_matches": [match.as_dict() for match in self.strongest_matches],
            "partial_matches": [match.as_dict() for match in self.partial_matches],
            "missing_requirements": [match.as_dict() for match in self.missing_requirements],
            "hard_blockers": self.hard_blockers,
            "evidence_coverage": self.evidence_coverage,
            "tailoring_recommendations": self.tailoring_recommendations,
            "all_matches": [match.as_dict() for match in self.all_matches],
        }


@dataclass(frozen=True)
class ResumeContentItem:
    text: str
    evidence_refs: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "evidence_refs": self.evidence_refs}


@dataclass(frozen=True)
class ResumeSkillItem:
    skill: str
    evidence_refs: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"skill": self.skill, "evidence_refs": self.evidence_refs}


@dataclass(frozen=True)
class ResumeLinkItem:
    label: str
    url: str
    evidence_refs: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "url": self.url, "evidence_refs": self.evidence_refs}


@dataclass(frozen=True)
class ResumeExperienceEntry:
    company: str
    title: str
    location: str | None
    start_date: str | None
    end_date: str | None
    summary: ResumeContentItem | None
    bullets: list[ResumeContentItem]
    evidence_refs: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "company": self.company,
            "title": self.title,
            "location": self.location,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "summary": self.summary.as_dict() if self.summary else None,
            "bullets": [bullet.as_dict() for bullet in self.bullets],
            "evidence_refs": self.evidence_refs,
        }


@dataclass(frozen=True)
class ResumeProjectEntry:
    name: str
    role: str | None
    link: str | None
    summary: ResumeContentItem | None
    bullets: list[ResumeContentItem]
    skills: list[ResumeSkillItem]
    evidence_refs: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "link": self.link,
            "summary": self.summary.as_dict() if self.summary else None,
            "bullets": [bullet.as_dict() for bullet in self.bullets],
            "skills": [skill.as_dict() for skill in self.skills],
            "evidence_refs": self.evidence_refs,
        }


@dataclass(frozen=True)
class ResumeEducationEntry:
    institution: str
    degree: str
    field_of_study: str | None
    start_date: str | None
    end_date: str | None
    evidence_refs: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "institution": self.institution,
            "degree": self.degree,
            "field_of_study": self.field_of_study,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "evidence_refs": self.evidence_refs,
        }


@dataclass(frozen=True)
class ResumeSourceModel:
    summary: ResumeContentItem | None
    skills: list[ResumeSkillItem]
    experience: list[ResumeExperienceEntry]
    projects: list[ResumeProjectEntry]
    education: list[ResumeEducationEntry]
    links: list[ResumeLinkItem]
    evidence_bullets: list[ResumeContentItem] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.as_dict() if self.summary else None,
            "skills": [skill.as_dict() for skill in self.skills],
            "experience": [entry.as_dict() for entry in self.experience],
            "projects": [entry.as_dict() for entry in self.projects],
            "education": [entry.as_dict() for entry in self.education],
            "links": [link.as_dict() for link in self.links],
            "evidence_bullets": [bullet.as_dict() for bullet in self.evidence_bullets],
        }


@dataclass(frozen=True)
class MasterResume:
    identity: str = DEFAULT_MASTER_RESUME_IDENTITY
    version: str = DEFAULT_MASTER_RESUME_VERSION
    page_limit: int = RESUME_PAGE_LIMIT
    source_status: str = "not_ingested"

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "version": self.version,
            "page_limit": self.page_limit,
            "source_status": self.source_status,
        }


@dataclass(frozen=True)
class ResumeTemplate:
    identity: str = DEFAULT_TEMPLATE_IDENTITY
    master_resume_identity: str = DEFAULT_MASTER_RESUME_IDENTITY
    page_limit: int = RESUME_PAGE_LIMIT
    layout_policy: str = "preserve_existing_layout"
    render_validation_status: str = "not_rendered"

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "master_resume_identity": self.master_resume_identity,
            "page_limit": self.page_limit,
            "layout_policy": self.layout_policy,
            "render_validation_status": self.render_validation_status,
        }


@dataclass(frozen=True)
class TailoringRecommendation:
    decision: str
    reasons: list[str]
    evidence_refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reasons": self.reasons,
            "evidence_refs": self.evidence_refs,
        }


@dataclass(frozen=True)
class ResumeChange:
    section: str
    action: str
    text: str
    evidence_refs: list[str]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "action": self.action,
            "text": self.text,
            "evidence_refs": self.evidence_refs,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class TailoredResume:
    job_id: int
    target_role: str
    master_resume: MasterResume
    template: ResumeTemplate
    page_limit: int
    page_validation_status: str
    recommendation: TailoringRecommendation
    changes: list[ResumeChange] = field(default_factory=list)
    preview: ResumeSourceModel | None = None
    output_pdf_path: str | None = None
    page_count: int | None = None
    evidence_refs_used: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "target_role": self.target_role,
            "master_resume": self.master_resume.as_dict(),
            "template": self.template.as_dict(),
            "page_limit": self.page_limit,
            "page_validation_status": self.page_validation_status,
            "recommendation": self.recommendation.as_dict(),
            "changes": [change.as_dict() for change in self.changes],
            "preview": self.preview.as_dict() if self.preview else None,
            "output_pdf_path": self.output_pdf_path,
            "page_count": self.page_count,
            "evidence_refs_used": self.evidence_refs_used,
        }


@dataclass(frozen=True)
class ResumeValidationIssue:
    code: str
    message: str
    evidence_refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "evidence_refs": self.evidence_refs}


@dataclass(frozen=True)
class ResumeValidationReport:
    status: str
    issues: list[ResumeValidationIssue]

    @property
    def ok(self) -> bool:
        return self.status == "valid" and not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class EvidenceIndex:
    profile_location_refs: list[str]
    work_auth_refs: list[str]
    work_auth_text: str
    skill_refs: dict[str, list[str]]
    experience_skill_refs: dict[str, list[str]]
    keyword_refs: dict[str, list[str]]
    leadership_refs: list[str]
    texts_by_ref: dict[str, list[str]]
    education_records: list[tuple[str, str]]
    explicit_years_experience: float | None
    documented_years_experience: float | None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = (item or "").strip()
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _fact_ref(fact: CandidateFact) -> str:
    return f"fact:{fact.id}" if fact.id is not None else fact.selector


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[\.\!\?])\s+|[\n\r]+|•| - ", text or "")
    return [part.strip(" -\t") for part in raw if part and part.strip(" -\t")]


def _normalize_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _skill_supported_by_evidence(skill: str, supported: str) -> bool:
    patterns = SKILL_PATTERNS.get(skill, ())
    if any(re.search(pattern, supported) for pattern in patterns):
        return True
    normalized_skill = _normalize_match_text(skill)
    normalized_supported = _normalize_match_text(supported)
    return bool(normalized_skill and normalized_skill in normalized_supported)


def _detect_skills(text: str) -> list[str]:
    found: list[str] = []
    lowered = (text or "").lower()
    for skill, patterns in SKILL_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            found.append(canonical_skill(skill))
    return _dedupe(found)


def _detect_keywords(text: str) -> list[str]:
    found = _detect_skills(text)
    lowered = (text or "").lower()
    for keyword, patterns in RESPONSIBILITY_KEYWORDS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            found.append(keyword)
    return _dedupe(found)


def _contains_any(text: str, cues: Iterable[str]) -> bool:
    lowered = (text or "").lower()
    return any(cue in lowered for cue in cues)


def _has_leadership_claim(text: str) -> bool:
    lowered = (text or "").lower()
    return any(re.search(pattern, lowered) for pattern in LEADERSHIP_PATTERNS)


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _descending_iso_sort_key(value: str | None) -> int:
    digits = str(value or "").replace("-", "")
    return -int(digits) if digits.isdigit() else 0


def _merge_day_ranges(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: item[0])
    merged: list[tuple[date, date]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _documented_years_experience(facts: list[CandidateFact], *, as_of: date | None = None) -> float | None:
    ranges: list[tuple[date, date]] = []
    end_fallback = as_of or date.today()
    for fact in facts:
        if not fact.verified or fact.category != "work_experience" or not isinstance(fact.value, dict):
            continue
        start = _parse_iso_date(fact.value.get("start_date"))
        if not start:
            continue
        end = _parse_iso_date(fact.value.get("end_date")) or end_fallback
        if end < start:
            continue
        ranges.append((start, end))
    merged = _merge_day_ranges(ranges)
    if not merged:
        return None
    days = sum((end - start).days + 1 for start, end in merged)
    return round(days / 365.25, 2)


def _fact_texts(fact: CandidateFact) -> list[str]:
    texts: list[str] = []
    value = fact.value
    if isinstance(value, str):
        texts.append(value)
    elif isinstance(value, list):
        texts.extend(str(item) for item in value if isinstance(item, str))
    elif isinstance(value, dict):
        for key in ("company", "title", "role", "name", "summary", "bullet", "institution", "degree", "field_of_study"):
            if isinstance(value.get(key), str) and value[key].strip():
                texts.append(value[key].strip())
        achievements = value.get("achievements")
        if isinstance(achievements, list):
            texts.extend(str(item).strip() for item in achievements if isinstance(item, str) and item.strip())
        skills = value.get("skills")
        if isinstance(skills, list):
            texts.extend(str(item).strip() for item in skills if isinstance(item, str) and item.strip())
        tags = value.get("tags")
        if isinstance(tags, list):
            texts.extend(str(item).strip() for item in tags if isinstance(item, str) and item.strip())
    return _dedupe(texts)


def _add_index_ref(index: dict[str, list[str]], key: str, ref: str) -> None:
    refs = index.setdefault(key, [])
    if ref not in refs:
        refs.append(ref)


def build_job_analysis(
    job: Job,
    *,
    location_signal: dict[str, Any],
    yoe_signal: dict[str, Any],
    work_auth_blocker: str | None,
) -> JobAnalysis:
    combined = " ".join(
        part for part in [job.title, job.description_text or "", job.location_raw or ""] if part
    )
    sentences = _split_sentences(combined)
    title_analysis = classify_title(normalize_title(job.title))

    required_skills: list[str] = []
    preferred_skills: list[str] = []
    required_experience_skills: set[str] = set()
    responsibilities: list[str] = []
    responsibility_requirements: list[JobRequirement] = []
    education_requirements: list[str] = []

    for sentence in sentences:
        skills = _detect_skills(sentence)
        is_required = _contains_any(sentence, REQUIRED_CUES)
        is_preferred = _contains_any(sentence, PREFERRED_CUES) and not is_required
        if skills and (is_required or is_preferred):
            target = required_skills if is_required else preferred_skills
            target.extend(skills)
            if is_required and (_contains_any(sentence, EXPERIENCE_CUES) or "production" in sentence.lower()):
                required_experience_skills.update(skills)

        lowered = sentence.lower()
        if (
            any(re.search(pattern, lowered) for pattern in EDUCATION_PATTERNS)
            and sentence not in education_requirements
        ):
            education_requirements.append(sentence[:200])

        if any(verb in lowered for verb in RESPONSIBILITY_VERBS):
            anchors = _detect_keywords(sentence)
            if anchors:
                responsibilities.append(sentence[:220])
                requirement = JobRequirement(
                    id=f"responsibility:{_slug(sentence)[:40]}",
                    category="responsibility",
                    text=sentence[:220],
                    required=True,
                    anchors=anchors,
                    evidence_mode="leadership" if _has_leadership_claim(sentence) else "experience",
                )
                if requirement.id not in {item.id for item in responsibility_requirements}:
                    responsibility_requirements.append(requirement)

    required_skills = _dedupe(required_skills)
    preferred_skills = [skill for skill in _dedupe(preferred_skills) if skill not in required_skills]
    responsibilities = _dedupe(responsibilities)[:8]
    education_requirements = _dedupe(education_requirements)[:6]

    required_min = yoe_signal.get("required_min")
    preferred_min = yoe_signal.get("preferred_min")
    seniority = "unknown"
    if title_analysis.early_career_signal or (required_min is not None and required_min <= 2):
        seniority = "early_career"
    elif title_analysis.is_senior or (required_min is not None and required_min >= 4):
        seniority = "senior"
    elif required_min == 3:
        seniority = "mid"

    domain_signals: list[str] = []
    lowered_combined = combined.lower()
    for label, patterns in DOMAIN_SIGNALS.items():
        if any(re.search(pattern, lowered_combined) for pattern in patterns):
            domain_signals.append(label)

    hard_blockers: list[str] = []
    if location_signal.get("status") == "ineligible":
        hard_blockers.append(str(location_signal.get("reason") or "location ineligible"))
    if work_auth_blocker:
        hard_blockers.append(work_auth_blocker)

    requirements: list[JobRequirement] = []
    for skill in required_skills:
        requirements.append(
            JobRequirement(
                id=f"required_skill:{_slug(skill)}",
                category="required_skill",
                text=skill,
                required=True,
                anchors=[skill],
                evidence_mode="experience" if skill in required_experience_skills else "skill",
            )
        )
    for skill in preferred_skills:
        requirements.append(
            JobRequirement(
                id=f"preferred_skill:{_slug(skill)}",
                category="preferred_skill",
                text=skill,
                required=False,
                anchors=[skill],
                evidence_mode="skill",
            )
        )
    if required_min is not None:
        requirements.append(
            JobRequirement(
                id="minimum_experience_years",
                category="minimum_experience",
                text=f"{required_min}+ years of experience",
                required=True,
                minimum_years=float(required_min),
                evidence_mode="years",
            )
        )
    if preferred_min is not None:
        requirements.append(
            JobRequirement(
                id="preferred_experience_years",
                category="preferred_experience",
                text=f"{preferred_min}+ years preferred",
                required=False,
                minimum_years=float(preferred_min),
                evidence_mode="years",
            )
        )
    for idx, entry in enumerate(education_requirements, start=1):
        requirements.append(
            JobRequirement(
                id=f"education:{idx}",
                category="education",
                text=entry,
                required=True,
                anchors=_dedupe(_detect_keywords(entry) + _education_anchors(entry)),
                evidence_mode="education",
            )
        )
    requirements.extend(responsibility_requirements)
    requirements.append(
        JobRequirement(
            id="location_eligibility",
            category="location",
            text=str(location_signal.get("reason") or job.location_raw or "location eligibility"),
            required=True,
            anchors=_dedupe(
                [str(item).lower() for item in location_signal.get("cities", [])]
                + [str(item).lower() for item in location_signal.get("countries", [])]
                + [str(location_signal.get("remote_scope") or "")]
            ),
            evidence_mode="location",
        )
    )
    if work_auth_blocker:
        requirements.append(
            JobRequirement(
                id="work_authorization",
                category="work_authorization",
                text=work_auth_blocker,
                required=True,
                anchors=_work_auth_anchors(work_auth_blocker),
                evidence_mode="work_authorization",
            )
        )

    resume_keywords = _dedupe(
        required_skills
        + preferred_skills
        + domain_signals
        + _detect_keywords(job.title)
        + _detect_keywords(job.description_text or "")
    )[:20]

    return JobAnalysis(
        job_id=job.id or 0,
        role_title=job.title,
        seniority=seniority,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        responsibilities=responsibilities,
        minimum_experience_years=float(required_min) if required_min is not None else None,
        preferred_experience_years=float(preferred_min) if preferred_min is not None else None,
        domain_signals=domain_signals,
        location_remote_eligibility=location_signal,
        education_requirements=education_requirements,
        hard_blockers=hard_blockers,
        resume_keywords=resume_keywords,
        requirements=requirements,
    )


def _education_anchors(text: str) -> list[str]:
    lowered = (text or "").lower()
    anchors: list[str] = []
    if "computer science" in lowered:
        anchors.append("computer science")
    if "computer engineering" in lowered:
        anchors.append("computer engineering")
    if any(token in lowered for token in ("bachelor", "b.tech", "btech", "b.e", "be", "bs")):
        anchors.append("bachelor")
    if any(token in lowered for token in ("master", "m.tech", "mtech", "ms")):
        anchors.append("master")
    if "phd" in lowered:
        anchors.append("phd")
    if "related field" in lowered:
        anchors.append("related field")
    return anchors


def _work_auth_anchors(text: str) -> list[str]:
    lowered = (text or "").lower()
    anchors: list[str] = []
    if "india" in lowered:
        anchors.append("india")
    if "united states" in lowered or "u.s." in lowered or "us " in lowered:
        anchors.append("us")
    if "canada" in lowered:
        anchors.append("canada")
    if "uk" in lowered or "united kingdom" in lowered:
        anchors.append("uk")
    if "eu" in lowered or "europe" in lowered:
        anchors.append("eu")
    if "visa sponsorship" in lowered or "sponsor" in lowered:
        anchors.append("sponsorship")
    if "security clearance" in lowered or "ts/sci" in lowered:
        anchors.append("clearance")
    return anchors


def _build_evidence_index(
    facts: list[CandidateFact],
    answers: list[VerifiedAnswer],
    *,
    as_of: date | None = None,
) -> EvidenceIndex:
    verified_facts = [fact for fact in facts if fact.verified]
    verified_answers = [answer for answer in answers if answer.verified]
    skill_refs: dict[str, list[str]] = {}
    experience_skill_refs: dict[str, list[str]] = {}
    keyword_refs: dict[str, list[str]] = {}
    texts_by_ref: dict[str, list[str]] = {}
    leadership_refs: list[str] = []
    education_records: list[tuple[str, str]] = []
    explicit_years: float | None = None

    for fact in verified_facts:
        ref = _fact_ref(fact)
        texts = _fact_texts(fact)
        if texts:
            texts_by_ref[ref] = texts
        if fact.selector == "personal_profile:years_experience" and isinstance(fact.value, (int, float)):
            explicit_years = float(fact.value)
        if fact.category == "skills":
            values: list[str] = []
            if isinstance(fact.value, str):
                values = [fact.value]
            elif isinstance(fact.value, list):
                values = [str(item) for item in fact.value if isinstance(item, str)]
            elif isinstance(fact.value, dict) and isinstance(fact.value.get("name"), str):
                values = [fact.value["name"]]
            for raw in values:
                for skill in _detect_skills(raw):
                    _add_index_ref(skill_refs, skill, ref)
        if fact.category in {"work_experience", "projects", "evidence"}:
            for text in texts:
                for skill in _detect_skills(text):
                    _add_index_ref(skill_refs, skill, ref)
                    _add_index_ref(experience_skill_refs, skill, ref)
                for keyword in _detect_keywords(text):
                    _add_index_ref(keyword_refs, keyword, ref)
            if _has_leadership_claim(" ".join(texts)):
                leadership_refs.append(ref)
        if fact.category == "projects" and isinstance(fact.value, dict):
            skills = fact.value.get("skills")
            if isinstance(skills, list):
                for raw in skills:
                    if not isinstance(raw, str):
                        continue
                    for skill in _detect_skills(raw):
                        _add_index_ref(skill_refs, skill, ref)
                        _add_index_ref(experience_skill_refs, skill, ref)
                        _add_index_ref(keyword_refs, skill, ref)
        if fact.category == "education" and isinstance(fact.value, dict):
            text = " ".join(
                str(fact.value.get(key) or "")
                for key in ("degree", "program", "field_of_study", "institution")
            ).strip()
            if text:
                education_records.append((ref, text.lower()))

    for answer in verified_answers:
        if answer.evidence_refs:
            texts_by_ref.setdefault(
                f"answer:{answer.id}" if answer.id is not None else answer.question_key,
                [answer.answer_text],
            )

    profile = build_application_profile(verified_facts)
    profile_location_refs: list[str] = []
    if "current_location" in profile.provenance:
        profile_location_refs.append(str(profile.provenance["current_location"]["selector"]))
    if "preferred_locations" in profile.provenance:
        profile_location_refs.append(str(profile.provenance["preferred_locations"]["selector"]))

    work_auth_refs: list[str] = []
    work_auth_text = ""
    if "work_authorization" in profile.provenance:
        work_auth_refs.append(str(profile.provenance["work_authorization"]["selector"]))
        work_auth_text = str(profile.work_authorization or "")
    for answer in verified_answers:
        if answer.question_key == "work_authorization":
            work_auth_text = work_auth_text or answer.answer_text
            work_auth_refs.extend(answer.evidence_refs)

    return EvidenceIndex(
        profile_location_refs=_dedupe(profile_location_refs),
        work_auth_refs=_dedupe(work_auth_refs),
        work_auth_text=work_auth_text.lower(),
        skill_refs=skill_refs,
        experience_skill_refs=experience_skill_refs,
        keyword_refs=keyword_refs,
        leadership_refs=_dedupe(leadership_refs),
        texts_by_ref=texts_by_ref,
        education_records=education_records,
        explicit_years_experience=explicit_years,
        documented_years_experience=_documented_years_experience(verified_facts, as_of=as_of),
    )


def _texts_for_refs(index: EvidenceIndex, refs: list[str]) -> str:
    collected: list[str] = []
    for ref in refs:
        collected.extend(index.texts_by_ref.get(ref, []))
    return " ".join(collected).lower()


def match_requirements(
    analysis: JobAnalysis,
    facts: list[CandidateFact],
    answers: list[VerifiedAnswer],
    *,
    as_of: date | None = None,
) -> list[RequirementMatch]:
    index = _build_evidence_index(facts, answers, as_of=as_of)
    matches: list[RequirementMatch] = []

    for requirement in analysis.requirements:
        refs: list[str] = []
        confidence = 0.0
        explanation = ""
        status = "unsupported"

        if requirement.evidence_mode == "skill":
            skill = requirement.anchors[0] if requirement.anchors else requirement.text
            exp_refs = index.experience_skill_refs.get(skill, [])
            direct_refs = index.skill_refs.get(skill, [])
            if exp_refs:
                refs = exp_refs
                status = "matched"
                confidence = 0.97
                explanation = "verified work/project evidence names this skill"
            elif direct_refs:
                refs = direct_refs
                status = "matched"
                confidence = 0.82
                explanation = "verified skill entry names this skill"
            else:
                explanation = "no verified evidence names this skill"
        elif requirement.evidence_mode == "experience":
            anchors = requirement.anchors or [requirement.text]
            exp_refs = _refs_for_anchors(index.experience_skill_refs, anchors)
            skill_refs = _refs_for_anchors(index.skill_refs, anchors)
            kw_refs = _refs_for_anchors(index.keyword_refs, anchors)
            if exp_refs or kw_refs:
                refs = _dedupe(exp_refs + kw_refs)
                status = "matched"
                confidence = 0.94
                explanation = "verified work/project evidence supports this experience requirement"
            elif skill_refs:
                refs = skill_refs
                status = "partial"
                confidence = 0.56
                explanation = "verified as a skill only; no verified hands-on evidence found"
            else:
                explanation = "no verified work/project evidence supports this requirement"
        elif requirement.evidence_mode == "years":
            documented = index.explicit_years_experience
            source = "explicit"
            if documented is None:
                documented = index.documented_years_experience
                source = "documented"
            need = float(requirement.minimum_years or 0.0)
            if documented is None:
                explanation = "no verified numeric experience total is available"
            elif documented >= need:
                refs = _experience_refs(facts)
                status = "matched"
                confidence = 0.92 if source == "explicit" else 0.8
                explanation = f"verified {source} experience supports {need:g}+ years"
            elif documented > 0 and documented >= max(0.0, need - 1.0):
                refs = _experience_refs(facts)
                status = "partial"
                confidence = 0.45
                explanation = f"verified {source} experience is {documented:g} years, below {need:g}"
            else:
                refs = _experience_refs(facts)
                explanation = (
                    f"verified {source} experience is {documented:g} years, below {need:g}"
                    if documented is not None
                    else "no verified numeric experience total is available"
                )
        elif requirement.evidence_mode == "education":
            anchors = requirement.anchors or _education_anchors(requirement.text)
            matched_refs = _education_refs(index.education_records, anchors)
            if matched_refs:
                refs = matched_refs
                status = "matched"
                confidence = 0.88
                explanation = "verified education evidence supports this requirement"
            elif index.education_records:
                refs = [record[0] for record in index.education_records]
                status = "partial"
                confidence = 0.4
                explanation = "verified education exists, but the field/degree is not explicit"
            else:
                explanation = "no verified education evidence is available"
        elif requirement.evidence_mode == "location":
            location = analysis.location_remote_eligibility
            refs = index.profile_location_refs
            if location.get("status") == "eligible":
                status = "matched"
                confidence = 0.85
                explanation = str(location.get("reason") or "location eligibility is explicit")
            elif location.get("status") == "unknown":
                status = "partial"
                confidence = 0.3
                explanation = str(location.get("reason") or "location eligibility is unknown")
            else:
                confidence = 0.0
                explanation = str(location.get("reason") or "location is ineligible")
        elif requirement.evidence_mode == "work_authorization":
            refs = index.work_auth_refs
            anchors = set(requirement.anchors)
            text = index.work_auth_text
            if not text:
                explanation = "no verified work authorization statement is available"
            elif "clearance" in anchors:
                explanation = "verified evidence does not claim a security clearance"
            elif "sponsorship" in anchors and ("india" in text or "authorized to work" in text):
                status = "partial"
                confidence = 0.3
                explanation = "verified authorization exists, but sponsorship eligibility is not explicit"
            elif not anchors or anchors & _work_auth_tokens(text):
                status = "matched"
                confidence = 0.8
                explanation = "verified work authorization statement covers this requirement"
            else:
                explanation = "verified work authorization does not cover this region or attestation"

        matches.append(
            RequirementMatch(
                requirement_id=requirement.id,
                category=requirement.category,
                text=requirement.text,
                status=status,
                evidence_refs=_dedupe(refs),
                confidence=confidence,
                explanation=explanation,
                required=requirement.required,
            )
        )

    return matches


def _experience_refs(facts: list[CandidateFact]) -> list[str]:
    return [_fact_ref(fact) for fact in facts if fact.verified and fact.category == "work_experience"]


def _education_refs(records: list[tuple[str, str]], anchors: list[str]) -> list[str]:
    matched: list[str] = []
    for ref, text in records:
        if not anchors:
            matched.append(ref)
            continue
        if any(anchor in text for anchor in anchors if anchor != "related field"):
            matched.append(ref)
            continue
        if "related field" in anchors and any(token in text for token in ("computer", "engineering", "software", "technology")):
            matched.append(ref)
    return _dedupe(matched)


def _refs_for_anchors(index: dict[str, list[str]], anchors: list[str]) -> list[str]:
    refs: list[str] = []
    for anchor in anchors:
        refs.extend(index.get(anchor, []))
    return _dedupe(refs)


def _work_auth_tokens(text: str) -> set[str]:
    tokens = set()
    lowered = (text or "").lower()
    if "india" in lowered:
        tokens.add("india")
    if "united states" in lowered or "u.s." in lowered or " us " in f" {lowered} ":
        tokens.add("us")
    if "canada" in lowered:
        tokens.add("canada")
    if "uk" in lowered or "united kingdom" in lowered:
        tokens.add("uk")
    if "eu" in lowered or "europe" in lowered:
        tokens.add("eu")
    if "sponsor" in lowered:
        tokens.add("sponsorship")
    return tokens


def build_fit_report(
    analysis: JobAnalysis,
    matches: list[RequirementMatch],
    source_model: ResumeSourceModel,
) -> FitReport:
    weights = {
        "required_skill": 3.0,
        "preferred_skill": 1.0,
        "responsibility": 1.5,
        "minimum_experience": 4.0,
        "preferred_experience": 2.0,
        "education": 1.5,
        "location": 2.0,
        "work_authorization": 3.0,
    }
    earned = 0.0
    possible = 0.0
    hard_blockers = list(analysis.hard_blockers)
    for match in matches:
        weight = weights.get(match.category, 1.0)
        if match.required or match.category in weights:
            possible += weight
        if match.status == "matched":
            earned += weight
        elif match.status == "partial":
            earned += weight * 0.5
        if match.required and match.status == "unsupported" and match.category in {
            "minimum_experience",
            "location",
            "work_authorization",
        }:
            hard_blockers.append(match.text if match.category == "minimum_experience" else match.explanation)

    score = int(round((earned / possible) * 100)) if possible else 0
    if hard_blockers:
        score = min(score, 35)

    strongest = sorted(
        [match for match in matches if match.status == "matched"],
        key=lambda item: (-weights.get(item.category, 1.0), -item.confidence, item.text),
    )[:6]
    partial = [match for match in matches if match.status == "partial"][:6]
    missing = [match for match in matches if match.status == "unsupported" and match.required][:8]

    evidence_refs = _dedupe(ref for match in matches if match.status != "unsupported" for ref in match.evidence_refs)
    recommendations = _build_tailoring_recommendations(matches, source_model)

    return FitReport(
        job_id=analysis.job_id,
        overall_fit_score=score,
        strongest_matches=strongest,
        partial_matches=partial,
        missing_requirements=missing,
        hard_blockers=_dedupe(hard_blockers),
        evidence_coverage={
            "requirements_total": len(matches),
            "matched": sum(1 for match in matches if match.status == "matched"),
            "partial": sum(1 for match in matches if match.status == "partial"),
            "unsupported": sum(1 for match in matches if match.status == "unsupported"),
            "evidence_ref_count": len(evidence_refs),
            "evidence_refs": evidence_refs,
        },
        tailoring_recommendations=recommendations,
        all_matches=matches,
    )


def _build_tailoring_recommendations(
    matches: list[RequirementMatch],
    source_model: ResumeSourceModel,
) -> list[str]:
    recommendations: list[str] = []
    matched_skills = [match.text for match in matches if match.category == "required_skill" and match.status == "matched"]
    partial_skills = [match.text for match in matches if match.category == "required_skill" and match.status == "partial"]
    missing = [match.text for match in matches if match.required and match.status == "unsupported"]

    if matched_skills:
        recommendations.append(f"Lead with verified evidence for {', '.join(matched_skills[:4])}.")
    for skill in partial_skills[:2]:
        recommendations.append(
            f"Keep {skill} conservative: verified as a skill, but not as clear hands-on experience."
        )
    for text in missing[:2]:
        recommendations.append(f"Do not add unsupported claims for {text}.")
    if source_model.summary is None:
        recommendations.append("No verified summary is stored yet; keep the summary blank or add one to the truth store.")
    return recommendations[:5]


def build_resume_source_model(
    facts: list[CandidateFact],
    answers: list[VerifiedAnswer],
) -> ResumeSourceModel:
    verified_facts = [fact for fact in facts if fact.verified]
    verified_answers = [answer for answer in answers if answer.verified]
    summary: ResumeContentItem | None = None

    for fact in verified_facts:
        if fact.selector in EXPLICIT_SUMMARY_KEYS and isinstance(fact.value, str) and fact.value.strip():
            summary = ResumeContentItem(text=fact.value.strip(), evidence_refs=[_fact_ref(fact)])
            break
    if summary is None:
        for answer in verified_answers:
            if answer.question_key in SUMMARY_ANSWER_KEYS and answer.answer_text.strip():
                summary = ResumeContentItem(
                    text=answer.answer_text.strip(),
                    evidence_refs=answer.evidence_refs or ([f"answer:{answer.id}"] if answer.id is not None else []),
                )
                break

    skills: list[ResumeSkillItem] = []
    for fact in verified_facts:
        ref = [_fact_ref(fact)]
        if fact.category == "skills":
            if isinstance(fact.value, str):
                items = _detect_skills(fact.value) or [canonical_skill(fact.value.strip().lower())]
                skills.extend(ResumeSkillItem(skill=item, evidence_refs=ref) for item in items if item)
            elif isinstance(fact.value, list):
                for item in fact.value:
                    if not isinstance(item, str):
                        continue
                    for skill in _detect_skills(item) or [canonical_skill(item.strip().lower())]:
                        skills.append(ResumeSkillItem(skill=skill, evidence_refs=ref))
            elif isinstance(fact.value, dict) and isinstance(fact.value.get("name"), str):
                skills.append(
                    ResumeSkillItem(skill=canonical_skill(fact.value["name"].strip().lower()), evidence_refs=ref)
                )
        elif fact.category == "projects" and isinstance(fact.value, dict):
            project_skills = fact.value.get("skills")
            if isinstance(project_skills, list):
                for item in project_skills:
                    if not isinstance(item, str):
                        continue
                    for skill in _detect_skills(item):
                        skills.append(ResumeSkillItem(skill=skill, evidence_refs=ref))
    skills = _dedupe_skill_items(skills)

    experience: list[ResumeExperienceEntry] = []
    for fact in sorted(
        [fact for fact in verified_facts if fact.category == "work_experience"],
        key=lambda item: (
            0 if (item.value or {}).get("sort_index") is not None else 1,
            int((item.value or {}).get("sort_index") or 9999),
            _descending_iso_sort_key((item.value or {}).get("end_date")),
            item.key,
        ),
    ):
        value = fact.value if isinstance(fact.value, dict) else {}
        ref = [_fact_ref(fact)]
        summary_item = None
        if isinstance(value.get("summary"), str) and value["summary"].strip():
            summary_item = ResumeContentItem(text=value["summary"].strip(), evidence_refs=ref)
        achievements = []
        if isinstance(value.get("achievements"), list):
            achievements = [
                ResumeContentItem(text=item.strip(), evidence_refs=ref)
                for item in value["achievements"]
                if isinstance(item, str) and item.strip()
            ]
        experience.append(
            ResumeExperienceEntry(
                company=str(value.get("company") or ""),
                title=str(value.get("title") or value.get("role") or ""),
                location=str(value.get("location") or "") or None,
                start_date=str(value.get("start_date") or "") or None,
                end_date=str(value.get("end_date") or "") or None,
                summary=summary_item,
                bullets=achievements,
                evidence_refs=ref,
            )
        )

    projects: list[ResumeProjectEntry] = []
    for fact in sorted(
        [fact for fact in verified_facts if fact.category == "projects"],
        key=lambda item: item.key,
    ):
        value = fact.value if isinstance(fact.value, dict) else {}
        ref = [_fact_ref(fact)]
        summary_item = None
        if isinstance(value.get("summary"), str) and value["summary"].strip():
            summary_item = ResumeContentItem(text=value["summary"].strip(), evidence_refs=ref)
        bullets = [
            ResumeContentItem(text=item.strip(), evidence_refs=ref)
            for item in (value.get("achievements") or [])
            if isinstance(item, str) and item.strip()
        ]
        project_skills = [
            ResumeSkillItem(skill=skill, evidence_refs=ref)
            for skill in _detect_skills(" ".join(str(item) for item in (value.get("skills") or [])))
        ]
        projects.append(
            ResumeProjectEntry(
                name=str(value.get("name") or ""),
                role=str(value.get("role") or "") or None,
                link=str(value.get("link") or "") or None,
                summary=summary_item,
                bullets=bullets,
                skills=project_skills,
                evidence_refs=ref,
            )
        )

    education: list[ResumeEducationEntry] = []
    for fact in sorted(
        [fact for fact in verified_facts if fact.category == "education"],
        key=lambda item: (
            0 if (item.value or {}).get("sort_index") is not None else 1,
            int((item.value or {}).get("sort_index") or 9999),
            _descending_iso_sort_key((item.value or {}).get("end_date")),
            item.key,
        ),
    ):
        value = fact.value if isinstance(fact.value, dict) else {}
        education.append(
            ResumeEducationEntry(
                institution=str(value.get("institution") or ""),
                degree=str(value.get("degree") or value.get("program") or ""),
                field_of_study=str(value.get("field_of_study") or "") or None,
                start_date=str(value.get("start_date") or "") or None,
                end_date=str(value.get("end_date") or "") or None,
                evidence_refs=[_fact_ref(fact)],
            )
        )

    links: list[ResumeLinkItem] = []
    profile = build_application_profile(verified_facts)
    for field_name, label in (
        ("linkedin_url", "LinkedIn"),
        ("github_url", "GitHub"),
        ("portfolio_url", "Portfolio"),
    ):
        url = getattr(profile, field_name)
        if url:
            provenance = profile.provenance.get(field_name, {})
            links.append(
                ResumeLinkItem(
                    label=label,
                    url=url,
                    evidence_refs=[str(provenance.get("selector"))] if provenance.get("selector") else [],
                )
            )

    evidence_bullets: list[ResumeContentItem] = []
    for fact in verified_facts:
        if fact.category != "evidence":
            continue
        ref = [_fact_ref(fact)]
        if isinstance(fact.value, str) and fact.value.strip():
            evidence_bullets.append(ResumeContentItem(text=fact.value.strip(), evidence_refs=ref))
        elif isinstance(fact.value, dict) and isinstance(fact.value.get("bullet"), str) and fact.value["bullet"].strip():
            evidence_bullets.append(ResumeContentItem(text=fact.value["bullet"].strip(), evidence_refs=ref))

    return ResumeSourceModel(
        summary=summary,
        skills=skills,
        experience=experience,
        projects=projects,
        education=education,
        links=links,
        evidence_bullets=evidence_bullets,
    )


def _dedupe_skill_items(items: list[ResumeSkillItem]) -> list[ResumeSkillItem]:
    merged: dict[str, list[str]] = {}
    for item in items:
        refs = merged.setdefault(item.skill, [])
        for ref in item.evidence_refs:
            if ref not in refs:
                refs.append(ref)
    return [ResumeSkillItem(skill=skill, evidence_refs=refs) for skill, refs in sorted(merged.items())]


def _default_master_resume() -> MasterResume:
    return MasterResume()


def _default_resume_template(master_resume: MasterResume) -> ResumeTemplate:
    return ResumeTemplate(master_resume_identity=master_resume.identity)


def _source_ref_sections(source_model: ResumeSourceModel) -> dict[str, str]:
    sections: dict[str, str] = {}
    if source_model.summary:
        for ref in source_model.summary.evidence_refs:
            sections[ref] = "summary"
    for skill in source_model.skills:
        for ref in skill.evidence_refs:
            sections.setdefault(ref, "skills")
    for entry in source_model.experience:
        for ref in entry.evidence_refs:
            sections[ref] = "experience"
        if entry.summary:
            for ref in entry.summary.evidence_refs:
                sections[ref] = "experience"
        for bullet in entry.bullets:
            for ref in bullet.evidence_refs:
                sections[ref] = "experience"
    for entry in source_model.projects:
        for ref in entry.evidence_refs:
            sections[ref] = "projects"
        if entry.summary:
            for ref in entry.summary.evidence_refs:
                sections[ref] = "projects"
        for bullet in entry.bullets:
            for ref in bullet.evidence_refs:
                sections[ref] = "projects"
        for skill in entry.skills:
            for ref in skill.evidence_refs:
                sections[ref] = "projects"
    for entry in source_model.education:
        for ref in entry.evidence_refs:
            sections[ref] = "education"
    for link in source_model.links:
        for ref in link.evidence_refs:
            sections[ref] = "links"
    for bullet in source_model.evidence_bullets:
        for ref in bullet.evidence_refs:
            sections[ref] = "evidence"
    return sections


def build_tailoring_recommendation(
    analysis: JobAnalysis,
    fit_report: FitReport,
    source_model: ResumeSourceModel,
) -> TailoringRecommendation:
    section_map = _source_ref_sections(source_model)
    unsupported_required = [
        match
        for match in fit_report.all_matches
        if match.required
        and match.status == "unsupported"
        and match.category in {"required_skill", "minimum_experience", "location", "work_authorization"}
    ]
    partial_required = [
        match
        for match in fit_report.all_matches
        if match.required
        and match.status == "partial"
        and match.category in {"required_skill", "minimum_experience", "location", "work_authorization"}
    ]
    if fit_report.hard_blockers or unsupported_required or partial_required or fit_report.overall_fit_score < 50:
        reasons = list(fit_report.hard_blockers)
        if unsupported_required:
            reasons.append(
                "Important requirements are unsupported: "
                + ", ".join(match.text for match in unsupported_required[:4])
            )
        if partial_required:
            reasons.append(
                "Important requirements are only partially supported: "
                + ", ".join(match.text for match in partial_required[:4])
            )
        evidence = _dedupe(
            ref
            for match in partial_required
            for ref in match.evidence_refs
        )
        return TailoringRecommendation(
            decision="poor_fit",
            reasons=_dedupe(reasons)[:5],
            evidence_refs=evidence,
        )

    highlightable = [
        match
        for match in fit_report.all_matches
        if match.status == "matched"
        and any(section_map.get(ref) in {"experience", "projects", "evidence"} for ref in match.evidence_refs)
    ]
    if len(highlightable) >= 2 or fit_report.partial_matches:
        focus = _dedupe(match.text for match in highlightable if match.category in {
            "required_skill",
            "preferred_skill",
            "responsibility",
        })
        reasons = []
        if focus:
            reasons.append(
                "Verified experience/project evidence can better surface "
                + ", ".join(focus[:4])
                + "."
            )
        if fit_report.partial_matches:
            reasons.append("Tailoring can emphasize verified strengths without manufacturing unsupported claims.")
        if not reasons:
            reasons.append("Relevant verified evidence exists beyond the base skills list.")
        return TailoringRecommendation(
            decision="tailor",
            reasons=reasons[:5],
            evidence_refs=_dedupe(ref for match in highlightable for ref in match.evidence_refs),
        )

    return TailoringRecommendation(
        decision="use_base",
        reasons=["Verified base resume evidence already covers the strongest supported requirements."],
        evidence_refs=_dedupe(ref for match in fit_report.strongest_matches for ref in match.evidence_refs),
    )


def tailor_resume(
    analysis: JobAnalysis,
    fit_report: FitReport,
    source_model: ResumeSourceModel,
) -> TailoredResume:
    matched_required = {
        match.text for match in fit_report.all_matches if match.category == "required_skill" and match.status == "matched"
    }
    partial_required = {
        match.text for match in fit_report.all_matches if match.category == "required_skill" and match.status == "partial"
    }
    preferred = {
        match.text for match in fit_report.all_matches if match.category == "preferred_skill" and match.status != "unsupported"
    }
    keywords = set(analysis.resume_keywords)

    def skill_rank(skill: str) -> tuple[int, str]:
        if skill in matched_required:
            return (0, skill)
        if skill in partial_required:
            return (1, skill)
        if skill in preferred:
            return (2, skill)
        return (3, skill)

    preview = ResumeSourceModel(
        summary=source_model.summary,
        skills=sorted(source_model.skills, key=lambda item: skill_rank(item.skill)),
        experience=_tailor_experience_entries(source_model.experience, matched_required, partial_required, keywords),
        projects=_tailor_project_entries(source_model.projects, matched_required, partial_required, keywords),
        education=source_model.education,
        links=source_model.links,
        evidence_bullets=_tailor_bullets(source_model.evidence_bullets, matched_required, partial_required, keywords),
    )
    recommendation = build_tailoring_recommendation(analysis, fit_report, source_model)
    master_resume = _default_master_resume()
    template = _default_resume_template(master_resume)

    return TailoredResume(
        job_id=analysis.job_id,
        target_role=analysis.role_title,
        master_resume=master_resume,
        template=template,
        page_limit=RESUME_PAGE_LIMIT,
        page_validation_status=PAGE_STATUS_NOT_RENDERED,
        recommendation=recommendation,
        changes=_build_change_plan(recommendation, preview, matched_required, preferred),
        preview=preview if recommendation.decision == "tailor" else None,
    )


def _build_change_plan(
    recommendation: TailoringRecommendation,
    preview: ResumeSourceModel,
    matched_required: set[str],
    preferred: set[str],
) -> list[ResumeChange]:
    if recommendation.decision != "tailor":
        return []

    changes: list[ResumeChange] = []
    prioritized_skills = [item for item in preview.skills if item.skill in matched_required or item.skill in preferred]
    if prioritized_skills:
        changes.append(
            ResumeChange(
                section="skills",
                action="prioritize",
                text=", ".join(item.skill for item in prioritized_skills[:8]),
                evidence_refs=_dedupe(ref for item in prioritized_skills[:8] for ref in item.evidence_refs),
                rationale="Move matching verified skills earlier within the existing template.",
            )
        )

    for entry in preview.experience[:2]:
        if entry.bullets:
            changes.append(
                ResumeChange(
                    section="experience",
                    action="highlight",
                    text=f"{entry.title} @ {entry.company}",
                    evidence_refs=_dedupe(entry.evidence_refs + [ref for bullet in entry.bullets for ref in bullet.evidence_refs]),
                    rationale="Promote verified bullets that best support the job requirements.",
                )
            )
    for entry in preview.projects[:2]:
        if entry.summary or entry.bullets:
            changes.append(
                ResumeChange(
                    section="projects",
                    action="highlight",
                    text=entry.name,
                    evidence_refs=_dedupe(entry.evidence_refs + [ref for bullet in entry.bullets for ref in bullet.evidence_refs]),
                    rationale="Surface verified project evidence relevant to the target role.",
                )
            )
    return changes[:6]


def _entry_score(texts: list[str], matched_required: set[str], partial_required: set[str], keywords: set[str]) -> int:
    text = " ".join(texts).lower()
    score = 0
    for skill in matched_required:
        if skill in _detect_skills(text):
            score += 4
    for skill in partial_required:
        if skill in _detect_skills(text):
            score += 2
    for keyword in keywords:
        if keyword in _detect_keywords(text):
            score += 1
    return score


def _tailor_experience_entries(
    entries: list[ResumeExperienceEntry],
    matched_required: set[str],
    partial_required: set[str],
    keywords: set[str],
) -> list[ResumeExperienceEntry]:
    ranked: list[tuple[int, ResumeExperienceEntry]] = []
    for entry in entries:
        texts = [entry.title, entry.company]
        if entry.summary:
            texts.append(entry.summary.text)
        texts.extend(item.text for item in entry.bullets)
        score = _entry_score(texts, matched_required, partial_required, keywords)
        bullets = _tailor_bullets(entry.bullets, matched_required, partial_required, keywords)
        if not bullets:
            bullets = entry.bullets[:2]
        ranked.append(
            (
                score,
                ResumeExperienceEntry(
                    company=entry.company,
                    title=entry.title,
                    location=entry.location,
                    start_date=entry.start_date,
                    end_date=entry.end_date,
                    summary=entry.summary,
                    bullets=bullets,
                    evidence_refs=entry.evidence_refs,
                ),
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1].end_date or "", item[1].company.lower()), reverse=True)
    return [entry for _, entry in ranked]


def _tailor_project_entries(
    entries: list[ResumeProjectEntry],
    matched_required: set[str],
    partial_required: set[str],
    keywords: set[str],
) -> list[ResumeProjectEntry]:
    ranked: list[tuple[int, ResumeProjectEntry]] = []
    for entry in entries:
        texts = [entry.name, entry.role or ""]
        if entry.summary:
            texts.append(entry.summary.text)
        texts.extend(item.text for item in entry.bullets)
        texts.extend(item.skill for item in entry.skills)
        score = _entry_score(texts, matched_required, partial_required, keywords)
        bullets = _tailor_bullets(entry.bullets, matched_required, partial_required, keywords)
        ranked.append(
            (
                score,
                ResumeProjectEntry(
                    name=entry.name,
                    role=entry.role,
                    link=entry.link,
                    summary=entry.summary,
                    bullets=bullets or entry.bullets[:2],
                    skills=sorted(entry.skills, key=lambda item: (item.skill not in matched_required, item.skill)),
                    evidence_refs=entry.evidence_refs,
                ),
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1].name.lower()))
    return [entry for _, entry in ranked]


def _tailor_bullets(
    bullets: list[ResumeContentItem],
    matched_required: set[str],
    partial_required: set[str],
    keywords: set[str],
) -> list[ResumeContentItem]:
    ranked: list[tuple[int, ResumeContentItem]] = []
    for bullet in bullets:
        score = _entry_score([bullet.text], matched_required, partial_required, keywords)
        if score > 0:
            ranked.append((score, bullet))
    ranked.sort(key=lambda item: (-item[0], item[1].text.lower()))
    return [bullet for _, bullet in ranked]


def validate_tailored_resume(
    resume: TailoredResume,
    source_model: ResumeSourceModel,
    *,
    as_of: date | None = None,
) -> ResumeValidationReport:
    issues: list[ResumeValidationIssue] = []
    allowed_page_statuses = {
        PAGE_STATUS_NOT_RENDERED,
        PAGE_STATUS_VALID,
        PAGE_STATUS_OVERFLOW,
        PAGE_STATUS_NOT_APPLICABLE,
        PAGE_STATUS_MISSING_OUTPUT,
    }
    if resume.page_limit != RESUME_PAGE_LIMIT:
        issues.append(
            ResumeValidationIssue(
                code="invalid_page_limit",
                message=f"resume variants currently support only page_limit={RESUME_PAGE_LIMIT}",
            )
        )
    if resume.page_validation_status not in allowed_page_statuses:
        issues.append(
            ResumeValidationIssue(
                code="invalid_page_validation_status",
                message=(
                    "page_validation_status must be one of "
                    f"{sorted(allowed_page_statuses)}"
                ),
            )
        )
    if resume.template.render_validation_status != resume.page_validation_status:
        issues.append(
            ResumeValidationIssue(
                code="template_render_status_mismatch",
                message="template render validation status must match the variant page validation status",
            )
        )
    if resume.template.page_limit != resume.page_limit or resume.master_resume.page_limit != resume.page_limit:
        issues.append(
            ResumeValidationIssue(
                code="page_limit_metadata_mismatch",
                message="master resume, template, and variant page limits must match",
            )
        )
    if resume.page_validation_status == PAGE_STATUS_NOT_RENDERED:
        if resume.output_pdf_path:
            issues.append(
                ResumeValidationIssue(
                    code="unexpected_output_pdf_path",
                    message="not_rendered variants must not point to an output PDF",
                )
            )
        if resume.page_count is not None:
            issues.append(
                ResumeValidationIssue(
                    code="unexpected_page_count",
                    message="not_rendered variants must not store a page count",
                )
            )
    elif resume.page_validation_status == PAGE_STATUS_NOT_APPLICABLE:
        if resume.recommendation.decision != "poor_fit":
            issues.append(
                ResumeValidationIssue(
                    code="invalid_not_applicable_page_status",
                    message="page_validation_status=not_applicable is only valid for poor_fit decisions",
                )
            )
        if resume.output_pdf_path:
            issues.append(
                ResumeValidationIssue(
                    code="unexpected_output_for_poor_fit",
                    message="poor_fit variants must not point to an output PDF",
                )
            )
        if resume.page_count is not None:
            issues.append(
                ResumeValidationIssue(
                    code="unexpected_page_count_for_poor_fit",
                    message="poor_fit variants must not store a page count",
                )
            )
    elif resume.page_validation_status == PAGE_STATUS_MISSING_OUTPUT:
        if not resume.output_pdf_path:
            issues.append(
                ResumeValidationIssue(
                    code="missing_output_pdf_path",
                    message="missing_output variants must retain the expected output path",
                )
            )
        if resume.page_count is not None:
            issues.append(
                ResumeValidationIssue(
                    code="unexpected_page_count_for_missing_output",
                    message="missing_output variants must not store a page count",
                )
            )
    else:
        if resume.recommendation.decision == "poor_fit":
            issues.append(
                ResumeValidationIssue(
                    code="unexpected_rendered_output_for_poor_fit",
                    message="poor_fit variants must not produce a rendered resume artifact",
                )
            )
        if not resume.output_pdf_path:
            issues.append(
                ResumeValidationIssue(
                    code="missing_output_pdf_path",
                    message="rendered variants must store an output PDF path",
                )
            )
        if resume.page_count is None or resume.page_count <= 0:
            issues.append(
                ResumeValidationIssue(
                    code="missing_page_count",
                    message="rendered variants must store a positive page count",
                )
            )
        elif resume.page_validation_status == PAGE_STATUS_VALID and resume.page_count != resume.page_limit:
            issues.append(
                ResumeValidationIssue(
                    code="validated_page_count_mismatch",
                    message="page_validation_status=valid requires the rendered PDF to match the page limit exactly",
                )
            )
        elif resume.page_validation_status == PAGE_STATUS_OVERFLOW and resume.page_count <= resume.page_limit:
            issues.append(
                ResumeValidationIssue(
                    code="overflow_page_count_mismatch",
                    message="page_validation_status=overflow requires the rendered PDF to exceed the page limit",
                )
            )
    if resume.recommendation.decision not in {"use_base", "tailor", "poor_fit"}:
        issues.append(
            ResumeValidationIssue(
                code="invalid_recommendation_decision",
                message=f"unknown tailoring recommendation {resume.recommendation.decision!r}",
            )
        )
    if resume.recommendation.decision != "tailor":
        if resume.preview is not None:
            issues.append(
                ResumeValidationIssue(
                    code="unexpected_preview_for_non_tailor",
                    message="use_base and poor_fit variants must not carry a tailored preview",
                )
            )
        if resume.changes:
            issues.append(
                ResumeValidationIssue(
                    code="unexpected_changes_for_non_tailor",
                    message="use_base and poor_fit variants must not carry change instructions",
                )
            )
        return ResumeValidationReport(status="invalid" if issues else "valid", issues=issues)
    if resume.preview is None:
        issues.append(
            ResumeValidationIssue(
                code="missing_preview_for_tailor",
                message="tailor variants must carry a text-only preview",
            )
        )
        return ResumeValidationReport(status="invalid", issues=issues)

    preview = resume.preview
    source_refs = _source_texts_by_ref(source_model)
    supported_years = _resume_documented_years(source_model, as_of=as_of)

    def validate_item(item: ResumeContentItem) -> None:
        if not item.evidence_refs:
            issues.append(
                ResumeValidationIssue(
                    code="missing_evidence_refs",
                    message=f"content item lacks evidence refs: {item.text}",
                )
            )
            return
        supported = " ".join(
            text
            for ref in item.evidence_refs
            for text in source_refs.get(ref, [])
        ).lower()
        if not supported.strip():
            issues.append(
                ResumeValidationIssue(
                    code="unknown_evidence_ref",
                    message=f"content item references missing evidence: {item.text}",
                    evidence_refs=item.evidence_refs,
                )
            )
            return

        for token in re.findall(r"\b\d+(?:\.\d+)?%?\b", item.text):
            if token not in supported:
                issues.append(
                    ResumeValidationIssue(
                        code="unsupported_numeric_claim",
                        message=f"numeric claim {token} is not present in the source evidence",
                        evidence_refs=item.evidence_refs,
                    )
                )
        for skill in _detect_skills(item.text):
            if not _skill_supported_by_evidence(skill, supported):
                issues.append(
                    ResumeValidationIssue(
                        code="unsupported_technology_claim",
                        message=f"technology claim {skill} is not present in the source evidence",
                        evidence_refs=item.evidence_refs,
                    )
                )
        if _has_leadership_claim(item.text) and not _has_leadership_claim(supported):
            issues.append(
                ResumeValidationIssue(
                    code="unsupported_leadership_claim",
                    message=f"leadership language is not supported by the source evidence: {item.text}",
                    evidence_refs=item.evidence_refs,
                )
            )
        for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", item.text.lower()):
            claim = float(match.group(1))
            if supported_years is None or claim - supported_years > 0.05:
                issues.append(
                    ResumeValidationIssue(
                        code="unsupported_years_claim",
                        message=f"years-of-experience claim {claim:g} is not supported by verified evidence",
                        evidence_refs=item.evidence_refs,
                    )
                )

    if preview.summary:
        validate_item(preview.summary)
    for item in preview.evidence_bullets:
        validate_item(item)
    for skill in preview.skills:
        supported = " ".join(
            text
            for ref in skill.evidence_refs
            for text in source_refs.get(ref, [])
        ).lower()
        if not skill.evidence_refs or not _skill_supported_by_evidence(skill.skill, supported):
            issues.append(
                ResumeValidationIssue(
                    code="unsupported_skill_entry",
                    message=f"skill entry {skill.skill} is not supported by its evidence refs",
                    evidence_refs=skill.evidence_refs,
                )
            )
    source_experience = {
        tuple(entry.evidence_refs): entry
        for entry in source_model.experience
    }
    for entry in preview.experience:
        expected = source_experience.get(tuple(entry.evidence_refs))
        if expected is None:
            issues.append(
                ResumeValidationIssue(
                    code="unknown_experience_entry",
                    message=f"experience entry {entry.company} / {entry.title} does not map to source evidence",
                    evidence_refs=entry.evidence_refs,
                )
            )
            continue
        for field_name in ("company", "title", "start_date", "end_date"):
            if getattr(entry, field_name) != getattr(expected, field_name):
                issues.append(
                    ResumeValidationIssue(
                        code="mutated_experience_identity",
                        message=f"{field_name} changed for {entry.company} / {entry.title}",
                        evidence_refs=entry.evidence_refs,
                    )
                )
        if entry.summary:
            validate_item(entry.summary)
        for bullet in entry.bullets:
            validate_item(bullet)

    source_projects = {
        tuple(entry.evidence_refs): entry
        for entry in source_model.projects
    }
    for entry in preview.projects:
        expected = source_projects.get(tuple(entry.evidence_refs))
        if expected is None:
            issues.append(
                ResumeValidationIssue(
                    code="unknown_project_entry",
                    message=f"project entry {entry.name} does not map to source evidence",
                    evidence_refs=entry.evidence_refs,
                )
            )
            continue
        for field_name in ("name", "role", "link"):
            if getattr(entry, field_name) != getattr(expected, field_name):
                issues.append(
                    ResumeValidationIssue(
                        code="mutated_project_identity",
                        message=f"{field_name} changed for project {entry.name}",
                        evidence_refs=entry.evidence_refs,
                    )
                )
        if entry.summary:
            validate_item(entry.summary)
        for bullet in entry.bullets:
            validate_item(bullet)

    status = "invalid" if issues else "valid"
    return ResumeValidationReport(status=status, issues=issues)


def _source_texts_by_ref(source_model: ResumeSourceModel) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}

    def add(refs: list[str], text: str) -> None:
        for ref in refs:
            mapping.setdefault(ref, [])
            if text and text not in mapping[ref]:
                mapping[ref].append(text)

    if source_model.summary:
        add(source_model.summary.evidence_refs, source_model.summary.text)
    for skill in source_model.skills:
        add(skill.evidence_refs, skill.skill)
    for entry in source_model.experience:
        add(entry.evidence_refs, " ".join(filter(None, [entry.company, entry.title, entry.location or ""])))
        if entry.summary:
            add(entry.summary.evidence_refs, entry.summary.text)
        for bullet in entry.bullets:
            add(bullet.evidence_refs, bullet.text)
    for entry in source_model.projects:
        add(entry.evidence_refs, " ".join(filter(None, [entry.name, entry.role or "", entry.link or ""])))
        if entry.summary:
            add(entry.summary.evidence_refs, entry.summary.text)
        for bullet in entry.bullets:
            add(bullet.evidence_refs, bullet.text)
        for skill in entry.skills:
            add(skill.evidence_refs, skill.skill)
    for entry in source_model.education:
        add(entry.evidence_refs, " ".join(filter(None, [entry.institution, entry.degree, entry.field_of_study or ""])))
    for link in source_model.links:
        add(link.evidence_refs, link.url)
    for bullet in source_model.evidence_bullets:
        add(bullet.evidence_refs, bullet.text)
    return mapping


def _resume_documented_years(source_model: ResumeSourceModel, *, as_of: date | None = None) -> float | None:
    ranges: list[tuple[date, date]] = []
    end_fallback = as_of or date.today()
    for entry in source_model.experience:
        start = _parse_iso_date(entry.start_date)
        if not start:
            continue
        end = _parse_iso_date(entry.end_date) or end_fallback
        if end < start:
            continue
        ranges.append((start, end))
    merged = _merge_day_ranges(ranges)
    if not merged:
        return None
    days = sum((end - start).days + 1 for start, end in merged)
    return round(days / 365.25, 2)


def truth_store_hash(facts: list[CandidateFact], answers: list[VerifiedAnswer]) -> str:
    payload = {
        "facts": [
            {
                "category": fact.category,
                "key": fact.key,
                "value": fact.value,
                "source": fact.source,
                "verified": fact.verified,
            }
            for fact in sorted(
                [item for item in facts if item.verified],
                key=lambda item: (item.category, item.key, json.dumps(item.value, sort_keys=True, ensure_ascii=True)),
            )
        ],
        "answers": [
            {
                "question_key": answer.question_key,
                "category": answer.category,
                "answer_text": answer.answer_text,
                "source": answer.source,
                "evidence_refs": answer.evidence_refs,
                "verified": answer.verified,
                "human_review_required": answer.human_review_required,
            }
            for answer in sorted(
                [item for item in answers if item.verified],
                key=lambda item: (item.question_key, item.category, item.answer_text),
            )
        ],
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
