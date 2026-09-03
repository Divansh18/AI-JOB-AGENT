"""Core domain entities. Plain dataclasses: no I/O, no ORM, no framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    NEW = "new"
    FILTERED_OUT = "filtered_out"
    RANKED = "ranked"
    DISMISSED = "dismissed"
    APPLIED = "applied"


class ApplicationStatus(str, Enum):
    TO_APPLY = "to_apply"
    APPLIED = "applied"
    NO_RESPONSE = "no_response"
    RECRUITER_REPLY = "recruiter_reply"
    SCREEN_SCHEDULED = "screen_scheduled"
    INTERVIEWING = "interviewing"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    GHOSTED = "ghosted"


# Allowed transitions. Enforced in services.ledger; kept here because it is
# domain policy, not persistence detail.
ALLOWED_TRANSITIONS: dict[ApplicationStatus, set[ApplicationStatus]] = {
    ApplicationStatus.TO_APPLY: {ApplicationStatus.APPLIED, ApplicationStatus.WITHDRAWN},
    ApplicationStatus.APPLIED: {
        ApplicationStatus.NO_RESPONSE, ApplicationStatus.REJECTED,
        ApplicationStatus.RECRUITER_REPLY, ApplicationStatus.WITHDRAWN,
        ApplicationStatus.GHOSTED, ApplicationStatus.SCREEN_SCHEDULED,
    },
    ApplicationStatus.NO_RESPONSE: {
        ApplicationStatus.GHOSTED, ApplicationStatus.RECRUITER_REPLY,
        ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN,
    },
    ApplicationStatus.RECRUITER_REPLY: {
        ApplicationStatus.SCREEN_SCHEDULED, ApplicationStatus.REJECTED,
        ApplicationStatus.GHOSTED, ApplicationStatus.WITHDRAWN,
    },
    ApplicationStatus.SCREEN_SCHEDULED: {
        ApplicationStatus.INTERVIEWING, ApplicationStatus.REJECTED,
        ApplicationStatus.GHOSTED, ApplicationStatus.WITHDRAWN,
    },
    ApplicationStatus.INTERVIEWING: {
        ApplicationStatus.OFFER, ApplicationStatus.REJECTED,
        ApplicationStatus.GHOSTED, ApplicationStatus.WITHDRAWN,
    },
    ApplicationStatus.OFFER: {ApplicationStatus.WITHDRAWN, ApplicationStatus.REJECTED},
    ApplicationStatus.REJECTED: set(),
    ApplicationStatus.WITHDRAWN: set(),
    ApplicationStatus.GHOSTED: {ApplicationStatus.RECRUITER_REPLY, ApplicationStatus.REJECTED},
}

TERMINAL_STATUSES = {
    ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN, ApplicationStatus.OFFER,
}
RESPONSE_STATUSES = {
    ApplicationStatus.RECRUITER_REPLY, ApplicationStatus.SCREEN_SCHEDULED,
    ApplicationStatus.INTERVIEWING, ApplicationStatus.OFFER,
}


@dataclass
class RawPosting:
    """What a source adapter returns, before normalization."""

    source: str
    external_id: str | None
    title: str
    company_name: str
    location_raw: str
    description_html: str | None = None
    description_text: str | None = None
    apply_url: str = ""
    canonical_url: str | None = None
    posted_at: datetime | None = None
    employment_hint: str | None = None
    company_id: int | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class Job:
    """A normalized posting."""

    fingerprint: str
    content_hash: str
    source: str
    title: str
    title_normalized: str
    company_name_raw: str
    company_normalized: str
    location_raw: str
    locations: list[str]
    country: str | None
    remote_type: str
    remote_scope: str
    employment_type: str
    description_text: str
    apply_url: str
    canonical_url: str | None
    posted_at: datetime | None
    first_seen_at: datetime
    external_id: str | None = None
    company_id: int | None = None
    source_id: int | None = None
    id: int | None = None
    raw: dict = field(default_factory=dict)

    @property
    def description_chars(self) -> int:
        return len(self.description_text or "")


@dataclass
class FilterResult:
    passed: bool
    rules_failed: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class ScoreComponents:
    semantic: float = 0.0
    title: float = 0.0
    freshness: float = 0.0
    location: float = 0.0
    skills: float = 0.0
    priority: float = 0.0
    adjustments: float = 0.0

    def total(self) -> float:
        return round(
            self.semantic + self.title + self.freshness
            + self.location + self.skills + self.priority + self.adjustments,
            2,
        )

    def as_dict(self) -> dict:
        return {
            "semantic": round(self.semantic, 2),
            "title": round(self.title, 2),
            "freshness": round(self.freshness, 2),
            "location": round(self.location, 2),
            "skills": round(self.skills, 2),
            "priority": round(self.priority, 2),
            "adjustments": round(self.adjustments, 2),
        }


@dataclass
class RankedJob:
    job: Job
    components: ScoreComponents
    score: float
    matched_skills_strong: list[str] = field(default_factory=list)
    matched_skills_familiar: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
