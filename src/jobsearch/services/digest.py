"""Assemble the daily digest.

Returns a plain DTO. Rendering is a separate concern, so a future web
dashboard can consume exactly the same structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from ..config.loader import AppConfig
from ..persistence.repositories import (
    ApplicationRepo,
    CompanyRepo,
    FilterRepo,
    JobRepo,
    LlmCallRepo,
    ScoreRepo,
)
from .health import company_health, source_health


@dataclass
class DigestCard:
    job_id: int
    title: str
    company: str
    location: str
    remote_type: str
    employment_type: str
    score: float
    components: dict
    flags: list[str]
    explanation: list[str]
    matched_strong: list[str]
    matched_familiar: list[str]
    age_hours: float
    apply_url: str
    source: str
    snippet: str
    yoe_summary: str | None = None


@dataclass
class DigestData:
    digest_date: str
    generated_at: str
    counts: dict = field(default_factory=dict)
    new_today: list[DigestCard] = field(default_factory=list)
    aging: list[DigestCard] = field(default_factory=list)
    needs_action: list[dict] = field(default_factory=list)
    source_health: list[dict] = field(default_factory=list)
    company_health: dict = field(default_factory=dict)
    rejected_sample: list[dict] = field(default_factory=list)
    filter_histogram: dict = field(default_factory=dict)
    spend_inr: float = 0.0
    encoder: str = ""


def _age_hours(first_seen: str, now: datetime) -> float:
    try:
        seen = datetime.fromisoformat(first_seen)
    except (ValueError, TypeError):
        return 0.0
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    return max(0.0, (now - seen).total_seconds() / 3600)


def _yoe_summary(signals: dict) -> str | None:
    yoe = (signals or {}).get("yoe") or {}
    verdict = yoe.get("verdict")
    if not verdict or verdict == "unknown":
        return None
    req, pref = yoe.get("required_min"), yoe.get("preferred_min")
    if req is not None:
        return f"{req}+ yrs required"
    if pref is not None:
        return f"{pref}+ yrs preferred"
    return verdict


def _row_to_card(row, now: datetime) -> DigestCard:
    matched = json.loads(row["matched"] or "{}")
    signals = json.loads(row["signals"] or "{}") if "signals" in row.keys() else {}
    text = row["description_text"] or ""
    return DigestCard(
        job_id=row["id"],
        title=row["title"],
        company=row["company_name_raw"],
        location=row["location_raw"] or "-",
        remote_type=row["remote_type"],
        employment_type=row["employment_type"],
        score=round(row["score"], 1),
        components=json.loads(row["components"] or "{}"),
        flags=json.loads(row["flags"] or "[]"),
        explanation=json.loads(row["explanation"] or "[]"),
        matched_strong=matched.get("strong", []),
        matched_familiar=matched.get("familiar", []),
        age_hours=_age_hours(row["first_seen_at"], now),
        apply_url=row["apply_url"],
        source=row["source_name"],
        snippet=(text[:230] + "...") if len(text) > 230 else text,
        yoe_summary=_yoe_summary(signals),
    )


def build(conn, config: AppConfig, *, top_n: int | None = None,
          for_date: str | None = None) -> DigestData:
    now = datetime.now(timezone.utc)
    top_n = top_n or config.settings.digest.top_n
    d = config.settings.digest

    score_repo, job_repo = ScoreRepo(conn), JobRepo(conn)
    filter_repo, app_repo = FilterRepo(conn), ApplicationRepo(conn)

    new_rows = score_repo.top(top_n, exclude_applied=True)
    new_today = [_row_to_card(r, now) for r in new_rows]

    seen_ids = {c.job_id for c in new_today}
    aging_cutoff = (now - timedelta(days=2)).isoformat()
    aging_floor = (now - timedelta(days=d.aging_days)).isoformat()
    aging_rows = [
        r for r in score_repo.top(60, min_score=d.aging_min_score, exclude_applied=True)
        if r["id"] not in seen_ids and aging_floor <= r["first_seen_at"] <= aging_cutoff
    ][:15]
    aging = [_row_to_card(r, now) for r in aging_rows]

    needs_action = [
        {"application_id": r["id"], "job_id": r["job_id"], "title": r["title"],
         "company": r["company_name_raw"], "status": r["status"],
         "updated_at": r["updated_at"]}
        for r in app_repo.needing_followup(d.followup_days)
    ]
    for r in app_repo.list():
        if r["status"] in ("recruiter_reply", "screen_scheduled", "interviewing"):
            needs_action.append({
                "application_id": r["id"], "job_id": r["job_id"], "title": r["title"],
                "company": r["company_name_raw"], "status": r["status"],
                "updated_at": r["updated_at"],
            })

    counts = {
        "jobs_total": job_repo.count(),
        "jobs_active": job_repo.count("is_active=1"),
        "duplicates": job_repo.duplicate_count(),
        "passed_filters": len(filter_repo.passed_job_ids()),
        "filtered_out": job_repo.count("status='filtered_out'"),
        "shown": len(new_today),
        "seen_last_24h": job_repo.count(
            "first_seen_at >= ?", [(now - timedelta(hours=24)).isoformat()]),
        "applications": len(app_repo.list()),
    }

    from ..embeddings.encoder import get_encoder

    return DigestData(
        digest_date=for_date or date.today().isoformat(),
        generated_at=now.isoformat(timespec="seconds"),
        counts=counts,
        new_today=new_today,
        aging=aging,
        needs_action=needs_action,
        source_health=source_health(conn),
        company_health=company_health(conn),
        rejected_sample=[
            {"id": r["id"], "title": r["title"], "company": r["company_name_raw"],
             "location": r["location_raw"], "rules": json.loads(r["rules_failed"])}
            for r in filter_repo.sample_rejected(5)
        ],
        filter_histogram=filter_repo.rule_histogram(),
        spend_inr=LlmCallRepo(conn).spend_inr(),
        encoder=get_encoder().model_name,
    )
