"""Application ledger with an enforced state machine."""

from __future__ import annotations

from datetime import datetime, timezone

from ..domain.models import ALLOWED_TRANSITIONS, RESPONSE_STATUSES, ApplicationStatus
from ..persistence.repositories import ApplicationRepo, JobRepo
from .audit import Audit


class LedgerError(Exception):
    pass


def mark_applied(conn, job_id: int, *, channel: str | None = None,
                 notes: str | None = None) -> int:
    """Record an application. Refuses duplicates outright."""
    job_repo, app_repo = JobRepo(conn), ApplicationRepo(conn)
    job = job_repo.row(job_id)
    if job is None:
        raise LedgerError(f"job {job_id} not found")

    existing = app_repo.get_by_job(job_id)
    if existing is not None:
        raise LedgerError(
            f"already recorded as application #{existing['id']} "
            f"(status={existing['status']}, at {existing['applied_at']})"
        )

    # Guard against applying twice to the same role via a duplicate record.
    dup = conn.execute(
        """SELECT a.id, a.status, j.title FROM job_duplicates d
           JOIN applications a ON a.job_id IN (d.canonical_job_id, d.duplicate_job_id)
           JOIN jobs j ON j.id = a.job_id
           WHERE d.canonical_job_id=? OR d.duplicate_job_id=?""",
        (job_id, job_id),
    ).fetchone()
    if dup is not None:
        raise LedgerError(
            f"a duplicate of this posting is already application #{dup['id']} ({dup['title']})"
        )

    app_id = app_repo.create(job_id, ApplicationStatus.APPLIED.value,
                             channel=channel, notes=notes,
                             applied_at=datetime.now(timezone.utc))
    job_repo.set_status(job_id, "applied")
    Audit(conn).human("application_recorded", "application", app_id,
                      job_id=job_id, title=job["title"], company=job["company_name_raw"])
    return app_id


def update_status(conn, app_id: int, new_status: str, note: str | None = None) -> None:
    app_repo = ApplicationRepo(conn)
    app = app_repo.get(app_id)
    if app is None:
        raise LedgerError(f"application {app_id} not found")
    try:
        current = ApplicationStatus(app["status"])
        target = ApplicationStatus(new_status)
    except ValueError as exc:
        raise LedgerError(f"unknown status: {exc}") from exc

    if target not in ALLOWED_TRANSITIONS.get(current, set()):
        allowed = sorted(s.value for s in ALLOWED_TRANSITIONS.get(current, set()))
        raise LedgerError(
            f"cannot move {current.value} -> {target.value}. Allowed: {allowed or 'none (terminal)'}"
        )
    app_repo.update_status(app_id, target.value, note)
    Audit(conn).human("status_changed", "application", app_id,
                      from_status=current.value, to_status=target.value, note=note)


def resolve_application_id(conn, identifier: int) -> int:
    """Accept either an application id or a job id."""
    app_repo = ApplicationRepo(conn)
    if app_repo.get(identifier) is not None:
        return identifier
    by_job = app_repo.get_by_job(identifier)
    if by_job is not None:
        return by_job["id"]
    raise LedgerError(f"no application found for id {identifier}")


def dismiss(conn, job_id: int, reason: str) -> None:
    job_repo = JobRepo(conn)
    if job_repo.row(job_id) is None:
        raise LedgerError(f"job {job_id} not found")
    job_repo.set_status(job_id, "dismissed")
    Audit(conn).human("job_dismissed", "job", job_id, reason=reason)


def funnel_stats(conn) -> dict:
    app_repo = ApplicationRepo(conn)
    counts = app_repo.funnel()
    total = sum(counts.values())
    responses = sum(c for s, c in counts.items()
                    if s in {x.value for x in RESPONSE_STATUSES})
    return {
        "applications": total,
        "responses": responses,
        "response_rate": round(100 * responses / total, 1) if total else 0.0,
        "by_status": counts,
    }


def time_to_apply_hours(conn) -> list[float]:
    rows = conn.execute(
        """SELECT j.first_seen_at, a.applied_at FROM applications a
           JOIN jobs j ON j.id=a.job_id WHERE a.applied_at IS NOT NULL"""
    ).fetchall()
    out = []
    for r in rows:
        try:
            seen = datetime.fromisoformat(r["first_seen_at"])
            applied = datetime.fromisoformat(r["applied_at"])
            out.append(max(0.0, (applied - seen).total_seconds() / 3600))
        except (ValueError, TypeError):
            continue
    return out
