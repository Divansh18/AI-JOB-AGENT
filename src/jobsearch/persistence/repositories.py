"""Data access. All SQL lives here; services never write SQL directly."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from ..domain.models import Job
from .db import iso, parse_dt, utcnow


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        fingerprint=row["fingerprint"],
        content_hash=row["content_hash"],
        source=row["source_name"],
        source_id=row["source_id"],
        company_id=row["company_id"],
        company_name_raw=row["company_name_raw"],
        company_normalized=row["company_normalized"],
        external_id=row["external_id"],
        title=row["title"],
        title_normalized=row["title_normalized"],
        location_raw=row["location_raw"],
        locations=json.loads(row["locations"]),
        country=row["country"],
        remote_type=row["remote_type"],
        remote_scope=row["remote_scope"],
        employment_type=row["employment_type"],
        description_text=row["description_text"],
        apply_url=row["apply_url"],
        canonical_url=row["canonical_url"],
        posted_at=parse_dt(row["posted_at"]),
        first_seen_at=parse_dt(row["first_seen_at"]),
        raw=json.loads(row["raw"]) if row["raw"] else {},
    )


class CompanyRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert(self, *, name: str, name_normalized: str, ats_type: str, ats_token: str,
               board_url: str | None = None, priority: str = "normal",
               tags: list[str] | None = None, hq_country: str | None = None,
               active: bool = True, verified_at: str | None = None,
               added_via: str = "manual", notes: str | None = None) -> int:
        now = iso(utcnow())
        self.conn.execute(
            """
            INSERT INTO companies (name, name_normalized, ats_type, ats_token, board_url,
                                   priority, tags, hq_country, active, verified_at,
                                   added_at, added_via, verification_note)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ats_type, ats_token) DO UPDATE SET
                name=excluded.name, name_normalized=excluded.name_normalized,
                board_url=excluded.board_url, priority=excluded.priority,
                tags=excluded.tags, hq_country=excluded.hq_country,
                active=excluded.active, verified_at=excluded.verified_at,
                verification_note=excluded.verification_note
            """,
            (name, name_normalized, ats_type, ats_token, board_url, priority,
             json.dumps(tags or []), hq_country, int(active), verified_at, now,
             added_via, notes),
        )
        row = self.conn.execute(
            "SELECT id FROM companies WHERE ats_type=? AND ats_token=?", (ats_type, ats_token)
        ).fetchone()
        return row["id"]

    def active(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM companies WHERE active=1 ORDER BY priority DESC, name"
        ).fetchall()

    def all(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM companies ORDER BY name").fetchall()

    def deactivate_missing(self, keep_keys: set[tuple[str, str]]) -> int:
        rows = self.conn.execute("SELECT id, ats_type, ats_token FROM companies WHERE active=1").fetchall()
        n = 0
        for r in rows:
            if (r["ats_type"], r["ats_token"]) not in keep_keys:
                self.conn.execute("UPDATE companies SET active=0 WHERE id=?", (r["id"],))
                n += 1
        return n

    def record_fetch(self, company_id: int, ok: bool, note: str | None = None) -> None:
        now = iso(utcnow())
        if ok:
            self.conn.execute(
                "UPDATE companies SET last_fetched_at=?, last_ok_at=?, consecutive_failures=0 WHERE id=?",
                (now, now, company_id),
            )
        else:
            self.conn.execute(
                """UPDATE companies SET last_fetched_at=?,
                   consecutive_failures=consecutive_failures+1,
                   verification_note=COALESCE(?, verification_note) WHERE id=?""",
                (now, note, company_id),
            )

    def failing(self, threshold: int = 3) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM companies WHERE active=1 AND consecutive_failures>=? ORDER BY consecutive_failures DESC",
            (threshold,),
        ).fetchall()

    def priority_map(self) -> dict[int, str]:
        return {r["id"]: r["priority"] for r in self.conn.execute("SELECT id, priority FROM companies")}


class SourceRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def ensure(self, name: str, kind: str) -> int:
        self.conn.execute(
            "INSERT OR IGNORE INTO sources (name, kind) VALUES (?,?)", (name, kind)
        )
        return self.conn.execute("SELECT id FROM sources WHERE name=?", (name,)).fetchone()["id"]

    def start_run(self, source_name: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO source_runs (source_name, started_at, status) VALUES (?,?,?)",
            (source_name, iso(utcnow()), "running"),
        )
        return cur.lastrowid

    def finish_run(self, run_id: int, *, status: str, fetched: int, new_jobs: int,
                   errors: int, error_detail: str | None = None) -> None:
        self.conn.execute(
            """UPDATE source_runs SET finished_at=?, status=?, fetched=?, new_jobs=?,
               errors=?, error_detail=? WHERE id=?""",
            (iso(utcnow()), status, fetched, new_jobs, errors, error_detail, run_id),
        )

    def recent_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM source_runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()

    def health(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT source_name,
                   MAX(started_at) AS last_run,
                   SUM(fetched)    AS fetched,
                   SUM(new_jobs)   AS new_jobs,
                   SUM(errors)     AS errors,
                   MAX(CASE WHEN status='ok' THEN started_at END) AS last_ok
            FROM source_runs GROUP BY source_name ORDER BY source_name
            """
        ).fetchall()


class JobRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert(self, job: Job) -> tuple[int, bool]:
        """Insert or update a job. Returns (job_id, is_new).

        first_seen_at is never overwritten: it is our authoritative freshness
        signal and the basis of the time-to-apply metric.
        """
        existing = self.conn.execute(
            "SELECT id, content_hash FROM jobs WHERE fingerprint=?", (job.fingerprint,)
        ).fetchone()
        now = iso(utcnow())
        if existing:
            self.conn.execute(
                """UPDATE jobs SET last_seen_at=?, is_active=1, content_hash=?,
                   description_text=?, description_chars=?, apply_url=?, title=?,
                   title_normalized=?, location_raw=?, locations=?, country=?,
                   remote_type=?, remote_scope=?, employment_type=?
                   WHERE id=?""",
                (now, job.content_hash, job.description_text, job.description_chars,
                 job.apply_url, job.title, job.title_normalized, job.location_raw,
                 json.dumps(job.locations), job.country, job.remote_type,
                 job.remote_scope, job.employment_type, existing["id"]),
            )
            return existing["id"], False

        cur = self.conn.execute(
            """INSERT INTO jobs (fingerprint, content_hash, source_id, source_name,
                company_id, company_name_raw, company_normalized, external_id, title,
                title_normalized, location_raw, locations, country, remote_type,
                remote_scope, employment_type, description_text, description_chars,
                apply_url, canonical_url, posted_at, first_seen_at, last_seen_at, raw)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (job.fingerprint, job.content_hash, job.source_id, job.source,
             job.company_id, job.company_name_raw, job.company_normalized,
             job.external_id, job.title, job.title_normalized, job.location_raw,
             json.dumps(job.locations), job.country, job.remote_type, job.remote_scope,
             job.employment_type, job.description_text, job.description_chars,
             job.apply_url, job.canonical_url, iso(job.posted_at),
             iso(job.first_seen_at), now, json.dumps(job.raw)[:20000]),
        )
        return cur.lastrowid, True

    def get(self, job_id: int) -> Job | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def row(self, job_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def all_active(self, exclude_duplicates: bool = True) -> list[Job]:
        sql = "SELECT * FROM jobs WHERE is_active=1"
        if exclude_duplicates:
            sql += " AND id NOT IN (SELECT duplicate_job_id FROM job_duplicates)"
        return [_row_to_job(r) for r in self.conn.execute(sql).fetchall()]

    def by_status(self, status: str) -> list[Job]:
        return [_row_to_job(r) for r in self.conn.execute(
            "SELECT * FROM jobs WHERE status=?", (status,)).fetchall()]

    def set_status(self, job_id: int, status: str) -> None:
        self.conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))

    def count(self, where: str = "1=1", params: Iterable[Any] = ()) -> int:
        return self.conn.execute(f"SELECT COUNT(*) c FROM jobs WHERE {where}", tuple(params)).fetchone()["c"]

    def add_duplicate(self, canonical_id: int, duplicate_id: int, method: str, similarity: float) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO job_duplicates
               (duplicate_job_id, canonical_job_id, method, similarity, created_at)
               VALUES (?,?,?,?,?)""",
            (duplicate_id, canonical_id, method, similarity, iso(utcnow())),
        )

    def id_by_fingerprint(self, fingerprint: str) -> int | None:
        row = self.conn.execute("SELECT id FROM jobs WHERE fingerprint=?", (fingerprint,)).fetchone()
        return row["id"] if row else None

    def duplicate_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) c FROM job_duplicates").fetchone()["c"]


class FilterRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save(self, job_id: int, result, filters_version: int) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO filter_results
               (job_id, passed, rules_failed, signals, notes, filters_version, evaluated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (job_id, int(result.passed), json.dumps(result.rules_failed),
             json.dumps(result.signals), json.dumps(result.notes),
             filters_version, iso(utcnow())),
        )

    def get(self, job_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM filter_results WHERE job_id=?", (job_id,)).fetchone()

    def passed_job_ids(self) -> list[int]:
        return [r["job_id"] for r in self.conn.execute(
            "SELECT job_id FROM filter_results WHERE passed=1").fetchall()]

    def rule_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for r in self.conn.execute("SELECT rules_failed FROM filter_results WHERE passed=0"):
            for rule in json.loads(r["rules_failed"]):
                hist[rule] = hist.get(rule, 0) + 1
        return dict(sorted(hist.items(), key=lambda kv: -kv[1]))

    def sample_rejected(self, limit: int = 5) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT j.id, j.title, j.company_name_raw, j.location_raw, f.rules_failed
               FROM filter_results f JOIN jobs j ON j.id=f.job_id
               WHERE f.passed=0 ORDER BY RANDOM() LIMIT ?""", (limit,)
        ).fetchall()

    def clear(self) -> None:
        self.conn.execute("DELETE FROM filter_results")


class ScoreRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save_rank(self, job_id: int, ranked, rank_version: int) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO job_scores
               (job_id, stage, score, components, rank_version, flags, explanation,
                matched, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (job_id, "rank", ranked.score, json.dumps(ranked.components.as_dict()),
             rank_version, json.dumps(ranked.flags), json.dumps(ranked.explanation),
             json.dumps({"strong": ranked.matched_skills_strong,
                         "familiar": ranked.matched_skills_familiar}),
             iso(utcnow())),
        )

    def top(self, limit: int, *, min_score: float = 0.0, since: str | None = None,
            exclude_applied: bool = True, stage: str = "rank") -> list[sqlite3.Row]:
        sql = """
            SELECT j.*, s.score, s.components, s.flags, s.explanation, s.matched,
                   f.signals, c.priority AS company_priority
            FROM job_scores s
            JOIN jobs j ON j.id = s.job_id
            LEFT JOIN filter_results f ON f.job_id = j.id
            LEFT JOIN companies c ON c.id = j.company_id
            WHERE s.stage=? AND s.score >= ?
              AND j.id NOT IN (SELECT duplicate_job_id FROM job_duplicates)
              AND j.status NOT IN ('dismissed')
        """
        params: list[Any] = [stage, min_score]
        if exclude_applied:
            sql += " AND j.id NOT IN (SELECT job_id FROM applications)"
        if since:
            sql += " AND j.first_seen_at >= ?"
            params.append(since)
        sql += " ORDER BY s.score DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(sql, tuple(params)).fetchall()

    def clear(self, stage: str = "rank") -> None:
        self.conn.execute("DELETE FROM job_scores WHERE stage=?", (stage,))

    def get(self, job_id: int, stage: str = "rank") -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM job_scores WHERE job_id=? AND stage=?", (job_id, stage)).fetchone()


class EmbeddingRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save(self, job_id: int, model: str, vector: bytes, dim: int) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO job_embeddings (job_id, model, dim, vector, created_at)
               VALUES (?,?,?,?,?)""",
            (job_id, model, dim, vector, iso(utcnow())),
        )

    def get_many(self, job_ids: list[int]) -> dict[int, bytes]:
        if not job_ids:
            return {}
        marks = ",".join("?" * len(job_ids))
        rows = self.conn.execute(
            f"SELECT job_id, vector FROM job_embeddings WHERE job_id IN ({marks})", job_ids
        ).fetchall()
        return {r["job_id"]: r["vector"] for r in rows}

    def missing(self, job_ids: list[int]) -> list[int]:
        have = set(self.get_many(job_ids).keys())
        return [j for j in job_ids if j not in have]


class ApplicationRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_by_job(self, job_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM applications WHERE job_id=?", (job_id,)).fetchone()

    def get(self, app_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM applications WHERE id=?", (app_id,)).fetchone()

    def create(self, job_id: int, status: str, *, channel: str | None = None,
               notes: str | None = None, applied_at: datetime | None = None) -> int:
        now = iso(utcnow())
        cur = self.conn.execute(
            """INSERT INTO applications (job_id, status, applied_at, channel, notes,
                                         created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (job_id, status, iso(applied_at) if applied_at else now, channel, notes, now, now),
        )
        app_id = cur.lastrowid
        self.add_event(app_id, None, status, notes)
        return app_id

    def update_status(self, app_id: int, new_status: str, note: str | None = None) -> None:
        old = self.get(app_id)
        self.conn.execute(
            "UPDATE applications SET status=?, updated_at=? WHERE id=?",
            (new_status, iso(utcnow()), app_id),
        )
        self.add_event(app_id, old["status"] if old else None, new_status, note)

    def add_event(self, app_id: int, from_status: str | None, to_status: str,
                  note: str | None = None) -> None:
        self.conn.execute(
            """INSERT INTO application_events (application_id, from_status, to_status,
                                               occurred_at, note) VALUES (?,?,?,?,?)""",
            (app_id, from_status, to_status, iso(utcnow()), note),
        )

    def list(self, status: str | None = None) -> list[sqlite3.Row]:
        sql = """SELECT a.*, j.title, j.company_name_raw, j.apply_url, j.first_seen_at
                 FROM applications a JOIN jobs j ON j.id=a.job_id"""
        params: tuple = ()
        if status:
            sql += " WHERE a.status=?"
            params = (status,)
        sql += " ORDER BY a.updated_at DESC"
        return self.conn.execute(sql, params).fetchall()

    def needing_followup(self, days: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT a.*, j.title, j.company_name_raw, j.apply_url
               FROM applications a JOIN jobs j ON j.id=a.job_id
               WHERE a.status IN ('applied','no_response')
                 AND julianday('now') - julianday(a.updated_at) >= ?
               ORDER BY a.updated_at ASC""", (days,)
        ).fetchall()

    def funnel(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) c FROM applications GROUP BY status").fetchall()
        return {r["status"]: r["c"] for r in rows}


class EventRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def log(self, actor: str, action: str, entity_type: str | None = None,
            entity_id: str | int | None = None, payload: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, actor, action, entity_type, entity_id, payload) VALUES (?,?,?,?,?,?)",
            (iso(utcnow()), actor, action, entity_type,
             str(entity_id) if entity_id is not None else None, json.dumps(payload or {})),
        )

    def tail(self, n: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (n,)).fetchall()


class DigestRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save(self, digest_date: str, path: str, counts: dict) -> None:
        self.conn.execute(
            """INSERT INTO digests (digest_date, path, counts, created_at) VALUES (?,?,?,?)
               ON CONFLICT(digest_date) DO UPDATE SET path=excluded.path,
               counts=excluded.counts, created_at=excluded.created_at""",
            (digest_date, path, json.dumps(counts), iso(utcnow())),
        )


class LlmCallRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def log(self, **kw) -> None:
        self.conn.execute(
            """INSERT INTO llm_calls (ts, purpose, model, prompt_hash, job_id, input_tokens,
               cached_input_tokens, output_tokens, cost_usd, cost_inr, from_cache)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (iso(utcnow()), kw.get("purpose", ""), kw.get("model", ""),
             kw.get("prompt_hash", ""), kw.get("job_id"), kw.get("input_tokens", 0),
             kw.get("cached_input_tokens", 0), kw.get("output_tokens", 0),
             kw.get("cost_usd", 0.0), kw.get("cost_inr", 0.0), int(kw.get("from_cache", False))),
        )

    def spend_inr(self, since: str | None = None) -> float:
        sql = "SELECT COALESCE(SUM(cost_inr),0) s FROM llm_calls"
        params: tuple = ()
        if since:
            sql += " WHERE ts >= ?"
            params = (since,)
        return float(self.conn.execute(sql, params).fetchone()["s"])
