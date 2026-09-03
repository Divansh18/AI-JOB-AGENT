"""Pipeline orchestration.

Each stage is idempotent and independently re-runnable, reading and writing
state in SQLite. Re-running the whole pipeline must never create duplicates
or reset first_seen_at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ..config.loader import AppConfig
from ..domain import filters as filter_engine
from ..domain.dedupe import find_duplicates
from ..domain.extract import extract_yoe
from ..domain.ranking import percentile_normalize, score_job
from ..domain.taxonomy import classify_title
from ..embeddings.encoder import cosine, from_blob, get_encoder, to_blob
from ..persistence.repositories import (
    CompanyRepo,
    EmbeddingRepo,
    FilterRepo,
    JobRepo,
    ScoreRepo,
    SourceRepo,
)
from ..sources.base import FetchContext, FetchReport
from ..sources.http import HttpClient
from ..sources.registry import build_adapters
from .audit import Audit
from .normalization import normalize_posting

JD_EMBED_CHARS = 3000


@dataclass
class StageCounts:
    fetched: int = 0
    new_jobs: int = 0
    updated_jobs: int = 0
    duplicates: int = 0
    filtered_out: int = 0
    passed_filters: int = 0
    ranked: int = 0
    errors: int = 0
    per_source: dict = field(default_factory=dict)
    filter_histogram: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "fetched": self.fetched, "new_jobs": self.new_jobs,
            "updated_jobs": self.updated_jobs, "duplicates": self.duplicates,
            "filtered_out": self.filtered_out, "passed_filters": self.passed_filters,
            "ranked": self.ranked, "errors": self.errors,
            "per_source": self.per_source, "filter_histogram": self.filter_histogram,
        }


def discover(conn, config: AppConfig, *, only_source: str | None = None,
             limit_per_company: int | None = None,
             include_aggregators: bool = True) -> StageCounts:
    """Fetch from all enabled sources and persist normalized jobs."""
    counts = StageCounts()
    audit = Audit(conn)
    job_repo, company_repo, source_repo = JobRepo(conn), CompanyRepo(conn), SourceRepo(conn)

    companies = [dict(r) for r in company_repo.active()]
    http = HttpClient(
        user_agent=config.settings.http.user_agent,
        timeout=config.settings.http.timeout_seconds,
        max_retries=config.settings.http.max_retries,
        requests_per_second=config.settings.http.requests_per_second,
    )
    try:
        for adapter in build_adapters(http, include_aggregators=include_aggregators):
            if only_source and adapter.name != only_source:
                continue
            ok, reason = adapter.available()
            if not ok:
                counts.per_source[adapter.name] = {"skipped": reason}
                audit.system("source_skipped", source=adapter.name, reason=reason)
                continue

            source_id = source_repo.ensure(adapter.name, adapter.kind)
            run_id = source_repo.start_run(adapter.name)
            report = FetchReport(source=adapter.name)
            new_here = 0
            ctx = FetchContext(companies=companies, settings=config.settings,
                               limit_per_company=limit_per_company)
            try:
                for posting in adapter.fetch(ctx, report):
                    job = normalize_posting(posting)
                    job.source_id = source_id
                    if job.company_id is None:
                        job.company_id = posting.company_id
                    _, is_new = job_repo.upsert(job)
                    if is_new:
                        new_here += 1
                        counts.new_jobs += 1
                    else:
                        counts.updated_jobs += 1
                status = "ok" if report.errors == 0 else "partial"
            except Exception as exc:  # one bad adapter must not kill the run
                report.errors += 1
                report.error_detail.append(f"{type(exc).__name__}: {exc}")
                status = "error"

            counts.fetched += report.fetched
            counts.errors += report.errors
            counts.per_source[adapter.name] = {
                "fetched": report.fetched, "new": new_here, "errors": report.errors,
                "detail": report.error_detail[:5],
            }
            source_repo.finish_run(run_id, status=status, fetched=report.fetched,
                                   new_jobs=new_here, errors=report.errors,
                                   error_detail="; ".join(report.error_detail[:10]) or None)

        # Record per-company fetch outcomes for health reporting.
        for company in companies:
            if "_ok" in company:
                company_repo.record_fetch(company["id"], bool(company["_ok"]))
    finally:
        http.close()

    audit.system("discover_complete", **{k: v for k, v in counts.as_dict().items()
                                         if k != "per_source"})
    return counts


def dedupe(conn) -> int:
    """Link duplicate jobs to a canonical record. Idempotent."""
    job_repo = JobRepo(conn)
    jobs = job_repo.all_active(exclude_duplicates=False)
    links = find_duplicates(jobs)
    by_fp = {j.fingerprint: j.id for j in jobs}
    n = 0
    for link in links:
        canonical_id = by_fp.get(link.canonical_fingerprint)
        duplicate_id = by_fp.get(link.duplicate_fingerprint)
        if canonical_id and duplicate_id and canonical_id != duplicate_id:
            job_repo.add_duplicate(canonical_id, duplicate_id, link.method, link.similarity)
            n += 1
    Audit(conn).system("dedupe_complete", links=n)
    return n


def apply_filters(conn, config: AppConfig, *, rerun: bool = False) -> StageCounts:
    """Evaluate hard filters over all non-duplicate active jobs."""
    counts = StageCounts()
    job_repo, filter_repo = JobRepo(conn), FilterRepo(conn)
    if rerun:
        filter_repo.clear()

    now = datetime.now(timezone.utc)
    for job in job_repo.all_active(exclude_duplicates=True):
        if not rerun and filter_repo.get(job.id) is not None:
            continue
        result = filter_engine.evaluate(job, config.filters, now=now)
        filter_repo.save(job.id, result, config.filters.version)
        if result.passed:
            counts.passed_filters += 1
            if job_repo.row(job.id)["status"] in ("new", "filtered_out"):
                job_repo.set_status(job.id, "ranked")
        else:
            counts.filtered_out += 1
            job_repo.set_status(job.id, "filtered_out")
    counts.filter_histogram = filter_repo.rule_histogram()
    Audit(conn).system("filter_complete", passed=counts.passed_filters,
                       filtered_out=counts.filtered_out)
    return counts


def _embed_missing(conn, config: AppConfig, jobs) -> dict[int, np.ndarray]:
    """Ensure every job has an embedding; return id -> vector."""
    emb_repo = EmbeddingRepo(conn)
    encoder = get_encoder()
    ids = [j.id for j in jobs]
    missing_ids = set(emb_repo.missing(ids))

    if missing_ids:
        to_encode = [j for j in jobs if j.id in missing_ids]
        texts = [f"{j.title}. {j.description_text[:JD_EMBED_CHARS]}" for j in to_encode]
        for start in range(0, len(texts), 64):
            batch = to_encode[start : start + 64]
            vectors = encoder.encode(texts[start : start + 64])
            for job, vec in zip(batch, vectors):
                emb_repo.save(job.id, encoder.model_name, to_blob(vec), encoder.dim)

    blobs = emb_repo.get_many(ids)
    return {jid: from_blob(blob, encoder.dim) for jid, blob in blobs.items()}


def rank(conn, config: AppConfig, *, rerun: bool = False) -> StageCounts:
    """Score every job that passed the filters."""
    counts = StageCounts()
    job_repo, filter_repo = JobRepo(conn), FilterRepo(conn)
    score_repo, company_repo = ScoreRepo(conn), CompanyRepo(conn)
    if rerun:
        score_repo.clear("rank")

    passed_ids = set(filter_repo.passed_job_ids())
    jobs = [j for j in job_repo.all_active(exclude_duplicates=True) if j.id in passed_ids]
    if not jobs:
        return counts

    vectors = _embed_missing(conn, config, jobs)
    encoder = get_encoder()
    profile_vec = encoder.encode([config.profile.matching_text()])[0]

    ordered = [j for j in jobs if j.id in vectors]
    matrix = np.vstack([vectors[j.id] for j in ordered]) if ordered else np.zeros((0, encoder.dim))
    sims = cosine(matrix, profile_vec).tolist()
    norms = (percentile_normalize(sims) if config.ranking.semantic_percentile_normalize
             else [max(0.0, min(1.0, s)) for s in sims])

    priorities = company_repo.priority_map()
    now = datetime.now(timezone.utc)
    for job, norm in zip(ordered, norms):
        row = filter_repo.get(job.id)
        signals = json.loads(row["signals"]) if row else {}
        ranked = score_job(
            job, profile=config.profile, config=config.ranking,
            semantic_norm=norm, signals=signals,
            company_priority=priorities.get(job.company_id, "normal"), now=now,
        )
        score_repo.save_rank(job.id, ranked, config.ranking.version)
        counts.ranked += 1

    Audit(conn).system("rank_complete", ranked=counts.ranked,
                       encoder=encoder.model_name)
    return counts


def run_all(conn, config: AppConfig, *, skip_discover: bool = False,
            rerun: bool = False, limit_per_company: int | None = None,
            include_aggregators: bool = True) -> StageCounts:
    total = StageCounts()
    if not skip_discover:
        d = discover(conn, config, limit_per_company=limit_per_company,
                     include_aggregators=include_aggregators)
        total.fetched, total.new_jobs = d.fetched, d.new_jobs
        total.updated_jobs, total.errors = d.updated_jobs, d.errors
        total.per_source = d.per_source
    total.duplicates = dedupe(conn)
    f = apply_filters(conn, config, rerun=rerun)
    total.filtered_out, total.passed_filters = f.filtered_out, f.passed_filters
    total.filter_histogram = f.filter_histogram
    r = rank(conn, config, rerun=rerun)
    total.ranked = r.ranked
    return total
