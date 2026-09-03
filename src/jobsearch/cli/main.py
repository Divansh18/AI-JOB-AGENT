"""Command-line interface.

The CLI is a thin adapter: it parses arguments, calls a service, and formats
the result. It contains no business logic, which is what allows a web
dashboard to be added later against the same service layer.
"""

from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..config.loader import load_config, project_root
from ..persistence.db import connect, migrate
from ..persistence.repositories import (
    ApplicationRepo,
    CompanyRepo,
    EventRepo,
    FilterRepo,
    JobRepo,
    ScoreRepo,
)
from ..services import digest as digest_service
from ..services import health as health_service
from ..services import ledger as ledger_service
from ..services import pipeline as pipeline_service

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Personal job discovery, ranking and application ledger.")
companies_app = typer.Typer(no_args_is_help=True, help="Manage the target company list.")
jobs_app = typer.Typer(no_args_is_help=True, help="Browse and manage jobs.")
db_app = typer.Typer(no_args_is_help=True, help="Database maintenance.")
app.add_typer(companies_app, name="companies")
app.add_typer(jobs_app, name="jobs")
app.add_typer(db_app, name="db")

console = Console()


def _ctx():
    config = load_config()
    conn = connect(config.db_path)
    migrate(conn)
    return config, conn


def _emit(payload: dict, as_json: bool) -> bool:
    if as_json:
        console.print_json(json.dumps(payload, default=str))
        return True
    return False


# --- setup -----------------------------------------------------------------


@app.command()
def init() -> None:
    """Create the database and apply migrations."""
    config = load_config()
    conn = connect(config.db_path)
    applied = migrate(conn)
    console.print(f"[green]ok[/] database at {config.db_path}")
    console.print(f"    migrations applied: {applied or 'none pending'}")
    console.print(f"    companies in config: {len(config.companies.companies)}")


@app.command()
def doctor(json_out: bool = typer.Option(False, "--json")) -> None:
    """Validate configuration and report system health."""
    config, conn = _ctx()
    checks = health_service.doctor(conn, config)
    if _emit({"checks": [{"level": l, "check": c, "detail": d} for l, c, d in checks]}, json_out):
        return
    table = Table(title="doctor", show_lines=False)
    table.add_column(""); table.add_column("check"); table.add_column("detail")
    marks = {"ok": "[green]OK[/]", "warn": "[yellow]WARN[/]", "fail": "[red]FAIL[/]"}
    for level, check, detail in checks:
        table.add_row(marks.get(level, level), check, str(detail))
    console.print(table)
    if any(l == "fail" for l, _, _ in checks):
        raise typer.Exit(1)


# --- companies -------------------------------------------------------------


@companies_app.command("discover")
def companies_discover(
    names_file: Path = typer.Option(..., "--names-file", exists=True),
    ats: str = typer.Option("greenhouse,lever,ashby", "--ats"),
    limit: int = typer.Option(0, "--limit", help="0 = no limit"),
    out: Path = typer.Option(Path("config/companies.candidates.yaml"), "--out"),
) -> None:
    """Probe company names against public ATS boards and record verified hits."""
    from ..config.schemas import CompaniesFile
    from ..services.companies import (
        load_companies_file,
        merge_entries,
        probe_company,
        write_companies_file,
    )
    from ..sources.http import HttpClient

    config, conn = _ctx()
    names = [l.strip() for l in names_file.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    if limit:
        names = names[:limit]
    ats_types = [a.strip() for a in ats.split(",") if a.strip()]

    known = {(c.ats, c.token) for c in config.companies.companies}
    candidates_path = config.root / out
    existing_candidates = load_companies_file(candidates_path)
    known |= {(c.ats, c.token) for c in existing_candidates.companies}

    http = HttpClient(user_agent=config.settings.http.user_agent,
                      timeout=config.settings.http.timeout_seconds,
                      max_retries=2,
                      requests_per_second=config.settings.http.requests_per_second)
    hits, checked = [], 0
    try:
        with console.status("probing...") as status:
            for name in names:
                checked += 1
                status.update(f"[{checked}/{len(names)}] {name} — {len(hits)} verified")
                hit = probe_company(http, name, ats_types)
                if hit and (hit.ats, hit.token) not in known:
                    hits.append(hit)
                    known.add((hit.ats, hit.token))
    except KeyboardInterrupt:
        console.print("[yellow]interrupted - saving what was verified so far[/]")
    finally:
        http.close()

    entries = [h.to_entry() for h in hits]
    merged, added = merge_entries(existing_candidates, entries)
    write_companies_file(candidates_path, merged)
    EventRepo(conn).log("system", "companies_discovered", payload={
        "checked": checked, "verified": len(hits), "added": added})
    console.print(f"[green]done[/] probed {checked} names, verified {len(hits)}, "
                  f"added {added} new candidates")
    console.print(f"    -> {candidates_path}")
    console.print("    review with: [bold]jobsearch companies review[/]")


@companies_app.command("review")
def companies_review(
    candidates: Path = typer.Option(Path("config/companies.candidates.yaml"), "--candidates"),
    accept_all: bool = typer.Option(False, "--accept-all",
                                    help="Accept every verified candidate without prompting"),
) -> None:
    """Accept or reject discovered candidates into companies.yaml."""
    from ..services.companies import (
        load_companies_file,
        merge_entries,
        write_companies_file,
    )

    config, conn = _ctx()
    cand_path = config.root / candidates
    main_path = config.root / "config" / "companies.yaml"
    cand = load_companies_file(cand_path)
    if not cand.companies:
        console.print("[yellow]no candidates to review[/]")
        return

    main = load_companies_file(main_path)
    known = {(c.ats, c.token) for c in main.companies}
    pending = [c for c in cand.companies if (c.ats, c.token) not in known]
    console.print(f"{len(pending)} candidates pending review\n")

    accepted, rejected = [], []
    for i, entry in enumerate(pending, 1):
        if accept_all:
            accepted.append(entry)
            continue
        console.print(f"[bold]{i}/{len(pending)}[/] {entry.name}  "
                      f"[dim]{entry.ats}/{entry.token}[/]")
        console.print(f"    {entry.notes or ''}")
        choice = typer.prompt("  [y]es / [n]o / [h]igh priority / [q]uit", default="y")
        c = choice.strip().lower()[:1]
        if c == "q":
            break
        if c == "n":
            rejected.append(entry)
        elif c == "h":
            entry.priority = "high"
            accepted.append(entry)
        else:
            accepted.append(entry)

    merged, added = merge_entries(main, accepted)
    write_companies_file(main_path, merged)
    remaining = [c for c in cand.companies
                 if (c.ats, c.token) not in {(a.ats, a.token) for a in accepted}
                 and (c.ats, c.token) not in {(r.ats, r.token) for r in rejected}]
    cand.companies = remaining
    write_companies_file(cand_path, cand)
    EventRepo(conn).log("human", "companies_reviewed", payload={
        "accepted": len(accepted), "rejected": len(rejected)})
    console.print(f"\n[green]accepted {added}[/], rejected {len(rejected)}, "
                  f"{len(remaining)} left pending")
    console.print("    run [bold]jobsearch companies sync[/] to load them")


@companies_app.command("sync")
def companies_sync() -> None:
    """Reconcile companies.yaml into the database."""
    from ..services.companies import sync_to_db

    config, conn = _ctx()
    result = sync_to_db(conn, config.companies)
    console.print(f"[green]synced[/] {result['synced']} companies "
                  f"({result['deactivated']} deactivated)")


@companies_app.command("add")
def companies_add(name: str, ats: str = typer.Option(...), token: str = typer.Option(...),
                  priority: str = typer.Option("normal"),
                  verify: bool = typer.Option(True, "--verify/--no-verify")) -> None:
    """Add a single company, verifying its board by default."""
    from ..config.schemas import CompanyEntry
    from ..services.companies import (
        load_companies_file,
        merge_entries,
        write_companies_file,
    )
    from ..sources.http import HttpClient
    from ..sources.registry import PROBE_ENDPOINTS, extract_probe_result

    config, _ = _ctx()
    if verify:
        http = HttpClient(user_agent=config.settings.http.user_agent)
        payload, status = http.get_json(PROBE_ENDPOINTS[ats].format(token=token))
        http.close()
        count, titles = extract_probe_result(ats, payload)
        if count < 1:
            console.print(f"[red]could not verify[/] {ats}/{token} (HTTP {status}, {count} jobs)")
            raise typer.Exit(1)
        console.print(f"[green]verified[/] {count} live postings, e.g. {titles[0][:60]!r}")

    path = config.root / "config" / "companies.yaml"
    main = load_companies_file(path)
    merged, added = merge_entries(
        main, [CompanyEntry(name=name, ats=ats, token=token, priority=priority)])
    write_companies_file(path, merged)
    console.print(f"[green]added[/] {name}" if added else "[yellow]already present[/]")


@companies_app.command("list")
def companies_list(failing: bool = typer.Option(False, "--failing"),
                   json_out: bool = typer.Option(False, "--json")) -> None:
    """List tracked companies."""
    config, conn = _ctx()
    repo = CompanyRepo(conn)
    rows = repo.failing() if failing else repo.all()
    if _emit({"companies": [dict(r) for r in rows]}, json_out):
        return
    table = Table(title=f"companies ({len(rows)})")
    for col in ("name", "ats", "token", "prio", "active", "fails", "last ok"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["name"], r["ats_type"], r["ats_token"], r["priority"],
                      "yes" if r["active"] else "no", str(r["consecutive_failures"]),
                      (r["last_ok_at"] or "-")[:10])
    console.print(table)


@companies_app.command("suggest")
def companies_suggest(min_count: int = typer.Option(2, "--min-count")) -> None:
    """Suggest untracked companies seen repeatedly in ingested jobs."""
    config, conn = _ctx()
    rows = conn.execute(
        """SELECT company_name_raw, COUNT(*) c FROM jobs
           WHERE company_id IS NULL AND status != 'filtered_out'
           GROUP BY company_normalized HAVING c >= ?
           ORDER BY c DESC LIMIT 60""", (min_count,)).fetchall()
    if not rows:
        console.print("[yellow]no suggestions yet[/]")
        return
    table = Table(title="companies worth tracking")
    table.add_column("company"); table.add_column("relevant postings")
    for r in rows:
        table.add_row(r["company_name_raw"], str(r["c"]))
    console.print(table)
    console.print("\nAdd the ones you want, then verify:\n"
                  "  [bold]jobsearch companies discover --names-file <file>[/]")


# --- pipeline --------------------------------------------------------------


@app.command()
def discover(source: str = typer.Option(None, "--source"),
             limit_per_company: int = typer.Option(0, "--limit-per-company")) -> None:
    """Fetch postings from all enabled sources."""
    config, conn = _ctx()
    counts = pipeline_service.discover(
        conn, config, only_source=source,
        limit_per_company=limit_per_company or None)
    console.print(f"[green]fetched[/] {counts.fetched}  "
                  f"new={counts.new_jobs}  updated={counts.updated_jobs}  "
                  f"errors={counts.errors}")
    for name, info in counts.per_source.items():
        console.print(f"    {name}: {info}")


@app.command("pipeline")
def pipeline_cmd(
    action: str = typer.Argument("run"),
    skip_discover: bool = typer.Option(False, "--skip-discover"),
    rerun: bool = typer.Option(False, "--rerun", help="Recompute filters and scores"),
    limit_per_company: int = typer.Option(0, "--limit-per-company"),
    no_aggregators: bool = typer.Option(False, "--no-aggregators"),
) -> None:
    """Run the full pipeline: discover -> dedupe -> filter -> rank."""
    if action != "run":
        console.print(f"[red]unknown action {action!r}[/]")
        raise typer.Exit(1)
    config, conn = _ctx()
    import time

    started = time.monotonic()
    counts = pipeline_service.run_all(
        conn, config, skip_discover=skip_discover, rerun=rerun,
        limit_per_company=limit_per_company or None,
        include_aggregators=not no_aggregators)
    elapsed = time.monotonic() - started
    console.print(
        f"[green]pipeline complete[/] in {elapsed:.1f}s\n"
        f"    fetched={counts.fetched} new={counts.new_jobs} updated={counts.updated_jobs}\n"
        f"    duplicates={counts.duplicates} filtered_out={counts.filtered_out} "
        f"passed={counts.passed_filters} ranked={counts.ranked} errors={counts.errors}")
    if counts.filter_histogram:
        top = list(counts.filter_histogram.items())[:6]
        console.print("    filter hits: " + ", ".join(f"{k}={v}" for k, v in top))


@app.command("filter")
def filter_cmd(rerun: bool = typer.Option(False, "--rerun"),
               explain: int = typer.Option(0, "--explain", help="Explain one job id")) -> None:
    """Apply hard filters, or explain one job's filter outcome."""
    config, conn = _ctx()
    if explain:
        from ..domain.filters import RULE_DESCRIPTIONS

        row = FilterRepo(conn).get(explain)
        job = JobRepo(conn).row(explain)
        if job is None:
            console.print(f"[red]job {explain} not found[/]")
            raise typer.Exit(1)
        console.print(f"[bold]{job['title']}[/] — {job['company_name_raw']}")
        if row is None:
            console.print("[yellow]not yet evaluated[/]")
            return
        passed = bool(row["passed"])
        console.print(f"verdict: {'[green]PASSED[/]' if passed else '[red]EXCLUDED[/]'}")
        for rid in json.loads(row["rules_failed"]):
            console.print(f"  [red]x[/] {rid}: {RULE_DESCRIPTIONS.get(rid, '?')}")
        for note in json.loads(row["notes"]):
            console.print(f"  [dim]- {note}[/]")
        console.print("\nsignals:")
        console.print_json(row["signals"])
        return
    counts = pipeline_service.apply_filters(conn, config, rerun=rerun)
    console.print(f"[green]passed[/] {counts.passed_filters}  "
                  f"[dim]filtered_out[/] {counts.filtered_out}")
    for rule, n in list(counts.filter_histogram.items())[:10]:
        console.print(f"    {rule}: {n}")


@app.command("rank")
def rank_cmd(rerun: bool = typer.Option(False, "--rerun")) -> None:
    """Score jobs that passed the filters."""
    config, conn = _ctx()
    counts = pipeline_service.rank(conn, config, rerun=rerun)
    console.print(f"[green]ranked[/] {counts.ranked} jobs")


@app.command("dedupe")
def dedupe_cmd() -> None:
    """Link duplicate postings to a canonical record."""
    _, conn = _ctx()
    console.print(f"[green]linked[/] {pipeline_service.dedupe(conn)} duplicates")


# --- jobs ------------------------------------------------------------------


@jobs_app.command("list")
def jobs_list(status: str = typer.Option(None, "--status"),
              min_score: float = typer.Option(0.0, "--min-score"),
              limit: int = typer.Option(20, "--limit"),
              json_out: bool = typer.Option(False, "--json")) -> None:
    """List ranked jobs."""
    config, conn = _ctx()
    rows = ScoreRepo(conn).top(limit, min_score=min_score, exclude_applied=False)
    if status:
        rows = [r for r in rows if r["status"] == status]
    if _emit({"jobs": [{"id": r["id"], "title": r["title"],
                        "company": r["company_name_raw"], "score": r["score"],
                        "url": r["apply_url"]} for r in rows]}, json_out):
        return
    table = Table(title=f"jobs ({len(rows)})")
    for col in ("id", "score", "title", "company", "location", "age", "src"):
        table.add_column(col)
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    for r in rows:
        try:
            seen = datetime.fromisoformat(r["first_seen_at"])
            age = f"{(now - seen).total_seconds() / 3600:.0f}h"
        except (ValueError, TypeError):
            age = "-"
        table.add_row(str(r["id"]), f"{r['score']:.0f}", r["title"][:44],
                      r["company_name_raw"][:22], (r["location_raw"] or "-")[:22],
                      age, r["source_name"])
    console.print(table)


@jobs_app.command("show")
def jobs_show(job_id: int) -> None:
    """Show a job with its score breakdown and filter signals."""
    config, conn = _ctx()
    job = JobRepo(conn).row(job_id)
    if job is None:
        console.print(f"[red]job {job_id} not found[/]")
        raise typer.Exit(1)
    console.print(f"\n[bold]{job['title']}[/]")
    console.print(f"{job['company_name_raw']} — {job['location_raw'] or '-'} "
                  f"({job['remote_type']}, {job['employment_type']})")
    console.print(f"[dim]{job['apply_url']}[/]")
    console.print(f"[dim]source={job['source_name']}  first_seen={job['first_seen_at'][:16]}  "
                  f"status={job['status']}[/]\n")

    score = ScoreRepo(conn).get(job_id)
    if score:
        console.print(f"score [bold]{score['score']:.1f}[/]")
        for k, v in json.loads(score["components"]).items():
            if v:
                console.print(f"    {k:12} {v:6.2f}")
        matched = json.loads(score["matched"] or "{}")
        if matched.get("strong"):
            console.print(f"    [green]core skills[/]: {', '.join(matched['strong'])}")
        if matched.get("familiar"):
            console.print(f"    [yellow]developing[/]: {', '.join(matched['familiar'])}")
        for f in json.loads(score["flags"] or "[]"):
            console.print(f"    [yellow]flag[/] {f}")

    triage = ScoreRepo(conn).get(job_id, stage="triage")
    if triage:
        console.print(f"\n[bold]LLM triage[/] ({triage['model']}): {triage['score']:.0f}")
        console.print(f"    {triage['rationale']}")

    fr = FilterRepo(conn).get(job_id)
    if fr:
        console.print(f"\nfilters: {'[green]passed[/]' if fr['passed'] else '[red]excluded[/]'}")
        for rid in json.loads(fr["rules_failed"]):
            console.print(f"    [red]x[/] {rid}")
    console.print(f"\n[dim]{(job['description_text'] or '')[:1200]}...[/]")


@jobs_app.command("add")
def jobs_add(url: str, company: str = typer.Option(None, "--company"),
             title: str = typer.Option(None, "--title"),
             location: str = typer.Option(None, "--location")) -> None:
    """Manually ingest a job posting by URL."""
    from ..services.normalization import normalize_posting
    from ..sources.http import HttpClient
    from ..sources.manual import build_posting

    config, conn = _ctx()
    http = HttpClient(user_agent=config.settings.http.user_agent)
    resp = http.get(url)
    http.close()
    if resp is None or resp.status_code != 200:
        console.print(f"[red]could not fetch[/] (HTTP {getattr(resp, 'status_code', 'n/a')}). "
                      "Pass --title/--company to add it manually.")
        raise typer.Exit(1)
    posting = build_posting(url, resp.text, company=company, title=title, location=location)
    job = normalize_posting(posting)
    from ..persistence.repositories import SourceRepo

    job.source_id = SourceRepo(conn).ensure("manual", "manual")
    job_id, is_new = JobRepo(conn).upsert(job)
    EventRepo(conn).log("human", "job_added_manually", "job", job_id, {"url": url})
    console.print(f"[green]{'added' if is_new else 'updated'}[/] job {job_id}: {job.title}")
    console.print("    run [bold]jobsearch pipeline run --skip-discover[/] to score it")


@jobs_app.command("dismiss")
def jobs_dismiss(job_id: int, reason: str = typer.Option(..., "--reason")) -> None:
    """Dismiss a job so it stops appearing in digests."""
    _, conn = _ctx()
    try:
        ledger_service.dismiss(conn, job_id, reason)
    except ledger_service.LedgerError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    console.print(f"[green]dismissed[/] job {job_id}")


@jobs_app.command("open")
def jobs_open(job_id: int) -> None:
    """Open a job's apply URL in the browser."""
    _, conn = _ctx()
    job = JobRepo(conn).row(job_id)
    if job is None:
        console.print(f"[red]job {job_id} not found[/]")
        raise typer.Exit(1)
    console.print(job["apply_url"])
    webbrowser.open(job["apply_url"])


# --- digest ----------------------------------------------------------------


@app.command()
def digest(top: int = typer.Option(0, "--top"),
           open_browser: bool = typer.Option(False, "--open"),
           date_str: str = typer.Option(None, "--date")) -> None:
    """Generate the daily HTML digest."""
    from ..render.digest import write_digest
    from ..persistence.repositories import DigestRepo

    config, conn = _ctx()
    data = digest_service.build(conn, config, top_n=top or None, for_date=date_str)
    path = write_digest(data, config.digest_dir)
    DigestRepo(conn).save(data.digest_date, str(path), data.counts)
    console.print(f"[green]digest[/] {path}")
    console.print(f"    {data.counts['shown']} shown, {data.counts['passed_filters']} passed "
                  f"filters, {data.counts['seen_last_24h']} new in 24h")
    if open_browser:
        webbrowser.open(f"file://{path}")


# --- ledger ----------------------------------------------------------------


@app.command()
def apply(job_id: int, channel: str = typer.Option(None, "--channel"),
          note: str = typer.Option(None, "--note")) -> None:
    """Record that you applied to a job."""
    _, conn = _ctx()
    try:
        app_id = ledger_service.mark_applied(conn, job_id, channel=channel, notes=note)
    except ledger_service.LedgerError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    console.print(f"[green]recorded[/] application #{app_id} for job {job_id}")


@app.command()
def status(identifier: int, new_status: str, note: str = typer.Option(None, "--note")) -> None:
    """Update an application's status (accepts a job id or an application id)."""
    _, conn = _ctx()
    try:
        app_id = ledger_service.resolve_application_id(conn, identifier)
        ledger_service.update_status(conn, app_id, new_status, note)
    except ledger_service.LedgerError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    console.print(f"[green]updated[/] application #{app_id} -> {new_status}")


@app.command()
def ledger(status_filter: str = typer.Option(None, "--status"),
           needs_followup: bool = typer.Option(False, "--needs-followup"),
           json_out: bool = typer.Option(False, "--json")) -> None:
    """List tracked applications."""
    config, conn = _ctx()
    repo = ApplicationRepo(conn)
    rows = (repo.needing_followup(config.settings.digest.followup_days)
            if needs_followup else repo.list(status_filter))
    if _emit({"applications": [dict(r) for r in rows]}, json_out):
        return
    table = Table(title=f"applications ({len(rows)})")
    for col in ("app", "job", "title", "company", "status", "updated"):
        table.add_column(col)
    for r in rows:
        table.add_row(str(r["id"]), str(r["job_id"]), r["title"][:40],
                      r["company_name_raw"][:22], r["status"], (r["updated_at"] or "")[:10])
    console.print(table)


@app.command()
def stats(json_out: bool = typer.Option(False, "--json")) -> None:
    """Funnel counts, response rate and spend."""
    from statistics import median

    from ..persistence.repositories import LlmCallRepo

    config, conn = _ctx()
    funnel = ledger_service.funnel_stats(conn)
    tta = ledger_service.time_to_apply_hours(conn)
    job_repo = JobRepo(conn)
    payload = {
        "jobs_tracked": job_repo.count(),
        "duplicates_merged": job_repo.duplicate_count(),
        "passed_filters": len(FilterRepo(conn).passed_job_ids()),
        "filtered_out": job_repo.count("status='filtered_out'"),
        **funnel,
        "median_time_to_apply_hours": round(median(tta), 1) if tta else None,
        "llm_spend_inr": round(LlmCallRepo(conn).spend_inr(), 2),
    }
    if _emit(payload, json_out):
        return
    table = Table(title="stats")
    table.add_column("metric"); table.add_column("value", justify="right")
    for k, v in payload.items():
        if k == "by_status":
            continue
        table.add_row(k, str(v))
    console.print(table)
    if funnel["by_status"]:
        console.print("\nby status: " + ", ".join(f"{k}={v}" for k, v in funnel["by_status"].items()))


# --- optional LLM triage ---------------------------------------------------


@app.command()
def triage(top: int = typer.Option(25, "--top"),
           dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Optional LLM second opinion on top-ranked jobs. Disabled by default."""
    from ..llm.triage import run_triage

    config, conn = _ctx()
    report = run_triage(conn, config, top_n=top, dry_run=dry_run)
    if report.skipped_reason:
        console.print(f"[yellow]skipped[/] {report.skipped_reason}")
        console.print("    enable with llm.enabled: true in config/settings.yaml")
        return
    console.print(f"[green]triage[/] provider={report.provider} "
                  f"succeeded={report.succeeded} failed={report.failed} "
                  f"cached={report.from_cache} cost=INR {report.cost_inr:.2f}")
    for err in report.errors[:5]:
        console.print(f"    [red]{err}[/]")


# --- audit -----------------------------------------------------------------


@app.command()
def audit(n: int = typer.Option(30, "-n")) -> None:
    """Show the tail of the audit log."""
    _, conn = _ctx()
    table = Table(title="audit log")
    for col in ("ts", "actor", "action", "entity", "payload"):
        table.add_column(col)
    for r in EventRepo(conn).tail(n):
        table.add_row(r["ts"][:19], r["actor"], r["action"],
                      f"{r['entity_type'] or ''}:{r['entity_id'] or ''}".strip(":"),
                      (r["payload"] or "")[:60])
    console.print(table)


@db_app.command("migrate")
def db_migrate() -> None:
    """Apply pending migrations."""
    config = load_config()
    conn = connect(config.db_path)
    console.print(f"applied: {migrate(conn) or 'none pending'}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
