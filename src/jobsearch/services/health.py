"""Source and configuration health checks."""

from __future__ import annotations

from ..config.loader import AppConfig
from ..persistence.repositories import CompanyRepo, JobRepo, LlmCallRepo, SourceRepo


def source_health(conn) -> list[dict]:
    return [dict(r) for r in SourceRepo(conn).health()]


def company_health(conn) -> dict:
    repo = CompanyRepo(conn)
    all_rows = repo.all()
    active = [r for r in all_rows if r["active"]]
    failing = repo.failing(threshold=3)
    by_ats: dict[str, int] = {}
    for r in active:
        by_ats[r["ats_type"]] = by_ats.get(r["ats_type"], 0) + 1
    return {
        "total": len(all_rows),
        "active": len(active),
        "failing": [dict(r) for r in failing],
        "by_ats": by_ats,
    }


def doctor(conn, config: AppConfig) -> list[tuple[str, str, str]]:
    """Return (level, check, detail). Level is ok | warn | fail."""
    out: list[tuple[str, str, str]] = []

    if not config.profile.summary_for_matching.strip():
        out.append(("fail", "profile.summary_for_matching", "empty - semantic ranking will be poor"))
    else:
        out.append(("ok", "profile.summary_for_matching", f"{len(config.profile.summary_for_matching)} chars"))

    n_strong = len(config.profile.skills.strong)
    out.append((("ok" if n_strong >= 5 else "warn"), "profile.skills.strong", f"{n_strong} skills"))

    if hasattr(config.profile, "years_experience"):
        out.append(("fail", "profile", "years_experience must not be present"))
    else:
        out.append(("ok", "profile.experience_stage", config.profile.experience_stage))

    ch = company_health(conn)
    level = "ok" if ch["active"] >= 150 else ("warn" if ch["active"] > 0 else "fail")
    out.append((level, "companies.active", f"{ch['active']} active ({ch['by_ats']})"))
    if ch["failing"]:
        out.append(("warn", "companies.failing",
                    f"{len(ch['failing'])} with >=3 consecutive failures"))

    jr = JobRepo(conn)
    out.append(("ok", "jobs.total", str(jr.count())))
    out.append(("ok", "jobs.duplicates", str(jr.duplicate_count())))

    llm = config.settings.llm
    out.append((("ok" if not llm.enabled else "warn"), "llm.enabled",
                f"{llm.enabled} (model={llm.model}, cap=INR {llm.daily_cap_inr}/day)"))
    out.append(("ok", "llm.spend_inr_total", f"{LlmCallRepo(conn).spend_inr():.2f}"))

    try:
        from ..embeddings.encoder import get_encoder
        out.append(("ok", "embeddings.encoder", get_encoder().model_name))
    except Exception as exc:
        out.append(("warn", "embeddings.encoder", f"fallback in use: {exc}"))

    return out
