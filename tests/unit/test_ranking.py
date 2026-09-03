from datetime import datetime, timedelta, timezone

from jobsearch.domain.ranking import (
    match_skills, percentile_normalize, score_job,
)
from tests.conftest import make_job

JD = ("We are hiring an engineer to build REST APIs with Python and FastAPI, "
      "a React frontend, and PostgreSQL. Docker experience helps. "
      "Exposure to AWS and Redis is a bonus. ") * 4


def _score(job, profile, ranking, **kw):
    return score_job(job, profile=profile, config=ranking,
                     semantic_norm=kw.pop("semantic_norm", 0.5),
                     signals=kw.pop("signals", {}), **kw)


def test_score_is_bounded_and_deterministic(profile, ranking):
    job = make_job(description_text=JD)
    a = _score(job, profile, ranking)
    b = _score(job, profile, ranking)
    assert a.score == b.score
    assert 0.0 <= a.score <= 100.0


def test_strong_and_familiar_skills_are_reported_separately(profile):
    strong, familiar = match_skills(JD, profile)
    assert "python" in strong and "fastapi" in strong and "postgresql" in strong
    assert "aws" in familiar and "redis" in familiar
    assert not set(strong) & set(familiar), "a skill must not appear in both tiers"


def test_familiar_skills_count_less_than_strong(profile, ranking):
    """A developing skill must never be weighted as production experience."""
    strong_jd = "We use Python, FastAPI, PostgreSQL, Docker, React.js and TypeScript. " * 6
    familiar_jd = "We use AWS, Redis, Elasticsearch and LLM orchestration. " * 6
    s = _score(make_job(description_text=strong_jd), profile, ranking)
    f = _score(make_job(description_text=familiar_jd), profile, ranking)
    assert s.components.skills > f.components.skills


def test_freshness_decays(profile, ranking):
    now = datetime.now(timezone.utc)
    fresh = make_job(description_text=JD, first_seen_at=now, posted_at=now)
    old_dt = now - timedelta(days=10)
    old = make_job(description_text=JD, first_seen_at=old_dt, posted_at=old_dt)
    assert (_score(fresh, profile, ranking, now=now).components.freshness
            > _score(old, profile, ranking, now=now).components.freshness)


def test_preferred_city_beats_unclear_location(profile, ranking):
    pref = make_job(description_text=JD, locations=["Bengaluru"], country="IN")
    vague = make_job(description_text=JD, locations=[], country=None,
                     remote_type="unknown")
    assert (_score(pref, profile, ranking).components.location
            > _score(vague, profile, ranking).components.location)


def test_remote_india_beats_global_remote(profile, ranking):
    india = _score(
        make_job(description_text=JD, location_raw="Remote, India", locations=[], country="IN", remote_type="remote"),
        profile,
        ranking,
        signals={"location": {"bucket": "india_remote", "status": "eligible", "reason": "remote within India"}},
    )
    global_remote = _score(
        make_job(description_text=JD, location_raw="Remote", locations=[], country=None, remote_type="remote"),
        profile,
        ranking,
        signals={"location": {"bucket": "remote_global", "status": "eligible", "reason": "explicitly global remote"}},
    )
    assert india.components.location > global_remote.components.location


def test_unknown_location_is_flagged_and_scored_lower(profile, ranking):
    unknown = _score(
        make_job(description_text=JD, location_raw="Remote", locations=[], country=None, remote_type="remote"),
        profile,
        ranking,
        signals={"location": {"bucket": "unknown_remote", "status": "unknown", "reason": "remote eligibility unclear"}},
    )
    india = _score(
        make_job(description_text=JD, location_raw="Bengaluru, India", locations=["Bengaluru"], country="IN"),
        profile,
        ranking,
        signals={"location": {"bucket": "india_city", "status": "eligible", "reason": "India location stated", "cities": ["Bengaluru"]}},
    )
    assert unknown.components.location < india.components.location
    assert "location eligibility unknown" in unknown.flags


def test_yoe_adjustments_follow_policy(profile, ranking):
    job = make_job(description_text=JD)
    scores = {}
    for verdict in ("strong", "ok", "unknown", "penalty", "soft_high"):
        r = _score(job, profile, ranking, signals={"yoe": {"verdict": verdict}})
        scores[verdict] = r.score
    assert scores["strong"] > scores["ok"] >= scores["unknown"]
    assert scores["unknown"] > scores["penalty"] > scores["soft_high"]


def test_internship_is_penalised_not_removed(profile, ranking):
    ft = _score(make_job(description_text=JD), profile, ranking)
    intern = _score(make_job(description_text=JD, employment_type="internship"),
                    profile, ranking)
    assert intern.score < ft.score
    assert intern.score > 0
    assert "internship" in intern.flags


def test_percentile_normalize_spreads_a_narrow_band():
    """bge cosines cluster in ~0.60-0.85; ranking must still discriminate."""
    raw = [0.61, 0.62, 0.63, 0.64, 0.65]
    out = percentile_normalize(raw)
    assert min(out) == 0.0 and max(out) == 1.0
    assert out == sorted(out)


def test_percentile_normalize_edge_cases():
    assert percentile_normalize([]) == []
    assert percentile_normalize([0.7]) == [0.5]


def test_components_sum_to_total(profile, ranking):
    r = _score(make_job(description_text=JD), profile, ranking)
    parts = r.components.as_dict()
    assert abs(sum(parts.values()) - r.components.total()) < 0.05
