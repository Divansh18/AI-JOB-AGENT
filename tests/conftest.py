from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jobsearch.config.schemas import (  # noqa: E402
    Filters, Profile, RankingConfig, SkillSet,
)
from jobsearch.domain.models import Job  # noqa: E402
from jobsearch.persistence.db import connect, migrate  # noqa: E402


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.db")
    migrate(c)
    yield c
    c.close()


@pytest.fixture
def profile() -> Profile:
    return Profile(
        experience_stage="early_career",
        target_roles=["Software Engineer", "Backend Engineer", "AI Engineer"],
        skills=SkillSet(
            strong=["python", "fastapi", "react.js", "postgresql", "docker", "typescript"],
            familiar=["aws", "redis", "elasticsearch", "llm orchestration"],
        ),
        preferred_locations=["Bengaluru", "Delhi NCR", "Pune"],
        summary_for_matching="Early-career full-stack and AI engineer building "
                             "React and FastAPI applications.",
    )


@pytest.fixture
def filters() -> Filters:
    return Filters()


@pytest.fixture
def ranking() -> RankingConfig:
    return RankingConfig()


def make_job(**kw) -> Job:
    """Build a Job with sensible defaults; override any field."""
    now = datetime.now(timezone.utc)
    defaults = dict(
        fingerprint="fp-" + kw.get("title", "x"),
        content_hash="ch",
        source="greenhouse",
        title="Software Engineer",
        title_normalized="software engineer",
        company_name_raw="Acme",
        company_normalized="acme",
        location_raw="Bengaluru, India",
        locations=["Bengaluru"],
        country="IN",
        remote_type="onsite",
        remote_scope="unknown",
        employment_type="full_time",
        description_text="We build products with Python, FastAPI and React. " * 12,
        apply_url="https://example.com/jobs/1",
        canonical_url="https://example.com/jobs/1",
        posted_at=now,
        first_seen_at=now,
        id=1,
    )
    defaults.update(kw)
    if "title" in kw and "title_normalized" not in kw:
        from jobsearch.domain.normalize import normalize_title
        defaults["title_normalized"] = normalize_title(kw["title"])
    return Job(**defaults)


@pytest.fixture
def job_factory():
    return make_job
