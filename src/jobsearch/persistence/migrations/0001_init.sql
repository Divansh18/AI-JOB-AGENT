-- Phase 0 schema.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS companies (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT NOT NULL,
    name_normalized      TEXT NOT NULL,
    ats_type             TEXT NOT NULL,
    ats_token            TEXT NOT NULL,
    board_url            TEXT,
    careers_url          TEXT,
    website              TEXT,
    hq_country           TEXT,
    size_hint            TEXT,
    tags                 TEXT NOT NULL DEFAULT '[]',
    priority             TEXT NOT NULL DEFAULT 'normal',
    active               INTEGER NOT NULL DEFAULT 1,
    verified_at          TEXT,
    verification_note    TEXT,
    added_at             TEXT NOT NULL,
    added_via            TEXT NOT NULL DEFAULT 'manual',
    last_fetched_at      TEXT,
    last_ok_at           TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    UNIQUE (ats_type, ats_token)
);
CREATE INDEX IF NOT EXISTS idx_companies_active ON companies(active);
CREATE INDEX IF NOT EXISTS idx_companies_norm   ON companies(name_normalized);

CREATE TABLE IF NOT EXISTS sources (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL UNIQUE,
    kind                TEXT NOT NULL,
    enabled             INTEGER NOT NULL DEFAULT 1,
    config              TEXT NOT NULL DEFAULT '{}',
    last_run_at         TEXT,
    last_status         TEXT,
    last_error          TEXT,
    total_postings_seen INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS jobs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint        TEXT NOT NULL UNIQUE,
    content_hash       TEXT NOT NULL,
    source_id          INTEGER REFERENCES sources(id),
    source_name        TEXT NOT NULL,
    company_id         INTEGER REFERENCES companies(id),
    company_name_raw   TEXT NOT NULL,
    company_normalized TEXT NOT NULL,
    external_id        TEXT,
    title              TEXT NOT NULL,
    title_normalized   TEXT NOT NULL,
    location_raw       TEXT NOT NULL DEFAULT '',
    locations          TEXT NOT NULL DEFAULT '[]',
    country            TEXT,
    remote_type        TEXT NOT NULL DEFAULT 'unknown',
    remote_scope       TEXT NOT NULL DEFAULT 'unknown',
    employment_type    TEXT NOT NULL DEFAULT 'unknown',
    description_text   TEXT NOT NULL DEFAULT '',
    description_chars  INTEGER NOT NULL DEFAULT 0,
    apply_url          TEXT NOT NULL DEFAULT '',
    canonical_url      TEXT,
    posted_at          TEXT,
    first_seen_at      TEXT NOT NULL,
    last_seen_at       TEXT NOT NULL,
    is_active          INTEGER NOT NULL DEFAULT 1,
    status             TEXT NOT NULL DEFAULT 'new',
    raw                TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_jobs_status     ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_first_seen ON jobs(first_seen_at);
CREATE INDEX IF NOT EXISTS idx_jobs_company    ON jobs(company_normalized);
CREATE INDEX IF NOT EXISTS idx_jobs_canonical  ON jobs(canonical_url);

CREATE TABLE IF NOT EXISTS job_duplicates (
    duplicate_job_id INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    canonical_job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    method           TEXT NOT NULL,
    similarity       REAL NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dupes_canonical ON job_duplicates(canonical_job_id);

CREATE TABLE IF NOT EXISTS filter_results (
    job_id          INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    passed          INTEGER NOT NULL,
    rules_failed    TEXT NOT NULL DEFAULT '[]',
    signals         TEXT NOT NULL DEFAULT '{}',
    notes           TEXT NOT NULL DEFAULT '[]',
    filters_version INTEGER NOT NULL,
    evaluated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_filter_passed ON filter_results(passed);

CREATE TABLE IF NOT EXISTS job_scores (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage         TEXT NOT NULL,
    score         REAL NOT NULL,
    components    TEXT NOT NULL DEFAULT '{}',
    rank_version  INTEGER NOT NULL DEFAULT 1,
    model         TEXT,
    rationale     TEXT,
    matched       TEXT,
    missing       TEXT,
    flags         TEXT NOT NULL DEFAULT '[]',
    explanation   TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL,
    UNIQUE (job_id, stage, rank_version)
);
CREATE INDEX IF NOT EXISTS idx_scores_score ON job_scores(stage, score DESC);

CREATE TABLE IF NOT EXISTS job_embeddings (
    job_id     INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS applications (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id         INTEGER NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    status         TEXT NOT NULL,
    applied_at     TEXT,
    resume_variant TEXT,
    channel        TEXT,
    notes          TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_apps_status ON applications(status);

CREATE TABLE IF NOT EXISTS application_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    from_status    TEXT,
    to_status      TEXT NOT NULL,
    occurred_at    TEXT NOT NULL,
    note           TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    entity_type TEXT,
    entity_id   TEXT,
    payload     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT NOT NULL,
    purpose             TEXT NOT NULL,
    model               TEXT NOT NULL,
    prompt_hash         TEXT NOT NULL,
    job_id              INTEGER,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cost_usd            REAL NOT NULL DEFAULT 0,
    cost_inr            REAL NOT NULL DEFAULT 0,
    from_cache          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS digests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_date TEXT NOT NULL UNIQUE,
    path        TEXT NOT NULL,
    counts      TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name  TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL,
    fetched      INTEGER NOT NULL DEFAULT 0,
    new_jobs     INTEGER NOT NULL DEFAULT 0,
    errors       INTEGER NOT NULL DEFAULT 0,
    error_detail TEXT
);
