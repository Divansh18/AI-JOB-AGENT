CREATE TABLE IF NOT EXISTS resume_wording_artifacts (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id               INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    resume_variant_id    INTEGER REFERENCES resume_variants(id) ON DELETE SET NULL,
    purpose              TEXT NOT NULL,
    prompt_version       TEXT NOT NULL,
    provider             TEXT NOT NULL,
    model                TEXT NOT NULL,
    context_fingerprint  TEXT NOT NULL,
    prompt_hash          TEXT NOT NULL,
    source_truth_hash    TEXT NOT NULL,
    request_json         TEXT NOT NULL DEFAULT '{}',
    suggestions_json     TEXT NOT NULL DEFAULT '[]',
    validation_status    TEXT NOT NULL CHECK (validation_status IN ('valid', 'invalid', 'blocked', 'skipped', 'failed')),
    validation_errors    TEXT NOT NULL DEFAULT '[]',
    errors_json          TEXT NOT NULL DEFAULT '[]',
    allowed_evidence_refs TEXT NOT NULL DEFAULT '[]',
    from_cache           INTEGER NOT NULL DEFAULT 0,
    cost_inr             REAL NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    UNIQUE (job_id, purpose, prompt_version, provider, model, context_fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_resume_wording_artifacts_job
    ON resume_wording_artifacts(job_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_resume_wording_artifacts_variant
    ON resume_wording_artifacts(resume_variant_id);
CREATE INDEX IF NOT EXISTS idx_resume_wording_artifacts_fingerprint
    ON resume_wording_artifacts(context_fingerprint);
