CREATE TABLE IF NOT EXISTS llm_artifacts (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id                INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    purpose               TEXT NOT NULL,
    prompt_version        TEXT NOT NULL,
    provider              TEXT NOT NULL,
    model                 TEXT NOT NULL,
    context_fingerprint   TEXT NOT NULL,
    prompt_hash           TEXT NOT NULL,
    source_truth_hash     TEXT NOT NULL,
    status                TEXT NOT NULL CHECK (status IN ('valid', 'invalid', 'blocked', 'failed', 'skipped')),
    input_json            TEXT NOT NULL,
    output_json           TEXT NOT NULL DEFAULT '{}',
    validation_errors     TEXT NOT NULL DEFAULT '[]',
    evidence_refs         TEXT NOT NULL DEFAULT '[]',
    from_cache            INTEGER NOT NULL DEFAULT 0,
    cost_inr              REAL NOT NULL DEFAULT 0,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    UNIQUE (job_id, purpose, prompt_version, provider, model, context_fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_llm_artifacts_job
    ON llm_artifacts(job_id, purpose, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_artifacts_fingerprint
    ON llm_artifacts(purpose, context_fingerprint);
