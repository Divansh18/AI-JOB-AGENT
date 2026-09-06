CREATE TABLE IF NOT EXISTS application_answer_drafts (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id          INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    job_id                  INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    question_key            TEXT NOT NULL,
    question_text           TEXT NOT NULL,
    question_classification TEXT NOT NULL,
    answer_text             TEXT NOT NULL DEFAULT '',
    evidence_refs           TEXT NOT NULL DEFAULT '[]',
    confidence              REAL NOT NULL DEFAULT 0,
    review_required         INTEGER NOT NULL DEFAULT 1,
    validation_status       TEXT NOT NULL CHECK (validation_status IN ('valid', 'invalid', 'blocked', 'skipped', 'failed')),
    validation_errors       TEXT NOT NULL DEFAULT '[]',
    provider                TEXT NOT NULL,
    model                   TEXT NOT NULL,
    prompt_version          TEXT NOT NULL,
    context_fingerprint     TEXT NOT NULL,
    prompt_hash             TEXT NOT NULL,
    source_truth_hash       TEXT NOT NULL,
    from_cache              INTEGER NOT NULL DEFAULT 0,
    cost_inr                REAL NOT NULL DEFAULT 0,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    UNIQUE (application_id, question_key, prompt_version, provider, model, context_fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_application_answer_drafts_app
    ON application_answer_drafts(application_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_application_answer_drafts_job
    ON application_answer_drafts(job_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_application_answer_drafts_fingerprint
    ON application_answer_drafts(context_fingerprint);
