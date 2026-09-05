CREATE TABLE IF NOT EXISTS application_plans (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id    INTEGER NOT NULL UNIQUE REFERENCES applications(id) ON DELETE CASCADE,
    job_id            INTEGER NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    state             TEXT NOT NULL CHECK (state IN ('planned', 'ready_for_review', 'submitted', 'blocked', 'failed')),
    plan_json         TEXT NOT NULL,
    resume_variant_id INTEGER REFERENCES resume_variants(id) ON DELETE SET NULL,
    resume_path       TEXT,
    review_required   INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_application_plans_state
    ON application_plans(state, updated_at DESC);
