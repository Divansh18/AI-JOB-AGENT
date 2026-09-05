CREATE TABLE IF NOT EXISTS application_autofill_runs (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id              INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    ats                         TEXT NOT NULL,
    url                         TEXT NOT NULL,
    status                      TEXT NOT NULL,
    result_json                 TEXT NOT NULL,
    fields_detected             TEXT NOT NULL DEFAULT '[]',
    fields_filled               TEXT NOT NULL DEFAULT '[]',
    unresolved_fields           TEXT NOT NULL DEFAULT '[]',
    sensitive_fields            TEXT NOT NULL DEFAULT '[]',
    resume_attached             INTEGER NOT NULL DEFAULT 0,
    human_intervention_required INTEGER NOT NULL DEFAULT 1,
    errors                      TEXT NOT NULL DEFAULT '[]',
    submitted                   INTEGER NOT NULL DEFAULT 0 CHECK (submitted = 0),
    created_at                  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_application_autofill_runs_app
    ON application_autofill_runs(application_id, created_at DESC);
