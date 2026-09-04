CREATE TABLE IF NOT EXISTS resume_variants (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id               INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    created_at           TEXT NOT NULL,
    fit_score            REAL NOT NULL,
    source_truth_version INTEGER NOT NULL,
    source_truth_hash    TEXT NOT NULL,
    master_resume_identity TEXT NOT NULL,
    master_resume_version  TEXT NOT NULL,
    template_identity      TEXT NOT NULL,
    page_limit             INTEGER NOT NULL DEFAULT 1,
    tailoring_decision     TEXT NOT NULL,
    tailoring_reasons      TEXT NOT NULL DEFAULT '[]',
    tailoring_evidence_refs TEXT NOT NULL DEFAULT '[]',
    content_json         TEXT NOT NULL,
    validation_status    TEXT NOT NULL,
    validation_errors    TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_resume_variants_job_created
    ON resume_variants(job_id, created_at DESC);
