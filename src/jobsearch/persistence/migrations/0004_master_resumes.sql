CREATE TABLE IF NOT EXISTS master_resumes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    identity         TEXT NOT NULL,
    version          INTEGER NOT NULL,
    file_path        TEXT NOT NULL,
    active           INTEGER NOT NULL DEFAULT 0,
    page_limit       INTEGER NOT NULL DEFAULT 1,
    file_hash        TEXT NOT NULL,
    extracted_text   TEXT,
    sections_json    TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_ingested_at TEXT,
    UNIQUE (identity, version)
);
CREATE INDEX IF NOT EXISTS idx_master_resumes_identity_version
    ON master_resumes(identity, version);
CREATE UNIQUE INDEX IF NOT EXISTS idx_master_resumes_active
    ON master_resumes(active) WHERE active = 1;
