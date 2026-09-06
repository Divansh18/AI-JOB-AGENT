ALTER TABLE application_answer_drafts
    ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending_review'
    CHECK (review_status IN ('pending_review', 'approved', 'rejected'));

ALTER TABLE application_answer_drafts
    ADD COLUMN reviewer_source TEXT;

ALTER TABLE application_answer_drafts
    ADD COLUMN reviewed_at TEXT;

ALTER TABLE application_answer_drafts
    ADD COLUMN review_note TEXT;

CREATE INDEX IF NOT EXISTS idx_application_answer_drafts_review
    ON application_answer_drafts(application_id, review_status, updated_at DESC);

ALTER TABLE resume_wording_artifacts
    ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending_review'
    CHECK (review_status IN ('pending_review', 'approved', 'rejected'));

ALTER TABLE resume_wording_artifacts
    ADD COLUMN reviewer_source TEXT;

ALTER TABLE resume_wording_artifacts
    ADD COLUMN reviewed_at TEXT;

ALTER TABLE resume_wording_artifacts
    ADD COLUMN review_note TEXT;

ALTER TABLE resume_wording_artifacts
    ADD COLUMN promoted_application_id INTEGER REFERENCES applications(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_resume_wording_artifacts_review
    ON resume_wording_artifacts(job_id, review_status, updated_at DESC);
