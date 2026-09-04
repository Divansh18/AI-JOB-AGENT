ALTER TABLE resume_variants ADD COLUMN output_pdf_path TEXT;
ALTER TABLE resume_variants ADD COLUMN page_count INTEGER;
ALTER TABLE resume_variants ADD COLUMN page_validation_status TEXT NOT NULL DEFAULT 'not_rendered';
ALTER TABLE resume_variants ADD COLUMN output_evidence_refs TEXT NOT NULL DEFAULT '[]';
ALTER TABLE resume_variants ADD COLUMN render_fingerprint TEXT;

CREATE INDEX IF NOT EXISTS idx_resume_variants_render_fingerprint
    ON resume_variants(render_fingerprint);
