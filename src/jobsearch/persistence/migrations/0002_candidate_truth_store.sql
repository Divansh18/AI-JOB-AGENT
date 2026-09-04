CREATE TABLE IF NOT EXISTS candidate_facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    category    TEXT NOT NULL,
    fact_key    TEXT NOT NULL,
    value_json  TEXT NOT NULL,
    source      TEXT NOT NULL,
    verified    INTEGER NOT NULL DEFAULT 0,
    confidence  REAL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE (category, fact_key)
);
CREATE INDEX IF NOT EXISTS idx_candidate_facts_category ON candidate_facts(category, verified);

CREATE TABLE IF NOT EXISTS candidate_answers (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    question_key          TEXT NOT NULL,
    category              TEXT NOT NULL DEFAULT 'general',
    answer_text           TEXT NOT NULL,
    source                TEXT NOT NULL,
    evidence_refs         TEXT NOT NULL DEFAULT '[]',
    verified              INTEGER NOT NULL DEFAULT 0,
    human_review_required INTEGER NOT NULL DEFAULT 0,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    UNIQUE (question_key, category)
);
CREATE INDEX IF NOT EXISTS idx_candidate_answers_question ON candidate_answers(question_key, verified);
