-- memorykit schema — the vector-memory map index.
-- Markdown is the source of truth; this database is a disposable map.
-- Delete it any time and rebuild from markdown (see README: rebuilding).
-- One table for entries, one for derivation edges, one for feedback.
-- `status = 'rejected'` is the reversible quarantine (soft-delete).
-- Automation NEVER hard-deletes rows; eviction is a human decision.

CREATE TABLE IF NOT EXISTS memory_item (
    id               TEXT PRIMARY KEY NOT NULL,
    created          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    project          TEXT NOT NULL DEFAULT 'turns',
    session_key      TEXT,
    type             TEXT NOT NULL,              -- 'fact' | 'note' | 'rule'
    content          TEXT NOT NULL,              -- stored text, provenance header included
    salience         INTEGER NOT NULL DEFAULT 5, -- 1-10 importance
    confidence       REAL NOT NULL DEFAULT 1.0,  -- 0.0-1.0, see DESIGN.md
    metadata         TEXT,                       -- JSON: provenance, supersedes, keys, parents
    embedding        TEXT,                       -- JSON list of floats, or NULL
    status           TEXT NOT NULL DEFAULT 'confirmed', -- 'confirmed' | 'rejected'
    superseded_by    TEXT,                       -- id of the entry that replaced this one
    valid_from       TEXT NOT NULL,
    valid_to         TEXT,
    learned_at       TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_item_project ON memory_item (project);
CREATE INDEX IF NOT EXISTS idx_item_status ON memory_item (status);
CREATE INDEX IF NOT EXISTS idx_item_superseded ON memory_item (superseded_by);
CREATE INDEX IF NOT EXISTS idx_item_accessed ON memory_item (last_accessed_at);

-- Causal DAG: subject was derived from object (used by cascade pruning).
CREATE TABLE IF NOT EXISTS memory_edge (
    id         TEXT PRIMARY KEY NOT NULL,
    created    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    project    TEXT NOT NULL DEFAULT 'turns',
    subject_id TEXT NOT NULL,
    predicate  TEXT NOT NULL,              -- currently only 'derived_from'
    object_id  TEXT NOT NULL,
    source_id  TEXT
);

CREATE INDEX IF NOT EXISTS idx_edge_object ON memory_edge (object_id, predicate);

-- Explicit retrieval feedback: does this entry actually help?
CREATE TABLE IF NOT EXISTS memory_feedback (
    id          TEXT PRIMARY KEY NOT NULL,
    created     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    memory_id   TEXT NOT NULL REFERENCES memory_item(id) ON DELETE CASCADE,
    rating      TEXT NOT NULL,              -- '+1' | '-1'
    comment     TEXT,
    FOREIGN KEY (memory_id) REFERENCES memory_item(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_memory ON memory_feedback (memory_id);
