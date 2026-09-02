CREATE EXTENSION IF NOT EXISTS vector;

-- Clean slate
DROP TABLE IF EXISTS child_chunks CASCADE;
DROP TABLE IF EXISTS parent_chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

-- ── Documents ──────────────────────────────
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,     
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Parent Chunks (full rule context, NO embeddings) ──
CREATE TABLE parent_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    page INTEGER,
    metadata JSONB
);

-- ── Child Chunks (small searchable units WITH embeddings) ──
CREATE TABLE child_chunks (
    id SERIAL PRIMARY KEY,
    parent_id INTEGER NOT NULL REFERENCES parent_chunks(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB
);

-- ── Indexes ────────────────────────────────
CREATE INDEX idx_child_embedding
    ON child_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Fast parent lookup when joining child → parent
CREATE INDEX idx_child_parent
    ON child_chunks(parent_id);

-- Fast document filtering (WHERE document_id = X)
CREATE INDEX idx_child_document
    ON child_chunks(document_id);