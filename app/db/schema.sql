CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing tables 
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

-- Documents table 
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Parent chunks: large context blocks (no embeddings needed)
CREATE TABLE parent_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    page INTEGER,
    metadata JSONB  -- Stores article/section/rule info
);

-- Child chunks: small searchable units (with embeddings)
CREATE TABLE child_chunks (
    id SERIAL PRIMARY KEY,
    parent_id INTEGER REFERENCES parent_chunks(id) ON DELETE CASCADE,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB
);

-- Index for fast vector similarity search
CREATE INDEX idx_child_embedding 
ON child_chunks USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Index for fast parent lookup
CREATE INDEX idx_child_parent ON child_chunks(parent_id);