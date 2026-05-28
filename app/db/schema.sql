CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    page INTEGER,
    type TEXT DEFAULT 'text',
    embedding vector(768)
);

-- Index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);