CREATE EXTENSION vector;

CREATE TABLE documents (
	id SERIAL PRIMARY KEY,
	name TEXT
); 

CREATE TABLE chunks (
	id SERIAL PRIMARY KEY,
	document_id INTEGER,
	content TEXT,
	page INTEGER,
	type TEXT,
	embedding vector (786)
);