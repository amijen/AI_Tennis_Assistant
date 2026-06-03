"""
Database insertion functions for documents, parent chunks, and child chunks.

Parent-child structure:
- Documents → has many parent chunks
- Parent chunks → has many child chunks
- Only child chunks have embeddings (parents are returned for context)
"""

from sqlalchemy import text
import json


def insert_document(conn, name: str) -> int:
    """
    Insert a document record and return its auto-generated ID.
    
    Args:
        conn: SQLAlchemy connection (from engine.begin())
        name: Display name of the document (e.g., "ITF Rules")
    
    Returns:
        The new document's ID (integer).
    """
    result = conn.execute(
        text("INSERT INTO documents (name) VALUES (:name) RETURNING id"),
        {"name": name}
    )
    return result.fetchone()[0]


def insert_parent_chunk(
    conn, 
    document_id: int, 
    content: str, 
    page: int, 
    metadata: dict
) -> int:
    """
    Insert a parent chunk and return its ID.

    Parent chunks store the FULL rule/article text.
    They have no embedding — they're retrieved by their children.
    
    Args:
        conn: SQLAlchemy connection
        document_id: ID of the parent document
        content: Full text of the parent chunk
        page: Page number where the chunk starts
        metadata: Dict with header, doc_type, etc.
    
    Returns:
        The new parent chunk's ID.
    """
    result = conn.execute(
        text("""
            INSERT INTO parent_chunks (document_id, content, page, metadata)
            VALUES (:doc_id, :content, :page, :metadata)
            RETURNING id
        """),
        {
            "doc_id": document_id,
            "content": content,
            "page": page,
            "metadata": json.dumps(metadata)
        }
    )
    return result.fetchone()[0]


def insert_child_chunk(
    conn,
    parent_id: int,
    document_id: int,
    content: str,
    embedding: list,
    metadata: dict
):
    """
    Insert a child chunk with embedding.

    Child chunks are small text pieces with embeddings.
    They link back to their parent for context retrieval.
    
    Args:
        conn: SQLAlchemy connection
        parent_id: ID of the parent chunk
        document_id: ID of the document (for filtering)
        content: Text content of the child
        embedding: 768-dimensional vector (list of floats)
        metadata: Dict inherited from parent + child_index
    """
    conn.execute(
        text("""
            INSERT INTO child_chunks (parent_id, document_id, content, embedding, metadata)
            VALUES (:parent_id, :doc_id, :content, CAST(:embedding AS vector), :metadata)
        """),
        {
            "parent_id": parent_id,
            "doc_id": document_id,
            "content": content,
            "embedding": str(embedding),
            "metadata": json.dumps(metadata)
        }
    )