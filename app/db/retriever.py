"""
Retrieves relevant chunks from the database using vector similarity search.

Strategy: Parent-Child Retrieval
- Searches CHILD chunks (small, precise embedding matches)
- Returns PARENT chunks (full rule/article context)
- Deduplicates: if multiple children of the same parent match, parent is returned once
"""
from sqlalchemy import text
from app.db.database import engine
from app.ingestion.embedder import embed_text


def search_similar_chunks(
    query: str,
    top_k: int = 5,
    document_filter: str | None = None,
    min_similarity : float = 0.3
) -> list[dict]:
    """
    Search for relevant content using semantic similarity.

    Args:
        query: User's search query (natural language).
        top_k: Number of unique parent chunks to return.
        document_filter: Optional document name (e.g., "ITF Rules").

    Returns:
        List of dicts with parent content, page, document, similarity score.
    """
    # Embed the query (using same model as ingestion)
    query_embedding = embed_text(query)

    # Build SQL
    # - Inner query: ranks children by similarity, keeps best per parent
    # - Outer query: orders parents by their best child's similarity
    base_sql = """
        WITH ranked_children AS (
            SELECT 
                c.parent_id,
                c.embedding <=> CAST(:query_embedding AS vector) AS distance,
                ROW_NUMBER() OVER (
                    PARTITION BY c.parent_id 
                    ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
                ) AS rn
            FROM child_chunks c
            JOIN documents d ON c.document_id = d.id
            {document_filter_clause}
        )
        SELECT
            p.id AS parent_id,
            p.content AS parent_content,
            p.metadata AS parent_metadata,
            p.page,
            d.name AS document_name,
            1 - rc.distance AS similarity
        FROM ranked_children rc
        JOIN parent_chunks p ON p.id = rc.parent_id
        JOIN documents d ON p.document_id = d.id
        WHERE rc.rn = 1
            AND (1 - rc.distance) >= :min_similarity
        ORDER BY rc.distance
        LIMIT :top_k
    """

    # Inject the document filter clause if needed
    document_filter_clause = ""
    params = {
        "query_embedding": str(query_embedding),
        "top_k": top_k,
        "min_similarity": min_similarity,      
    }
    if document_filter:
        document_filter_clause = "WHERE d.name = :document_filter"
        params["document_filter"] = document_filter

    final_sql = base_sql.format(document_filter_clause=document_filter_clause)

    # Execute
    with engine.connect() as conn:
        result = conn.execute(text(final_sql), params)
        rows = result.fetchall()

    #  Format results
    return [
        {
            "content": row.parent_content,
            "page": row.page,
            "document": row.document_name,
            "similarity": round(float(row.similarity), 4),
            "metadata": row.parent_metadata,
        }
        for row in rows
    ]