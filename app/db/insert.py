from sqlalchemy import text

def insert_document(conn, name):
    result = conn.execute(
        text("INSERT INTO documents (name) VALUES (:name) RETURNING id"),
        {"name": name}
    )
    return result.fetchone()[0]

def insert_chunk(conn, document_id, content, page, type_, embedding):
    conn.execute(
        text("""
             INSERT INTO chuncks (document_id, content, page, type, embedding)
             VALUES (:doc_id, :content, :page, :type, :embedding)
        """),
        {
            "doc_id": document_id,
            "content": content,
            "page": page,
            "type": type_,
            "embedding": str(embedding)
        }
    )

    