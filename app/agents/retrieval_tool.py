from app.db.retriever import search_similar_chunks

def search_rules(query: str, document: str="") -> str:
    """
    Search the tennis rulebooks for relevant excerpts. 

    Args:
        query: The search query (e.g., "tie-break rules")
        document: Optional filter - "ITF Rules" or "Grand Slam Rules" 
                  (leave empty to search both)
    
    Returns:
        A formatted string with the top retrieved chunks.
    """
    # Normalize document filter
    doc_filter = document.strip() if document.strip() else None

    # Validate document filter 
    if doc_filter and doc_filter not in ["ITF Rules", "Grand Slam Rules"]:
        if 'itf' in doc_filter.lower():
            doc_filter = "ITF Rules"
        elif "grand" in doc_filter.lower():
            doc_filter = "Grand Slam Rules"
        else:
            doc_filter = None # Invalid filter, search both

    results = search_similar_chunks(
        query = query,
        top_k = 5,
        document_filter = doc_filter
    )

    if not results:
        return "No relevant rules were found for this query. Try rephrasing or using different keywords."
    
    # Format results 
    formatted = []
    for i, r in enumerate(results, start=1):
        formatted.append(
            f"[Excerpt {i}] "
            f"(Source: {r['document']}, page {r['page']}, "
            f"similarity: {r['similarity']:.2f})\n"
            f"{r['content']}"
        )
    return "\n\n".join(formatted)