from app.db.retriever import search_similar_chunks 

def search_rules(
        query: str,
        document_filter: str | None = None,
        top_k: int = 4
) -> list[dict]:
    """
    Search the tennis rulebook for relevant excerpts.

    Args: 
        query: Natural language question.
        document_filter: "ITF Rules", "Grand Slam Rules", or None for both.
        top_k: Number of parent chunks to return.

    Returns: 
        List of result dicts with content, page, document, similarity.
    """

    return search_similar_chunks(
        query = query, 
        top_k = top_k,
        document_filter = document_filter,
        min_similarity = 0.35
    )

def format_context(results: list[dict]) -> str: 
    """
    Fromat retrieval results into a readable context block for the LLM.
    """

    if not results: 
        return "NO RESULTS"

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[Excerpt {i}]\n"
            f"Source: {r['document']}, page {r['page']}\n"
            f"Relevance: {r['similarity']:.2f}\n"
            f"{r['content']}\n"
        )

    return "\n---\n".join(lines)

def truncate_context(
        results: list[dict],
        max_total_chars: int = 4000,
) -> list[dict]:

    """
    Truncate results to fit within the LLM's context window.
    Cuts at sentence boundaries when possible.
    """

    if not results: 
        return results

    budget_per_chunk = max_total_chars // len(results)
    truncated = []

    for r in results: 
        content = r["content"]
        if len(content) > budget_per_chunk :
            content = content[:budget_per_chunk]
            # Try to cut at the last complete sentence
            last_period = content.rfind(".")
            if last_period > budget_per_chunk * 0.7 : 
                content = content[:last_period + 1]

        truncated.append({**r, "content": content})

    return truncated