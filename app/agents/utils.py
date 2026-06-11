def truncate(
    results: list[dict],
    max_total_chars: int = 3000     
) -> list[dict]:
    """
    Truncate results to fit within token budget.
    Keeps the BEGINNING of each chunk (most relevant part).
    Distributes budget evenly across all results.
    
    chars → tokens rough ratio: 1 token ≈ 3.5 chars
    3000 chars ≈ 857 tokens → leaves plenty of room for prompt + answer
    """
    if not results:
        return results

    # Budget per chunk
    budget_per_chunk = max_total_chars // len(results)

    truncated = []
    for r in results:
        content = r["content"]
        if len(content) > budget_per_chunk:
            content = content[:budget_per_chunk]
            # Cut at last complete sentence to avoid mid-sentence truncation
            last_period = content.rfind(".")
            if last_period > budget_per_chunk * 0.7:   # only if not too short
                content = content[:last_period + 1]

        truncated.append({**r, "content": content})

    return truncated