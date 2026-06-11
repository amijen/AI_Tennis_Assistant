from app.db.retriever import search_similar_chunks

def compare_rules(topic: str) -> str:
    """
    Retrieve rule excerpts from both ITF and Grand Slam rulebooks for comparaison. 

    Args:
        topic: The topic to compare (e.g., "tie-break", "medical timeout")

    Returns: 
        Side-by-side excerpts from both rulebooks.
    """
    itf_results = search_similar_chunks(
        query = topic,
        top_k = 3,
        document_filter = "ITF Rules"
    )
    gs_results = search_similar_chunks(
        query = topic,
        top_k = 3,
        document_filter = "Grand Slam Rules"
    )
    output = [f"Comparison of '{topic}' between rulebooks:\n"]

    # ITF section
    output.append("=== ITF Rules of Tennis ===")
    if itf_results: 
        for r in itf_results:
            output.append(f"\n[Page {r['page']}, similarity: {r['similarity']:.2f}]\n{r['content']}")
    else: 
        output.append("- No relevant content found in ITF Rules.\n")
    
    # Grand Slam section
    output.append("\n## Grand Slam Rulebook:")
    if gs_results: 
        for r in gs_results:
            output.append(f"\n[Page {r['page']}, similarity: {r['similarity']:.2f}]\n{r['content']}")
    else: 
        output.append("- No relevant content found in Grand Slam Rulebook.\n")
    return "\n".join(output)