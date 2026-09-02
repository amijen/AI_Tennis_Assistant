from app.db.retriever import search_similar_chunks 

def test_retrieval():
    test_queries = [
        ("What are the dimensions of the tennis court?", "ITF Rules"),
        ("What is the penalty for coaching during a match?", "Grand Slam Rules"),
        ("How does the final set tie-break work?", None),  # Searches both
    ]

    for query, doc_filter in test_queries: 
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Filter: {doc_filter or 'BOTH DOCUMENTS'}")
        print(f"{'='*60}")

        results = search_similar_chunks(query, top_k=2, document_filter=doc_filter)

        if not results:
            print("❌ No chunks found! Check similarity threshold or DB data.")
            continue

        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Source: {r['document']} | Page: {r['page']} | Score: {r['similarity']:.4f}")
            print(f"Excerpt:\n{r['content'][:300]}...")

if __name__ == "__main__":
    test_retrieval()