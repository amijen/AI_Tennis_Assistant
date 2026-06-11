from app.db.retriever import search_similar_chunks 
from typing import List

def print_result(query: str, results: List[dict]):
    print(f"\n{'='*70}")
    print(f"🔎 Query: {query}")
    print(f"{'='*70}")
    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] 📄 {r['document']} (page {r['page']}) — similarity: {r['similarity']}")
        print(f"    {r['content'][:200]}...")


if __name__ == "__main__":
    queries = [
        "How many sets are played in a men's Grand Slam match?",
        "What is the tie-break rule?",
        "What happens if a player is injured during the match?",
        "Can a player receive coaching during a match?",
    ]

    for q in queries:
        results = search_similar_chunks(q, top_k=3)
        print_result(q, results)

    print("\n\n🎯 Testing with document filter (ITF Rules only):")
    results = search_similar_chunks(
        "How many sets?",
        top_k=3,
        document_filter="ITF Rules"
    )
    print_result("How many sets? [ITF only]", results)