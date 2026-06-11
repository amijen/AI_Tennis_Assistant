"""
Test retrieval quality with WELL-FORMED queries (not 'men').
This isolates: is the problem in retrieval or in query formation?
"""
from app.db.retriever import search_similar_chunks


def test(query: str, expected_keywords: list[str]):
    print(f"\n{'='*70}")
    print(f"🔎 Query: '{query}'")
    print(f"   Expected to find: {expected_keywords}")
    print(f"{'='*70}")
    
    results = search_similar_chunks(query, top_k=5)
    
    for i, r in enumerate(results, 1):
        content_lower = r['content'].lower()
        found_keywords = [kw for kw in expected_keywords if kw.lower() in content_lower]
        marker = "✅" if found_keywords else "❌"
        
        print(f"\n[{i}] {marker} {r['document']} p.{r['page']} (sim: {r['similarity']})")
        if found_keywords:
            print(f"    Found: {found_keywords}")
        print(f"    Preview: {r['content'][:200]}...")


# Test 1: Men's Grand Slam sets — should find "best of 5"
test(
    "How many sets in men's singles Grand Slam match best of five",
    ["best of five", "best of 5", "five sets", "5 sets"]
)

# Test 2: Tie-break — should find the actual tie-break rule
test(
    "tie-break game scoring rules first player to win seven points",
    ["tie-break", "seven points", "margin of two"]
)

# Test 3: Warm-up time — should find specific duration
test(
    "warm-up period duration minutes before match starts",
    ["warm-up", "minute", "before the match"]
)

# Test 4: Coaching — should find coaching rule
test(
    "coaching during match prohibited signals communication",
    ["coach", "coaching", "signal"]
)


if __name__ == "__main__":
    print("Running retrieval quality tests with WELL-FORMED queries...")