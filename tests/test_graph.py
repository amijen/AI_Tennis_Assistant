from app.graph.builder import agent_graph


def test_graph():
    questions = [
        "How does the tie-break work?",
        "What are the dimensions of the court according to ITF?",
        "What is the coaching rule at Wimbledon?",
        "Who won Roland Garros in 2024?",  # Should get a weak answer or no results
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")

        result = agent_graph.invoke({
            "question": q,
            "retrieved_context": "",
            "retrieval_grade": "",
            "retry_count": 0,
            "answer": "",
            "steps": [],
        })

        print(f"\nA: {result['answer'][:500]}")
        print(f"\nSteps: {result['steps']}")


if __name__ == "__main__":
    test_graph()