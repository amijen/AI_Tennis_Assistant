from typing import TypedDict

class AgentState(TypedDict):
    # Input 
    question: str

    # Routing 
    document_filter: str | None # "ITF Rules", "Grand Slam Rules", or None (both)

    # Retrieval 
    retrieved_context: str # Formatted chunks from the database
    retrieval_grade: str   # "relevant" or "irrelevant" (set by grade node)
    retry_count: int       # How many times we've re-retrieved (max 1)

    # Output 
    answer: str            # Final synthesized answer 

    # Debugging 
    steps: list[dict]      # Log of what happened at each step

def initial_state(question: str) -> AgentState:
    """
    Build a fresh state dict for a new question.
    Keeps API code from duplicating this everywhere.
    """
    return {
        "question": question,
        "document_filter": None,
        "reformulated_query": "",
        "retrieved_context": "",
        "retrieval_grade": "",
        "retry_count": 0,
        "top_similarity": 0.0,
        "answer": "",
        "steps": [],
    }