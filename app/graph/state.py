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