import logging 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.graph.state import AgentState
from app.graph.tools import search_rules, format_context, truncate_context
from app.graph.prompts import (
    GRADE_PROMPT,
    REFORMULATE_PROMPT,
    SYNTHESIS_PROMPT,
    NO_RESULTS_ANSWER,
)

logger = logging.getLogger(__name__)

llm = ChatGroq(
    model = settings.groq_model,
    temperature = 0,
    api_key = settings.groq_api_key,
    max_tokens = 1024,
    streaming=True, 
)

# ============================================================
# NODE 1: ROUTE (deterministic, no LLM)
# ============================================================

# Keywords that indicate Grand Slam specific questions
GS_KEYWORDS = [
    "grand slam", "wimbledon", "us open", "u.s. open",
    "french open", "australian open", "roland garros",
    "roland-garros", "flinders park", "all england",
]

# Keywords that indicate ITF specific questions
ITF_KEYWORDS = [
    "itf", "international tennis federation",
    "rules of tennis", "itf rule",
]

def route_node(state: AgentState) -> dict:
    """
    Determine which document(s) to search based on keywords.
    """
    question_lower = state["question"].lower()

    # Check for explicit ITF mention
    if any(kw in question_lower for kw in ITF_KEYWORDS):
        doc_filter = "ITF Rules"
        reason = "ITF keyword detected"

    # Check for explicit Grand Slam mention
    elif any(kw in question_lower for kw in GS_KEYWORDS):
        doc_filter = "Grand Slam Rules"
        reason = "Grand Slam keyword detected"

    # Default: search both
    else:
        doc_filter = None
        reason = "No specific source mentioned, searching both"

    logger.info(f"Route: {reason} → filter={doc_filter}")

    return {
        "document_filter": doc_filter,
        "retry_count": 0,
        "steps": [{"action": "route", "filter": doc_filter, "reason": reason}],
    }

# ============================================================
# NODE 2: RETRIEVE
# ============================================================
def retrieve_node(state: AgentState) -> dict: 
    """
    Search the database for relevant rule excerpts.
    Uses the reformulated query if available (on retry), otherwise the original.
    """
    query = state.get("reformulated_query") or state["question"]
    doc_filter = state["document_filter"]

    logger.info(f"Retrieving: query='{query}', filter={doc_filter}")

    results = search_rules(
        query = query,
        document_filter = doc_filter,
        top_k = 4
    )

    results = truncate_context(results, max_total_chars = 4000)
    context = format_context(results)

    step = {
        "action": "retrieve",
        "query": query,
        "filter": doc_filter,
        "num_results": len(results),
    }

    return {
        "retrieved_context": context,
        "steps": state.get("steps", []) + [step],
    }

# ============================================================
# NODE 3: GRADE (LLM decides if retrieval is relevant)
# ============================================================

def grade_node(state: AgentState) -> dict:
    """
    Ask the LLM whether the retrieved context is relevant to the question.
    """

    context = state["retrieved_context"]

    # Skip grading if no results
    if context == "NO_RESULTS":
        return {
            "retrieval_grade": "irrelevant",
            "steps": state.get("steps", []) + [{"action": "grade", "result": "no_results"}],
        }

    prompt = ChatPromptTemplate.from_template(GRADE_PROMPT)
    chain = prompt | llm | StrOutputParser()

    grade = chain.invoke({
        "question" : state["question"],
        "context": context,
    }).strip().lower()

    # Normalize the response
    if "relevant" in grade and "irrelevant" not in grade:
        grade = "relevant"
    else:
        grade = "irrelevant"

    logger.info(f"Grade: {grade}")

    return {
        "retrieval_grade": grade,
        "steps": state.get("steps", []) + [{"action": "grade", "result": grade}],
    }

# ============================================================
# NODE 4: REFORMULATE (only called if grade is "irrelevant")
# ============================================================
def reformulate_node(state: AgentState) -> dict: 
    """
    Ask the LLM to rephrase the question for better retrieval.
    """

    prompt = ChatPromptTemplate.from_template(REFORMULATE_PROMPT)
    chain = prompt | llm | StrOutputParser()

    new_query = chain.invoke({
        "question": state["question"],
    }).strip()

    logger.info(f"Reformulated: '{state['question']}' → '{new_query}'")

    return {
        "reformulated_query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "steps": state.get("steps", []) + [
            {"action": "reformulate", "new_query": new_query}
        ],
    }

# ============================================================
# NODE 5: SYNTHESIZE (LLM generates the final answer)
# ============================================================

def synthesize_node(state: AgentState) -> dict:
    """
    Generate the final answer from the retrieved context.
    """
    context = state["retrieved_context"]

    if context == "NO_RESULTS":
        return {
            "answer": NO_RESULTS_ANSWER,
            "steps": state.get("steps", []) + [{"action": "synthesize", "mode": "no_results"}],
        }

    prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "question": state["question"],
        "context": context,
    })

    return {
        "answer": answer,
        "steps": state.get("steps", []) + [{"action": "synthesize", "mode": "full"}],
    }