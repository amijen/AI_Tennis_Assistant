import logging 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.graph.state import AgentState
from app.graph.tools import search_rules, format_context, truncate_context
from app.graph.prompts import (
    CONTEXTUALIZE_PROMPT,
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

def _format_history(history: list[dict], max_turns: int = 4) -> str:
    """
    Format the last N conversation into a string.
    """
    if not history:
        return "No previous conversation."

    recent = history[-max_turns*2:]
    lines = []
    for turn in recent:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")

    return "\n".join(lines)

# ============================================================
# NODE 1: ROUTE & CONTEXTUALIZE
# ============================================================

def route_node(state: AgentState) -> dict:
    """
    1. If chat history exists, rewrite question into a standalone query.
    2. Route to ITF / Grand Slam / Both.
    """
    question = state["question"]
    history = state.get("chat_history", [])

    # 1. Standalone question resolution
    standalone = question
    if history: 
        prompt = ChatPromptTemplate.from_template(CONTEXTUALIZE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        standalone = chain.invoke({
            "chat_history" : _format_history(history),
            "question" : question,
        }).strip()
        logger.info(f"Contextualized: '{question}' → '{standalone}'")

    # 2. Keyword routing on the standalone question

    q_lower = standalone.lower()
    if any(kw in q_lower for kw in ITF_KEYWORDS):
        doc_filter = "ITF Rules"
        reason = "ITF keyword detected"
    elif any(kw in q_lower for kw in GS_KEYWORDS):
        doc_filter = "Grand Slam Rules"
        reason = "Grand Slam keyword detected"
    else:
        doc_filter = None
        reason = "Searching both rulebooks"

    return {
        "standalone_question": standalone,
        "document_filter": doc_filter,
        "retry_count": 0,
        "steps": [{"action": "route", "filter": doc_filter, "standalone": standalone, "reason": reason}],
    }

# ============================================================
# NODE 2: RETRIEVE
# ============================================================
def retrieve_node(state: AgentState) -> dict: 
    """
    Search the database for relevant rule excerpts.
    Uses the reformulated query if available (on retry), otherwise the original.
    """
    query = state.get("reformulated_query") or state.get("standalone_question") or state["question"]
    doc_filter = state["document_filter"]

    logger.info(f"Retrieving: query='{query}', filter={doc_filter}")

    results = search_rules(
        query = query,
        document_filter = doc_filter,
        top_k = 4
    )

    results = truncate_context(results, max_total_chars = 4000)
    context = format_context(results)

    top_score = results[0]["similarity"] if results else 0.0

    step = {
        "action": "retrieve",
        "query": query,
        "filter": doc_filter,
        "num_results": len(results),
        "top_score": top_score,
    }

    return {
        "retrieved_context": context,
        "top_similarity" : top_score,
        "steps": state.get("steps", []) + [step],
    }

# ============================================================
# NODE 3: GRADE (LLM decides if retrieval is relevant)
# ============================================================


def grade_node(state: AgentState) -> dict:
    """
    Ask the LLM whether the retrieved context is relevant to the question.
    """

    context = state.get("retrieved_context", "NO_RESULTS")
    top_score = state.get("top_similarity", 0.0)
    steps = state.get("steps", [])

    # Skip grading if no results
    if context == "NO_RESULTS":
        return {
            "retrieval_grade": "irrelevant",
            "steps": steps + [{"action": "grade", "result": "no_results", "cost": "free"}],
        }

    prompt = ChatPromptTemplate.from_template(GRADE_PROMPT)
    chain = prompt | llm | StrOutputParser()

    grade = chain.invoke({
        "question" : state.get("standalone_question") or state["question"],
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
        "steps": steps + [{"action": "grade", "result": grade, "cost": "llm"}],
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
        "question": state.get("standalone_question") or state["question"],
    }).strip()

    logger.info(f"Reformulated: '{state['question']}' → '{new_query}'")

    return {
        "reformulated_query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "steps": state.get("steps", []) + [{"action": "reformulate", "new_query": new_query}],
    }

# ============================================================
# NODE 5: SYNTHESIZE (LLM generates the final answer)
# ============================================================

def synthesize_node(state: AgentState) -> dict:
    """
    Generate the final answer from the retrieved context.
    """
    context = state.get("retrieved_context", "NO_RESULTS")
    history = state.get("chat_history", [])

    if context == "NO_RESULTS":
        answer = NO_RESULTS_ANSWER
    else:
        prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "chat_history": _format_history(history),
            "question": state["question"],
            "context": context,
        })

    # Append current turn to chat history (keeps last 10 messages)
    updated_history = list(history) + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": answer},
    ]
    updated_history = updated_history[-10:]

    return {
        "answer": answer,
        "chat_history": updated_history,
        "steps": state.get("steps", []) + [{"action": "synthesize"}],
    }