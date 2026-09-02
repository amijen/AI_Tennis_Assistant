import logging 
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import AgentState
from app.graph.nodes import (
    route_node,
    retrieve_node,
    grade_node,
    reformulate_node,
    synthesize_node,
)

logger = logging.getLogger(__name__)


def _should_retry(state: AgentState) -> str: 
    """
    Conditional edge after grading.
    - If relevant → synthesize
    - If irrelevant → reformulate
    """
    if state["retrieval_grade"] == "relevant": 
        return "synthesize"

    if state.get("retry_count", 0) < 2 :
        return "reformulate"

    logger.warning("Retrieval still irrelevant after retry, synthesizing anyway")
    return "synthesize"

def build_graph() -> StateGraph: 
    """
    Build and compile the agent graph.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("synthesize", synthesize_node)

    # Add edges
    # START → route → retrieve → grade
    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "grade")

    # grade → conditional (synthesize OR reformulate)
    graph.add_conditional_edges(
        "grade",
        _should_retry,
        {
            "synthesize": "synthesize",
            "reformulate": "reformulate"
        }
    )

    # reformulate → retrieve (loop back for retry)
    graph.add_edge("reformulate", "retrieve")
    # synthesize → END
    graph.add_edge("synthesize", END)

    # Enable checkpointer for thread-level state retention
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    logger.info("Agent graph compiled with MemorySaver checkpointer")

    return compiled

agent_graph = build_graph()