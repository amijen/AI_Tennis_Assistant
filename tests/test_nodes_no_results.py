"""Regression: no retrieval results must skip the LLM and use the fallback answer."""
import pytest

pytestmark = pytest.mark.unit


def test_format_context_empty_returns_sentinel(graph_modules):
    tools, _, _ = graph_modules
    assert tools.format_context([]) == tools.NO_RESULTS


def test_grade_skips_llm_when_no_results(graph_modules):
    tools, nodes, _ = graph_modules
    state = {"question": "q", "retrieved_context": tools.NO_RESULTS, "steps": []}

    result = nodes.grade_node(state)  # would hit the network if the LLM were called

    assert result["retrieval_grade"] == "irrelevant"


def test_synthesize_returns_fallback_when_no_results(graph_modules):
    tools, nodes, prompts = graph_modules
    state = {
        "question": "q",
        "retrieved_context": tools.NO_RESULTS,
        "chat_history": [],
        "steps": [],
    }

    result = nodes.synthesize_node(state)

    assert result["answer"] == prompts.NO_RESULTS_ANSWER
