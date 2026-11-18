"""Regression: route_node must clear per-question state left by the checkpointer."""
import pytest

pytestmark = pytest.mark.unit


def test_route_clears_stale_reformulated_query(graph_modules):
    _, nodes, _ = graph_modules
    stale_state = {
        "question": "How long is the warm-up?",
        "chat_history": [],  # empty -> no contextualize LLM call
        "reformulated_query": "service let net cord",
        "retrieval_grade": "irrelevant",
        "top_similarity": 0.41,
        "retrieved_context": "old excerpts",
        "retry_count": 2,
    }

    update = nodes.route_node(stale_state)

    assert update["reformulated_query"] == ""
    assert update["retrieval_grade"] == ""
    assert update["top_similarity"] == 0.0
    assert update["retrieved_context"] == ""
    assert update["retry_count"] == 0


def test_retrieve_uses_new_question_after_route(graph_modules, monkeypatch):
    _, nodes, _ = graph_modules
    seen = {}

    def fake_search(query, document_filter=None, top_k=4):
        seen["query"] = query
        return []

    monkeypatch.setattr(nodes, "search_rules", fake_search)

    state = {
        "question": "How long is the warm-up?",
        "chat_history": [],
        "reformulated_query": "service let net cord",
        "steps": [],
    }
    state.update(nodes.route_node(state))  # what LangGraph does between nodes
    nodes.retrieve_node(state)

    assert seen["query"] == "How long is the warm-up?"
