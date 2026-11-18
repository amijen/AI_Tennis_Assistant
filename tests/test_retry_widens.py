"""Regression 1.5: a retry must search both rulebooks instead of repeating the routed filter."""
import pytest

pytestmark = pytest.mark.unit


def _run_retrieve(nodes, monkeypatch, retry_count, doc_filter):
    seen = {}

    def fake_search(query, document_filter=None, top_k=4):
        seen["filter"], seen["top_k"] = document_filter, top_k
        return []

    monkeypatch.setattr(nodes, "search_rules", fake_search)
    state = {
        "question": "q",
        "standalone_question": "q",
        "document_filter": doc_filter,
        "retry_count": retry_count,
        "steps": [],
    }
    result = nodes.retrieve_node(state)
    return seen, result["steps"][-1]


def test_first_search_uses_routed_filter(graph_modules, monkeypatch):
    _, nodes, _ = graph_modules
    seen, step = _run_retrieve(nodes, monkeypatch, 0, "ITF Rules")
    assert seen == {"filter": "ITF Rules", "top_k": 4}
    assert step["widened"] is False


@pytest.mark.parametrize("retry", [1, 2])
def test_retry_drops_filter_keeps_top_k(graph_modules, monkeypatch, retry):
    _, nodes, _ = graph_modules
    seen, step = _run_retrieve(nodes, monkeypatch, retry, "ITF Rules")
    assert seen == {"filter": None, "top_k": 4}
    assert step["filter"] is None
    assert step["widened"] is True


def test_retry_when_already_searching_both_is_not_widened(graph_modules, monkeypatch):
    _, nodes, _ = graph_modules
    seen, step = _run_retrieve(nodes, monkeypatch, 1, None)
    assert seen == {"filter": None, "top_k": 4}
    assert step["widened"] is False
