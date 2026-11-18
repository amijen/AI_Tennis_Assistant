import importlib
import sys
import types

import pytest

_RELOADED = ("app.config", "app.graph.tools", "app.graph.nodes")


@pytest.fixture
def graph_modules(monkeypatch):
    """
    Import app.graph.{tools,nodes,prompts} without a DB, network or model.

    nodes.py builds a ChatGroq client at import and tools.py imports the
    retriever (which loads the embedding model), so the retriever is stubbed
    and dummy settings are provided. Returns (tools, nodes, prompts).
    """
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")

    fake_retriever = types.ModuleType("app.db.retriever")
    fake_retriever.search_similar_chunks = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "app.db.retriever", fake_retriever)

    for name in _RELOADED:
        monkeypatch.delitem(sys.modules, name, raising=False)

    tools = importlib.import_module("app.graph.tools")
    nodes = importlib.import_module("app.graph.nodes")
    prompts = importlib.import_module("app.graph.prompts")
    yield tools, nodes, prompts

    # Don't leak stub-bound modules into other tests.
    for name in _RELOADED:
        sys.modules.pop(name, None)
