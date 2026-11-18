"""Regression: keyword routing uses whole-word matching and searches both docs when both are named."""
import pytest

pytestmark = pytest.mark.unit

CASES = [
    ("Compare ITF and Wimbledon coaching", None),
    ("What does the ITF say about a let?", "ITF Rules"),
    ("Wimbledon dress code?", "Grand Slam Rules"),
    ("What is the U.S. Open tiebreak rule?", "Grand Slam Rules"),
    ("Is a focus open serve legal?", None),
    ("Is the benefit of the doubt given on line calls?", None),
    ("How many sets in a match?", None),
]


@pytest.mark.parametrize("question,expected", CASES)
def test_route_filter(graph_modules, question, expected):
    _, nodes, _ = graph_modules
    assert nodes._route_filter(question)[0] == expected
