"""Regression 1.4: each child chunk must carry the page its own text is on."""
import re

import pytest

from app.ingestion.splitter import process_document

pytestmark = pytest.mark.unit


def _sentences(tag: str, n: int, topic: str) -> str:
    return " ".join(f"[{tag}] Sentence {i} about the {topic}." for i in range(n))


def _fake_itf_pages() -> list[dict]:
    # ITF skips the first 4 pages, so content starts on page 5.
    # Rule 1 spans pages 5-7; rule 2 starts on page 7.
    return [
        {"page": 5, "text": "1. THE COURT\n" + _sentences("P5", 25, "court")},
        {"page": 6, "text": _sentences("P6", 25, "court")},
        {"page": 7, "text": _sentences("P7", 25, "court") + "\n2. THE NET\n" + _sentences("P7N", 10, "net")},
    ]


def test_child_pages_follow_their_own_text():
    parents = process_document(_fake_itf_pages(), "ITF")
    rule1 = next(p for p in parents if p.metadata["rule"].startswith("1."))

    assert rule1.page == 5  # parent still starts on page 5
    seen_pages = set()
    for child in rule1.children:
        first_tag = re.search(r"\[P(\d)\]", child.content)
        # A child holding only the header line has no tag; it is on page 5.
        expected = int(first_tag.group(1)) if first_tag else 5
        assert child.page == expected, child.content
        seen_pages.add(child.page)
    assert seen_pages == {5, 6, 7}


def test_parent_starting_mid_page_gets_that_page():
    parents = process_document(_fake_itf_pages(), "ITF")
    rule2 = next(p for p in parents if p.metadata["rule"].startswith("2."))

    assert rule2.page == 7
    assert all(c.page == 7 for c in rule2.children)
