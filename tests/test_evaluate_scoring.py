"""Offline checks for the evaluation harness: scoring helpers and test-set integrity."""
import pytest

from scripts.evaluate import (
    DOCUMENTS,
    evidence_found,
    facts_present,
    first_hit_rank,
    has_citation,
    header_of,
    is_refusal,
    load_test_set,
    matches_evidence,
    route_conflict,
)

pytestmark = pytest.mark.unit

ITF_COACHING = {"document": "ITF Rules", "metadata": {"rule": "30. COACHING"}}
GS_COACHING = {"document": "Grand Slam Rules", "metadata": {"article": "ARTICLE III", "section": "L. COACHING AND COACHES"}}
ITF_COURT = {"document": "ITF Rules", "metadata": {"rule": "1. THE COURT"}}
EV_ITF = {"document": "ITF Rules", "header": "30. COACHING"}
EV_GS = {"document": "Grand Slam Rules", "header": "L. COACHING AND COACHES"}


def test_header_of_handles_itf_and_grand_slam_and_missing():
    assert header_of({"rule": "1. THE COURT"}).startswith("1. THE COURT")
    assert "L. COACHING" in header_of(GS_COACHING["metadata"])
    assert header_of(None) == " / "


def test_matches_requires_same_document_and_header():
    assert matches_evidence(ITF_COACHING, EV_ITF)
    assert not matches_evidence(ITF_COURT, EV_ITF)
    # same header text but wrong rulebook must not match
    assert not matches_evidence({**ITF_COACHING, "document": "Grand Slam Rules"}, EV_ITF)


def test_first_hit_rank_is_one_based_and_none_on_miss():
    assert first_hit_rank([ITF_COURT, ITF_COACHING], [EV_ITF]) == 2
    assert first_hit_rank([ITF_COURT], [EV_ITF]) is None


def test_evidence_found_needs_every_entry():
    assert evidence_found([ITF_COACHING, GS_COACHING], [EV_ITF, EV_GS])
    assert not evidence_found([ITF_COACHING], [EV_ITF, EV_GS])
    assert not evidence_found([ITF_COACHING], [])  # no evidence is not a hit


def test_route_conflict():
    assert route_conflict("ITF Rules", [EV_GS])          # ITF filter hides the GS answer
    assert not route_conflict(None, [EV_GS])             # searching both is always safe
    assert not route_conflict("ITF Rules", [EV_ITF])
    assert route_conflict("ITF Rules", [EV_ITF, EV_GS])  # a both-docs question needs None


def test_facts_present_ignores_case_and_whitespace():
    assert facts_present(["twenty -five"], ["a maximum of  Twenty\n -five (25) seconds"])
    assert not facts_present(["90"], ["only 25 here"])
    assert facts_present([], ["anything"])


def test_has_citation_matches_documented_format():
    assert has_citation("Yes. Source: ITF Rules, page 17")
    assert has_citation("Source: [Grand Slam Rules], page [56]")
    assert not has_citation("According to the rules, yes.")


def test_is_refusal():
    assert is_refusal("I could not find relevant information about this topic.")
    assert is_refusal("The context does not contain this information.")
    assert not is_refusal("A foot fault happens when... Source: ITF Rules, page 11")
    assert is_refusal("exact fallback", no_results_answer="exact fallback")


def test_test_set_is_well_formed():
    items = load_test_set()
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for item in items:
        assert item["question"].strip()
        assert item["expected_filter"] in (None, *DOCUMENTS)
        for ev in item["evidence"]:
            assert ev["document"] in DOCUMENTS and ev["header"].strip()
        if item["category"] == "out_of_scope":
            assert item["evidence"] == []
        else:
            assert item["evidence"], f"{item['id']} has no evidence"
    categories = {i["category"] for i in items}
    assert {"single_doc", "both_docs", "follow_up", "out_of_scope", "colloquial", "misrouted"} <= categories
