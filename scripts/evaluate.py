"""
Evaluation harness for the tennis RAG chatbot.

Layers (cheap and deterministic first):
  1. Routing   - no LLM, no DB: does the keyword router pick the expected filter,
                 and does it ever exclude the rulebook that holds the answer?
  2. Retrieval - DB + local embeddings, no LLM: is the right rule in the top-k?
  3. Answers   - LLM (opt-in with --answers, uses the Groq quota): keywords,
                 citation format, and refusal on out-of-scope questions.

Usage:
    python -m scripts.evaluate                       # layers 1 + 2
    python -m scripts.evaluate --answers --delay 4   # + layer 3 (slow, uses Groq)
    python -m scripts.evaluate --out plan/baseline.json

Exit code is non-zero when a score is below its --min-* threshold.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

TEST_SET_PATH = Path("data/eval/test_set.json")
TOP_K = 4              # what retrieve_node uses
CONTEXT_CHARS = 4000   # what retrieve_node passes to truncate_context
DOCUMENTS = ("ITF Rules", "Grand Slam Rules")

REFUSAL_MARKERS = (
    "could not find", "couldn't find", "not contain", "does not contain",
    "doesn't contain", "no information", "not enough information",
    "not covered", "cannot find", "can't find", "not specified",
    "outside the scope", "not mentioned", "no relevant",
    "does not include", "doesn't include", "only contains",
)
CITATION_RE = re.compile(r"source:\s*\[?[^\n\],]+\]?,\s*page\s*\[?\d+", re.IGNORECASE)


# ============================================================
# PURE SCORING HELPERS (unit-tested, no DB / LLM)
# ============================================================

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def header_of(metadata: dict | None) -> str:
    """'<rule or article> / <section>' built from a parent chunk's metadata."""
    metadata = metadata or {}
    main = metadata.get("rule") or metadata.get("article") or ""
    return f"{main} / {metadata.get('section') or ''}"


def matches_evidence(result: dict, evidence: dict) -> bool:
    """A retrieved parent satisfies an evidence entry if document and header match."""
    return (
        result.get("document") == evidence["document"]
        and evidence["header"].lower() in header_of(result.get("metadata")).lower()
    )


def first_hit_rank(results: list[dict], evidence: list[dict]) -> int | None:
    """1-based rank of the first result matching ANY evidence entry, else None."""
    for rank, result in enumerate(results, start=1):
        if any(matches_evidence(result, ev) for ev in evidence):
            return rank
    return None


def evidence_found(results: list[dict], evidence: list[dict]) -> bool:
    """True when EVERY evidence entry is matched by at least one result."""
    return bool(evidence) and all(
        any(matches_evidence(r, ev) for r in results) for ev in evidence
    )


def route_conflict(doc_filter: str | None, evidence: list[dict]) -> bool:
    """True when the filter excludes a rulebook that holds required evidence."""
    return doc_filter is not None and any(ev["document"] != doc_filter for ev in evidence)


def facts_present(keywords: list[str], texts: list[str]) -> bool:
    """All answer keywords appear somewhere in the given texts."""
    blob = normalize(" ".join(texts))
    return all(k.lower() in blob for k in keywords)


def has_citation(answer: str) -> bool:
    return bool(CITATION_RE.search(answer))


def is_refusal(answer: str, no_results_answer: str = "") -> bool:
    low = answer.lower()
    if no_results_answer and answer.strip() == no_results_answer.strip():
        return True
    return any(marker in low for marker in REFUSAL_MARKERS)


def rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_test_set(path: Path = TEST_SET_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ============================================================
# LAYER 1: ROUTING
# ============================================================

def evaluate_routing(items: list[dict]) -> dict:
    from app.graph.nodes import _route_filter

    rows = []
    for item in items:
        if item.get("history"):
            continue  # needs the LLM contextualizer; covered by --answers
        doc_filter, _ = _route_filter(item["question"])
        rows.append({
            "id": item["id"],
            "filter": doc_filter,
            "expected": item["expected_filter"],
            "correct": doc_filter == item["expected_filter"],
            "conflict": route_conflict(doc_filter, item["evidence"]),
        })
    return {
        "n": len(rows),
        "accuracy": rate([r["correct"] for r in rows]),
        "conflicts": sum(r["conflict"] for r in rows),
        "rows": rows,
    }


# ============================================================
# LAYER 2: RETRIEVAL
# ============================================================

def evaluate_retrieval(items: list[dict]) -> dict:
    from app.graph.nodes import _route_filter
    from app.graph.tools import search_rules, truncate_context

    rows, oos_rows = [], []
    for item in items:
        if item.get("history"):
            continue  # needs the LLM contextualizer; covered by --answers
        question = item["question"]
        doc_filter, _ = _route_filter(question)
        results = search_rules(query=question, document_filter=doc_filter, top_k=TOP_K)

        if not item["evidence"]:  # out of scope: ideally nothing passes the threshold
            oos_rows.append({
                "id": item["id"],
                "n_results": len(results),
                "top_similarity": results[0]["similarity"] if results else 0.0,
            })
            continue

        found = evidence_found(results, item["evidence"])
        rank = first_hit_rank(results, item["evidence"])

        # Retry simulation (bug 1.5): if the first pass missed, search both rulebooks.
        recovered = None
        if not found and doc_filter is not None:
            wide = search_rules(query=question, document_filter=None, top_k=TOP_K)
            recovered = evidence_found(wide, item["evidence"])

        # What the LLM actually sees: truncated parents (bug 4).
        seen = truncate_context(results, max_total_chars=CONTEXT_CHARS)
        keywords = item.get("answer_keywords", [])
        evidence_results = [r for r in results if any(matches_evidence(r, e) for e in item["evidence"])]
        evidence_seen = [s for s, r in zip(seen, results) if r in evidence_results]
        rows.append({
            "id": item["id"],
            "category": item["category"],
            "filter": doc_filter,
            "found": found,
            "rank": rank,
            "recovered_by_widening": recovered,
            "fact_in_retrieved": facts_present(keywords, [r["content"] for r in evidence_results]) if keywords else None,
            "fact_in_context": facts_present(keywords, [s["content"] for s in evidence_seen]) if keywords else None,
            "top_similarity": results[0]["similarity"] if results else 0.0,
        })

    by_category: dict[str, float | None] = {}
    for cat in sorted({r["category"] for r in rows}):
        by_category[cat] = rate([r["found"] for r in rows if r["category"] == cat])

    return {
        "n": len(rows),
        f"hit@{TOP_K}": rate([r["found"] for r in rows]),
        "mrr": mean([1 / r["rank"] if r["rank"] else 0.0 for r in rows]),
        "by_category": by_category,
        "recovered_by_widening": rate([r["recovered_by_widening"] for r in rows if r["recovered_by_widening"] is not None]),
        "fact_in_retrieved": rate([r["fact_in_retrieved"] for r in rows if r["fact_in_retrieved"] is not None]),
        "fact_in_context": rate([r["fact_in_context"] for r in rows if r["fact_in_context"] is not None]),
        "oos_false_retrieval": rate([o["n_results"] > 0 for o in oos_rows]),
        "rows": rows,
        "oos_rows": oos_rows,
    }


# ============================================================
# LAYER 3: ANSWERS (LLM)
# ============================================================

def evaluate_answers(items: list[dict], delay: float) -> dict:
    from app.graph.builder import agent_graph
    from app.graph.prompts import NO_RESULTS_ANSWER

    run_id = int(time.time())
    rows = []
    for n, item in enumerate(items):
        config = {"configurable": {"thread_id": f"eval-{run_id}-{item['id']}"}}
        state = {"question": item["question"], "chat_history": item.get("history", [])}
        result, error = None, None
        for attempt in range(2):  # one retry after a rate-limit pause
            try:
                result = agent_graph.invoke(state, config=config)
                error = None
                break
            except Exception as exc:  # noqa: BLE001 - report and continue
                error = f"{type(exc).__name__}: {str(exc)[:120]}"
                if attempt == 0 and "rate" in str(exc).lower():
                    print(f"   rate limited on {item['id']}, waiting 30s...")
                    time.sleep(30)
                    continue
                break

        row = {"id": item["id"], "category": item["category"], "error": error}
        if result is not None:
            answer = result.get("answer", "")
            steps = result.get("steps", [])
            row["answer"] = answer
            row["retries"] = sum(1 for s in steps if s.get("action") == "reformulate")
            if item["evidence"]:
                row["keywords_ok"] = facts_present(item.get("answer_keywords", []), [answer])
                row["citation_ok"] = has_citation(answer)
                row["passed"] = row["keywords_ok"] and row["citation_ok"]
            else:
                row["refused"] = is_refusal(answer, NO_RESULTS_ANSWER)
                row["passed"] = row["refused"]
        rows.append(row)
        print(f"   [{n + 1}/{len(items)}] {item['id']}: {'ERROR ' + error if error else ('pass' if row['passed'] else 'FAIL')}")
        if n < len(items) - 1:
            time.sleep(delay)

    scored = [r for r in rows if not r["error"]]
    in_scope = [r for r in scored if "keywords_ok" in r]
    oos = [r for r in scored if "refused" in r]
    return {
        "n": len(rows),
        "errors": len(rows) - len(scored),
        "pass_rate": rate([r["passed"] for r in scored]),
        "keyword_rate": rate([r["keywords_ok"] for r in in_scope]),
        "citation_rate": rate([r["citation_ok"] for r in in_scope]),
        "refusal_rate": rate([r["refused"] for r in oos]),
        "avg_retries": mean([float(r["retries"]) for r in scored]),
        "rows": rows,
    }


# ============================================================
# REPORT
# ============================================================

def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:5.1f}%"


def print_report(report: dict) -> None:
    routing, retrieval = report["routing"], report["retrieval"]
    print("\n" + "=" * 64)
    print("ROUTING  (n=%d, follow-ups skipped: need LLM)" % routing["n"])
    print("  accuracy vs expected filter :", pct(routing["accuracy"]))
    print("  routes that EXCLUDE the answer's rulebook:", routing["conflicts"])
    for r in routing["rows"]:
        if not r["correct"] or r["conflict"]:
            print(f"    - {r['id']}: got {r['filter']!r}, expected {r['expected']!r}"
                  + ("  [excludes answer's rulebook]" if r["conflict"] else ""))

    print("\nRETRIEVAL  (n=%d, top_k=%d)" % (retrieval["n"], TOP_K))
    print("  hit@%d (all evidence in top-k) :" % TOP_K, pct(retrieval[f"hit@{TOP_K}"]))
    print("  MRR                         :", "n/a" if retrieval["mrr"] is None else f"{retrieval['mrr']:.3f}")
    for cat, value in retrieval["by_category"].items():
        print(f"    {cat:<14}: {pct(value)}")
    print("  misses recovered by widening:", pct(retrieval["recovered_by_widening"]))
    print("  answer fact in retrieved text:", pct(retrieval["fact_in_retrieved"]))
    print("  answer fact after truncation :", pct(retrieval["fact_in_context"]))
    print("  out-of-scope with >=1 result :", pct(retrieval["oos_false_retrieval"]))
    for r in retrieval["rows"]:
        if not r["found"]:
            print(f"    - MISS {r['id']} (filter={r['filter']!r}, top_sim={r['top_similarity']:.2f}, "
                  f"widening_recovers={r['recovered_by_widening']})")

    answers = report.get("answers")
    if answers:
        print("\nANSWERS  (n=%d, errors=%d)" % (answers["n"], answers["errors"]))
        print("  overall pass rate :", pct(answers["pass_rate"]))
        print("  keywords present  :", pct(answers["keyword_rate"]))
        print("  citation format   :", pct(answers["citation_rate"]))
        print("  refusal on out-of-scope:", pct(answers["refusal_rate"]))
        print("  avg retries/question   :", "n/a" if answers["avg_retries"] is None else f"{answers['avg_retries']:.2f}")
        for r in answers["rows"]:
            if not r["error"] and not r["passed"]:
                print(f"    - FAIL {r['id']}: keywords={r.get('keywords_ok')} citation={r.get('citation_ok')} refused={r.get('refused')}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-set", default=str(TEST_SET_PATH))
    parser.add_argument("--answers", action="store_true", help="also run the LLM answer layer (uses Groq quota)")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between questions in --answers (rate limit)")
    parser.add_argument("--ids", help="comma-separated question ids to run")
    parser.add_argument("--out", help="write the full report as JSON")
    parser.add_argument("--min-routing", type=float, default=0.85)
    parser.add_argument("--min-hit", type=float, default=0.80)
    parser.add_argument("--min-answer", type=float, default=0.70)
    args = parser.parse_args(argv)

    items = load_test_set(Path(args.test_set))
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [i for i in items if i["id"] in wanted]

    report = {"routing": evaluate_routing(items), "retrieval": evaluate_retrieval(items)}
    if args.answers:
        print("Running answer layer (LLM)...")
        report["answers"] = evaluate_answers(items, args.delay)

    print_report(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report written to {args.out}")

    failures = []
    if (report["routing"]["accuracy"] or 0) < args.min_routing:
        failures.append("routing accuracy")
    if (report["retrieval"][f"hit@{TOP_K}"] or 0) < args.min_hit:
        failures.append(f"hit@{TOP_K}")
    if args.answers and (report["answers"]["pass_rate"] or 0) < args.min_answer:
        failures.append("answer pass rate")
    if failures:
        print("BELOW THRESHOLD:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
