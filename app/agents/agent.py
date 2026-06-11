"""
Tennis Rules Agent — Deterministic Chain Architecture.

Architecture:
1. Classify intent (LLM) ← keep this
2. Route in Python (deterministic)
3. Retrieve with raw question (NO LLM query rewriting)
4. Synthesize answer (LLM)
"""
import json
import os
import re
import dotenv
dotenv.load_dotenv()
from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.db.retriever import search_similar_chunks
from app.agents.utils import truncate


GROQ_MODEL = os.getenv("GROQ_MODEL")
CLASSIF_MODEL = os.getenv("CLASSIF_MODEL")
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")
if not GROQ_MODEL:
    raise ValueError("GROQ_MODEL not found in .env")
if not CLASSIF_MODEL:
    raise ValueError("CLASSIF_MODEL not found in .env")


# ============================================================
# INTENT CLASSIFICATION 
# ============================================================

def classify_intent(question: str) -> Literal["search_itf", "search_gs", "compare", "refuse"]:
    """
    Use LLM to classify the question intent.
    """
    system_prompt = """
    You are a helpful AI assistant for a tennis application. 

    Your task is to classify the user question into ONE category. You have 4 categories to choose from: 
    1. search_itf: When the user asks about ITF rules only. 
        Examples: "What does ITF say about...", "According to ITF rules..."
    2. search_gs: The user EXPLICITLY asks about Grand Slam rules, 
       OR mentions a specific Grand Slam tournament (Wimbledon, US Open, 
       French Open, Australian Open).
       Examples: "What are Wimbledon's rules on...", "US Open tie-break rule"
    3. compare: General tennis rules questions that do NOT specify 
       a source. These search BOTH ITF and Grand Slam documents.
       Examples: "What is the warm-up time?", "How does the tie-break work?",
       "What are the coaching rules?", "How many sets in a match?",
       "What are the medical timeout rules?", "Is there an entry fee?"
    4. refuse: The question is NOT about tennis rules.
       Examples: player rankings, match results, tournament history, news, winner names.

    IMPORTANT:
    - If the user does NOT mention "ITF" or a specific tournament, default to compare.
    - Do NOT assume a general question is ITF-only.
    - NEVER return anything other than these 4 exact values.
    - NEVER return "unknown".

    Your output should be in a structured json format like so:
    {{
        "chain_of_thought": "Why did you pick this category?",
        "decision": "search_itf OR search_gs OR compare OR refuse"
    }}

    INPUT: {question}
    OUTPUT: 
    """
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    llm = ChatGroq(
        model=CLASSIF_MODEL,
        temperature=0,
        api_key=API_KEY,
        max_tokens=256
    )
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip()

    try:
        parsed = json.loads(result)
        decision = parsed.get("decision", "").strip().lower()
    except json.JSONDecodeError:
        print(f"   JSON parse failed, defaulting to compare")
        return "compare"

    valid_decisions = {"search_itf", "search_gs", "compare", "refuse"}
    if decision not in valid_decisions:
        print(f"   Invalid decision '{decision}', defaulting to compare")
        return "compare"
    
    print(f"   Intent: {decision}")
    return decision


# ============================================================
# QUERY NORMALIZATION (NO LLM — simple cleanup)
# ============================================================

STOPWORDS = {
    "what", "how", "why", "when", "where", "which",
    "is", "are", "was", "were", "do", "does", "did",
    "the", "a", "an", "to", "of", "in", "about",
    "explain", "tell", "me", "please", "can", "could",
    "would", "should", "there", "any", "i", "want",
    "know", "understand", "describe"
}


def normalize_query(question: str) -> str:
    """
    Simple deterministic normalization:
    - lowercase
    - fix common typos
    - remove stopwords
    - remove punctuation
    """
    q = question.lower().strip()

    # Remove punctuation
    q = re.sub(r"[^a-z0-9\s\-/]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    # Remove stopwords
    tokens = [t for t in q.split() if t not in STOPWORDS]

    return " ".join(tokens[:10])

def _dedupe_results(results: list[dict]) -> list[dict]:
    """Remove duplicate chunks, keeping highest similarity."""
    seen = {}
    for r in results:
        key = (r["document"], r["page"], r["content"][:100])
        if key not in seen or r["similarity"] > seen[key]["similarity"]:
            seen[key] = r
    return sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)


def retrieve_rules(
    question: str,
    document_filter: str | None = None,
    top_k: int = 4
) -> list[dict]:
    """
    Robust retrieval strategy:
    1. Try raw question first
    2. Try normalized query as fallback
    3. Merge and dedupe results
    4. Keep best matches
    """
    all_results = []

    # ── Try 1: raw question ───────────────────────────────────────────────
    raw_results = search_similar_chunks(
        question,
        top_k=top_k,
        document_filter=document_filter
    )
    all_results.extend(raw_results)
    print(f"   Raw query: {len(raw_results)} results")

    # ── Try 2: normalized query (if different) ────────────────────────────
    normalized = normalize_query(question)
    if normalized and normalized != question.lower().strip():
        print(f"   Normalized query: '{normalized}'")
        norm_results = search_similar_chunks(
            normalized,
            top_k=top_k,
            document_filter=document_filter
        )
        all_results.extend(norm_results)
        print(f"   Normalized: {len(norm_results)} results")

    # ── Merge and dedupe ──────────────────────────────────────────────────
    merged = _dedupe_results(all_results)

    # Prefer strong results (similarity >= 0.5)
    strong = [r for r in merged if r["similarity"] >= 0.5]
    if strong:
        return strong[:top_k]

    return merged[:top_k]


def format_context(results: list[dict], section_title: str) -> str:
    """Format results into a clear context block for the LLM."""
    if not results:
        return ""

    lines = [f"=== {section_title} ==="]

    for i, r in enumerate(results, 1):
        lines.append(
            f"\n[Excerpt {i}]\n"
            f"Source: {r['document']}, page {r['page']}\n"
            f"Relevance: {r['similarity']:.2f}\n"
            f"{r['content']}"
        )

    return "\n".join(lines)


# ============================================================
# SEARCH FUNCTIONS
# ============================================================

def search_itf(question: str) -> str:
    """Search ITF Rules only."""
    print(f"\nSearching ITF Rules...")
    
    results = retrieve_rules(
        question,
        document_filter="ITF Rules",
        top_k=3
    )

    if not results:
        return "NO_RESULTS"

    results = truncate(results, max_total_chars=3000)
    return format_context(results, "ITF Rules")


def search_gs(question: str) -> str:
    """Search Grand Slam Rules only."""
    print(f"\nSearching Grand Slam Rules...")
    
    results = retrieve_rules(
        question,
        document_filter="Grand Slam Rules",
        top_k=3
    )

    if not results:
        return "NO_RESULTS"

    results = truncate(results, max_total_chars=3000)
    return format_context(results, "Grand Slam Rules")


def compare_rules(question: str) -> str:
    """Search BOTH ITF and Grand Slam Rules."""
    print(f"\n🔍 Searching both rulebooks...")
    
    itf_results = retrieve_rules(
        question,
        document_filter="ITF Rules",
        top_k=2
    )
    gs_results = retrieve_rules(
        question,
        document_filter="Grand Slam Rules",
        top_k=2
    )

    if not itf_results and not gs_results:
        return "NO_RESULTS"

    itf_results = truncate(itf_results, max_total_chars=1800)
    gs_results = truncate(gs_results, max_total_chars=1800)

    blocks = []

    if itf_results:
        blocks.append(format_context(itf_results, "ITF Rules"))

    if gs_results:
        blocks.append(format_context(gs_results, "Grand Slam Rules"))

    return "\n\n".join(blocks)


# ============================================================
# SYNTHESIS (LLM generates final answer)
# ============================================================

def synthesize_answer(question: str, context: str) -> str:
    """Use LLM to synthesize a clear answer from retrieved context."""
    
    if context == "NO_RESULTS":
        return (
            "I could not find relevant information about this topic "
            "in the ITF or Grand Slam rulebooks. "
            "Please try rephrasing your question."
        )

    system_prompt = """
You are a tennis rules expert.

Answer the user's question using ONLY the context below.

CONTEXT:
{context}

RULES:
1. Answer directly and clearly.
2. Use ONLY the provided context — no outside knowledge.
3. If one excerpt gives a direct answer and another is only general,
   prefer the direct answer.
4. If ITF provides exact specifications and Grand Slam only refers 
   to "Rules of Tennis", answer from ITF with specific details.
5. If both documents contain relevant information, mention both.
6. Always cite sources exactly as: "Source: [Document Name], page [X]"
7. Never write generic placeholders like [Document] or [X].
8. Only say "I cannot find this information" as an ABSOLUTE LAST RESORT
   when the context is truly unrelated to the question.

QUESTION: {question}
ANSWER:
"""

    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=API_KEY,
        max_tokens=1024
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})


# ============================================================
# ORCHESTRATOR
# ============================================================

def ask_agent(question: str) -> dict:
    """
    Main entry point.
    1. Classify intent (LLM)
    2. Route to appropriate search
    3. Synthesize answer (LLM)
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # Step 1: Classify
    intent = classify_intent(question)
    
    # Step 2: Route and retrieve
    if intent == "refuse":
        return {
            "answer": (
                "I can only answer questions about official tennis rules "
                "(ITF Rulebook or Grand Slam Rulebook). I cannot help with "
                "tournament results, player history, rankings, or news."
            ),
            "steps": [{"action": "refuse", "reason": "out_of_scope"}]
        }
    
    elif intent == "search_itf":
        context = search_itf(question)
        answer = synthesize_answer(question, context)
        return {
            "answer": answer,
            "steps": [{"tool": "search_itf", "query": question}]
        }
    
    elif intent == "search_gs":
        context = search_gs(question)
        answer = synthesize_answer(question, context)
        return {
            "answer": answer,
            "steps": [{"tool": "search_gs", "query": question}]
        }
    
    elif intent == "compare":
        context = compare_rules(question)
        answer = synthesize_answer(question, context)
        return {
            "answer": answer,
            "steps": [{"tool": "search_both", "query": question}]
        }

    else:
        return {
            "answer": "I encountered an unexpected error. Please try rephrasing.",
            "steps": [{"error": "invalid_intent", "intent": intent}]
        }


# ============================================================
# LEGACY COMPATIBILITY
# ============================================================

def build_agent(**kwargs):
    """
    Legacy compatibility for test scripts that call build_agent().
    """
    class DummyAgent:
        def invoke(self, inputs):
            result = ask_agent(inputs["input"])
            return {
                "output": result["answer"],
                "intermediate_steps": []
            }
    return DummyAgent()