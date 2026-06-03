"""
Tennis Rules Agent — Deterministic Chain Architecture.

Instead of an autonomous agent (which fails intermittently with Groq),
we use a deterministic router:
1. Classify intent (LLM)
2. Route in Python (deterministic)
3. Call tool explicitly (no JSON parsing)
4. Synthesize answer (LLM)

This is production-grade LangChain: RunnableSequence + explicit orchestration.
"""
import os
import dotenv
dotenv.load_dotenv()
from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.db.retriever import search_similar_chunks


GROQ_MODEL = os.getenv("GROQ_MODEL")
CLASSIF_MODEL = os.getenv("CLASSIF_MODEL")
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")
if not GROQ_MODEL:
    raise ValueError("GROQ_MODEL not found in .env")
if not CLASSIF_MODEL:
    raise ValueError("CLASSIF_MODEL not found in .env")


def classify_intent(question: str) -> Literal["search_itf", "search_gs", "compare", "refuse"]:
    """
    Use LLM to classify the question intent.
    """
    # Out-of-scope check
    out_of_scope_keywords = ["won", "winner", "champion", "score", "result", 
                            "wimbledon 20", "ranking", "atp", "wta", 
                            "career", "born", "age of", "history of"]
    if any(kw in question.lower() for kw in out_of_scope_keywords):
        return "refuse"
    
    # HEURISTIC: Check for comparison signals BEFORE asking LLM (saves a call)
    comparison_signals = [
        "compare", "comparison", "vs", "versus",
        "difference", "differ", "differs", "different",
        "between itf", "between grand slam", "between the two",
        "how do.*differ", "how does.*differ"
    ]
    question_lower = question.lower()
    if any(signal in question_lower for signal in comparison_signals):
        return "compare"
    
    # Use LLM for the remaining nuanced classifications
    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant for a tennis application. Your task is to classify this tennis question into ONE category:
- search_itf: General tennis rules, ITF rules, scoring, court, serve, equipment
- search_gs: Grand Slam specific questions (Wimbledon, US Open, etc.)
- refuse: Not about tennis rules (history, results, players)

Question: "{question}"

Return ONLY the category name, nothing else.
""")
    
    llm = ChatGroq(
        model=CLASSIF_MODEL,
        temperature=0,
        api_key=API_KEY,
        max_tokens=10
    )
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().lower()
    
    valid = ["search_itf", "search_gs", "refuse"]
    return result if result in valid else "search_itf"


# ============================================================
# TOOL CALLING (Explicit Python functions, not @tool)
# ============================================================

def _extract_search_query(question: str) -> str:
    """Extract the best search query from a natural language question."""
    extract_prompt = ChatPromptTemplate.from_template("""You convert tennis questions into short search queries.

ONLY output the search query. NO explanation. NO prefix. NO "Search query:".
Just the keywords, 5-10 words including specific terminology.

Question: How many sets in a men's Grand Slam match?
number of sets best of five men singles match

Question: Can a player receive coaching during a match?
coaching during match prohibited rules

Question: What's the tie-break rule?
tie-break game scoring seven points margin

Question: What is the warm-up time?
warm-up time duration five minutes before match starts

Question: What does the rulebook say about warm-up?
warm-up duration time before match starts

Question: What are the coaching rules?
coaching during match permitted prohibited

Question: {question}
""")
    
    llm = ChatGroq(
        model=CLASSIF_MODEL,
        temperature=0,
        api_key=API_KEY,
        max_tokens=30,
        stop=["\n\n", "Question:"]
    )
    
    chain = extract_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip()
    
    cleanup_prefixes = [
        "search query:", "query:", "answer:", "based on", 
        "the best search query", "i'll provide", "to convert"
    ]
    result_lower = result.lower()
    for prefix in cleanup_prefixes:
        if result_lower.startswith(prefix):
            return question
    
    return result

def search_itf(question: str) -> str:
    """Search ITF Rules with query optimization."""
    query = _extract_search_query(question)
    print(f"   🔍 Extracted query: '{query}'")  # For debugging
    
    results = search_similar_chunks(query, top_k=4, document_filter="ITF Rules")
    if not results:
        return "No relevant rules found in ITF Rules."
    
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[Excerpt {i}] (ITF Rules, page {r['page']})\n{r['content'][:500]}..."
        )
    return "\n\n".join(formatted)


def search_gs(question: str) -> str:
    """Search Grand Slam Rules with query optimization."""
    query = _extract_search_query(question)
    print(f"   🔍 Extracted query: '{query}'")
    
    results = search_similar_chunks(query, top_k=4, document_filter="Grand Slam Rules")
    if not results:
        return "No relevant rules found in Grand Slam Rulebook."
    
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[Excerpt {i}] (Grand Slam Rules, page {r['page']})\n{r['content'][:500]}..."
        )
    return "\n\n".join(formatted)


def compare_rules(question: str) -> str:
    """Compare topic between both rulebooks."""
    topic_prompt = ChatPromptTemplate.from_template("""Extract ONLY the topic keywords from this comparison question.
NO explanation. NO prefix. NO list. Just a single 2-5 word phrase.

Question: Compare the warm-up time between ITF and Grand Slam
warm-up time duration

Question: How do the coaching rules differ between rulebooks?
coaching rules during match

Question: What's the difference in medical timeouts?
medical timeout

Question: Compare the tie-break format
tie-break game format

Question: {question}
""")
    
    llm = ChatGroq(
        model=CLASSIF_MODEL,
        temperature=0,
        api_key=API_KEY,
        max_tokens=15,
        stop=["\n\n", "Question:", "1.", "2."]
    )
    
    chain = topic_prompt | llm | StrOutputParser()
    topic = chain.invoke({"question": question}).strip()
    
    # Defensive: if topic looks like a list, fall back
    if topic.startswith(("1.", "2.", "-", "*")) or "\n" in topic:
        # Extract clean topic from the question itself
        topic = question.lower().replace("compare", "").replace("the", "").replace("between itf and grand slam", "").replace("difference", "").strip()
    
    print(f"   🔍 Extracted topic: '{topic}'")
    
    itf_results = search_similar_chunks(topic, top_k=3, document_filter="ITF Rules")
    gs_results = search_similar_chunks(topic, top_k=3, document_filter="Grand Slam Rules")
    
    output = [f"Comparison of '{topic}':\n"]
    output.append("=== ITF Rules ===")
    if itf_results:
        for r in itf_results:
            output.append(f"[Page {r['page']}]\n{r['content'][:400]}...")
    else:
        output.append("No content found.")
    
    output.append("\n=== Grand Slam Rules ===")
    if gs_results:
        for r in gs_results:
            output.append(f"[Page {r['page']}]\n{r['content'][:400]}...")
    else:
        output.append("No content found.")
    
    return "\n\n".join(output)

# ============================================================
# SYNTHESIS (LLM generates final answer)
# ============================================================

def synthesize_answer(question: str, context: str) -> str:
    """
    Use LLM to synthesize a clear answer from retrieved context.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a tennis rules expert. Answer the user's question based ONLY on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer concisely and directly
- Always cite your source as: "Source: [Document], page [X]"
- If the context doesn't contain the answer, say: "I cannot find this information in the rulebooks."
- Do not use outside knowledge

Answer:
""")
    
    llm = ChatGroq(
        model=GROQ_MODEL,  # Big model for quality synthesis
        temperature=0.1,
        api_key=API_KEY,
        max_tokens=512
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})


# ============================================================
#  ORCHESTRATOR 
# ============================================================

def ask_agent(question: str) -> dict:
    """
    Main entry point. Deterministic routing + LLM synthesis.
    
    This replaces the fragile AgentExecutor with explicit control flow.
    """
    # Step 1: Classify
    intent = classify_intent(question)
    
    # Step 2: Route and execute
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
            "steps": [{"tool": "compare_rules", "topic": question}]
        }
    
    else:
        # Should never happen, but defensive
        return {
            "answer": "I encountered an unexpected error. Please try rephrasing.",
            "steps": [{"error": "invalid_intent", "intent": intent}]
        }


def build_agent(**kwargs):
    """
    Legacy compatibility for test scripts that call build_agent().
    We return a dummy object that has an invoke method.
    """
    class DummyAgent:
        def invoke(self, inputs):
            result = ask_agent(inputs["input"])
            # Format to match old AgentExecutor output structure
            return {
                "output": result["answer"],
                "intermediate_steps": []  # We don't track steps the same way
            }
    return DummyAgent()