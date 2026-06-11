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
import json
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
    # Use LLM for the remaining nuanced classifications
    system_prompt = """
    You are a helpful AI assistant for a tennis application. 

    Your task is to classify the user question into ONE category.  You have 4 categories to choose from: 
    1. search_itf: General tennis rules, ITF rules, scoring, court, serve, equipment...
    2. search_gs: Grand Slam specific questions (Wimbeldon, US Open, French Open, Australian Open)
    3. compare: To compare the tennis rules between ITF and Grand Slam. This is used when the answer is present in both documents, and the user did not specify the category to answer from. 
    4. refuse: The user must ask a question about tennis rules only: ITF and/or Grand Slam. 

    The user is not allowed to : 
    1. Ask questions anything else about tennis rules. 
    2. Ask questions about the winners, names of players, comparisons between players, or history of tournaments.

    Your output should be in a structured json format like so. Each key is a string and each value is a string. Make sure to follow the format exactly:
    {{
        "chain of thought": "go over each of the categories above and write some of your thoughts about what category is this input relevant to.",
        "decision": "search_itf OR search_gs OR compare OR refus. Pick one of those, and only write the word."
    }}

    INPUT: {question}
    OUTPUT: 
    """
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    llm = ChatGroq(
        model=CLASSIF_MODEL,
        temperature=0,
        api_key=API_KEY,
        max_tokens=1024
    )
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().lower()

    result = json.loads(result)
    decision = result.get("decision")
    return decision 


# ============================================================
# TOOL CALLING 
# ============================================================

def _extract_search_query(question: str) -> str:
    """Extract the best search query from a natural language question."""
    prompt = """
    You are a helpful AI assistant for a tennis application. 

    Your task is to convert the tennis questions into short search queries. 

    RULES:
    1. Output ONLY the search query. 
    2. Use ONLY keywords, 5-10 words maximum including specific terminology.
    3. Do not explain anything. 
    4. Do not add prefix.
    5. Do not add "Search query:".

    EXAMPLES:
    1. 
    Question: How many sets in a men's Grand Slam match?
    Output: number of sets in men's Grand Slam match.

    2. 
    Question: Can a player receive coaching during a match?
    Output: coaching during match prohibited rules

    3. 
    Question: What's the tie-break rule?
    Output: tie-break rule

    4. 
    Question: What is the warm-up time?
    Output: warm-up time

    5. 
    Question: What does the rulebook say about warm-up?
    Output: warm-up duration time before match starts

    6. 
    Question: What are the coaching rules?
    Output: coaching during match permitted prohibited


    INPUT: {question}
    Output: 
    """
    extract_prompt = ChatPromptTemplate.from_template(prompt)
    
    llm = ChatGroq(
        model=CLASSIF_MODEL,
        temperature=0,
        api_key=API_KEY,
        max_tokens=128,
    )
    
    chain = extract_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip()
    
    return result

def search_itf(question: str) -> str:
    """Search ITF Rules with query optimization."""
    query = _extract_search_query(question)
    print(f"   Extracted query: '{query}'")  # For debugging
    
    results = search_similar_chunks(query, top_k=5, document_filter="ITF Rules")
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
    print(f"   Extracted query: '{query}'")
    
    results = search_similar_chunks(query, top_k=5, document_filter="Grand Slam Rules")
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
    prompt = """
    You are a helpful AI assistant for a tennis application. 

    Your task is to extract ONLY the topic keywords from this comparison question.

    RULES:
    1. Output ONLY the topic's keyword. 
    2. Use ONLY keywords, 5-10 words maximum including specific terminology.
    3. Do not explain anything. 
    4. Do not add prefix.
    5. Do not add "Topic Keyword:".

    EXAMPLES:
    1. 
    Question: How many sets in a men's Grand Slam match?
    Output: number of sets

    2. 
    Question: Can a player receive coaching during a match?
    Output: coaching during match prohibited rules

    3. 
    Question: What's the tie-break rule?
    Output: tie-break rule

    4. 
    Question: What is the warm-up time?
    Output: warm-up time

    5. 
    Question: What does the rulebook say about warm-up?
    Output: warm-up duration time before match starts

    6. 
    Question: What are the coaching rules?
    Output: coaching during match permitted prohibited

    7. 
    Question: What's the difference in medical timeouts?
    Output: medical timeout


    INPUT: {question}
    Output: 
    """
    topic_prompt = ChatPromptTemplate.from_template(prompt)
    
    llm = ChatGroq(
        model=CLASSIF_MODEL,
        temperature=0,
        api_key=API_KEY,
        max_tokens=128,
    )
    
    chain = topic_prompt | llm | StrOutputParser()
    topic = chain.invoke({"question": question}).strip()
    
    
    print(f"   Extracted topic: '{topic}'")
    
    itf_results = search_similar_chunks(topic, top_k=5, document_filter="ITF Rules")
    gs_results = search_similar_chunks(topic, top_k=5, document_filter="Grand Slam Rules")
    
    output = [f"Comparison of '{topic}':\n"]
    output.append("=== ITF Rules ===")
    if itf_results:
        for r in itf_results:
            output.append(f"[Page {r['page']}]\n{r['content']}...")
    else:
        output.append("No content found.")
    
    output.append("\n=== Grand Slam Rules ===")
    if gs_results:
        for r in gs_results:
            output.append(f"[Page {r['page']}]\n{r['content']}...")
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
    system_prompt = """
    You are a tennis rules expert. 

    Your task is to answer the user's question based ONLY on the provided context. 

    CONTEXT:
    {context}

    RULES:
    1. Answer concisely and directly.
    2. Always cite your source as: "Source: [Document], page [X]"
    3. If the context doesn't contain the answer, say: "I cannot find this information in the rulebooks."
    4. Do not use outside knowledge.

    INPUT: {question}
    Output: 
    """
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    llm = ChatGroq(
        model=GROQ_MODEL,  # Big model for quality synthesis
        temperature=0.1,
        api_key=API_KEY,
        max_tokens=1024
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