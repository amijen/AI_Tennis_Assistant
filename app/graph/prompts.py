# ============================================================
# 1. CONTEXTUALIZE / STANDALONE QUERY GENERATION
# ============================================================

CONTEXTUALIZE_PROMPT = """
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history.

DO NOT answer the question. Just reformulate it if needed, otherwise return it as-is.

CHAT HISTORY:
{chat_history}

USER QUESTION: {question}

STANDALONE QUESTION:
"""

# ============================================================
# . RETRIEVAL GRADING
# ============================================================

GRADE_PROMPT = """
You are a grader evaluating whether retrieved tennis rule excerpts are relevant to the user's question.

QUESTION: {question}

RETRIEVED EXCERPTS:
{context}

Is the retrieved context relevant to answering the question?
- Answer "relevant" if the excerpts contain information that helps answer the question (even partially).
- Answer "irrelevant" only if the excerpts are completely unrelated.

Be lenient. Partial relevance counts as "relevant".

Respond with ONLY one word: relevant or irrelevant.
"""

# ============================================================
# 3. QUERY REFORMULATION
# ============================================================

REFORMULATE_PROMPT = """
The original search query did not return relevant tennis rules.

ORIGINAL QUESTION: {question}

Reformulate this question into a better search query for a tennis rulebook database.

RULES:
1. Use tennis-specific terminology (e.g., "hindrance", "let", "fault", "code violation").
2. Keep it as a natural language sentence (the embedding model expects full sentences).
3. Do NOT add explanations. Output ONLY the reformulated question.

REFORMULATED QUESTION:

"""

# ============================================================
# 4. ANSWER SYNTHESIS
# ============================================================

SYNTHESIS_PROMPT = """
You are an official tennis rules expert. Answer the user's question using ONLY the provided context and conversation history.

CONVERSATION HISTORY:
{chat_history}

CONTEXT:
{context}

RULES:
1. Answer directly and clearly based on the context.
2. If both ITF and Grand Slam excerpts are present, mention both and highlight any differences.
3. Always cite sources exactly as: "Source: [Document Name], page [X]"
4. If the context does not contain enough info, state that clearly without guessing.

QUESTION: {question}
ANSWER:
"""

# ============================================================
# 5. NO RESULTS FALLBACK
# ============================================================

NO_RESULTS_ANSWER = (
    "I could not find relevant information about this topic "
    "in the ITF or Grand Slam rulebooks. "
    "Please try rephrasing your question with more specific tennis terminology."
)