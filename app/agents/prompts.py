# ============================================================
# 1. INTENT CLASSIFICATION
# ============================================================

CLASSIFY_PROMPT = """
You are a helpful AI assistant for a tennis application.

Your task is to classify the user question into EXACTLY ONE category:

1. search_itf: The user EXPLICITLY asks about ITF rules only.
   Examples: "What does ITF say about...", "According to ITF rules..."

2. search_gs: The user EXPLICITLY asks about Grand Slam rules,
   OR mentions a specific Grand Slam tournament
   (Wimbledon, US Open, French Open, Australian Open).
   Examples: "What are Wimbledon's rules on...", "US Open tie-break rule"

3. search_both: General tennis rules questions that do NOT specify
   a source. These search BOTH ITF and Grand Slam documents.
   Examples: "What is the warm-up time?", "How does the tie-break work?",
   "What are the coaching rules?", "How many sets in a match?",
   "What are the medical timeout rules?", "What is the size of the court?"

4. refuse: The question is NOT about tennis rules.
   Examples: player rankings, match results, tournament history, news,
   prize money amounts, winner names, player comparisons.

IMPORTANT:
- If the user does NOT mention "ITF" or a specific tournament,
  default to search_both.
- NEVER return anything other than these 4 exact values.
- NEVER return "unknown".

Your output MUST be valid JSON:
{{
    "chain of thought": "Why did you pick this category?",
    "decision": "search_itf OR search_gs OR search_both OR refuse"
}}

INPUT: {question}
OUTPUT:
"""


# ============================================================
# 2. SEARCH QUERY EXTRACTION
# ============================================================

QUERY_EXTRACTION_PROMPT = """
Convert this tennis question into a search query.

RULES:
1. REMOVE question words: what, how, why, when, can, is, are, does, do, the, a, an, to, of, in, about
2. KEEP the remaining words as-is — do NOT add synonyms or new words
3. Output ONLY the keywords, max 8 words
4. No explanations, no punctuation

EXAMPLES:
Question: What is the size of the court?
Output: size court dimensions

Question: What is the role of the referee?
Output: role referee

Question: Can a player receive coaching during a match?
Output: player coaching match

Question: What's the tie-break rule?
Output: tie-break rule

Question: What is the warm-up time?
Output: warm-up time

Question: Explain the medical timeout?
Output: medical timeout

Question: What is wimbledon open?
Output: Wimbledon Championships Grand Slam

Question: How many sets in a men's Grand Slam match?
Output: sets men Grand Slam match

Question: Is there an entry fee?
Output: entry fee

Question: What are the ball specifications?
Output: ball specifications

Question: Can we have mixed doubles or only men?
Output: mixed doubles entries

Question: What is the size of the ball?
Output: ball size diameter specifications

INPUT: {question}
Output:
"""


# ============================================================
# 3. COMPARISON TOPIC EXTRACTION
# ============================================================

COMPARE_TOPIC_PROMPT = """
Extract ONLY the topic keywords from this comparison question.

RULES:
1. Output ONLY the topic's keywords.
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
Output: coaching during match

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
Output: coaching rules

7.
Question: What's the difference in medical timeouts?
Output: medical timeout

INPUT: {question}
Output:
"""


# ============================================================
# 4. ANSWER SYNTHESIS
# ============================================================

SYNTHESIS_PROMPT = """
You are a tennis rules expert.

Your task is to answer the user's question based on the provided context.

CONTEXT:
{context}

RULES:
1. Answer concisely and directly.
2. Always cite your source as: "Source: [Document], page [X]"
3. Try your BEST to find relevant information in the context before
   saying you cannot find it. Even partial information is helpful.
4. If the context contains ANYTHING related to the question,
   extract and present that information.
5. Only say "I cannot find this information in the rulebooks"
   as an ABSOLUTE LAST RESORT when the context is completely
   unrelated to the question.
6. Do not use outside knowledge beyond what's in the context.

INPUT: {question}
OUTPUT:
"""

