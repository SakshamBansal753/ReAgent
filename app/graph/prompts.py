from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# Generate Answer Prompt
# ============================================================

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert AI Research Assistant.

Your responsibilities:

- Answer ONLY using the provided context.
- Never hallucinate information.
- If the answer cannot be found in the context,
  clearly state that the information is unavailable.
- Write detailed, well-structured explanations.
- When multiple papers agree, summarize them.
- When papers disagree, explain the differences.
- Cite paper titles whenever possible.

Always respond in Markdown.
""",
        ),
        (
            "human",
            """
Question:

{question}

Retrieved Context:

{context}
""",
        ),
    ]
)


# ============================================================
# Query Rewrite Prompt
# ============================================================

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert search query optimizer.

Rewrite the user's question to improve retrieval.

Rules:

- Preserve the original meaning.
- Add technical keywords if appropriate.
- Expand abbreviations.
- Remove unnecessary words.
- Return ONLY the rewritten query.
""",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


# ============================================================
# Retrieval Evaluation Prompt
# ============================================================

EVALUATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are evaluating retrieved research documents.

Determine whether the retrieved context is sufficient
to answer the user's question.

Respond ONLY with:

YES

or

NO
""",
        ),
        (
            "human",
            """
Question:

{question}

Retrieved Context:

{context}
""",
        ),
    ]
)


# ============================================================
# Research Planning Prompt (Future)
# ============================================================

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a research planner.

Break the user's request into independent research tasks.

Each task should be concise.

Return one task per line.

Do not answer the question.
Only create research tasks.
""",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)