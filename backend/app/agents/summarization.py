from .base import call_llm

SYSTEM_PROMPT = """You are a research assistant that drafts answers from document excerpts.

Rules:
- Base every claim on the provided excerpts.
- Do not use outside knowledge.
- If excerpts don't contain the answer, say so clearly.
- Reference excerpts inline using the format [Excerpt N], where N is the excerpt number.
- Be concise and factual. No filler.

Your output is a draft. Citations will be verified separately."""


def summarize(question: str, chunks: list[dict]) -> str:
    """Draft an answer from retrieved chunks."""
    if not chunks:
        return "No relevant excerpts were found in the documents to answer this question."

    context = "\n\n".join([
        f"[Excerpt {i+1} | source: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    user_prompt = f"""Document excerpts:

{context}

---

Question: {question}

Draft an answer using only the excerpts. Reference excerpts inline as [Excerpt N]."""

    return call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.2)