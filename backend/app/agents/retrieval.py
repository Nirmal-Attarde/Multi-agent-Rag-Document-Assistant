from .base import call_llm
from ..retriever import Retriever

REWRITE_SYSTEM_PROMPT = """You are a query rewriting agent for a research paper retrieval system.

Your job: rewrite the user's natural-language question into a search query that will match the technical vocabulary used in academic papers.

Rules:
- Use formal/technical synonyms (e.g., "heart attack" -> "myocardial infarction")
- Include domain-specific terms likely to appear in the source
- Keep it concise: 5-15 words
- Output ONLY the rewritten query, nothing else. No explanation, no quotes."""


class RetrievalAgent:
    def __init__(self):
        self.retriever = Retriever()

    def rewrite_query(self, query: str) -> str:
        rewritten = call_llm(REWRITE_SYSTEM_PROMPT, query, temperature=0.3)
        return rewritten.strip().strip('"').strip("'")

    def retrieve(self, query: str, top_k: int = 5, rewrite: bool = True) -> dict:
        """Retrieve chunks. Returns {chunks, original_query, rewritten_query}."""
        if rewrite:
            rewritten = self.rewrite_query(query)
        else:
            rewritten = query

        chunks = self.retriever.search(rewritten, top_k=top_k)

        return {
            "chunks": chunks,
            "original_query": query,
            "rewritten_query": rewritten,
        }