import os
from dotenv import load_dotenv
from groq import Groq
from .retriever import Retriever

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a research assistant that answers questions using ONLY the provided document excerpts.

Rules:
- Base every claim on the excerpts. Do not use outside knowledge.
- If the excerpts don't contain the answer, say so clearly. Don't guess.
- When you make a claim, cite the source like this: [source: filename.pdf].
- Keep answers concise and factual. No filler.
- If excerpts are contradictory, point that out."""


class RAGAgent:
    """Single-agent RAG: retrieve relevant chunks, then answer using them."""

    def __init__(self):
        self.retriever = Retriever()
        self.llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def answer(self, question: str, top_k: int = 5) -> dict:
        # 1. Retrieve relevant chunks
        chunks = self.retriever.search(question, top_k=top_k)

        # 2. Format chunks into a context block
        context = "\n\n".join([
            f"[Excerpt {i+1} | source: {c['source']}]\n{c['text']}"
            for i, c in enumerate(chunks)
        ])

        # 3. Build the prompt
        user_message = f"""Document excerpts:

{context}

---

Question: {question}

Answer using only the excerpts above. Cite sources inline."""

        # 4. Call the LLM
        response = self.llm.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,  # low for factual Q&A
        )

        answer_text = response.choices[0].message.content

        # 5. Return answer + the sources we used
        return {
            "answer": answer_text,
            "sources": [
                {"source": c["source"], "score": c["score"], "preview": c["text"][:150]}
                for c in chunks
            ],
        }