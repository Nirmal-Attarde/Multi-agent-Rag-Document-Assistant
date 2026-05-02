from .agents.triage import triage
from .agents.retrieval import RetrievalAgent
from .agents.summarization import summarize
from .agents.citation import verify_and_format


class Orchestrator:
    def __init__(self):
        self.retrieval_agent = RetrievalAgent()

    def run(self, query: str) -> dict:
        trace = {"agents": []}

        # 1. Triage
        triage_result = triage(query)
        trace["agents"].append({"agent": "triage", "output": triage_result})

        if not triage_result["should_answer"] or triage_result["category"] == "out_of_scope":
            return {
                "answer": (
                    "This question doesn't appear to be about the indexed research papers. "
                    f"({triage_result['reasoning']})"
                ),
                "sources": [],
                "trace": trace,
            }

        # 2. Retrieval (with query rewriting)
        top_k = 8 if triage_result["category"] == "summarize" else 5
        retrieval_result = self.retrieval_agent.retrieve(query, top_k=top_k, rewrite=True)
        trace["agents"].append({
            "agent": "retrieval",
            "output": {
                "rewritten_query": retrieval_result["rewritten_query"],
                "num_chunks": len(retrieval_result["chunks"]),
                "top_score": retrieval_result["chunks"][0]["score"] if retrieval_result["chunks"] else None,
            },
        })

        chunks = retrieval_result["chunks"]

        # Score-threshold short-circuit: if retrieval is bad, refuse early
        if not chunks or chunks[0]["score"] < 0.30:
            return {
                "answer": "I couldn't find relevant content in the indexed documents to answer this question.",
                "sources": [],
                "trace": trace,
            }

        # 3. Summarization
        draft = summarize(query, chunks)
        trace["agents"].append({"agent": "summarization", "output": {"draft_length": len(draft)}})

        # 4. Citation verification + formatting
        cited = verify_and_format(draft, chunks)
        trace["agents"].append({
            "agent": "citation",
            "output": cited.get("verification"),
        })

        return {
            "answer": cited["answer"],
            "sources": [
                {"source": c["source"], "score": c["score"], "preview": c["text"][:150]}
                for c in chunks
            ],
            "trace": trace,
        }