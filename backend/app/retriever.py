import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "rag-documents"


class Retriever:
    """Wraps Pinecone + the embedding model into a single search interface."""

    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(INDEX_NAME)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top-k most relevant chunks for a query."""
        query_vector = self.embedder.encode(query).tolist()
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )
        chunks = []
        for match in results["matches"]:
            chunks.append({
                "text": match["metadata"]["text"],
                "source": match["metadata"]["source"],
                "score": float(match["score"]),
                "chunk_index": match["metadata"].get("chunk_index", 0),
            })
        return chunks