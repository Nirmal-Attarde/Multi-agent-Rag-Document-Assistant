import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "rag-documents"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

# Generic warm-up queries that should work on most research papers.
# After running once, edit these to match YOUR specific papers.
queries = [
    "What causes heart attack?",
]

for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 70)
    query_vector = embedder.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    for i, match in enumerate(results["matches"], 1):
        source = match["metadata"]["source"]
        text = match["metadata"]["text"][:200].replace("\n", " ")
        print(f"{i}. [score={match['score']:.3f}] {source}")
        print(f"   {text}...")