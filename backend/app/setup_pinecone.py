import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "rag-documents"
EMBEDDING_DIMENSION = 384  # matches the all-MiniLM-L6-v2 model we'll use

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if INDEX_NAME in existing_indexes:
    print(f"Index '{INDEX_NAME}' already exists. Skipping creation.")
else:
    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index created.")

print("\nAll indexes in your Pinecone account:")
for idx in pc.list_indexes():
    print(f"  - {idx['name']} (dimension: {idx['dimension']})")