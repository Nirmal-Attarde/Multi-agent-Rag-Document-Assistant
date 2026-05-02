# import os
# import glob
# import uuid
# from dotenv import load_dotenv
# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer
# from pypdf import PdfReader

# load_dotenv()

# INDEX_NAME = "rag-documents"
# DOCUMENTS_DIR = "data/documents"

# # Tuned for research papers: ~150 words per chunk with ~30 word overlap.
# # Big enough to capture a full thought; small enough to be precise.
# CHUNK_SIZE = 1000      # characters per chunk (~150-200 words)
# CHUNK_OVERLAP = 200    # characters of overlap (~30-40 words)

# BATCH_SIZE = 100       # vectors per Pinecone upsert call (stays under 4MB)


# def chunk_text(text: str, size: int, overlap: int) -> list[str]:
#     """Split text into overlapping character chunks."""
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + size
#         chunk = text[start:end].strip()
#         if chunk:
#             chunks.append(chunk)
#         start = end - overlap
#     return chunks


# def read_pdf(path: str) -> str:
#     """Extract all text from a PDF file."""
#     reader = PdfReader(path)
#     pages = []
#     for page in reader.pages:
#         text = page.extract_text() or ""
#         pages.append(text)
#     return "\n\n".join(pages)


# def read_txt(path: str) -> str:
#     """Read a plain text file, tolerating odd encodings."""
#     with open(path, "r", encoding="utf-8", errors="replace") as f:
#         return f.read()


# def load_documents(directory: str) -> list[dict]:
#     """Read every supported file in a directory."""
#     docs = []
#     paths = sorted(glob.glob(f"{directory}/*"))
#     for path in paths:
#         filename = os.path.basename(path)
#         if filename.startswith("."):
#             continue  # skip hidden files

#         ext = os.path.splitext(filename)[1].lower()
#         try:
#             if ext == ".pdf":
#                 text = read_pdf(path)
#             elif ext == ".txt":
#                 text = read_txt(path)
#             else:
#                 print(f"  skipping unsupported file: {filename}")
#                 continue
#         except Exception as e:
#             print(f"  failed to read {filename}: {e}")
#             continue

#         if not text.strip():
#             print(f"  warning: {filename} produced no text (scanned PDF?)")
#             continue

#         docs.append({"source": filename, "text": text, "char_count": len(text)})
#     return docs


# def main():
#     print("Loading embedding model...")
#     embedder = SentenceTransformer("all-MiniLM-L6-v2")

#     print("Connecting to Pinecone...")
#     pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#     index = pc.Index(INDEX_NAME)

#     print("Clearing existing vectors from index...")
#     try:
#         index.delete(delete_all=True)
#         print("  cleared.")
#     except Exception as e:
#         print(f"  (nothing to clear: {type(e).__name__})")

#     print(f"\nLoading documents from '{DOCUMENTS_DIR}'...")
#     documents = load_documents(DOCUMENTS_DIR)
#     print(f"Found {len(documents)} document(s).")

#     vectors_to_upsert = []
#     for doc in documents:
#         chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
#         print(f"  {doc['source']}: {doc['char_count']:,} chars -> {len(chunks)} chunk(s)")

#         # Embed in one batch per document for efficiency
#         embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

#         for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
#             vectors_to_upsert.append({
#                 "id": str(uuid.uuid4()),
#                 "values": embedding,
#                 "metadata": {
#                     "source": doc["source"],
#                     "chunk_index": i,
#                     "text": chunk,
#                 },
#             })

#     print(f"\nUpserting {len(vectors_to_upsert)} vectors to Pinecone in batches of {BATCH_SIZE}...")
#     for i in range(0, len(vectors_to_upsert), BATCH_SIZE):
#         batch = vectors_to_upsert[i : i + BATCH_SIZE]
#         index.upsert(vectors=batch)
#         print(f"  upserted {i + len(batch):,} / {len(vectors_to_upsert):,}")

#     stats = index.describe_index_stats()
#     print(f"\nDone. Index now contains {stats['total_vector_count']:,} vectors.")


# if __name__ == "__main__":
#     main()



import os
import glob
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

INDEX_NAME = "rag-documents"
DOCUMENTS_DIR = "data/documents"

# RecursiveCharacterTextSplitter splits on paragraph -> sentence -> word boundaries.
# These sizes are tuned for research papers: ~150-200 words per chunk.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

BATCH_SIZE = 100  # vectors per Pinecone upsert call


def load_documents(directory: str) -> list[dict]:
    """Load all supported files using LangChain document loaders.

    Each file produces a list of LangChain Documents (one per PDF page,
    or one for a whole text file). We flatten them and tag each with the
    source filename for later citation.
    """
    docs = []
    paths = sorted(glob.glob(f"{directory}/*"))

    for path in paths:
        filename = os.path.basename(path)
        if filename.startswith("."):
            continue

        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
                pages = loader.load()
            elif ext == ".txt":
                loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)
                pages = loader.load()
            else:
                print(f"  skipping unsupported file: {filename}")
                continue
        except Exception as e:
            print(f"  failed to read {filename}: {e}")
            continue

        for page in pages:
            if page.page_content.strip():
                docs.append({
                    "source": filename,
                    "page": page.metadata.get("page", 0),
                    "text": page.page_content,
                })

    return docs


def main():
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    print("Clearing existing vectors from index...")
    try:
        index.delete(delete_all=True)
        print("  cleared.")
    except Exception as e:
        print(f"  (nothing to clear: {type(e).__name__})")

    print(f"\nLoading documents from '{DOCUMENTS_DIR}'...")
    raw_docs = load_documents(DOCUMENTS_DIR)
    print(f"Found {len(raw_docs)} page/section(s) across documents.")

    # Initialize LangChain's smarter text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # try paragraph, then sentence, then word
        length_function=len,
    )

    vectors_to_upsert = []
    chunk_count_by_source = {}

    for doc in raw_docs:
        chunks = splitter.split_text(doc["text"])
        chunk_count_by_source[doc["source"]] = chunk_count_by_source.get(doc["source"], 0) + len(chunks)

        if not chunks:
            continue

        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors_to_upsert.append({
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "source": doc["source"],
                    "page": doc["page"],
                    "chunk_index": i,
                    "text": chunk,
                },
            })

    print("\nChunks per source:")
    for source, count in chunk_count_by_source.items():
        print(f"  {source}: {count} chunk(s)")

    print(f"\nUpserting {len(vectors_to_upsert)} vectors to Pinecone in batches of {BATCH_SIZE}...")
    for i in range(0, len(vectors_to_upsert), BATCH_SIZE):
        batch = vectors_to_upsert[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"  upserted {i + len(batch):,} / {len(vectors_to_upsert):,}")

    stats = index.describe_index_stats()
    print(f"\nDone. Index now contains {stats['total_vector_count']:,} vectors.")


if __name__ == "__main__":
    main()