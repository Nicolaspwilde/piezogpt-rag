from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ----------------------------------------
# Configuration
# ----------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# ----------------------------------------
# Paths
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = BASE_DIR / "chroma_db"

# ----------------------------------------
# Load Embedding Model
# ----------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.\n")

# ----------------------------------------
# Load ChromaDB
# ----------------------------------------

db = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = db.get_collection(
    name="piezo_book"
)

print(
    f"Database contains {collection.count()} chunks\n"
)

# ----------------------------------------
# Ask Question
# ----------------------------------------

question = input("Ask a question: ").strip()

if not question:
    print("No question provided.")
    exit()

# ----------------------------------------
# Embed Question
# ----------------------------------------

query_embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

# ----------------------------------------
# Search ChromaDB
# ----------------------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K
)

# ----------------------------------------
# Display Results
# ----------------------------------------

print("\n" + "=" * 80)
print("Top Retrieved Chunks")
print("=" * 80)

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

for i, (doc, meta, dist) in enumerate(
    zip(documents, metadatas, distances),
    start=1
):

    print(f"\nResult {i}")
    print("-" * 80)

    print(f"Page      : {meta['page']}")
    print(f"Chunk ID   : {meta.get('chunk_id', 'N/A')}")
    print(f"Distance  : {dist:.4f}")

    print("\nDocument:\n")
    print(doc[:700])

    print("\n" + "-" * 80)