import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ----------------------------------------
# Configuration
# ----------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

# ----------------------------------------
# Paths
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    BASE_DIR
    / "Databank"
    / "output"
    / "chunks.json"
)

CHROMA_PATH = BASE_DIR / "chroma_db"

# ----------------------------------------
# Load Chunks
# ----------------------------------------

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("=" * 60)
print("PiezoGPT Vector Database Builder")
print("=" * 60)

print(f"Loaded chunks : {len(chunks)}")
print(f"Embedding model : {EMBEDDING_MODEL}")
print(f"Batch size : {BATCH_SIZE}")
print()

# ----------------------------------------
# Load Local Embedding Model
# ----------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded successfully.")
print()

# ----------------------------------------
# Create Chroma Database
# ----------------------------------------

db = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = db.get_or_create_collection(
    name="piezo_book"
)

# ----------------------------------------
# Resume Support
# ----------------------------------------

existing = collection.get(
    include=[]
)

existing_ids = set(existing["ids"])

remaining_chunks = [
    chunk
    for chunk in chunks
    if str(chunk["chunk_id"]) not in existing_ids
]

print(f"Already stored : {len(existing_ids)}")
print(f"Remaining      : {len(remaining_chunks)}")
print()

if not remaining_chunks:
    print("All chunks are already embedded.")
    print(f"Total in DB : {collection.count()}")
    exit()

# ----------------------------------------
# Generate Embeddings in Batches
# ----------------------------------------

embedded = 0

for start in tqdm(
    range(
        0,
        len(remaining_chunks),
        BATCH_SIZE
    ),
    desc="Embedding batches"
):

    batch = remaining_chunks[
        start:start + BATCH_SIZE
    ]

    texts = [
        chunk["text"]
        for chunk in batch
    ]

    # Generate embeddings locally
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    embeddings = embeddings.tolist()

    # ------------------------------------
    # Prepare Chroma records
    # ------------------------------------

    ids = [
        str(chunk["chunk_id"])
        for chunk in batch
    ]

    documents = [
        chunk["text"]
        for chunk in batch
    ]

    metadatas = [
        {
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "source": chunk.get(
                "source",
                "Linear Theory of Piezoelectricity"
            )
        }
        for chunk in batch
    ]

    # ------------------------------------
    # Store batch
    # ------------------------------------

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    embedded += len(batch)

# ----------------------------------------
# Summary
# ----------------------------------------

print()
print("=" * 60)
print("Embedding Complete")
print("=" * 60)

print(f"New Embeddings : {embedded}")
print(f"Skipped        : {len(existing_ids)}")
print(f"Failed         : 0")
print(f"Total in DB    : {collection.count()}")

print("=" * 60)