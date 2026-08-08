import json
import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm

# ----------------------------------------
# Configuration
# ----------------------------------------

MAX_RETRIES = 5
INITIAL_WAIT = 2  # seconds
REQUEST_DELAY = 0.2  # seconds between successful requests

# ----------------------------------------
# Load API Key
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# ----------------------------------------
# Load Chunks
# ----------------------------------------

CHUNKS_FILE = BASE_DIR / "Databank" / "output" / "chunks.json"

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"\nLoaded {len(chunks)} chunks")

# ----------------------------------------
# Create Chroma Database
# ----------------------------------------

db = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = db.get_or_create_collection(
    name="piezo_book"
)

# ----------------------------------------
# Resume Support
# ----------------------------------------

existing = collection.get(include=[])

existing_ids = set(existing["ids"])

print(f"Already stored : {len(existing_ids)}")
print(f"Remaining      : {len(chunks) - len(existing_ids)}\n")

# ----------------------------------------
# Generate Embeddings
# ----------------------------------------

embedded = 0
skipped = 0
failed = 0

for chunk in tqdm(chunks):

    chunk_id = str(chunk["chunk_id"])

    # Skip already processed chunks
    if chunk_id in existing_ids:
        skipped += 1
        continue

    wait_time = INITIAL_WAIT

    embedding = None

    for attempt in range(MAX_RETRIES):

        try:

            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=chunk["text"]
            )

            embedding = response.embeddings[0].values

            break

        except Exception as e:

            print(
                f"\nChunk {chunk_id} | Attempt {attempt + 1}/{MAX_RETRIES}"
            )
            print(e)

            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {wait_time} seconds...\n")
                time.sleep(wait_time)
                wait_time *= 2

    # Failed after all retries
    if embedding is None:
        failed += 1
        print(f"Skipping chunk {chunk_id}\n")
        continue

    # Store in ChromaDB
    collection.add(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[chunk["text"]],
        metadatas=[
            {
                "page": chunk["page"]
            }
        ]
    )

    existing_ids.add(chunk_id)
    embedded += 1

    time.sleep(REQUEST_DELAY)

# ----------------------------------------
# Summary
# ----------------------------------------

print("\n" + "=" * 50)
print("Embedding Complete")
print("=" * 50)
print(f"New Embeddings : {embedded}")
print(f"Skipped        : {skipped}")
print(f"Failed         : {failed}")
print(f"Total in DB    : {collection.count()}")
print("=" * 50)
