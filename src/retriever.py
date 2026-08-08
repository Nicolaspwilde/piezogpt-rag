import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

# ----------------------------------------
# Load API Key
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# ----------------------------------------
# Load ChromaDB
# ----------------------------------------

db = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = db.get_collection("piezo_book")

print(f"Database contains {collection.count()} chunks\n")

# ----------------------------------------
# Ask Question
# ----------------------------------------

question = input("Ask a question: ")

# ----------------------------------------
# Embed Question
# ----------------------------------------

response = client.models.embed_content(
    model="gemini-embedding-2",
    contents=question
)

query_embedding = response.embeddings[0].values

# ----------------------------------------
# Search
# ----------------------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
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
        start=1):

    print(f"\nResult {i}")
    print("-" * 80)

    print(f"Page      : {meta['page']}")
    print(f"Distance  : {dist:.4f}")

    print("\nDocument:\n")
    print(doc[:700])

    print("\n" + "-" * 80)
