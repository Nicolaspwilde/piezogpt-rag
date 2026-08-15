import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Models are tried in this order.
# If one is temporarily unavailable, the next one is tried.
GENERATION_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

TOP_K = 5

MAX_RETRIES_PER_MODEL = 2
RETRY_WAIT_SECONDS = 5


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = BASE_DIR / "chroma_db"
ENV_FILE = BASE_DIR / ".env"


# ============================================================
# Load Environment
# ============================================================

load_dotenv(ENV_FILE)

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found.\n"
        f"Expected .env file at:\n{ENV_FILE}"
    )


# ============================================================
# Gemini Client
# ============================================================

client = genai.Client(
    api_key=api_key
)


# ============================================================
# Load Embedding Model
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# Load ChromaDB
# ============================================================

db = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = db.get_collection(
    name="piezo_book"
)

print(
    f"Knowledge base contains "
    f"{collection.count()} chunks.\n"
)


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are PiezoGPT, a domain-specific question-answering
assistant based on the book:

"Linear Piezoelectric Plate Vibrations"
by H. F. Tiersten.

Your job is to answer questions using ONLY the
provided textbook context.

IMPORTANT RULES:

1. Use only information supported by the provided
   textbook context.

2. Do not invent equations, definitions, facts,
   interpretations, or references.

3. Do not rely on outside knowledge to fill gaps.

4. Preserve mathematical notation and equations
   as accurately as possible.

5. When equations are provided, explain the symbols
   only when the provided context supports the
   explanation.

6. If the retrieved context is insufficient to answer
   the question confidently, explicitly say:

   "The provided textbook context does not contain
   enough information to answer this confidently."

7. Distinguish clearly between equations explicitly
   stated in the textbook and explanations derived
   from those equations.

8. Always provide the relevant textbook page numbers
   at the end of the answer.

9. Do not cite pages that do not actually support
   the answer.

10. Prefer precise technical explanations over
    generic explanations.
"""


# ============================================================
# Retrieve Relevant Context
# ============================================================

def retrieve_context(question):
    """
    Convert the user question into an embedding and
    retrieve the most relevant textbook chunks.
    """

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        page = metadata.get(
            "page",
            "Unknown"
        )

        chunk_id = metadata.get(
            "chunk_id",
            "Unknown"
        )

        context_parts.append(
            f"""
--- SOURCE {i} ---
Page: {page}
Chunk ID: {chunk_id}
Similarity Distance: {distance:.4f}

{document}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# Generate Answer
# ============================================================

def generate_answer(question, context):
    """
    Send the retrieved textbook context to Gemini.

    Models are attempted in order. Temporary failures
    automatically trigger retries and model fallback.
    """

    prompt = f"""
{SYSTEM_PROMPT}

============================================================
TEXTBOOK CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
ANSWER INSTRUCTIONS
============================================================

Answer the question using the textbook context above.

Structure the answer clearly.

For technical questions:

- State the answer directly.
- Include relevant equations when available.
- Explain the equations using only supported context.
- Avoid unnecessary general background.
- Clearly acknowledge when the context is insufficient.

At the end, include:

Sources:
- Page X
- Page Y

Only include pages that directly support your answer.
"""

    # --------------------------------------------------------
    # Try each Gemini model
    # --------------------------------------------------------

    for model_name in GENERATION_MODELS:

        print(
            f"Trying generation model: {model_name}"
        )

        # ----------------------------------------------------
        # Retry individual model
        # ----------------------------------------------------

        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                # ------------------------------------------------
                # Validate response
                # ------------------------------------------------

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                print(
                    f"✓ Answer generated using "
                    f"{model_name}\n"
                )

                return response.text

            except Exception as e:

                print(
                    f"Model {model_name} failed "
                    f"(attempt "
                    f"{attempt}/{MAX_RETRIES_PER_MODEL})"
                )

                print(f"Error: {e}")

                # --------------------------------------------
                # Retry
                # --------------------------------------------

                if attempt < MAX_RETRIES_PER_MODEL:

                    wait_time = (
                        RETRY_WAIT_SECONDS * attempt
                    )

                    print(
                        f"Retrying in "
                        f"{wait_time} seconds...\n"
                    )

                    time.sleep(wait_time)

        # ----------------------------------------------------
        # Current model failed completely
        # ----------------------------------------------------

        print(
            f"\nModel {model_name} unavailable."
        )

        print(
            "Trying next generation model...\n"
        )

    # --------------------------------------------------------
    # All models failed
    # --------------------------------------------------------

    raise RuntimeError(
        "All configured Gemini generation models failed.\n"
        f"Models attempted: {GENERATION_MODELS}"
    )


# ============================================================
# Main Chat Loop
# ============================================================

def main():

    print("=" * 70)
    print("PiezoGPT")
    print("=" * 70)

    print(
        "Ask questions about the Linear Theory "
        "of Piezoelectricity."
    )

    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:

        # ----------------------------------------------------
        # Get question
        # ----------------------------------------------------

        question = input("You: ").strip()

        if question.lower() in {
            "exit",
            "quit"
        }:

            print("\nGoodbye!")
            break

        if not question:
            continue

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        print(
            "\nRetrieving relevant textbook content..."
        )

        try:

            context = retrieve_context(
                question
            )

        except Exception as e:

            print(
                "\nError during retrieval:"
            )

            print(e)

            continue

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        print(
            "Generating answer...\n"
        )

        try:

            answer = generate_answer(
                question,
                context
            )

            print("=" * 70)
            print("PiezoGPT")
            print("=" * 70)

            print(answer)

            print()

        except Exception as e:

            print(
                "\nError generating answer:"
            )

            print(e)

            print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()