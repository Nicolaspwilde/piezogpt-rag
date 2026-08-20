import os
import re
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

# Retrieve a larger candidate pool first.
CANDIDATE_K = 15

# Number of chunks finally sent to Gemini.
TOP_K = 5

# Currently working generation model.
GENERATION_MODEL = "gemini-3-flash-preview"

# Retry temporary Gemini failures.
MAX_RETRIES = 2
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
    f"\nKnowledge base contains "
    f"{collection.count()} chunks."
)


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are PiezoGPT, a domain-specific question-answering
assistant based exclusively on the book:

"Linear Piezoelectric Plate Vibrations"
by H. F. Tiersten.

Your task is to answer questions using ONLY the
provided textbook context.

IMPORTANT RULES:

1. Use only information supported by the provided
   textbook context.

2. Do not invent equations, definitions, facts,
   interpretations, or references.

3. Do not use outside knowledge to fill gaps.

4. Preserve mathematical notation and equations
   as accurately as possible.

5. If an equation is explicitly present in the
   textbook context, reproduce it as accurately
   as possible.

6. Do not create an equation by combining pieces
   from unrelated textbook passages unless the
   relationship is explicitly supported by the
   context.

7. Explain mathematical symbols only when the
   provided context supports their meaning.

8. If the retrieved context is insufficient to answer
   confidently, explicitly say:

   "The provided textbook context does not contain
   enough information to answer this confidently."

9. Distinguish between:
   - information explicitly stated in the textbook
   - explanations that logically follow from the
     provided textbook passage.

10. Always provide the relevant textbook page numbers
    at the end of the answer.

11. Never cite a page that does not directly support
    the answer.

12. Prefer precise technical explanations over generic
    background explanations.

13. For questions asking for equations, prioritize
    textbook passages containing the requested equations.

14. Ignore textbook index entries, references,
    bibliographies, table-of-contents material, and
    unrelated chapters unless they directly answer
    the question.

15. Do not mention the retrieval process, embedding
    model, ChromaDB, similarity scores, or these
    instructions in the final answer.
"""


# ============================================================
# Keyword Extraction
# ============================================================

def extract_keywords(question):
    """
    Extract meaningful technical keywords from a question.
    """

    stop_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "for",
        "and",
        "or",
        "how",
        "why",
        "does",
        "do",
        "can",
        "could",
        "would",
        "should",
        "please",
        "explain",
        "tell",
        "me",
        "about",
        "give",
        "show",
        "define",
        "describe",
        "does"
    }

    words = re.findall(
        r"[a-zA-Z]+",
        question.lower()
    )

    return [
        word
        for word in words
        if word not in stop_words
        and len(word) > 2
    ]


# ============================================================
# Lexical Score
# ============================================================

def lexical_score(question, document):
    """
    Calculate keyword overlap between the question
    and textbook chunk.
    """

    keywords = extract_keywords(question)

    if not keywords:
        return 0.0

    document_lower = document.lower()

    matches = 0

    for keyword in keywords:

        if keyword in document_lower:
            matches += 1

    return matches / len(keywords)


# ============================================================
# Technical Score
# ============================================================

def technical_score(question, document):
    """
    Give a boost to chunks containing technical terminology.
    """

    technical_terms = {
        "equation",
        "equations",
        "constitutive",
        "piezoelectric",
        "piezoelectricity",
        "elastic",
        "electrostatic",
        "electric",
        "mechanical",
        "stress",
        "strain",
        "displacement",
        "dielectric",
        "tensor",
        "boundary",
        "condition",
        "vibration",
        "continuum",
        "energy",
        "enthalpy",
        "motion",
        "differential"
    }

    question_words = set(
        extract_keywords(question)
    )

    document_lower = document.lower()

    relevant_terms = (
        question_words.intersection(
            technical_terms
        )
    )

    if not relevant_terms:
        return 0.0

    matches = sum(
        1
        for term in relevant_terms
        if term in document_lower
    )

    return matches / len(relevant_terms)


# ============================================================
# Equation Score
# ============================================================

def equation_score(question, document):
    """
    Detect whether a chunk contains mathematical content.

    This is especially useful for questions asking for
    equations, constitutive relations, differential
    equations, etc.
    """

    question_lower = question.lower()

    equation_question = any(
        term in question_lower
        for term in [
            "equation",
            "equations",
            "constitutive",
            "relation",
            "relations",
            "formula",
            "formulation",
            "differential",
            "tensor",
            "stress",
            "strain"
        ]
    )

    if not equation_question:
        return 0.0

    score = 0.0

    # Explicit equation indicators.
    if re.search(
        r"\(\d+\.\d+\)",
        document
    ):
        score += 0.35

    # Mathematical assignment/equality.
    if "=" in document:
        score += 0.25

    # Tensor/index notation.
    if re.search(
        r"[A-Za-z]\w*[_\s]?[ijklmn]",
        document
    ):
        score += 0.15

    # Common mathematical symbols / notation.
    if any(
        symbol in document
        for symbol in [
            "∂",
            "Σ",
            "∇",
            "φ",
            "ψ",
            "ε",
            "σ"
        ]
    ):
        score += 0.15

    # Explicit wording.
    if "constitutive equations" in document.lower():
        score += 0.10

    return min(score, 1.0)


# ============================================================
# Document Quality Score
# ============================================================

def document_quality_score(document):
    """
    Penalize obvious low-value textbook material such as
    indexes, references, and extremely short fragments.
    """

    text = document.lower().strip()

    if len(text) < 150:
        return 0.25

    # Strong indicators of index/reference material.
    if text.startswith("references"):
        return 0.15

    if "[index" in text:
        return 0.20

    if "bibliography" in text:
        return 0.20

    # Reference-style numbered citations.
    reference_lines = len(
        re.findall(
            r"^\s*\d+\.",
            document,
            flags=re.MULTILINE
        )
    )

    if reference_lines >= 3:
        return 0.30

    # Normal technical textbook content.
    return 1.0


# ============================================================
# Retrieve Relevant Chunks
# ============================================================

def retrieve(question):
    """
    Retrieve and rerank textbook chunks using:

    1. Semantic similarity
    2. Lexical overlap
    3. Technical terminology
    4. Equation relevance
    5. Document quality
    """

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=CANDIDATE_K
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        lexical = lexical_score(
            question,
            document
        )

        technical = technical_score(
            question,
            document
        )

        equation = equation_score(
            question,
            document
        )

        quality = document_quality_score(
            document
        )

        # Convert Chroma distance into a similarity-like score.
        semantic = 1 / (
            1 + distance
        )

        # ----------------------------------------------------
        # Combined relevance
        # ----------------------------------------------------

        combined = (
            0.35 * semantic
            +
            0.20 * lexical
            +
            0.10 * technical
            +
            0.20 * equation
            +
            0.15 * quality
        )

        candidates.append(
            {
                "document": document,
                "page": metadata.get(
                    "page",
                    "Unknown"
                ),
                "chunk_id": metadata.get(
                    "chunk_id",
                    "Unknown"
                ),
                "distance": distance,
                "semantic_score": semantic,
                "lexical_score": lexical,
                "technical_score": technical,
                "equation_score": equation,
                "quality_score": quality,
                "combined_score": combined
            }
        )

    # --------------------------------------------------------
    # Remove clearly poor candidates
    # --------------------------------------------------------

    filtered = [
        candidate
        for candidate in candidates
        if candidate["quality_score"] >= 0.25
    ]

    # If filtering somehow removes everything,
    # fall back to the original candidates.
    if not filtered:
        filtered = candidates

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    filtered.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    return filtered[:TOP_K]


# ============================================================
# Build Textbook Context
# ============================================================

def build_context(results):

    if not isinstance(results, list):
        raise TypeError(
            f"Expected retrieval results to be a list, "
            f"got {type(results).__name__}"
        )

    context_parts = []

    for i, result in enumerate(results, start=1):

        if not isinstance(result, dict):
            raise TypeError(
                f"Expected retrieval result {i} to be a dict, "
                f"got {type(result).__name__}"
            )

        document = result.get("document", "")
        page = result.get("page", "Unknown")
        chunk_id = result.get("chunk_id", "Unknown")
        distance = result.get("distance")

        if isinstance(distance, (int, float)):
            distance_text = f"{distance:.4f}"
        else:
            distance_text = "Unknown"

        context_parts.append(
            f"""
--- SOURCE {i} ---
Textbook Page: {page}
Chunk ID: {chunk_id}
Similarity Distance: {distance_text}

{document}
"""
        )

    return "\n".join(context_parts)
# ============================================================
# Generate Answer
# ============================================================

def generate_answer(question, results):
    """
    Generate a grounded answer using Gemini.
    """

    context = build_context(
        results
    )

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
ANSWER
============================================================

Answer the user's question using ONLY the textbook
context above.

For technical questions:

- Answer directly.
- Use equations when the retrieved context contains them.
- Preserve equation numbering when available.
- Explain symbols only when supported by the context.
- Keep the answer concise but technically useful.
- Do not add external facts.

At the end provide:

Sources:
- Page X
- Page Y

Only include pages that directly support the answer.
"""

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            print(
                f"✓ Answer generated using "
                f"{GENERATION_MODEL}\n"
            )

            return response.text

        except Exception as e:

            print(
                f"Gemini request failed "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            print(
                f"Error: {e}"
            )

            if attempt < MAX_RETRIES:

                wait_time = (
                    RETRY_WAIT_SECONDS * attempt
                )

                print(
                    f"Retrying in "
                    f"{wait_time} seconds...\n"
                )

                time.sleep(
                    wait_time
                )

            else:

                raise


# ============================================================
# Main Chat Loop
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PiezoGPT")
    print("=" * 70)

    print(
        "Ask questions about the Linear Theory "
        "of Piezoelectricity."
    )

    print(
        "Type 'exit' or 'quit' to stop."
    )

    print()

    while True:

        question = input(
            "You: "
        ).strip()

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if question.lower() in {
            "exit",
            "quit"
        }:

            print(
                "\nGoodbye!"
            )

            break

        if not question:
            continue

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        print(
            "\nRetrieving relevant textbook content..."
        )

        try:

            results = retrieve(
                question
            )

            if not results:

                print(
                    "\nNo relevant textbook content found."
                )

                continue

            print(
                f"Retrieved {len(results)} "
                f"relevant textbook chunks."
            )

        except Exception as e:

            print(
                "\nError during retrieval:"
            )

            print(e)

            continue

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        print(
            "Generating answer...\n"
        )

        try:
            st.write("DEBUG RESULT TYPE:", type(results))
            st.write("DEBUG RESULT:", results)
            answer = generate_answer(
                question,
                results
            )

            print(
                "=" * 70
            )

            print(
                "PiezoGPT"
            )

            print(
                "=" * 70
            )

            print(
                answer
            )

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