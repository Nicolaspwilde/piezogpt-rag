import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieve a larger candidate pool first.
CANDIDATE_K = 20

# Number of results displayed.
TOP_K = 5

# Minimum final score.
MIN_SCORE = 0.35


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = BASE_DIR / "chroma_db"


# ============================================================
# Load embedding model
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
    f"\nDatabase contains "
    f"{collection.count()} chunks"
)


# ============================================================
# Keyword extraction
# ============================================================

def extract_keywords(question):
    """
    Extract meaningful technical keywords from the question.
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
        "define",
        "definition",
        "mean",
        "means"
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
# Lexical relevance
# ============================================================

def lexical_score(question, document):
    """
    Measure how many important question keywords
    appear in the document.
    """

    keywords = extract_keywords(question)

    if not keywords:
        return 0.0

    document_lower = document.lower()

    matches = 0

    for keyword in keywords:

        # Word-boundary matching prevents things such as
        # "piezo" matching unrelated longer words.
        if re.search(
            rf"\b{re.escape(keyword)}\b",
            document_lower
        ):
            matches += 1

    return matches / len(keywords)


# ============================================================
# Technical phrase relevance
# ============================================================

def technical_phrase_score(question, document):
    """
    Detect important multi-word technical phrases.

    Exact phrases are particularly useful for technical
    textbook questions because semantic similarity alone
    can retrieve conceptually related but less useful pages.
    """

    question_lower = question.lower()
    document_lower = document.lower()

    phrases = [
        "constitutive equation",
        "constitutive equations",
        "linear piezoelectricity",
        "piezoelectric constitutive",
        "differential equations",
        "equations of piezoelectricity",
        "linear theory",
        "piezoelectric constants",
        "electric enthalpy",
        "conservation of energy",
        "stress equation",
        "charge equation"
    ]

    matched = 0

    for phrase in phrases:

        if phrase in question_lower:

            if phrase in document_lower:
                matched += 1

    relevant_phrases = [
        phrase
        for phrase in phrases
        if phrase in question_lower
    ]

    if not relevant_phrases:
        return 0.0

    return matched / len(relevant_phrases)


# ============================================================
# Equation relevance
# ============================================================

def equation_score(question, document):
    """
    Detect whether a document contains mathematical
    equations.

    This is especially important for questions asking
    for equations, formulas, or constitutive relations.
    """

    question_lower = question.lower()
    document_lower = document.lower()

    equation_words = [
        "equation",
        "equations",
        "formula",
        "formulas",
        "expression",
        "expressions",
        "constitutive",
        "relation",
        "relations"
    ]

    asks_for_equation = any(
        word in question_lower
        for word in equation_words
    )

    if not asks_for_equation:
        return 0.0

    score = 0.0

    # --------------------------------------------------------
    # Explicit equation numbering
    # Example:
    #
    # (5.19)
    # (5.20)
    # --------------------------------------------------------

    if re.search(
        r"\(\d+\.\d+\)",
        document
    ):
        score += 0.35

    # --------------------------------------------------------
    # Tensor / constitutive notation
    # --------------------------------------------------------

    equation_patterns = [
        r"\bTij\s*=",
        r"\bDi\s*=",
        r"\bSkl\s*=",
        r"\bEk\s*=",
        r"\beikl",
        r"\beij",
        r"\bCijkl",
        r"\bC_",
        r"\bD\s*=",
        r"\bT\s*=",
    ]

    pattern_hits = 0

    for pattern in equation_patterns:

        if re.search(
            pattern,
            document
        ):
            pattern_hits += 1

    if pattern_hits > 0:

        score += min(
            0.65,
            pattern_hits * 0.13
        )

    return min(score, 1.0)


# ============================================================
# Content quality
# ============================================================

def content_quality_score(document):
    """
    Estimate whether a chunk contains useful explanatory
    textbook content rather than an index, reference,
    heading, or navigation page.
    """

    text = document.strip()
    text_lower = text.lower()

    # --------------------------------------------------------
    # Obvious junk
    # --------------------------------------------------------

    if text_lower.startswith("references"):
        return 0.0

    if text_lower.startswith("index"):
        return 0.0

    if text_lower.startswith("[index"):
        return 0.0

    if "table of contents" in text_lower:
        return 0.0

    # --------------------------------------------------------
    # Index-style content
    # --------------------------------------------------------

    index_hits = 0

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if re.search(
            r",\s*\d+(ff)?\s*$",
            line.lower()
        ):
            index_hits += 1

        elif re.search(
            r",\s*\d+-\d+",
            line.lower()
        ):
            index_hits += 1

    if index_hits >= 5:
        return 0.0

    if index_hits >= 3:
        return 0.25

    # --------------------------------------------------------
    # Reference-style content
    # --------------------------------------------------------

    reference_hits = 0

    reference_patterns = [
        r"^\d+\.\s+[A-Z]",
        r"\bphys\.\s+rev\.",
        r"\bj\.\s+acoust\.",
        r"\bieee\s+trans",
        r"\bmcgraw-hill\b"
    ]

    for pattern in reference_patterns:

        if re.search(
            pattern,
            text_lower,
            flags=re.MULTILINE
        ):
            reference_hits += 1

    if reference_hits >= 3:
        return 0.0

    # --------------------------------------------------------
    # Very short heading chunks
    # --------------------------------------------------------

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if (
        len(lines) <= 5
        and len(text) < 250
    ):
        return 0.25

    # --------------------------------------------------------
    # Part / chapter title-only pages
    # --------------------------------------------------------

    if (
        "part i" in text_lower
        and len(text) < 300
    ):
        return 0.25

    if (
        "part ii" in text_lower
        and len(text) < 300
    ):
        return 0.25

    if (
        "part iii" in text_lower
        and len(text) < 300
    ):
        return 0.25

    # --------------------------------------------------------
    # Good normal content
    # --------------------------------------------------------

    return 1.0


# ============================================================
# Retrieve
# ============================================================

def retrieve(question):

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        lexical = lexical_score(
            question,
            document
        )

        technical = technical_phrase_score(
            question,
            document
        )

        equation = equation_score(
            question,
            document
        )

        quality = content_quality_score(
            document
        )

        # ----------------------------------------------------
        # Semantic similarity
        #
        # Lower Chroma distance = better.
        # ----------------------------------------------------

        semantic = 1 / (
            1 + distance
        )

        # ----------------------------------------------------
        # Combined score
        #
        # Semantic remains dominant.
        #
        # Lexical:
        # exact terminology.
        #
        # Technical:
        # exact technical phrases.
        #
        # Equation:
        # mathematical relevance.
        #
        # Quality:
        # suppress junk pages.
        # ----------------------------------------------------

        combined = (
            0.50 * semantic
            +
            0.15 * lexical
            +
            0.15 * technical
            +
            0.20 * equation
        )

        # Apply content quality as a multiplier.
        combined *= quality

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
    # Sort
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Filter weak candidates
    # --------------------------------------------------------

    filtered = [
        candidate
        for candidate in candidates
        if candidate["combined_score"] >= MIN_SCORE
    ]

    # --------------------------------------------------------
    # Return final results
    # --------------------------------------------------------

    return filtered[:TOP_K]


# ============================================================
# Interactive testing
# ============================================================

def main():

    while True:

        question = input(
            "\nAsk a question: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit"
        }:

            print("\nGoodbye!")
            break

        if not question:
            continue

        print(
            "\n" + "=" * 90
        )

        print(
            "Top Retrieved Chunks"
        )

        print(
            "=" * 90
        )

        results = retrieve(question)

        if not results:

            print(
                "\nNo relevant textbook content found."
            )

            continue

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nResult {i}"
            )

            print(
                "-" * 90
            )

            print(
                f"Page            : "
                f"{result['page']}"
            )

            print(
                f"Chunk ID         : "
                f"{result['chunk_id']}"
            )

            print(
                f"Chroma distance  : "
                f"{result['distance']:.4f}"
            )

            print(
                f"Semantic score   : "
                f"{result['semantic_score']:.4f}"
            )

            print(
                f"Lexical score    : "
                f"{result['lexical_score']:.4f}"
            )

            print(
                f"Technical score  : "
                f"{result['technical_score']:.4f}"
            )

            print(
                f"Equation score   : "
                f"{result['equation_score']:.4f}"
            )

            print(
                f"Quality score    : "
                f"{result['quality_score']:.4f}"
            )

            print(
                f"Combined score   : "
                f"{result['combined_score']:.4f}"
            )

            print("\nDocument:\n")

            print(
                result["document"]
            )


if __name__ == "__main__":
    main()