import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Configuration
# ============================================================

START_PAGE = 22

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

MIN_PAGE_LENGTH = 50


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEXT_FILE = (
    BASE_DIR
    / "Databank"
    / "output"
    / "piezo_text.txt"
)

OUTPUT_JSON = (
    BASE_DIR
    / "Databank"
    / "output"
    / "chunks.json"
)


# ============================================================
# Read extracted text
# ============================================================

print("Loading extracted text...")

with open(
    TEXT_FILE,
    "r",
    encoding="utf-8"
) as f:

    text = f.read()

print(
    f"Loaded {len(text):,} characters"
)


# ============================================================
# Split into pages
# ============================================================

pattern = r"========== PAGE (\d+) =========="

parts = re.split(
    pattern,
    text
)

# re.split returns:
#
# [
#     "",
#     "1",
#     "page 1 text",
#     "2",
#     "page 2 text",
#     ...
# ]


# ============================================================
# Text splitter
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "; ",
        ", ",
        " ",
        ""
    ]
)


# ============================================================
# Generate chunks
# ============================================================

chunks = []

chunk_id = 1

pages_processed = 0
pages_skipped_front_matter = 0
pages_skipped_empty = 0


for i in range(
    1,
    len(parts),
    2
):

    page_number = int(parts[i])

    page_text = parts[i + 1].strip()


    # --------------------------------------------------------
    # Skip front matter
    # --------------------------------------------------------

    if page_number < START_PAGE:

        pages_skipped_front_matter += 1

        continue


    # --------------------------------------------------------
    # Skip empty / nearly empty pages
    # --------------------------------------------------------

    if len(page_text) < MIN_PAGE_LENGTH:

        pages_skipped_empty += 1

        continue


    pages_processed += 1


    # --------------------------------------------------------
    # Clean excessive whitespace
    # --------------------------------------------------------

    page_text = re.sub(
        r"[ \t]+",
        " ",
        page_text
    )

    page_text = re.sub(
        r"\n{3,}",
        "\n\n",
        page_text
    )

    page_text = page_text.strip()


    # --------------------------------------------------------
    # Split page into chunks
    # --------------------------------------------------------

    page_chunks = splitter.split_text(
        page_text
    )


    # --------------------------------------------------------
    # Store chunks
    # --------------------------------------------------------

    for chunk in page_chunks:

        chunk = chunk.strip()

        if not chunk:
            continue

        chunks.append(
            {
                "chunk_id": chunk_id,
                "page": page_number,
                "text": chunk,
                "char_count": len(chunk)
            }
        )

        chunk_id += 1


# ============================================================
# Save chunks
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunks,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# Statistics
# ============================================================

if chunks:

    character_counts = [
        chunk["char_count"]
        for chunk in chunks
    ]

    average_chunk_size = (
        sum(character_counts)
        / len(character_counts)
    )

    smallest_chunk = min(
        character_counts
    )

    largest_chunk = max(
        character_counts
    )

else:

    average_chunk_size = 0
    smallest_chunk = 0
    largest_chunk = 0


# ============================================================
# Summary
# ============================================================

print()
print("=" * 60)
print("Chunking Complete")
print("=" * 60)

print(
    f"Pages processed              : "
    f"{pages_processed}"
)

print(
    f"Front matter pages skipped   : "
    f"{pages_skipped_front_matter}"
)

print(
    f"Empty pages skipped          : "
    f"{pages_skipped_empty}"
)

print(
    f"Total chunks                 : "
    f"{len(chunks)}"
)

print(
    f"Chunk size                   : "
    f"{CHUNK_SIZE}"
)

print(
    f"Chunk overlap                : "
    f"{CHUNK_OVERLAP}"
)

print(
    f"Average chunk length         : "
    f"{average_chunk_size:.1f} characters"
)

print(
    f"Smallest chunk               : "
    f"{smallest_chunk} characters"
)

print(
    f"Largest chunk                : "
    f"{largest_chunk} characters"
)

print(
    f"Saved to                     : "
    f"{OUTPUT_JSON}"
)

print("=" * 60)