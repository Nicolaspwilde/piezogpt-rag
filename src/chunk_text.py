import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------
# Paths
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TEXT_FILE = BASE_DIR / "Databank"/ "output" / "piezo_text.txt"

OUTPUT_JSON = BASE_DIR / "Databank"/ "output" / "chunks.json"

# ----------------------------
# Read extracted text
# ----------------------------

with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# ----------------------------
# Split into pages
# ----------------------------

pattern = r"========== PAGE (\d+) =========="

parts = re.split(pattern, text)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = []

chunk_id = 1

# re.split returns:
# ["", page_number, page_text, page_number, page_text...]

for i in range(1, len(parts), 2):

    page_number = int(parts[i])

    page_text = parts[i + 1].strip()

    page_chunks = splitter.split_text(page_text)

    for chunk in page_chunks:

        chunks.append(
            {
                "chunk_id": chunk_id,
                "page": page_number,
                "text": chunk
            }
        )

        chunk_id += 1

# ----------------------------
# Save JSON
# ----------------------------

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:

    json.dump(chunks, f, indent=4, ensure_ascii=False)

print("=" * 40)
print(f"Total Chunks : {len(chunks)}")
print(f"Saved to : {OUTPUT_JSON}")
print("=" * 40)