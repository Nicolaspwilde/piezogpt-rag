import fitz
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "Linear Theory of Piezoeletricity.pdf"
OUTPUT_PATH = BASE_DIR / "Databank"/ "output" / "piezo_text.txt"

# Create output directory if it doesn't exist
OUTPUT_PATH.parent.mkdir(exist_ok=True)

# Open PDF
document = fitz.open(PDF_PATH)

print("=" * 50)
print("PDF Loaded Successfully")
print(f"Total Pages: {len(document)}")
print("=" * 50)

all_text = []

# Read every page
for page_number, page in enumerate(document, start=1):
    print(f"Reading Page {page_number}")
    text = page.get_text()

    all_text.append(
        f"\n\n========== PAGE {page_number} ==========\n\n{text}"
    )

# Save extracted text
with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    file.write("".join(all_text))

print("\nExtraction Completed!")
print(f"Saved to: {OUTPUT_PATH}")