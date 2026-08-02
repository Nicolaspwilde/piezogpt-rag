# piezogpt-rag

A focused Python RAG (Retrieval-Augmented Generation) pipeline for extracting knowledge from a PDF ("Linear Theory of Piezoelectricity") and preparing it for retrieval and generation. It provides reproducible scripts to extract text, split it into chunks, and scaffold embedding and retrieval steps so you can build a retriever-backed chatbot using Google Gemini or another model.

## Features
- PDF → plain text extraction (page markers preserved)
- Text chunking with overlap (800-char chunks, 150-char overlap by default)
- Examples showing how to call Google Gemini (google-genai) for model listing and generation
- SVG diagrams in `assets/diagrams/` for a visual overview

## Stack
- **Language:** Python 3.x
- **Notable libraries:** PyMuPDF (fitz), langchain_text_splitters, python-dotenv, google-genai

## Quickstart (final-ready)
These commands show the shortest path from a fresh clone to extracting and chunking the PDF.

1. Clone and enter the repository
```bash
git clone https://github.com/Nicolaspwilde/piezogpt-rag.git
cd piezogpt-rag
```

2. (Recommended) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate     # Windows (PowerShell)
```

3. Install dependencies
- Create a `requirements.txt` with the following contents (or run the pip command below):
```
python-dotenv
PyMuPDF
langchain-text-splitters
google-genai
```
- Install:
```bash
pip install -r requirements.txt
# OR
pip install python-dotenv PyMuPDF langchain-text-splitters google-genai
```

4. Configure credentials
- Create a `.env` file at the repo root containing:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```
- The example Gemini scripts (`src/test_gemini.py`, `src/list_models.py`) rely on this env var.

5. Place the PDF to process
- Put your PDF (the repository expects `Linear Theory of Piezoeletricity.pdf`) in the `data/` directory.

6. Run extraction and chunking
```bash
python src/read_data.py     # -> Databank/output/piezo_text.txt
python src/chunk_text.py    # -> Databank/output/chunks.json
```

7. Optional: test Gemini connectivity
```bash
python src/test_gemini.py
```

## Next steps (embeddings, vector store, chatbot)
- `src/create_embeddings.py` — (placeholder) convert `Databank/output/chunks.json` into embeddings using your chosen model/API (e.g., Gemini via google-genai).
- `src/vector_store.py` — (placeholder) build or persist a vector store (FAISS, Chroma, or other) from the embeddings.
- `src/retriever.py` / `src/chatbot.py` — (placeholder) wire retrieval + generation to answer user queries with context from the vector store.

If you want, I can implement a minimal `create_embeddings.py` that:
- loads `chunks.json`,
- calls Gemini to get embeddings per chunk,
- writes `Databank/output/embeddings.json`.

I can also add a small `chatbot.py` example that uses FAISS locally and queries Gemini for answer generation.

## Files of interest
- `src/read_data.py` — PDF → flattened text extraction (uses PyMuPDF)
- `src/chunk_text.py` — text → chunks.json (langchain_text_splitters)
- `src/test_gemini.py`, `src/list_models.py` — example uses of google-genai
- `assets/diagrams/` — `pipeline.svg`, `architecture.svg` for visuals

## Recommendations
- Rename `requirements.txt.txt` (if present) to `requirements.txt` and populate it as shown above.
- Avoid committing large generated outputs in `Databank/output/`. Consider adding `Databank/output/` to `.gitignore` if you want outputs local only.
- Add basic error handling (file-not-found, missing env vars) to the scripts before sharing widely.
- Add LICENSE and CONTRIBUTING.md if you expect external contributions.

## Visuals & diagrams
Pipeline overview and repository architecture diagrams are embedded in this README and stored under `assets/diagrams/`:

![Pipeline overview](assets/diagrams/pipeline.svg)

![Architecture & data flow](assets/diagrams/architecture.svg)

## Contributing
- Open an issue if you want a feature (e.g., FAISS integration, hosted API, web UI).
- PRs are welcome — keep changes focused and include tests/usage notes where appropriate.

## License
No license file is included in the repository. Add a LICENSE (e.g., MIT) if you want to make this project reusable by others.
