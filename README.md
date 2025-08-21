# PDF Knowledge-Base Chatbot

This project turns a folder full of **unstructured PDFs** into an interactive chatbot that can answer questions and return **both relevant text *and* cropped images** exactly as they appear in the source documents — no hallucinations, no missing data.

---
## 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---
## 2. Add your PDFs
Place every PDF you want indexed into:
```
./data/pdfs/
```
(You can change the location in `config.py` → `PDF_DIR`)

---
## 3. Build the vector store
```bash
python build_corpus.py
```
What happens under the hood:
1. **Text extraction** – removes repetitive headers/footers and splits the remaining text into ~300-char overlapping chunks.
2. **Image extraction** – skips tiny images (logos etc.), crops meaningful figures and saves them to `./data/processed/images/`.
3. **Multi-modal embeddings** – every text chunk **and** every image is embedded with the same CLIP model (`ViT-B/32`, 512-dim).
4. **Faiss index** + JSON metadata are written to `./data/processed/faiss.index` and `metadata.json`.

> Re-run whenever you add / update PDFs.

---
## 4. Start the chatbot
```bash
python app.py
```
Then open <http://localhost:8000>.

* Type any question – the backend embeds your query, retrieves the most similar vectors, and sends the matched text & image URLs back to the frontend.
* Images are served straight from `./data/processed/images/` under `/static/images/<file>`.

---
## Project layout
```
├── app.py                 # Flask backend
├── build_corpus.py        # One-time/offline corpus builder
├── config.py              # Central config
├── embedding_utils.py     # CLIP embedder + Faiss store helper
├── requirements.txt       # Python deps
├── templates/
│   └── index.html         # Simple WhatsApp-like UI (CSS/JS inline)
├── data/
│   ├── pdfs/              # <-- put PDFs here
│   └── processed/
│       ├── images/        # auto-cropped images
│       ├── faiss.index    # vectors
│       └── metadata.json  # meta about every vector
└── README.md
```

---
## Customisation notes

* **Change model** – update `CLIP_MODEL_NAME` in `config.py` (must be available in `sentence-transformers`).
* **Header/footer tolerance** – tweak `HEADER_FOOTER_MAX_LINES` and detection threshold.
* **Minimum image size** – raise or lower `MIN_IMAGE_AREA` to include/exclude smaller pictures.
* **Chunk granularity** – `TEXT_CHUNK_SIZE` & `TEXT_CHUNK_OVERLAP`.

---
## Why this works for the supplied PDFs
The sample PDFs you provided combine dense tables, photos, and corporate header/footer elements.

* **Tables & text** – extracted as raw text; preserved line-breaks keep table alignment readable in chat.
* **Photos & sketches** – cropped individually; small logos (<10 000 px²) discarded.
* **Legal headers/footers** – appear on ≥60 % of pages so are automatically filtered.
* **CLIP retrieval** – embeds both natural language questions ("What is the tolerance limit for US ring sizes?") **and** images (if user pastes an image in future extensions) into the *same* space, guaranteeing aligned similarity search.

---
## Next steps / ideas
1. Swap Faiss for a distributed vector DB (Weaviate, Qdrant, etc.) if the corpus grows.
2. Add OCR for scanned PDFs (e.g. `pytesseract`) before text extraction.
3. Enhance image cropping with object-detection (YOLO) to isolate gems/components inside large product shots.
4. Build WhatsApp Web integration using Twilio or Cloud API – the Flask `/chat` endpoint already returns plain JSON.
