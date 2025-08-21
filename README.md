AI PDF Analyst - Text+Image Search Chat

Overview
- Joint text+image retrieval over PDFs using CLIP (openai/clip-vit-base-patch32) and FAISS.
- Heuristics remove headers/footers and logo-like images; crops useful figures/tables.
- Maps each cropped image to its nearest text on the same page.
- WhatsApp-like UI over Flask. Index builds automatically on first request or via CLI.

Quick Start
1) Place PDFs under the directory pointed to `PDF_DIR` (default: `/workspace/pdfs`).
   - Copy `.env.example` to `.env` and edit `PDF_DIR` as needed.
2) Install dependencies:
   - `pip install -r requirements.txt`
3) Run the app:
   - `python app.py`
   - First load will build crops and the FAISS index (`storage/clip.index`, `storage/items.json`).

Manual Index Build
```bash
python build_corpus.py --pdf_dir /path/to/pdfs --storage_dir /workspace/storage --crops_dir /workspace/static/crops --force
```

How it Works
- Extraction (`build_corpus.py`):
  - Reads each PDF page with PyMuPDF; collects text blocks excluding header/footer bands and tiny noise.
  - Crops image blocks that are not tiny, not extreme-aspect separators, and not page-repeated logos (MD5-based de-dup + frequency filter).
  - Associates each cropped image to the closest text block by geometric distance on the page.
  - Embeds all text and images with CLIP; vectors are L2-normalized and added to a FAISS Inner Product index.
- Serving (`app.py`):
  - On first request, builds the index if missing.
  - Encodes user query with CLIP text encoder, searches FAISS, and returns matched text with its linked image crops and/or standalone image matches.
- UI (`templates/index.html`):
  - Simple chat view. Sends `/query` POST with the text; renders relevant text and cropped images grid.

Design Notes
- Keeping text and image items separate avoids contamination; retrieval happens jointly via a shared vector space.
- No full-page images are returned—only cropped figures/tables near matched text.
- Logos are filtered via size/aspect and cross-page frequency.

Tips for the Provided Sample PDFs
- Size charts and standard guidelines with tables/images are cropped and indexed as figures; queries like "US finger ring size 6" or "stone weight printing rule" should return the exact blocks and nearby images.
- If a table is entirely vector graphics, it is treated as an image block and retrieved.

Improving Accuracy (optional)
- Add table detection for vector tables with layoutparser/detectron2 to prioritize table crops.
- Add OCR (e.g., easyocr/rapidocr) for embedded/non-extractable text inside images.
- Use a reranker (e.g., bge-reranker) on the top-50 to boost precision without changing the index.

Configuration
- Change `PDF_DIR` in `.env` or environment variables.
- All crops are saved under `static/crops/`; the FAISS index and items metadata under `storage/`.

API
- POST `/query` { "query": "text" } -> { results: [ { kind: "text", text, images: [{url}] }, { kind: "image", image:{url}, caption } ] }
- POST `/reindex` -> forces rebuild.

1.Smart home interior setup 

Link:https://wokwi.com/projects/363034882611849217

2.Smart home outer areas-automated plant watering system

Link:https://wokwi.com/projects/362960322581025793
