"""Build vector store (Faiss) from PDFs in the data directory.
Run: python build_corpus.py
"""
from __future__ import annotations

import os
import hashlib
import json
from collections import Counter
from typing import List, Dict, Any

import fitz  # PyMuPDF
import numpy as np
from tqdm import tqdm

import config
from embedding_utils import Embedder, VectorStore


def ensure_dirs():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.IMAGE_DIR, exist_ok=True)


def hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def detect_headers_footers(doc: fitz.Document, max_lines: int = config.HEADER_FOOTER_MAX_LINES):
    """Return sets of header and footer lines (strings) that appear in most pages."""
    header_cnt = Counter()
    footer_cnt = Counter()
    for page in doc:
        blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, text, block_no, block_type)
        # Sort blocks by y0 (top to bottom)
        blocks_sorted = sorted(blocks, key=lambda b: b[1])
        top_lines = []
        bot_lines = []
        # Collect top lines
        collected = 0
        for b in blocks_sorted:
            if collected >= max_lines:
                break
            text = b[4].strip()
            if text:
                header_cnt[text] += 1
                collected += 1
        # Bottom lines - from end
        collected = 0
        for b in reversed(blocks_sorted):
            if collected >= max_lines:
                break
            text = b[4].strip()
            if text:
                footer_cnt[text] += 1
                collected += 1
    min_freq = int(0.6 * len(doc))  # appears in at least 60% of pages
    headers = {t for t, c in header_cnt.items() if c >= min_freq}
    footers = {t for t, c in footer_cnt.items() if c >= min_freq}
    return headers, footers


def process_pdf(pdf_path: str):
    """Extract text chunks and image paths with metadata from a single PDF"""
    doc = fitz.open(pdf_path)
    headers, footers = detect_headers_footers(doc)

    text_chunks: List[Dict[str, Any]] = []
    images_meta: List[Dict[str, Any]] = []

    for page_num, page in enumerate(doc, start=1):
        # --- Text extraction ---
        blocks = page.get_text("blocks")
        blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))  # y0 then x0
        page_text_lines = []
        for b in blocks_sorted:
            text = b[4].strip()
            if not text:
                continue
            if text in headers or text in footers:
                continue  # skip repetitive header/footer
            page_text_lines.append(text)
        page_text = "\n".join(page_text_lines)
        # Chunking
        idx = 0
        while idx < len(page_text):
            chunk = page_text[idx: idx + config.TEXT_CHUNK_SIZE]
            if chunk.strip():
                text_chunks.append(
                    {
                        "type": "text",
                        "content": chunk,
                        "page": page_num,
                        "source": os.path.basename(pdf_path),
                    }
                )
            idx += config.TEXT_CHUNK_SIZE - config.TEXT_CHUNK_OVERLAP

        # --- Image extraction ---
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            width, height = base_image["width"], base_image["height"]
            if width * height < config.MIN_IMAGE_AREA:
                continue  # skip tiny images like logos
            # Save image
            img_hash = hash_bytes(image_bytes)
            ext = base_image.get("ext", "png")
            img_filename = f"{img_hash}.{ext}"
            img_path = os.path.join(config.IMAGE_DIR, img_filename)
            if not os.path.exists(img_path):
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
            images_meta.append(
                {
                    "type": "image",
                    "image_path": img_path,
                    "page": page_num,
                    "source": os.path.basename(pdf_path),
                }
            )
    doc.close()
    return text_chunks, images_meta


def build_corpus():
    ensure_dirs()
    pdf_files = [
        os.path.join(config.PDF_DIR, f)
        for f in os.listdir(config.PDF_DIR)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print(f"No PDFs found in {config.PDF_DIR}. Add files and retry.")
        return

    all_entries: List[Dict[str, Any]] = []
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        txt, imgs = process_pdf(pdf_path)
        all_entries.extend(txt)
        all_entries.extend(imgs)

    # Prepare embeddings
    text_entries = [e for e in all_entries if e["type"] == "text"]
    img_entries = [e for e in all_entries if e["type"] == "image"]

    print(f"Total text chunks: {len(text_entries)}; images: {len(img_entries)}")

    vs = VectorStore(dim=config.EMBED_DIM)

    # Batch embed text
    BATCH = 64
    for i in tqdm(range(0, len(text_entries), BATCH), desc="Embedding text"):
        batch_text = [e["content"] for e in text_entries[i: i+BATCH]]
        vecs = Embedder.embed_text(batch_text)
        vs.add(vecs, text_entries[i: i+BATCH])

    # Batch embed images
    for i in tqdm(range(0, len(img_entries), BATCH), desc="Embedding images"):
        batch_imgs = [e["image_path"] for e in img_entries[i: i+BATCH]]
        vecs = Embedder.embed_images(batch_imgs)
        vs.add(vecs, img_entries[i: i+BATCH])

    # Persist
    vs.save()
    print("Corpus built and saved to", config.OUTPUT_DIR)


if __name__ == "__main__":
    build_corpus()