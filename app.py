import os
import json
import time
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_from_directory, render_template
from dotenv import load_dotenv

import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


load_dotenv()


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(APP_ROOT, "storage")
STATIC_DIR = os.path.join(APP_ROOT, "static")
CROPS_DIR = os.path.join(STATIC_DIR, "crops")
TEMPLATES_DIR = os.path.join(APP_ROOT, "templates")

PDF_DIR = os.environ.get("PDF_DIR", os.path.join(APP_ROOT, "pdfs"))

INDEX_PATH = os.path.join(STORAGE_DIR, "clip.index")
ITEMS_PATH = os.path.join(STORAGE_DIR, "items.json")


os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class ClipSearch:
    def __init__(self, index_path: str, items_path: str):
        self.index_path = index_path
        self.items_path = items_path
        self.model = None
        self.processor = None
        self.index = None
        self.items: List[Dict[str, Any]] = []
        self._load_model()
        self._load_index()

    def _load_model(self) -> None:
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device())
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _load_index(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.items_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.items_path, "r", encoding="utf-8") as f:
                self.items = json.load(f)
        else:
            self.index = None
            self.items = []

    def has_index(self) -> bool:
        return self.index is not None and len(self.items) > 0

    @torch.no_grad()
    def embed_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(_device()) for k, v in inputs.items()}
        outputs = self.model.get_text_features(**inputs)
        vectors = outputs.detach().cpu().numpy().astype("float32")
        faiss.normalize_L2(vectors)
        return vectors

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        if not self.has_index():
            return []
        qv = self.embed_text([query])
        scores, idxs = self.index.search(qv, top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            item = self.items[idx]
            results.append({"score": float(score), **item})
        return results


clip_search = ClipSearch(INDEX_PATH, ITEMS_PATH)


def ensure_index_ready() -> None:
    """Build the index at runtime if it does not exist."""
    if clip_search.has_index():
        return
    # Defer heavy import to runtime only if needed
    from build_corpus import build_index
    build_index(pdf_dir=PDF_DIR, storage_dir=STORAGE_DIR, crops_dir=CROPS_DIR)
    clip_search._load_index()


app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)


@app.route("/")
def index() -> Any:
    ensure_index_ready()
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query() -> Any:
    ensure_index_ready()
    data = request.get_json(force=True)
    user_query = (data or {}).get("query", "").strip()
    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    raw_results = clip_search.search(user_query, top_k=25)

    # Collate text and images sensibly
    response_items: List[Dict[str, Any]] = []
    seen_text_ids = set()
    seen_image_ids = set()

    for item in raw_results:
        if item["type"] == "text":
            if item["id"] in seen_text_ids:
                continue
            seen_text_ids.add(item["id"])
            images = [
                {"url": f"/static/crops/{img['file']}", "page": img.get("page"), "bbox": img.get("bbox")}
                for img in item.get("linked_images", [])[:4]
            ]
            response_items.append({
                "kind": "text",
                "text": item.get("content", ""),
                "source": {"page": item.get("page"), "bbox": item.get("bbox")},
                "images": images,
            })
        elif item["type"] == "image":
            if item["id"] in seen_image_ids:
                continue
            seen_image_ids.add(item["id"])
            response_items.append({
                "kind": "image",
                "image": {"url": f"/static/crops/{item['file']}", "page": item.get("page"), "bbox": item.get("bbox")},
                "caption": item.get("nearest_text", ""),
            })

    # Trim to a reasonable payload
    if len(response_items) > 12:
        response_items = response_items[:12]

    return jsonify({
        "results": response_items,
        "meta": {"returned": len(response_items), "total_candidates": len(raw_results)},
    })


@app.route("/reindex", methods=["POST"])  # optional rebuild endpoint
def reindex() -> Any:
    start = time.time()
    from build_corpus import build_index
    build_index(pdf_dir=PDF_DIR, storage_dir=STORAGE_DIR, crops_dir=CROPS_DIR, force_rebuild=True)
    clip_search._load_index()
    return jsonify({"status": "ok", "took_sec": round(time.time() - start, 2)})


@app.route("/static/crops/<path:filename>")
def serve_crop(filename: str):
    return send_from_directory(CROPS_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

