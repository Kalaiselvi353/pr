import json
import os
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

import config


class Embedder:
    """Utility class that wraps the CLIP model for both text and image embedding."""

    _model: SentenceTransformer = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            cls._model = SentenceTransformer(config.CLIP_MODEL_NAME)
        return cls._model

    @classmethod
    def embed_text(cls, texts: List[str]) -> np.ndarray:
        model = cls.get_model()
        with torch.no_grad():
            emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")

    @classmethod
    def embed_images(cls, image_paths: List[str]) -> np.ndarray:
        model = cls.get_model()
        images = [Image.open(p).convert("RGB") for p in image_paths]
        with torch.no_grad():
            emb = model.encode(images, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")


class VectorStore:
    """Simple Faiss-backed vector store combining text and image vectors."""

    def __init__(self, dim: int = config.EMBED_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity (after normalization)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        assert len(vectors) == len(metadatas)
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = config.TOP_K):
        if len(self.metadata) == 0:
            return []
        D, I = self.index.search(query_vector.astype("float32"), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results

    def save(self, index_path: str = config.INDEX_PATH, meta_path: str = config.META_PATH):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str = config.INDEX_PATH, meta_path: str = config.META_PATH):
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Index or metadata file not found. Run build_corpus.py first.")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)