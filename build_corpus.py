import os
import io
import json
import hashlib
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
import fitz  # PyMuPDF
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _clip_loader():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device())
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def _normalize(vecs: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vecs)
    return vecs


def _is_header_footer(bbox: List[float], page_w: float, page_h: float) -> bool:
    top, bottom = bbox[1], bbox[3]
    margin = max(40.0, page_h * 0.06)
    return top < margin or bottom > page_h - margin


def _small_or_logo_like(bbox: List[float], page_w: float, page_h: float) -> bool:
    # Reject tiny or very thin/long images (likely separators or logos)
    x0, y0, x1, y1 = bbox
    area = (x1 - x0) * (y1 - y0)
    page_area = page_w * page_h
    if area < page_area * 0.01:
        return True
    aspect = (x1 - x0) / max(1.0, (y1 - y0))
    if aspect > 8 or aspect < 0.125:
        return True
    # Top-left corner small images are often brand logos
    if x0 < page_w * 0.2 and y0 < page_h * 0.2 and area < page_area * 0.03:
        return True
    return False


def _save_crop(page: fitz.Page, bbox: List[float], out_dir: str, base_name: str, idx: int, dpi: int = 200) -> str:
    rect = fitz.Rect(*bbox)
    pix = page.get_pixmap(clip=rect, dpi=dpi, annots=False)
    img_path = f"{base_name}_{idx}.png"
    full_path = os.path.join(out_dir, img_path)
    pix.save(full_path)
    return img_path


def extract_pdf(pdf_path: str, crops_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract text blocks and cropped images from PDF with heuristics.

    Returns two lists: text_items and image_items with metadata.
    """
    doc = fitz.open(pdf_path)
    text_items: List[Dict[str, Any]] = []
    image_items: List[Dict[str, Any]] = []

    # Track repeats for images to filter logo-like duplicates
    image_hash_counts: Dict[str, int] = {}
    per_page_text_blocks: Dict[int, List[Dict[str, Any]]] = {}

    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        page_w, page_h = page.rect.width, page.rect.height
        page_dict = page.get_text("dict")

        blocks = page_dict.get("blocks", [])
        cur_page_text_blocks: List[Dict[str, Any]] = []

        # First pass: text blocks
        for b in blocks:
            if b.get("type") == 0:  # text block
                bbox = b.get("bbox")
                if not bbox:
                    continue
                if _is_header_footer(bbox, page_w, page_h):
                    continue
                # Concatenate spans
                texts: List[str] = []
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        s = span.get("text", "").strip()
                        if s:
                            texts.append(s)
                content = " ".join(texts).strip()
                if not content or len(content) < 3:
                    continue
                cur_page_text_blocks.append({
                    "type": "text",
                    "content": content,
                    "page": page_num,
                    "bbox": bbox,
                })

        per_page_text_blocks[page_num] = cur_page_text_blocks
        text_items.extend(cur_page_text_blocks)

        # Second pass: image blocks with cropping
        img_idx = 0
        for b in blocks:
            if b.get("type") == 1:  # image block
                bbox = b.get("bbox")
                if not bbox:
                    continue
                if _small_or_logo_like(bbox, page_w, page_h):
                    continue
                # Crop and hash to de-duplicate
                tmp_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_num}"
                file_name = _save_crop(page, bbox, crops_dir, tmp_name, img_idx)
                img_idx += 1
                with open(os.path.join(crops_dir, file_name), "rb") as f:
                    h = _hash_bytes(f.read())
                image_hash_counts[h] = image_hash_counts.get(h, 0) + 1
                image_items.append({
                    "type": "image",
                    "file": file_name,
                    "hash": h,
                    "page": page_num,
                    "bbox": bbox,
                })

    # Filter repeated small logos (appear on 50%+ pages)
    pages_count = len(doc)
    filtered_images: List[Dict[str, Any]] = []
    for img in image_items:
        h = img["hash"]
        if image_hash_counts.get(h, 0) >= max(3, int(pages_count * 0.5)):
            # likely a logo/decoration; skip
            try:
                os.remove(os.path.join(crops_dir, img["file"]))
            except Exception:
                pass
            continue
        filtered_images.append(img)

    # Map images to nearest text block on the same page
    def rect_center(bb: List[float]) -> Tuple[float, float]:
        return (0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3]))

    for img in filtered_images:
        page_blocks = per_page_text_blocks.get(img["page"], [])
        cx, cy = rect_center(img["bbox"])
        best = None
        best_d = 1e9
        for tb in page_blocks:
            tx, ty = rect_center(tb["bbox"])
            d = (tx - cx) ** 2 + (ty - cy) ** 2
            if d < best_d:
                best_d = d
                best = tb
        if best is not None:
            img["nearest_text"] = best["content"]
            best.setdefault("linked_images", []).append({
                "file": img["file"],
                "page": img["page"],
                "bbox": img["bbox"],
            })

    return text_items, filtered_images


@torch.no_grad()
def _embed_all(text_items: List[Dict[str, Any]], image_items: List[Dict[str, Any]], crops_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    model, processor = _clip_loader()

    # Text embeddings
    texts = [t["content"] for t in text_items]
    if texts:
        text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(_device()) for k, v in text_inputs.items()}
        text_feats = model.get_text_features(**text_inputs).detach().cpu().numpy().astype("float32")
    else:
        text_feats = np.zeros((0, 512), dtype="float32")

    # Image embeddings
    image_tensors: List[torch.Tensor] = []
    image_order: List[Dict[str, Any]] = []
    for img in image_items:
        try:
            pil = Image.open(os.path.join(crops_dir, img["file"]))
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            image_order.append(img)
            image_tensors.append(pil)
        except Exception:
            continue
    if image_tensors:
        image_inputs = processor(images=image_tensors, return_tensors="pt")
        image_inputs = {k: v.to(_device()) for k, v in image_inputs.items()}
        image_feats = model.get_image_features(**image_inputs).detach().cpu().numpy().astype("float32")
    else:
        image_feats = np.zeros((0, 512), dtype="float32")

    # Normalize and combine
    all_vecs = np.concatenate([text_feats, image_feats], axis=0)
    _normalize(all_vecs)

    items: List[Dict[str, Any]] = []
    # Assign IDs and serialize minimal metadata
    for i, t in enumerate(text_items):
        item = {"id": f"text_{i}", "type": "text", "content": t["content"], "page": t["page"], "bbox": t["bbox"], "linked_images": t.get("linked_images", [])}
        items.append(item)
    base = len(text_items)
    for j, img in enumerate(image_order):
        item = {"id": f"image_{j}", "type": "image", "file": img["file"], "page": img["page"], "bbox": img["bbox"], "nearest_text": img.get("nearest_text", "")}
        items.append(item)

    return all_vecs, items


def build_index(pdf_dir: str, storage_dir: str, crops_dir: str, force_rebuild: bool = False) -> None:
    ensure_dirs(storage_dir, crops_dir)
    index_path = os.path.join(storage_dir, "clip.index")
    items_path = os.path.join(storage_dir, "items.json")

    if os.path.exists(index_path) and os.path.exists(items_path) and not force_rebuild:
        return

    pdf_paths = []
    for root, _dirs, files in os.walk(pdf_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, f))

    text_items_all: List[Dict[str, Any]] = []
    image_items_all: List[Dict[str, Any]] = []

    for pdf in pdf_paths:
        t_items, i_items = extract_pdf(pdf, crops_dir)
        text_items_all.extend(t_items)
        image_items_all.extend(i_items)

    vecs, items = _embed_all(text_items_all, image_items_all, crops_dir)

    # Build FAISS index
    if vecs.shape[0] == 0:
        # create empty index with known dim 512
        index = faiss.IndexFlatIP(512)
    else:
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)

    faiss.write_index(index, index_path)
    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build joint text-image index from PDFs")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDFs")
    parser.add_argument("--storage_dir", type=str, default="storage")
    parser.add_argument("--crops_dir", type=str, default=os.path.join("static", "crops"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    build_index(args.pdf_dir, args.storage_dir, args.crops_dir, force_rebuild=args.force)

