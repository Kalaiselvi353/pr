import os

# Base directories (customize as needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")  # Place your PDFs here
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss.index")
META_PATH = os.path.join(OUTPUT_DIR, "metadata.json")

# Embedding / model settings
CLIP_MODEL_NAME = "clip-ViT-B-32"  # Provided by sentence-transformers
EMBED_DIM = 512  # CLIP base model dimension

# Text processing settings
TEXT_CHUNK_SIZE = 300  # characters per chunk
TEXT_CHUNK_OVERLAP = 50  # characters overlap between chunks

# PDF parsing heuristics
HEADER_FOOTER_MAX_LINES = 3  # Top/bottom lines to consider header/footer when repeated
MIN_IMAGE_AREA = 10_000  # Skip images smaller than this pixel area (filter logos)

# Retrieval settings
TOP_K = 5  # default number of results returned for a query

# Flask settings
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True